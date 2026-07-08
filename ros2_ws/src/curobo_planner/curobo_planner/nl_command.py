"""Natural-language target CLI: type "go to the bottle" -> the arm plans there.

Resolves a free-text phrase to ONE of the scene's named targets, then publishes
that target name to the planner's ~/command topic (the planner does the cuRobo
planning + execution). Two resolvers:

  * Claude (if ANTHROPIC_API_KEY is set and the `anthropic` package is installed):
    a forced tool-use call constrained to an enum of the scene's target names, so
    the model must return exactly one valid target (or "none").
  * Offline fallback: keyword / substring matching against target names + keywords.
    Works with zero API keys.

Usage:
    ros2 run curobo_planner goto "go to the bottle"     # one-shot
    ros2 run curobo_planner goto                        # interactive prompt
    ros2 run curobo_planner goto --list                 # print known targets
Special phrases pass straight through: "home", or "pose: x y z r p yaw".
"""

import os
import re
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from std_msgs.msg import String

from curobo_planner.scene import load_scene

DEFAULT_MODEL = 'claude-haiku-4-5'  # fast + cheap for one-word intent; override with --model

# Set after a permanent Claude failure (bad key/model) so interactive use
# doesn't re-pay a doomed API round-trip on every phrase.
_claude_disabled = False


def _find_scene_file(explicit):
    if explicit:
        return explicit
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('curobo_planner'), 'config', 'scene.yaml')


def resolve_offline(phrase, scene):
    """Token match phrase -> target name, or None. Word-boundary tokens (not raw
    substrings) so 'stop' doesn't match 'top' and 'interest' doesn't match 'rest'."""
    tokens = set(re.findall(r'[a-z0-9]+', phrase.lower()))
    best, best_score = None, 0
    for t in scene.targets:
        score = 0
        for kw in [t.name.lower()] + t.keywords:
            for sub in re.split(r'[^a-z0-9]+', kw):
                if len(sub) >= 3 and sub in tokens:
                    score = max(score, len(sub))
        if score > best_score:
            best, best_score = t.name, score
    return best


def resolve_claude(phrase, scene, model):
    """Force Claude to pick one target name via constrained tool use. Returns
    a target name, 'home', 'none', or None if Claude is unavailable/failed."""
    global _claude_disabled
    if _claude_disabled or not os.environ.get('ANTHROPIC_API_KEY'):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    names = scene.target_names + ['home', 'none']
    catalog = '\n'.join(
        '- %s: %s (keywords: %s)' % (t.name, t.description, ', '.join(t.keywords))
        for t in scene.targets)
    catalog += '\n- home: retract the arm to its safe home joint pose'
    tool = {
        'name': 'select_target',
        'description': 'Select the single scene target the user wants the robot arm to move to.',
        'strict': True,  # enum becomes a guarantee, not a suggestion
        'input_schema': {
            'type': 'object',
            'properties': {
                'target': {
                    'type': 'string',
                    'enum': names,
                    'description': "The chosen target name, or 'none' if no target fits.",
                },
            },
            'required': ['target'],
            'additionalProperties': False,
        },
    }
    try:
        # Short budget: this is a one-word intent call and a working offline
        # fallback exists — never leave the user staring at a frozen prompt.
        client = anthropic.Anthropic(timeout=10.0, max_retries=1)
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            tools=[tool],
            tool_choice={'type': 'tool', 'name': 'select_target'},
            messages=[{
                'role': 'user',
                'content': (
                    'Robot arm targets:\n%s\n\n'
                    'The user said: "%s"\n'
                    'Pick the single best target.' % (catalog, phrase)),
            }],
        )
    except Exception as exc:
        kind = type(exc).__name__
        if kind in ('AuthenticationError', 'PermissionDeniedError', 'NotFoundError'):
            _claude_disabled = True  # permanent: bad key or bad --model
            print('  (Claude disabled for this session: %s — using offline matching)'
                  % kind, file=sys.stderr)
        else:
            print('  (Claude unreachable: %s — using offline matching)' % kind,
                  file=sys.stderr)
        return None
    for block in resp.content:
        if block.type == 'tool_use' and block.name == 'select_target':
            picked = block.input.get('target')
            return picked if picked in names else None
    return None


class GotoClient(Node):
    def __init__(self):
        super().__init__('goto_client')
        self.pub = self.create_publisher(String, '/curobo_planner/command', 10)
        # Latched QoS to match the planner (so a fast terminal reply isn't lost
        # to discovery timing); the stale latched value is drained in send().
        self.create_subscription(
            String, '/curobo_planner/status', self._status_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._last_status = None

    def _status_cb(self, msg):
        self._last_status = msg.data

    def wait_for_planner(self, timeout=5.0):
        """Block until the planner's command subscription is matched, so the
        (volatile) command isn't published before discovery and dropped."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.pub.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def send(self, target):
        if not self.wait_for_planner():
            return False
        # Absorb the stale latched status (delivered once, at subscription
        # match), THEN clear, THEN publish — so the next status we see belongs
        # to this command. Event-driven, not a fixed sleep: the planner always
        # has a latched value ('ready' at minimum), so the first command waits
        # only for its arrival (~ms) and later commands in an interactive
        # session pay nothing (the previous response already satisfied it).
        deadline = time.monotonic() + 0.3
        while rclpy.ok() and self._last_status is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self._last_status = None
        self.pub.publish(String(data=target))
        return True

    def wait_status(self, timeout=180.0, quiet_timeout=75.0):
        """Wait for the TERMINAL status. Interim statuses ('...' prefix) are
        printed and extend the wait: multi-segment commands (retreat +
        standoff + final approach) execute motions BETWEEN plans, so the
        terminal reply can be a minute-plus out — each interim message
        proves the planner is alive and working this command."""
        hard = time.monotonic() + timeout
        deadline = time.monotonic() + quiet_timeout
        while rclpy.ok() and time.monotonic() < min(hard, deadline):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._last_status is not None:
                s = self._last_status
                self._last_status = None
                if s.startswith('...'):
                    print('  planner: %s' % s)
                    deadline = time.monotonic() + quiet_timeout
                    continue
                return s
        return None


def _resolve(phrase, scene, model):
    """Return (target_name_or_passthrough, how)."""
    low = phrase.strip().lower()
    if low.startswith('pose:'):
        return phrase.strip(), 'passthrough'
    if low in ('check', 'test', 'selftest'):
        return 'check', 'passthrough'
    # "home", "go home", "return home"... all mean the home command.
    if 'home' in re.findall(r'[a-z0-9]+', low):
        return 'home', 'passthrough'
    picked = resolve_claude(phrase, scene, model)
    how = 'claude'
    if picked is None:
        picked = resolve_offline(phrase, scene)
        how = 'offline'
    if picked == 'home':
        return 'home', how
    if not picked or picked == 'none':
        return None, how
    return picked, how


def main(args=None):
    argv = [a for a in (args or sys.argv[1:])]
    model = DEFAULT_MODEL
    scene_file = None
    list_only = False
    rest = []
    it = iter(argv)
    for a in it:
        if a == '--model':
            model = next(it, DEFAULT_MODEL)
        elif a == '--scene':
            scene_file = next(it, None)
        elif a == '--list':
            list_only = True
        else:
            rest.append(a)

    scene = load_scene(_find_scene_file(scene_file))

    if list_only:
        print('Known targets:')
        for t in scene.targets:
            print('  %-16s %s' % (t.name, t.description))
        print('  (also: home, "pose: x y z roll pitch yaw")')
        return

    rclpy.init()
    client = GotoClient()
    try:
        phrases = [' '.join(rest)] if rest else None
        interactive = phrases is None
        if interactive:
            print('Type a command ("go to the bottle", "home", "quit"). Targets: %s'
                  % ', '.join(scene.target_names))
        while rclpy.ok():
            if interactive:
                try:
                    phrase = input('> ').strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if phrase.lower() in ('quit', 'exit', 'q'):
                    break
                if not phrase:
                    continue
            else:
                phrase = phrases[0]

            target, how = _resolve(phrase, scene, model)
            if target is None:
                print('  could not resolve "%s" to a target (%s).' % (phrase, how))
            else:
                if how != 'passthrough':
                    print('  [%s] "%s" -> %s' % (how, phrase, target))
                if not client.send(target):
                    print('  planner not found on /curobo_planner/command '
                          '(is the demo launched?)')
                else:
                    status = client.wait_status()
                    print('  planner: %s' % (status or 'no response (is the planner running?)'))
            if not interactive:
                break
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

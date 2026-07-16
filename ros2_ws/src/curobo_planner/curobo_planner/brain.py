"""The brain: Claude picks tools based on live circumstance — system modulation.

    ros2 run curobo_planner brain        (started by the bringup launch)

Sits between the user's words (/rammp/task) and the planner. Instead of every
command mapping to a fixed authored target, Claude sees the LIVE world (object
positions from perception, geometry from the scene, what's in the gripper)
and picks from a hierarchy of tools — including creating its OWN endpoints
via move_tool, which the planner is free to refuse (the semantic/metric
split: the brain owns intent and coarse geometry, cuRobo owns collision
truth). Inspired by kinova-gemini's tool hierarchy; unlike it, every tool
here returns the planner's REAL verdict plus a fresh world snapshot, so the
model is never reasoning about a stale or imagined world.

Degrades honestly: without ANTHROPIC_API_KEY (or the anthropic package) the
brain forwards task text verbatim to the planner — exactly the pre-brain
behavior. STOP words never touch the API: forwarded to the planner
immediately, mid-task included.
"""

import json
import os
import re
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from std_msgs.msg import String

from curobo_planner.scene import load_scene

STOP_WORDS = {'stop', 'halt', 'freeze', 'cancel', 'estop'}
MAX_STEPS = 12          # tool calls per task — a runaway-loop backstop
PLANNER_TIMEOUT = 180.0

SYSTEM = """You are the decision layer of RAMMP, an assistive robot arm \
(Kinova Gen3 7-DoF + Robotiq 2F-85 gripper) that helps a person with motor \
disabilities perform daily-living tasks in their kitchen. You receive a task \
in their words plus a live snapshot of the world, and you act by calling \
tools. Be efficient and safe: every motion costs the user time, and unneeded \
motion erodes trust.

FRAME AND WORKSPACE (metres, base frame: arm base at origin, +z up, +x \
toward the back wall): the arm stands on a kitchen island whose top is at \
z=-0.07 — objects on the island have their base there. Reachable endpoints: \
horizontal distance 0.30-0.72 from the origin. orientation='down' (tool \
pointing down, for tabletop work) is reliable for z between -0.05 and 0.35; \
orientation='forward' (tool horizontal) for z between 0.25 and 0.60 (shelf \
and handle heights). Above z~0.35 tool-down becomes unreachable.

DIVISION OF LABOUR: you decide WHAT to do and roughly WHERE; the motion \
planner (cuRobo) owns collision truth and may REFUSE any endpoint. A refusal \
message is ground truth about the world, not an error to fight — adjust \
your approach ONCE (different point, different tool, different order); if it \
fails again, stop and report honestly with task_complete or ask_user.

RULES:
- Prefer reach/grasp: they use live perception and field-proven approach \
poses. Use move_tool for endpoints no tool covers (hovering near something, \
handover positions, pre-placement poses, nudging clear of clutter).
- GRASP STRATEGY: before grasping, call look(object) — you will SEE the \
object through the wrist camera. From the image, choose the most easily \
graspable PART (a handle, a rim, a narrow neck — or the whole body if it \
is small) and pass its bounding box to grasp as part_box \
[ymin, xmin, ymax, xmax], normalized 0-1000 on that exact image. Reason \
from what you see, not from what objects usually look like. If a grasp \
MISSED, look again before retrying — the object may have moved.
- The collision world is PADDED ~2 cm: a move_tool endpoint within ~3 cm \
of any object will be REFUSED (IK_FAIL) — that is honesty, not an error. \
Only reach/grasp may get close to an object (they carry exemptions). To \
place something down, release from 5-8 cm above the surface instead of \
trying to touch it.
- The world snapshot after each tool call is fresh perception — trust it \
over your memory. Objects can move; positions are live.
- While holding an object, plan where it goes BEFORE picking it up. To set \
something down: move_tool to 5-8 cm above the surface, then release.
- Never stack more than one retry on the same failing idea.
- NEVER ask the user questions. You work autonomously: when blocked after \
a retry, end with task_complete stating exactly what failed, what you \
tried, and what would unblock it. An honest failure report is the correct \
ending — a question is not.
- Speak briefly through say() at meaningful moments; always finish with \
task_complete (honest summary, including failures)."""

TOOLS = [
    {'name': 'reach', 'description':
        'Move the gripper to a named object or known place using live '
        'perception and the proven approach pose for it. Does not grasp.',
     'input_schema': {'type': 'object', 'required': ['name'], 'properties': {
         'name': {'type': 'string', 'description':
                  'object or target name, e.g. bottle, mug, cabinet_handle, '
                  'shelf_edge, pills, rest'}}}},
    {'name': 'look', 'description':
        'Point the wrist camera at an object and receive the IMAGE. Use it '
        'to choose the most easily graspable part before grasping.',
     'input_schema': {'type': 'object', 'required': ['object'], 'properties': {
         'object': {'type': 'string'}}}},
    {'name': 'grasp', 'description':
        'Pick up a free object: plan a grasp from live perception (point-'
        'cloud proposals + geometric fallback), close the gripper, verify '
        'the hold, lift 12 cm. Returns MISSED honestly if the gripper '
        'closed on air. If you called look first, pass part_box to aim the '
        'grasp at the part you chose in that image.',
     'input_schema': {'type': 'object', 'required': ['object'], 'properties': {
         'object': {'type': 'string'},
         'part_box': {'type': 'array', 'items': {'type': 'integer'},
                      'minItems': 4, 'maxItems': 4, 'description':
                      '[ymin, xmin, ymax, xmax] of the graspable part, '
                      'normalized 0-1000 on the look image'},
         'part_name': {'type': 'string', 'description':
                       'one word: which part the box covers'}}}},
    {'name': 'release', 'description':
        'Open the gripper, letting go of whatever it holds, at the current '
        'position. Position the tool first with move_tool.',
     'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'move_tool', 'description':
        'Move the gripper fingertip midpoint to an endpoint YOU choose '
        '(base frame, metres). The planner may refuse it — the refusal text '
        'says why. Use for custom positions no other tool covers.',
     'input_schema': {'type': 'object',
                      'required': ['x', 'y', 'z', 'orientation', 'why'],
                      'properties': {
         'x': {'type': 'number'}, 'y': {'type': 'number'},
         'z': {'type': 'number'},
         'orientation': {'enum': ['down', 'forward'], 'description':
                         'down = tool pointing down (tabletop); forward = '
                         'tool horizontal pointing +x-ish (shelves/handles)'},
         'yaw_deg': {'type': 'number', 'description':
                     'rotation about vertical, default 0'},
         'why': {'type': 'string', 'description':
                 'one line: what this endpoint accomplishes'}}}},
    {'name': 'home', 'description':
        'Return the arm to its safe home pose.',
     'input_schema': {'type': 'object', 'properties': {}}},
    {'name': 'say', 'description':
        'Speak a short sentence to the user (text-to-speech).',
     'input_schema': {'type': 'object', 'required': ['text'], 'properties': {
         'text': {'type': 'string'}}}},
    {'name': 'task_complete', 'description':
        'End the task with an honest one-or-two-sentence summary of what '
        'happened, including anything that failed.',
     'input_schema': {'type': 'object', 'required': ['summary'],
                      'properties': {'summary': {'type': 'string'}}}},
]


def _find_scene_yaml():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('curobo_planner'),
                        'config', 'scene.yaml')


class Brain(Node):

    def __init__(self):
        super().__init__('rammp_brain')
        self._cb = ReentrantCallbackGroup()
        # haiku 4.5, no extended thinking: sub-second tool decisions — the
        # right default for a robot loop (brain_model:=claude-sonnet-4-6 or
        # claude-opus-4-8 for harder multi-step tasks).
        self.model = self.declare_parameter('model', 'claude-haiku-4-5').value
        # Adaptive thinking adds seconds per decision — off by default.
        # NOTE: haiku-4-5 does NOT support adaptive thinking (API 400) —
        # only enable together with a sonnet-4-6+/opus-4-6+ brain_model.
        self.thinking = bool(self.declare_parameter('thinking', False).value)
        self.scene_file = self.declare_parameter('scene_file', '').value \
            or _find_scene_yaml()
        self._scene = load_scene(self.scene_file)

        self._cmd_pub = self.create_publisher(
            String, '/curobo_planner/command', 10)
        self._task_status_pub = self.create_publisher(
            String, '/rammp/task_status',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/rammp/task', self._task_cb, 10,
                                 callback_group=self._cb)
        self.create_subscription(String, '/curobo_planner/status',
                                 self._status_cb, 10,
                                 callback_group=self._cb)
        self._live = {}                 # label -> ([x,y,z], monotonic)
        try:
            from vision_msgs.msg import Detection3DArray
            self.create_subscription(Detection3DArray, '/perception/objects',
                                     self._perception_cb, 5,
                                     callback_group=self._cb)
        except ImportError:
            self.get_logger().warning('vision_msgs unavailable — world '
                                      'snapshots will use YAML poses only')
        # The brain's EYES: the proposer freezes a frame on 'look at X' and
        # publishes the JPEG here; the look tool hands it to the model so
        # it can choose the graspable part from what is actually visible.
        self._look_jpeg = None
        from sensor_msgs.msg import CompressedImage
        self.create_subscription(CompressedImage,
                                 '/grasp_proposer/look_image',
                                 self._look_cb, 1, callback_group=self._cb)

        self._statuses = []             # planner statuses since last command
        self._status_lock = threading.Lock()
        self._busy_lock = threading.Lock()
        self._busy = False
        self._abort = False
        self._held = None
        # Speech rides its own topic: /rammp/task_status is a strict
        # terminal-status protocol (first non-'...' message ends a task for
        # every listener) and say() mid-task must not terminate it.
        self._say_pub = self.create_publisher(String, '/rammp/say', 10)

        self._client = None
        if os.environ.get('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                # timeout*retries must stay under goto's 75 s quiet window
                self._client = anthropic.Anthropic(timeout=30.0, max_retries=1)
            except ImportError:
                pass
        mode = ('Claude %s' % self.model) if self._client else \
            'PASSTHROUGH (no ANTHROPIC_API_KEY / anthropic package)'
        # Latch an initial value so late subscribers' stale-drain logic has
        # something to absorb (mirrors the planner's 'ready').
        self._task_status_pub.publish(String(data='brain ready (%s)' % mode))
        self.get_logger().info('rammp brain up on /rammp/task — %s' % mode)

    # ------------------------------------------------------------- callbacks
    def _perception_cb(self, msg):
        now = time.monotonic()
        for det in msg.detections:
            if det.results:
                p = det.bbox.center.position
                self._live[det.results[0].hypothesis.class_id] = (
                    [float(p.x), float(p.y), float(p.z)], now)

    def _status_cb(self, msg):
        with self._status_lock:
            self._statuses.append(msg.data)

    def _look_cb(self, msg):
        self._look_jpeg = bytes(msg.data)

    def _task_cb(self, msg):
        text = msg.data.strip()
        if not text:
            return
        if set(re.findall(r'[a-z]+', text.lower())) & STOP_WORDS:
            # STOP never waits on anything — straight to the planner, and
            # abort whatever task is mid-flight.
            self._abort = True
            self._cmd_pub.publish(String(data=text))
            return
        # Claim the task ATOMICALLY here, not in the worker: two arrivals in
        # the thread-start window both saw busy=False, and a stop landing in
        # that window was clobbered by the worker's own abort reset.
        with self._busy_lock:
            if self._busy:
                self._task_status('Busy with the previous task; ignoring '
                                  '"%s". Say "stop" to abort it.' % text)
                return
            self._busy = True
            self._abort = False
        threading.Thread(target=self._run_task, args=(text,),
                         daemon=True).start()

    # ----------------------------------------------------------------- world
    def _world(self):
        self._scene = load_scene(self.scene_file)
        now = time.monotonic()
        lines = []
        for o in self._scene.objects:
            if not o.free:
                continue
            lv = self._live.get(o.name)
            if lv and now - lv[1] < 10.0:
                x, y = lv[0][0], lv[0][1]
                src = 'live'
            else:
                x, y = o.position[0], o.position[1]
                src = 'last known'
            dims = o.bounding_dims()
            lines.append('%s: at [%.2f, %.2f, %.2f] (%s), %s, %.0fx%.0fx%.0f mm'
                         % (o.name, x, y, o.position[2], src, o.type,
                            dims[0] * 1000, dims[1] * 1000, dims[2] * 1000))
        held = self._held or 'nothing'
        return ('WORLD (base frame, metres):\n- objects: \n  ' +
                '\n  '.join(lines) +
                '\n- gripper holds: %s'
                '\n- named places for reach(): %s'
                % (held, ', '.join(self._scene.target_names + ['home'])))

    # --------------------------------------------------------------- planner
    def _planner(self, text):
        """Send one planner command, return its terminal status (the goto
        protocol: '...'-prefixed statuses are interim)."""
        if self._abort:
            # NEVER publish after a stop — a fresh command legitimately
            # clears the planner's hold, so a post-stop tool command would
            # move a deliberately stopped arm.
            return 'ABORTED by user stop.'
        with self._status_lock:
            self._statuses.clear()
        self._cmd_pub.publish(String(data=text))
        deadline = time.monotonic() + PLANNER_TIMEOUT
        while time.monotonic() < deadline:
            if self._abort:
                return 'ABORTED by user stop.'
            with self._status_lock:
                while self._statuses:
                    s = self._statuses.pop(0)
                    if s.startswith('...'):
                        # relay interim progress so task listeners' quiet
                        # timers keep breathing during long motions
                        self._task_status_pub.publish(String(data=s))
                        continue
                    return s
            time.sleep(0.1)
        return 'TIMEOUT: the planner did not answer within %.0f s.' \
            % PLANNER_TIMEOUT

    # ------------------------------------------------------------------ task
    def _task_status(self, text):
        self.get_logger().info('[task] %s' % text)
        self._task_status_pub.publish(String(data=text))

    def _run_task(self, text):
        # busy/abort were claimed atomically in _task_cb — do NOT touch
        # _abort here (a stop issued in the thread-start window must stick)
        try:
            if self._client is None:
                # passthrough: exactly the pre-brain pipeline
                self._task_status(self._planner(text))
                return
            self._task_status('... thinking about "%s"' % text)
            self._loop(text)
        except Exception as exc:
            self._task_status('Task failed: %s' % exc)
        finally:
            self._busy = False

    def _execute(self, name, args):
        """One tool -> (result_text, ends_task)."""
        if name == 'reach':
            return self._planner('go to the %s' % args['name']), False
        if name == 'look':
            self._look_jpeg = None
            out = self._planner('look at the %s' % args['object'])
            time.sleep(0.3)     # the JPEG rides a separate topic
            if self._look_jpeg is not None and out.startswith('Looking at'):
                import base64
                return ([{'type': 'text', 'text':
                          out + ' Choose the most easily graspable part '
                          'and pass its [ymin, xmin, ymax, xmax] box '
                          '(0-1000 normalized on THIS image) to grasp.'},
                         {'type': 'image', 'source': {
                             'type': 'base64',
                             'media_type': 'image/jpeg',
                             'data': base64.b64encode(
                                 self._look_jpeg).decode()}}], False)
            return out, False
        if name == 'grasp':
            cmd = 'grasp the %s' % args['object']
            box = args.get('part_box')
            if box and len(box) == 4:
                cmd += ' box:%d,%d,%d,%d' % tuple(int(b) for b in box)
                self.get_logger().info(
                    '[brain grasp] %s part=%s box=%s'
                    % (args['object'], args.get('part_name', '?'), box))
            out = self._planner(cmd)
            if out.startswith('GRASPED'):
                self._held = args['object']
            return out, False
        if name == 'release':
            out = self._planner('release')
            if out.startswith('Released'):
                self._held = None
            return out, False
        if name == 'move_tool':
            pitch = 180.0 if args['orientation'] == 'down' else 90.0
            yaw = float(args.get('yaw_deg', 0.0))
            self.get_logger().info('[brain endpoint] %s -> [%.2f %.2f %.2f] %s'
                                   % (args.get('why', ''), args['x'],
                                      args['y'], args['z'],
                                      args['orientation']))
            if args['orientation'] == 'down':
                cmd = 'pose: %.3f %.3f %.3f 180 0 %.0f' % (
                    args['x'], args['y'], args['z'], yaw)
            else:
                cmd = 'pose: %.3f %.3f %.3f 0 90 %.0f' % (
                    args['x'], args['y'], args['z'], yaw)
            return self._planner(cmd), False
        if name == 'home':
            return self._planner('home'), False
        if name == 'say':
            # dedicated topic: task_status is a terminal-status protocol
            self.get_logger().info('[say] %s' % args['text'])
            self._say_pub.publish(String(data=args['text']))
            self._task_status_pub.publish(String(data='... %s' % args['text']))
            return 'said it', False
        if name == 'ask_user':
            self._task_status('QUESTION: %s' % args['question'])
            return '', True
        if name == 'task_complete':
            self._task_status(args['summary'])
            return '', True
        return 'unknown tool %r' % name, False

    def _loop(self, task):
        messages = [{'role': 'user', 'content':
                     'TASK: %s\n\n%s' % (task, self._world())}]
        for _ in range(MAX_STEPS):
            if self._abort:
                self._task_status('Task aborted.')
                return
            # The API call can take tens of seconds (adaptive thinking, SDK
            # retries) and a stop must NOT wait for it: stream the response
            # and poll the abort flag between events, closing the stream on
            # abort (field log: 40 s of "Busy... say stop" while a stopped
            # task sat inside messages.create). The heartbeat keeps
            # listeners' quiet timers breathing.
            self._task_status_pub.publish(String(data='... thinking'))
            resp = None
            think = {'thinking': {'type': 'adaptive'}} if self.thinking else {}
            with self._client.messages.stream(
                    model=self.model, max_tokens=16000,
                    system=SYSTEM, tools=TOOLS, messages=messages,
                    **think) as stream:
                next_beat = time.monotonic() + 5.0
                for _event in stream:
                    if self._abort:
                        break   # exiting the with closes the connection
                    if time.monotonic() > next_beat:
                        self._task_status_pub.publish(
                            String(data='... thinking'))
                        next_beat = time.monotonic() + 5.0
                if not self._abort:
                    resp = stream.get_final_message()
            if resp is None:
                self._task_status('Task aborted.')
                return
            # Only end_turn/tool_use are normal. A truncated or declined
            # response must NEVER read as success on an assistive robot.
            if resp.stop_reason == 'max_tokens':
                self._task_status('Task stopped: the model response was '
                                  'truncated — outcome unverified.')
                return
            if resp.stop_reason == 'refusal':
                self._task_status('Task stopped: the model declined this '
                                  'request.')
                return
            messages.append({'role': 'assistant', 'content': resp.content})
            tool_uses = [b for b in resp.content if b.type == 'tool_use']
            if not tool_uses:
                text = ' '.join(b.text for b in resp.content
                                if b.type == 'text').strip()
                self._task_status(text or 'Done.')
                return
            results = []
            ended = False
            for tu in tool_uses:
                if self._abort:
                    self._task_status('Task aborted.')
                    return
                out, ends = self._execute(tu.name, dict(tu.input))
                world = self._world()
                if isinstance(out, list):
                    # multimodal result (look): blocks + the fresh world
                    content = out + [{'type': 'text', 'text': world}]
                else:
                    content = '%s\n\n%s' % (out, world)
                results.append({'type': 'tool_result',
                                'tool_use_id': tu.id,
                                'content': content})
                if ends:
                    ended = True
                    break
            if ended:
                return
            messages.append({'role': 'user', 'content': results})
        self._task_status('Stopping: task hit the %d-step limit without '
                          'finishing — tell me how to continue.' % MAX_STEPS)


def main(args=None):
    rclpy.init(args=args)
    node = Brain()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

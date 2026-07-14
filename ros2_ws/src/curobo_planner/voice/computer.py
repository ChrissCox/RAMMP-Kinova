"""RAMMP voice control, native and OFFLINE: "computer, go to my bottle."

Runs on the DEV MACHINE (no ROS needed) — double-click the Desktop launcher
or:
    pip install vosk sounddevice pyttsx3 roslibpy pyyaml
    python computer.py --host 192.168.1.11

First run downloads the small Vosk English model (~40 MB) automatically.
Recognition is GRAMMAR-CONSTRAINED to this project's vocabulary, which makes
it fast and hard to mishear, and it runs fully offline (no browser, no cloud,
no mic permission prompts). Replies are printed and spoken.

    "computer, go to my bottle"   -> arm goes (fires when Vosk finalizes,
                                     ~0.3 s after you stop speaking)
    "computer"                    -> armed for 6 s, then say the command
    "stop"                        -> the arm halts — any word order, wake
                                     word optional, checked on partials too
                                     (two agreeing within 3/4 s). A stop
                                     with no STOPPED reply within 2.5 s is
                                     called out loud (dead-link watchdog).

The mic is MUTED while the app itself is speaking (half-duplex): an open
mic next to the speakers would otherwise feed our own phrases back into
the grammar-constrained recognizer, and none of our spoken strings may
contain a stop word for the same reason.

--direct sends commands straight to the planner (/curobo_planner/command),
skipping the brain — for when the brain node is not running. Do not mix
--direct voice with brain-mediated tasks (goto without --direct): the
planner's busy rejection lands on the shared status stream and the brain
would read it as its own tool verdict. While the link is up, STOP is
published to BOTH /rammp/task and /curobo_planner/command regardless of
mode: the brain latches its abort flag only for stops it sees, and the
planner stops fastest when told directly. If rosbridge is down, the refusal
is printed AND spoken — the arm is not listening to this microphone then.

Ctrl-C to quit. --list-mics to pick an input device (--mic N).
"""

import argparse
import json
import os
import queue
import re
import shutil
import sys
import time
import urllib.request
import zipfile

MODEL_URL = ('https://alphacephei.com/vosk/models/'
             'vosk-model-small-en-us-0.15.zip')
MODEL_DIR = os.path.join(os.path.expanduser('~'), '.rammp',
                         'vosk-model-small-en-us-0.15')

# Recognition is grammar-constrained, so 'computer' is the only wake token
# Vosk can ever emit — no phonetic variants needed.
WAKE = re.compile(r'\bcomputer\b')
# Client-side stop set = the grammar's stop words (planner_node also knows
# 'estop', but the small model has no such token so it can never be heard).
# tools/voice_gate_test.py cross-checks this regex against planner_node.py.
STOP_RE = re.compile(r'\b(stop|halt|freeze|cancel)\b')
# Grammar = every word the recognizer is allowed to hear. Small vocabulary =
# fast, accurate, offline. [unk] absorbs everything else. The OBJECT words
# come from scene.yaml (target names + keywords) so new targets are
# automatically hearable — a hardcoded list silently deafened the
# recognizer to 'pills' when that target was added. 'release'/'drop' and
# the grasp verbs are load-bearing: drop one and the recognizer goes deaf
# to it (a fork of this file once lost 'release' — held objects could
# never be let go by voice). tools/voice_gate_test.py cross-checks this
# list against planner_node.py's word sets.
FILLER = ('computer go to the my a please grab get open take pick '
          'fetch grasp release drop stop halt freeze cancel check home')
# Words that make an utterance FIREABLE on their own. The planner's grasp/
# release verbs stand alone ("drop" IS a complete release command), so they
# must be here — hearable-but-unfireable is the same deafness bug wearing
# a different hat.
KNOWN_WORDS = ('stop halt freeze cancel check home release drop grasp '
               'grab pick take get fetch open')
ARM_WINDOW_S = 6.0
COOLDOWN_S = 2.5
STOP_REPEAT_S = 1.0     # dedupe echoes of ONE stop; a repeat past this is
                        # honored again (the planner treats stop as idempotent)
PARTIAL_AGREE_S = 0.75  # two stop partials must agree within this window —
                        # partials arrive ~4x/s, so consecutive ones are
                        # ~0.25 s apart; a stale rumor must not linger
STARTUP_SKIP_S = 3.0    # the status topics are LATCHED: the first message
                        # right after our startup subscribe can be an old
                        # verdict from before this session (a longer window
                        # would eat the planner's live 'ready' instead)
STOP_ACK_S = 2.5        # the planner answers every stop with STOPPED before
                        # anything else; silence past this = dead link


def known_regex(objects):
    """The fire gate: scene words + verbs that stand alone as commands."""
    return re.compile(
        r'\b(%s)\b' % '|'.join(list(objects) + KNOWN_WORDS.split()))


class CommandGate:
    """The utterance state machine: stop, wake word, arming, cooldown.

    Pure logic — no I/O, no clock (callers pass a monotonic `now`) — so
    tools/voice_gate_test.py can drive every field-failure scenario offline.
    handle() returns (kind, text): ('stop', ...), ('fire', command),
    ('armed', None), ('ignored', command) for a cooldown drop the caller
    must report, or (None, None).

    Rules, each earned in the field or in review:
    - STOP outranks everything: the WHOLE text is scanned before the wake
      parse and before the cooldown (a debounce must never delay a safety
      stop, and "stop, computer" must work as well as "computer, stop").
      Stops fire on finals immediately, and on partials when two partials
      agree within PARTIAL_AGREE_S — grammar-constrained partials rewrite
      drastically mid-utterance (one misheard 'coffee' during "go home"),
      so a single partial is a rumor, two in quick succession are a shout.
      The rumor is timestamped, not latched: an unconfirmed one expires
      instead of lying in wait to confirm an unrelated rumor later.
    - A stop also resets the fire cooldown: the stop proves the user's
      next command is a correction, not an echo of the aborted one.
    - Non-stop partials never act (FINALS ONLY): Vosk finalizes ~0.3 s
      after speech ends, so the latency cost is negligible.
    - The cooldown debounces only real fires (the "say it twice" bug);
      arming is exempt, and a swallowed command is reported, not silent.
    """

    def __init__(self, known, cooldown_s=COOLDOWN_S,
                 arm_window_s=ARM_WINDOW_S, stop_repeat_s=STOP_REPEAT_S,
                 partial_agree_s=PARTIAL_AGREE_S):
        self.known = known
        self.cooldown_s = cooldown_s
        self.arm_window_s = arm_window_s
        self.stop_repeat_s = stop_repeat_s
        self.partial_agree_s = partial_agree_s
        self.armed_until = 0.0
        self.last_fire = -1e9
        self.last_stop = -1e9
        self._partial_stop_t = -1e9

    def reset_cooldown(self):
        """A refused publish must not lock out the retry."""
        self.last_fire = -1e9

    def reset_stop(self):
        """A refused STOP must not eat the user's immediate retry."""
        self.last_stop = -1e9

    def unfire(self, now):
        """A refused fire restores BOTH things handle() consumed: the
        cooldown AND the armed window — an armed-mode retry must not need
        the wake word again."""
        self.last_fire = -1e9
        self.armed_until = now + self.arm_window_s

    def handle(self, text, final, now):
        text = text.replace('[unk]', ' ').strip()
        if not text and not final:
            return (None, None)   # an [unk]-only partial is no evidence
            # against a stop rumor — only real words or a final clear it
        # STOP first — before the cooldown, before the wake-word parse.
        if STOP_RE.search(text):
            confirmed = final or (now - self._partial_stop_t
                                  <= self.partial_agree_s)
            self._partial_stop_t = -1e9 if final else now
            if confirmed and now - self.last_stop >= self.stop_repeat_s:
                self.last_stop = now
                self.armed_until = 0.0
                self.last_fire = -1e9   # the correction is not an echo
                return ('stop', text)
            return (None, None)
        self._partial_stop_t = -1e9     # any non-stop text kills the rumor
        if not final:
            return (None, None)
        m = WAKE.search(text)
        command = None
        if m:
            after = text[m.end():].strip(' ,.!?')
            if len(after) >= 3:
                command = after
            else:
                # Arming is exempt from the cooldown: swallowing it silently
                # lost BOTH halves of a two-step exchange.
                self.armed_until = now + self.arm_window_s
                return ('armed', None)
        elif now < self.armed_until and len(text) >= 3:
            command = text
        if command and (self.known.search(command) or len(command) >= 6):
            if now - self.last_fire < self.cooldown_s:
                return ('ignored', command)
            self.last_fire = now
            self.armed_until = 0.0
            return ('fire', command)
        return (None, None)


def scene_words(scene_path=None):
    """Target names + keywords from scene.yaml -> grammar/known words."""
    if scene_path is None:
        scene_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'config', 'scene.yaml')
    words = set()
    try:
        import yaml   # inside the try: a missing PyYAML degrades like a
        # missing file instead of killing the app before the fallback runs
        with open(scene_path, encoding='utf-8') as f:
            scene = yaml.safe_load(f)
        for t in scene.get('targets', []):
            for w in [t.get('name', '')] + list(t.get('keywords', [])):
                for sub in re.split(r'[^a-z]+', str(w).lower()):
                    if len(sub) >= 3:
                        words.add(sub)
        # free props are graspable by NAME ("grab the apple") even when no
        # authored target mentions them
        for o in scene.get('objects', []):
            if o.get('free', False):
                for sub in re.split(r'[^a-z]+', str(o.get('name', '')).lower()):
                    if len(sub) >= 3:
                        words.add(sub)
    except Exception as exc:
        hint = ' — fix: pip install pyyaml' if isinstance(exc, ImportError) \
            else ''
        print('WARNING: could not read %s (%s%s) — using a stale built-in '
              'word list; targets added since it froze are UNHEARABLE.'
              % (scene_path, exc, hint))
        words = set(('bottle water drink mug cup coffee tea cabinet handle '
                     'door cupboard shelf snack cereal rest ready pills pill '
                     'medicine meds medication').split())
    return sorted(words)


def ensure_model():
    """Download+extract the Vosk model ATOMICALLY.

    The old urlretrieve+extractall-in-place version had two failure holes:
    a stalled connection hung forever (no socket timeout), and a Ctrl-C
    mid-extract left a half-populated MODEL_DIR that the isdir() check
    trusted forever after (Vosk then failed with a cryptic Kaldi error).
    """
    marker = os.path.join(MODEL_DIR, 'am', 'final.mdl')
    if os.path.isdir(MODEL_DIR):
        if os.path.exists(marker):
            return MODEL_DIR
        sys.exit('Speech model at %s is INCOMPLETE (an earlier download was '
                 'interrupted). Delete it and rerun:  rm -rf "%s"  '
                 '(PowerShell: Remove-Item -Recurse -Force "%s")'
                 % (MODEL_DIR, MODEL_DIR, MODEL_DIR))
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    tmp_zip = MODEL_DIR + '.zip.part'
    tmp_dir = MODEL_DIR + '.extracting'
    print('First run: downloading the offline speech model (~40 MB)...')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, \
                open(tmp_zip, 'wb') as f:
            shutil.copyfileobj(r, f)
    except Exception as exc:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass
        sys.exit('Model download FAILED (%s). Check the network and rerun, '
                 'or unzip %s into %s yourself.'
                 % (exc, MODEL_URL, os.path.dirname(MODEL_DIR)))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        try:
            with zipfile.ZipFile(tmp_zip) as f:
                f.extractall(tmp_dir)
        except zipfile.BadZipFile as exc:
            # A truncated transfer ends WITHOUT a socket error, and a
            # captive portal happily serves HTML with a 200 — both land here.
            sys.exit('Model download was CORRUPT (%s) — truncated transfer '
                     'or a captive portal serving HTML instead of the zip. '
                     'Get on a real connection and rerun, or unzip %s into '
                     '%s yourself.'
                     % (exc, MODEL_URL, os.path.dirname(MODEL_DIR)))
        os.remove(tmp_zip)
        inner = os.path.join(tmp_dir, os.path.basename(MODEL_DIR))
        if not os.path.exists(os.path.join(inner, 'am', 'final.mdl')):
            sys.exit('Model zip did not contain the expected %s layout — '
                     'the URL may have changed. Got: %s'
                     % (os.path.basename(MODEL_DIR), os.listdir(tmp_dir)))
        os.rename(inner, MODEL_DIR)   # atomic on the same volume: the
        # isdir() check can never see halves
    except OSError as exc:
        # Disk full, or Windows Defender holding a handle inside the fresh
        # extract (a directory move fails while any child handle is open).
        sys.exit('Model install FAILED (%s: %s). Free disk space / wait a '
                 'moment and rerun after:  rm -rf "%s" "%s"  (PowerShell: '
                 'Remove-Item -Recurse -Force)'
                 % (type(exc).__name__, exc, tmp_dir, tmp_zip))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print('Model ready.')
    return MODEL_DIR


class Speaker:
    """TTS in a worker thread — pyttsx3's runAndWait blocks.

    Failures are NAMED, never swallowed: a silent mute is indistinguishable
    from a dead robot for a user who cannot see the console. A SAPI error
    wedges pyttsx3 (its run loop stays marked busy and every later
    runAndWait raises) AND pyttsx3 caches engines in a WeakValueDictionary,
    so a naive re-init hands back the SAME wedged object — recovery must
    drop every strong reference and collect before re-initializing (the
    rebuild happens after the except block so the in-flight traceback's
    frames release theirs too). The queue is small and drops the OLDEST
    phrase when full — narrating stale verdicts ("Stopped." while a newer
    command is already executing) actively misleads.
    """

    def __init__(self):
        import threading
        self._q = queue.Queue(maxsize=4)
        self._warned = False
        self._speaking = False
        threading.Thread(target=self._run, daemon=True).start()

    def busy(self):
        """True while a phrase is pending or being spoken — the main loop
        mutes the mic then (half-duplex), so our own voice can never fire
        the recognizer."""
        return self._speaking or not self._q.empty()

    def _engine(self):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty('rate', 185)
            self._warned = False
            return eng
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print('WARNING: text-to-speech unavailable (%s) — replies '
                      'will be PRINTED only.' % exc)
            return None

    def _run(self):
        import gc
        eng = self._engine()
        while True:
            text = self._q.get()
            self._speaking = True
            if eng is None:
                eng = self._engine()   # a speaker plugged in later comes back
            # Two attempts: the phrase that hit a transient SAPI error is
            # retried once on the rebuilt engine, not silently dropped.
            for attempt in (1, 2):
                if eng is None:
                    print('(speech LOST, no TTS engine: %s)' % text)
                    break
                failed = False
                try:
                    eng.say(text)
                    eng.runAndWait()
                except Exception as exc:
                    print('(speech failed: %s — rebuilding the TTS engine)'
                          % exc)
                    failed = True
                if not failed:
                    break
                eng = None       # outside the except: the traceback holds
                gc.collect()     # the wedged engine's frames until then
                eng = self._engine()
                if attempt == 2:
                    print('(speech LOST after retry: %s)' % text)
            self._speaking = False

    def say(self, text):
        try:
            self._q.put_nowait(text)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(text)
            except queue.Full:
                pass


def shorten(s):
    if s.startswith('STOPPED'):
        # Never SPEAK a stop word: with an open mic the recognizer would
        # hear our own 'Stopped.' as a fresh stop (half-duplex is the first
        # line of defense; this is the second).
        return 'Holding still.'
    m = re.match(r'Planned to ([a-z_ ()]+):', s, re.I)
    if m:
        return 'Going to the %s.' % m.group(1).replace('_', ' ').strip()
    m = re.search(r'Plan to ([a-z_ ()]+) FAILED', s, re.I)
    if m:
        return 'Could not reach the %s.' % m.group(1).replace('_', ' ').strip()
    return s.split('(')[0][:80]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='192.168.1.11', help='Jetson IP (rosbridge)')
    ap.add_argument('--port', type=int, default=9090)
    ap.add_argument('--mic', type=int, default=None, help='input device index')
    ap.add_argument('--list-mics', action='store_true')
    ap.add_argument('--scene', default=None, help='scene.yaml path (for vocabulary)')
    ap.add_argument('--direct', action='store_true',
                    help='publish commands straight to /curobo_planner/command '
                         '(skip the brain); while connected, STOP still goes '
                         'to both topics')
    args = ap.parse_args(argv)

    import sounddevice as sd
    if args.list_mics:
        print(sd.query_devices())
        return 0

    from vosk import Model, KaldiRecognizer
    import roslibpy

    objects = scene_words(args.scene)
    vocab = list(dict.fromkeys(FILLER.split() + objects))
    known = known_regex(objects)
    print('Vocabulary (%d words): %s' % (len(vocab), ' '.join(vocab)))

    model = Model(ensure_model())
    grammar = json.dumps(vocab + ['[unk]'])
    rec = KaldiRecognizer(model, 16000, grammar)

    client = roslibpy.Ros(host=args.host, port=args.port)
    try:
        # Half-open links (cable pull, Jetson power loss) otherwise sit
        # undetected for the full TCP timeout while everything still looks
        # connected: websocket pings surface them in seconds.
        client.factory.setProtocolOptions(autoPingInterval=2.0,
                                          autoPingTimeout=4.0)
    except Exception:
        pass   # older roslibpy/autobahn: degrade to TCP-level detection
    try:
        client.run()   # raises RosTimeoutError when unreachable — the old
    except Exception as exc:   # post-hoc is_connected check was dead code
        sys.exit('Could not reach rosbridge at ws://%s:%d (%s). Is the '
                 'Jetson bringup running?  ros2 launch mujoco_sim '
                 'mujoco_bringup.launch.py' % (args.host, args.port, exc))
    # Tasks go to the BRAIN (Claude picks tools; degrades to a planner
    # passthrough without an API key) unless --direct. Both status streams
    # are spoken: the planner's motion verdicts and the brain's summaries.
    task = roslibpy.Topic(client, '/rammp/task', 'std_msgs/String')
    planner = roslibpy.Topic(client, '/curobo_planner/command', 'std_msgs/String')
    cmd = planner if args.direct else task
    status = roslibpy.Topic(client, '/curobo_planner/status', 'std_msgs/String')
    task_status = roslibpy.Topic(client, '/rammp/task_status', 'std_msgs/String')
    say_topic = roslibpy.Topic(client, '/rammp/say', 'std_msgs/String')
    voice = Speaker()

    # Connectivity is tracked from roslibpy's own events: publish() on a
    # dropped websocket does NOT fail — it parks the message on a one-shot
    # 'ready' listener and replays it as a surprise burst on reconnect
    # (a queued stale motion command would clear a stop-hold). So commands
    # are REFUSED, out loud, while the link is down. (A publish racing the
    # close event by microseconds can still park — publish() re-checks
    # roslibpy's own flag to shrink that window as far as the API allows.)
    link = {'connected': True, 'shutting_down': False, 'fired_once': False,
            'startup_skip_until': time.monotonic() + STARTUP_SKIP_S}
    # Stop-ack watchdog: the planner answers every stop with STOPPED before
    # its command lock — no reply past the deadline means the link is dead
    # even though the publish "succeeded" (half-open TCP looks connected).
    pending = {'stop_deadline': None}

    # In passthrough mode the brain relays the planner's verdict verbatim on
    # its own status topic — dedupe so it is spoken once, not twice.
    last = {'s': None, 't': 0.0}
    # Both status topics are LATCHED (planner ~/status and /rammp/
    # task_status), so EACH replays its last verdict on every (re)subscribe
    # — replay tracking must be per topic, one shared slot muted only one.
    topic_states = []

    def on_close(_proto):
        if link['shutting_down']:
            return                   # terminate() emits 'close' too — a
        link['connected'] = False    # clean quit is not a lost connection
        print('LOST the rosbridge connection — commands will be refused '
              'until it returns (reconnecting automatically).')
        voice.say('Connection lost.')

    def on_ready(_proto):
        # Fires on RE-connects: roslibpy re-subscribes each topic ~1 s
        # later and each LATCHED status replays. The replay is the last
        # text seen ON THAT TOPIC — match by content, per topic; a time
        # window here once ate genuinely fresh verdicts.
        now = time.monotonic()
        for st in topic_states:
            st['resub_until'] = now + 5.0
        if not link['connected']:
            link['connected'] = True
            print('rosbridge connection restored.')
            voice.say('Connection restored.')
    client.on('close', on_close)
    client.on('ready', on_ready)

    def make_on_status(label):
        st = {'last_seen': None, 'first': True, 'resub_until': 0.0}
        topic_states.append(st)

        def on_status(msg):
            s = msg.get('data', '')
            now = time.monotonic()
            if now < st['resub_until'] and s == st['last_seen']:
                st['resub_until'] = 0.0
                print('  %s (reconnect replay, ignored): %s' % (label, s))
                return
            st['resub_until'] = 0.0   # first fresh message ends the watch
            st['last_seen'] = s
            # Startup: the latched pre-session verdict arrives right after
            # the subscribe. Skip one early message per topic — unless a
            # command of ours was SENT, which makes every status an answer.
            if (st['first'] and not link['fired_once']
                    and now < link['startup_skip_until']):
                st['first'] = False
                print('  %s (latched, pre-session): %s' % (label, s))
                return
            st['first'] = False
            if s.startswith('STOPPED'):
                pending['stop_deadline'] = None   # the stop-ack arrived
            if s == last['s'] and now - last['t'] < 3.0:
                return
            last['s'], last['t'] = s, now
            print('  %s: %s' % (label, s))
            if not s.startswith('...'):
                voice.say(shorten(s))
        return on_status
    status.subscribe(make_on_status('robot'))
    task_status.subscribe(make_on_status('robot'))

    def on_say(msg):
        s = msg.get('data', '')
        print('  robot says: %s' % s)
        voice.say(s)                 # the brain's own words, spoken verbatim
    say_topic.subscribe(on_say)

    # Bounded, drop-OLDEST: if the consumer stalls (console QuickEdit
    # selection freezes print for as long as the user drags), an unbounded
    # queue grows a backlog that delays every later utterance — including
    # stop — by the length of the stall. Drops are counted so the main
    # loop can reset the recognizer: splicing pre- and post-stall audio
    # lets grammar-constrained decoding fabricate a command from the seam.
    audio = queue.Queue(maxsize=40)   # ~10 s of 0.25 s blocks
    drops = {'n': 0}

    def on_audio(indata, frames, t, flags):
        if flags:   # driver/OS dropped frames (input_overflow) — the same
            drops['n'] += 1   # splice hazard as a queue drop; force a reset
        try:
            audio.put_nowait(bytes(indata))
        except queue.Full:
            drops['n'] += 1
            try:
                audio.get_nowait()
            except queue.Empty:
                pass
            try:
                audio.put_nowait(bytes(indata))
            except queue.Full:
                pass

    gate = CommandGate(known)

    def publish(topic, text):
        # Both our flag (event-driven) and roslibpy's own are checked:
        # neither closes the race completely, but is_connected flips on
        # the reactor thread earlier than our handler runs.
        if not (link['connected'] and client.is_connected):
            return False
        topic.publish(roslibpy.Message({'data': text}))
        return True

    def dispatch(text, final):
        kind, payload = gate.handle(text, final, time.monotonic())
        if kind == 'stop':
            # BOTH topics, always: the brain latches _abort only for stops
            # it sees on /rammp/task (otherwise its next tool command would
            # legitimately clear the planner's stop-hold), and the planner
            # reacts fastest when told directly. Publishes come FIRST —
            # a frozen console (QuickEdit drag) must never sit between the
            # stop decision and the send.
            sent_task = publish(task, 'stop')
            sent_planner = publish(planner, 'stop')
            if sent_task or sent_planner:
                link['fired_once'] = True   # sent, not merely attempted
                last['s'] = None   # the next verdict is fresh even if its
                # text repeats (the planner's STOPPED string is constant)
                pending['stop_deadline'] = time.monotonic() + STOP_ACK_S
            if sent_task and sent_planner:
                print('-> STOP (to brain and planner)')
            else:
                gate.reset_stop()   # the refusal must not eat the retry
                print('STOP NOT SENT%s — rosbridge is down. The arm is NOT '
                      'listening to this microphone right now.'
                      % (' EVERYWHERE (the link dropped mid-send)'
                         if sent_task or sent_planner else ''))
                # Spoken strings must never contain a stop word (echo!),
                # and delivery of a partial send is unknowable here — the
                # watchdog above settles it within STOP_ACK_S.
                voice.say('No connection. That may not have reached the arm.'
                          if sent_task or sent_planner else
                          'No connection. The arm did not hear that.')
        elif kind == 'fire':
            if publish(cmd, payload):
                link['fired_once'] = True
                last['s'] = None
                print('-> %s' % payload)
            else:
                gate.unfire(time.monotonic())   # no lockout on refusal, and
                # an armed-mode retry must not need the wake word again
                print('NOT SENT (rosbridge down): %s' % payload)
                voice.say('No connection.')
        elif kind == 'armed':
            print('  (armed — say a command)')
            voice.say('Yes?')
        elif kind == 'ignored':
            print('  (ignored — within %.1f s of the last command; '
                  'say it again in a moment)' % COOLDOWN_S)
            voice.say('Too soon. Say it again.')

    print('Listening (offline). Say "computer, go to my bottle" — '
          'Ctrl-C quits.%s' % (' [DIRECT: brain bypassed]' if args.direct else ''))
    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                           channels=1, callback=on_audio, device=args.mic) as stream:
        tts_gap = False
        try:
            while True:
                now = time.monotonic()
                if pending['stop_deadline'] and now > pending['stop_deadline']:
                    pending['stop_deadline'] = None
                    print('NO STOPPED reply from the planner within %.1f s — '
                          'the link is probably dead (half-open TCP looks '
                          'connected). Check the Jetson / network.'
                          % STOP_ACK_S)
                    voice.say('No reply from the arm. The link may be dead.')
                try:
                    # The timeout keeps Ctrl-C deliverable on Windows (an
                    # untimed Queue.get is uninterruptible there), lets us
                    # notice a dead microphone stream, and paces the stop-
                    # ack watchdog above.
                    data = audio.get(timeout=0.5)
                except queue.Empty:
                    if not stream.active:
                        sys.exit('Microphone stream STOPPED (device removed '
                                 'or audio error) — rerun with --list-mics '
                                 'and pick a working input with --mic N.')
                    continue
                if voice.busy():
                    # HALF-DUPLEX: never feed our own voice to the
                    # recognizer — an open mic next to the speakers would
                    # loop our phrases back as commands.
                    tts_gap = True
                    continue
                if tts_gap:
                    tts_gap = False
                    rec.Reset()   # do not splice across the muted stretch
                if drops['n']:
                    n, drops['n'] = drops['n'], 0
                    rec.Reset()
                    while True:   # the backlog is the stall's stale audio —
                        try:      # start clean from NOW
                            audio.get_nowait()
                        except queue.Empty:
                            break
                    print('(audio overrun: %d blocks dropped — recognizer '
                          'reset; say that again)' % n)
                    voice.say('I missed that. Try again.')
                    continue
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get('text', '')
                    # Empty finals dispatch too: they close the utterance,
                    # and the gate clears its stop rumor on them. dispatch
                    # runs BEFORE the (blocking, freezable) console print —
                    # nothing may sit between recognition and a stop send.
                    dispatch(text, final=True)
                    if text:
                        print('  heard: %s' % text)
                else:
                    # Partials are processed ONLY for stop words (see
                    # CommandGate): sub-second stop latency is worth regex
                    # work 4x/s; everything else waits for the final.
                    part = json.loads(rec.PartialResult()).get('partial', '')
                    if part:
                        dispatch(part, final=False)
        except KeyboardInterrupt:
            print()
    link['shutting_down'] = True
    status.unsubscribe()
    task_status.unsubscribe()
    say_topic.unsubscribe()
    client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

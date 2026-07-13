"""RAMMP voice control, native and OFFLINE: "computer, go to my bottle."

Runs on the DEV MACHINE (no ROS needed) — double-click the Desktop launcher
or:
    pip install vosk sounddevice pyttsx3 roslibpy pyyaml
    python computer.py --host 192.168.1.11

First run downloads the small Vosk English model (~40 MB) automatically.
Recognition is GRAMMAR-CONSTRAINED to this project's vocabulary, which makes
it fast and hard to mishear, and it runs fully offline (no browser, no cloud,
no mic permission prompts). Replies are printed and spoken.

    "computer, go to my bottle"   -> arm goes (fires on partial results:
                                     speech-end to motion well under a second)
    "computer"                    -> armed for 6 s, then say the command
    "computer, stop"              -> the arm halts immediately

Ctrl-C to quit. --list-mics to pick an input device (--mic N).
"""

import argparse
import json
import os
import queue
import re
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
# Grammar = every word the recognizer is allowed to hear. Small vocabulary =
# fast, accurate, offline. [unk] absorbs everything else. The OBJECT words
# come from scene.yaml (target names + keywords) so new targets are
# automatically hearable — a hardcoded list silently deafened the
# recognizer to 'pills' when that target was added.
FILLER = ('computer go to the my a please grab get open take pick '
          'fetch grasp release drop stop halt freeze cancel check home')
ARM_WINDOW_S = 6.0
COOLDOWN_S = 2.5


def scene_words(scene_path=None):
    """Target names + keywords from scene.yaml -> grammar/known words."""
    import yaml
    if scene_path is None:
        scene_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'config', 'scene.yaml')
    words = set()
    try:
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
        print('WARNING: could not read scene.yaml (%s) — using a stale '
              'built-in word list.' % exc)
        words = set(('bottle water drink mug cup coffee tea cabinet handle '
                     'door cupboard shelf snack cereal rest ready pills pill '
                     'medicine meds medication').split())
    return sorted(words)


def ensure_model():
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    z = MODEL_DIR + '.zip'
    print('First run: downloading the offline speech model (~40 MB)...')
    urllib.request.urlretrieve(MODEL_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(os.path.dirname(MODEL_DIR))
    os.remove(z)
    print('Model ready.')
    return MODEL_DIR


class Speaker:
    """TTS in a worker thread — pyttsx3's runAndWait blocks."""

    def __init__(self):
        import threading
        self._q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty('rate', 185)
        except Exception:
            eng = None
        while True:
            text = self._q.get()
            if eng is not None:
                try:
                    eng.say(text)
                    eng.runAndWait()
                except Exception:
                    pass

    def say(self, text):
        self._q.put(text)


def shorten(s):
    if s.startswith('STOPPED'):
        return 'Stopped.'
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
    args = ap.parse_args(argv)

    import sounddevice as sd
    if args.list_mics:
        print(sd.query_devices())
        return 0

    from vosk import Model, KaldiRecognizer
    import roslibpy

    objects = scene_words(args.scene)
    vocab = list(dict.fromkeys(FILLER.split() + objects))
    known = re.compile(r'\b(%s)\b' % '|'.join(
        objects + ['stop', 'halt', 'freeze', 'cancel', 'check', 'home']))
    print('Vocabulary (%d words): %s' % (len(vocab), ' '.join(vocab)))

    model = Model(ensure_model())
    grammar = json.dumps(vocab + ['[unk]'])
    rec = KaldiRecognizer(model, 16000, grammar)

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()
    if not client.is_connected:
        sys.exit('Could not reach rosbridge at ws://%s:%d — is the Jetson '
                 'bringup running?' % (args.host, args.port))
    cmd = roslibpy.Topic(client, '/curobo_planner/command', 'std_msgs/String')
    status = roslibpy.Topic(client, '/curobo_planner/status', 'std_msgs/String')
    voice = Speaker()

    first_status = [True]

    def on_status(msg):
        if first_status[0]:          # latched stale value arrives on subscribe
            first_status[0] = False
            return
        s = msg.get('data', '')
        print('  planner: %s' % s)
        if not s.startswith('...'):
            voice.say(shorten(s))
    status.subscribe(on_status)

    audio = queue.Queue()

    def on_audio(indata, frames, t, st):
        audio.put(bytes(indata))

    state = {'armed_until': 0.0, 'last_fire': 0.0, 'fired': False}

    def fire(text):
        state['last_fire'] = time.time()
        state['fired'] = True
        state['armed_until'] = 0.0
        print('-> %s' % text)
        cmd.publish(roslibpy.Message({'data': text}))

    def handle(text, final):
        now = time.time()
        # (No fired-flag consumption here: that was a leftover from partial-
        # result firing and it ATE every other final — "say it twice" bug.
        # The cooldown alone debounces.)
        if now - state['last_fire'] < COOLDOWN_S:
            return
        text = text.replace('[unk]', ' ').strip()
        m = WAKE.search(text)
        command = None
        if m:
            after = text[m.end():].strip(' ,.!?')
            if len(after) >= 3:
                command = after
            elif final:
                state['armed_until'] = now + ARM_WINDOW_S
                print('  (armed — say a command)')
                return
        elif now < state['armed_until'] and len(text) >= 3:
            command = text
        # FINALS ONLY: grammar-constrained partials are volatile (they force
        # the audio onto vocabulary words and rewrite drastically mid-
        # utterance — field log fired 'coffee' during "go home"). Vosk
        # finalizes ~0.3 s after you stop speaking, so the latency cost is
        # negligible; KNOWN still gates out wake-word-only fragments.
        if command and final and (known.search(command) or len(command) >= 6):
            fire(command)

    print('Listening (offline). Say "computer, go to my bottle" — Ctrl-C quits.')
    with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                           channels=1, callback=on_audio, device=args.mic):
        try:
            while True:
                data = audio.get()
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get('text', '')
                    if text:
                        print('  heard: %s' % text)
                        handle(text, final=True)
                else:
                    part = json.loads(rec.PartialResult()).get('partial', '')
                    if part:
                        handle(part, final=False)
        except KeyboardInterrupt:
            print()
    status.unsubscribe()
    client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

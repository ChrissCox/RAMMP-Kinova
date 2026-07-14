"""Offline test of the voice app: utterance state machine + vocabulary.

Runs anywhere (no ROS, no mic, no network; cases needing PyYAML or Vosk
skip with a named reason): drives computer.CommandGate through the exact
failure scenarios two review rounds found, and cross-checks the grammar,
the fire gate, and the stop regex against planner_node.py's word sets and
scene.yaml — the recognizer is grammar-constrained, so a word missing from
the grammar is a word the system is DEAF to, and a word the gate will not
fire is deaf with extra steps (a fork of computer.py once lost 'release':
held objects could never be let go by voice, and nothing complained).

    python3 tools/voice_gate_test.py        # bar: every case PASS
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE = os.path.join(ROOT, 'ros2_ws', 'src', 'curobo_planner', 'voice')
PLANNER = os.path.join(ROOT, 'ros2_ws', 'src', 'curobo_planner',
                       'curobo_planner', 'planner_node.py')
SCENE = os.path.join(ROOT, 'ros2_ws', 'src', 'curobo_planner',
                     'config', 'scene.yaml')
sys.path.insert(0, VOICE)
import computer  # noqa: E402  (stdlib-only at import time, by design)

# 'estop' is in planner_node's STOP_WORDS but the small acoustic model has
# no such token — it can never be heard, so it is exempt from the coverage
# checks below.
UNHEARABLE = {'estop'}

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def gate():
    return computer.CommandGate(computer.known_regex(['bottle', 'pills']))


def planner_words(setname):
    src = open(PLANNER, encoding='utf-8').read()
    m = re.search(r'^%s\s*=\s*\{([^}]*)\}' % setname, src, re.M)
    assert m, 'could not find %s in planner_node.py' % setname
    words = re.findall(r"['\"]([a-z ]+)['\"]", m.group(1))
    assert words, ('%s parsed to an EMPTY list — the coverage checks below '
                   'would pass vacuously; fix this parser' % setname)
    return words


# ----------------------------------------------------------- stop path
@case('stop within the cooldown of a fire is honored')
def _():
    g = gate()
    assert g.handle('computer go home', True, 0.0) == ('fire', 'go home')
    kind, _ = g.handle('computer stop', True, 1.9)   # inside COOLDOWN_S
    assert kind == 'stop', 'cooldown swallowed a stop'


@case('"stop, computer" word order stops (text before the wake word)')
def _():
    kind, _ = gate().handle('stop computer', True, 0.0)
    assert kind == 'stop'


@case('bare stop word, no wake word, not armed — still stops')
def _():
    kind, _ = gate().handle('halt', True, 0.0)
    assert kind == 'stop'


@case('stop clears the armed window')
def _():
    g = gate()
    assert g.handle('computer', True, 0.0)[0] == 'armed'
    assert g.handle('stop', True, 1.0)[0] == 'stop'
    assert g.handle('go to my bottle', True, 2.0) == (None, None), \
        'armed window survived a stop'


@case('stop resets the fire cooldown (the correction is not an echo)')
def _():
    g = gate()
    assert g.handle('computer go home', True, 0.0)[0] == 'fire'
    assert g.handle('stop', True, 1.0)[0] == 'stop'
    assert g.handle('computer go to my bottle', True, 2.0)[0] == 'fire', \
        'post-stop corrective command was refused as an echo'


@case('one partial with a stop word is a rumor (no fire)')
def _():
    g = gate()
    assert g.handle('stop', False, 0.0) == (None, None)


@case('two quick stop partials fire, the final echo is deduped')
def _():
    g = gate()
    assert g.handle('stop', False, 0.00) == (None, None)
    assert g.handle('stop', False, 0.25)[0] == 'stop'
    assert g.handle('stop', True, 0.50) == (None, None), \
        'the final echo of one stop was double-fired'


@case('a stop rumor EXPIRES: two partials far apart do not confirm')
def _():
    g = gate()
    assert g.handle('stop', False, 0.0) == (None, None)
    assert g.handle('stop', False, 2.0) == (None, None), \
        'a stale rumor confirmed an unrelated partial'
    assert g.handle('stop', False, 2.5)[0] == 'stop'   # these two agree


@case('an empty final clears the stop rumor (silence closes the utterance)')
def _():
    g = gate()
    assert g.handle('stop', False, 0.0) == (None, None)
    assert g.handle('', True, 0.3) == (None, None)
    assert g.handle('stop', False, 0.5) == (None, None), \
        'a rumor survived the empty final that ended its utterance'


@case('a spurious single stop partial does not poison the next utterance')
def _():
    g = gate()
    assert g.handle('coffee stop', False, 0.0) == (None, None)   # rumor
    assert g.handle('go home', False, 0.25) == (None, None)      # rewritten
    assert g.handle('computer go home', True, 0.8) == ('fire', 'go home')
    # and a LATER lone stop partial still needs its own confirmation
    assert g.handle('stop', False, 1.2) == (None, None)


@case('a repeated stop past the dedupe window is honored again')
def _():
    g = gate()
    assert g.handle('stop', True, 0.0)[0] == 'stop'
    assert g.handle('stop', True, 0.5) == (None, None)
    assert g.handle('stop', True, 2.0)[0] == 'stop'


@case('reset_stop() reopens the stop gate after a refused publish')
def _():
    g = gate()
    assert g.handle('stop', True, 0.0)[0] == 'stop'
    g.reset_stop()
    assert g.handle('stop', True, 0.5)[0] == 'stop', \
        'a refused stop locked out the immediate retry'


# ------------------------------------------------- commands and arming
@case('wake + command in one utterance fires')
def _():
    assert gate().handle('computer go to my pills', True, 0.0) == \
        ('fire', 'go to my pills')


@case('bare planner verbs fire ("computer, drop" is a complete release)')
def _():
    assert gate().handle('computer drop', True, 0.0) == ('fire', 'drop'), \
        'a hearable release verb did not fire — deafness with extra steps'


@case('arming is exempt from the cooldown; the follow-up then fires')
def _():
    g = gate()
    assert g.handle('computer go home', True, 0.0)[0] == 'fire'
    assert g.handle('computer', True, 1.5)[0] == 'armed', \
        'cooldown swallowed the wake-only arm'
    assert g.handle('go to my bottle', True, 4.0) == \
        ('fire', 'go to my bottle')


@case('a command inside the cooldown is reported, not silently dropped')
def _():
    g = gate()
    assert g.handle('computer go home', True, 0.0)[0] == 'fire'
    assert g.handle('computer check', True, 1.0) == ('ignored', 'check')


@case('"computer" + [unk] tail does not arm (mumble is not a wake)')
def _():
    g = gate()
    assert g.handle('computer [unk]', True, 0.0) == (None, None), \
        'mumbled wake armed the gate — junk finals could then fire bare'
    assert g.handle('go banana bottle', True, 1.0) == (None, None)
    assert g.handle('computer', True, 2.0)[0] == 'armed'   # clean wake still arms


@case('the armed window expires')
def _():
    g = gate()
    assert g.handle('computer', True, 0.0)[0] == 'armed'
    assert g.handle('go to my bottle', True, 7.0) == (None, None)


@case('non-stop partials never act (finals only)')
def _():
    g = gate()
    assert g.handle('computer go to my bottle', False, 0.0) == (None, None)
    assert g.handle('computer', False, 0.0) == (None, None), \
        'a partial armed the gate'


@case('reset_cooldown() reopens the gate after a refused publish')
def _():
    g = gate()
    assert g.handle('computer go home', True, 0.0)[0] == 'fire'
    g.reset_cooldown()
    assert g.handle('computer go home', True, 0.5)[0] == 'fire'


@case('unfire() restores the armed window a refused armed-mode fire consumed')
def _():
    g = gate()
    assert g.handle('computer', True, 0.0)[0] == 'armed'
    assert g.handle('go to my bottle', True, 1.0)[0] == 'fire'
    g.unfire(1.0)   # the publish was refused
    assert g.handle('go to my bottle', True, 2.0)[0] == 'fire', \
        'the armed-mode retry needed the wake word again'


@case('an [unk]-only partial is no evidence: the stop rumor survives it')
def _():
    g = gate()
    assert g.handle('stop', False, 0.0) == (None, None)      # rumor
    assert g.handle('[unk]', False, 0.25) == (None, None)    # no evidence
    assert g.handle('stop', False, 0.5)[0] == 'stop', \
        'an unrecognized-audio partial killed the stop rumor'


@case('[unk] noise is stripped, not fused into words')
def _():
    assert gate().handle('[unk] computer [unk] go home [unk]', True, 0.0) == \
        ('fire', 'go home')


# ------------------------------------------------- vocabulary coverage
@case('grammar covers every planner stop/grasp/release word (deafness check)')
def _():
    vocab = set(computer.FILLER.split())
    missing = []
    for setname in ('STOP_WORDS', 'GRASP_WORDS', 'RELEASE_WORDS'):
        for word in planner_words(setname):
            for tok in word.split():
                if tok not in vocab and tok not in UNHEARABLE:
                    missing.append('%s:%s' % (setname, tok))
    assert not missing, 'grammar is DEAF to planner words: %s' % missing


@case('STOP_RE covers every hearable planner stop word (client stop check)')
def _():
    missing = [w for w in planner_words('STOP_WORDS')
               if w not in UNHEARABLE and not computer.STOP_RE.search(w)]
    assert not missing, \
        'planner stop words the CLIENT treats as normal commands: %s' % missing


@case('every hearable grasp/release verb fires standalone (fire-gate check)')
def _():
    silent = []
    for setname in ('GRASP_WORDS', 'RELEASE_WORDS'):
        for w in planner_words(setname):
            if w in UNHEARABLE or ' ' in w:
                continue
            kind, payload = gate().handle('computer %s' % w, True, 0.0)
            if (kind, payload) != ('fire', w):
                silent.append('%s -> %s' % (w, kind))
    assert not silent, 'hearable verbs the gate will not fire: %s' % silent


@case('grammar covers scene.yaml free-prop names (grab-by-name check)')
def _():
    try:
        import yaml
    except ImportError:
        print('    (skip: pyyaml not installed here)')
        return
    scene = yaml.safe_load(open(SCENE, encoding='utf-8'))
    words = set(computer.scene_words())
    missing = []
    for o in scene.get('objects', []):
        if o.get('free', False):
            for sub in re.split(r'[^a-z]+', str(o.get('name', '')).lower()):
                if len(sub) >= 3 and sub not in words:
                    missing.append(sub)
    assert not missing, 'free props unhearable by name: %s' % missing


@case('grammar covers every scene.yaml target name + keyword')
def _():
    try:
        import yaml
    except ImportError:
        print('    (skip: pyyaml not installed here)')
        return
    scene = yaml.safe_load(open(SCENE, encoding='utf-8'))
    words = set(computer.scene_words())
    missing = []
    for t in scene.get('targets', []):
        for w in [t.get('name', '')] + list(t.get('keywords', [])):
            for sub in re.split(r'[^a-z]+', str(w).lower()):
                if len(sub) >= 3 and sub not in words:
                    missing.append(sub)
    assert not missing, 'targets unhearable: %s' % missing


@case('Vosk accepts the real grammar (skipped unless model is present)')
def _():
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError:
        print('    (skip: vosk not installed here)')
        return
    if not os.path.isdir(computer.MODEL_DIR):
        print('    (skip: model not downloaded at %s)' % computer.MODEL_DIR)
        return
    import json
    vocab = list(dict.fromkeys(computer.FILLER.split() + computer.scene_words()))
    rec = KaldiRecognizer(Model(computer.MODEL_DIR), 16000,
                          json.dumps(vocab + ['[unk]']))
    rec.AcceptWaveform(b'\x00' * 8000)   # a real decode tick, not just init
    assert rec is not None


def main():
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print('PASS  %s' % name)
        except AssertionError as exc:
            failed += 1
            print('FAIL  %s\n      %s' % (name, exc))
        except Exception as exc:   # a crash is a failure with a name too
            failed += 1
            print('FAIL  %s\n      %s: %s' % (name, type(exc).__name__, exc))
    total = len(CASES)
    print('%d/%d clean' % (total - failed, total))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

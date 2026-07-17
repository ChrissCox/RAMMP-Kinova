#!/usr/bin/env zsh
# GraspGen-X on the Jetson, the --no-deps way (the AnyGrasp playbook):
# its pyproject pins (numpy==1.26.4, torch<2.7, scene-synthesizer with an
# x86-only usd-core wheel) would fight the Jetson CUDA torch — so install
# the package bare and add ONLY the runtime deps, against system torch.
#
#   zsh ~/RAMMP-Kinova/tools/install_graspgen.zsh
#
# Idempotent: safe to re-run; each step skips what already exists.
# After it passes: restart the stack (tools/launch_stack.zsh) — the
# bringup now starts the GraspGen ZMQ server itself.
set -e

REPO=~/GraspGenX
VENV=~/graspgen_venv

echo "== 1/6 prerequisites =="
command -v git-lfs >/dev/null || {
    echo "git-lfs missing (checkpoint clone needs it): sudo apt install git-lfs && git lfs install"
    exit 1
}
python3 - <<'EOF'
import torch
v = torch.__version__
print('system torch %s (cuda %s)' % (v, torch.cuda.is_available()))
major, minor = (int(x) for x in v.split('.')[:2])
assert (major, minor) >= (2, 1), 'GraspGenX needs torch>=2.1 — found %s' % v
if (major, minor) >= (2, 7):
    print('WARNING: torch >= 2.7 — above GraspGenX\'s tested cap (<2.7); proceed and watch the smoke test')
EOF

echo "== 2/6 clone =="
[[ -d $REPO ]] || git clone https://github.com/NVlabs/GraspGenX.git $REPO

echo "== 3/6 venv (system-site-packages: reuse the Jetson torch) =="
[[ -d $VENV ]] || python3 -m venv $VENV --system-site-packages
source $VENV/bin/activate
# Their pinned build backend predates PEP 660 (no build_editable hook) —
# bring our own setuptools and build without isolation.
pip install -U 'setuptools>=64' wheel
pip install --no-deps --no-build-isolation -e $REPO
# Runtime dep subset (NOT the full pyproject list — no scene-synthesizer,
# no pyrender, no training stack). torch-geometric is pure-Python since
# their pin; ptv3vanilla backbone needs no spconv/MinkowskiEngine.
# transformers is VENV-LOCAL and OLD on purpose: diffusers 0.11.1 needs
# huggingface-hub 0.25 (cached_download), the SYSTEM transformers 4.57
# demands hub>=0.34 — a venv 4.46.3 shadows it and accepts hub 0.25.
# (Converged on the Jetson 2026-07-17; the import check below verifies.)
pip install torch-geometric 'diffusers==0.11.1' 'huggingface-hub==0.25.2' \
    'transformers==4.46.3' 'timm==1.0.15' 'trimesh==4.5.3' \
    hydra-core omegaconf einops h5py scikit-learn webdataset tensordict \
    addict pyzmq msgpack msgpack-numpy

echo "== 4/6 import check (names every missing module — install and re-run) =="
python - <<'EOF'
import importlib, sys
missing = []
for mod in ('graspgenx', 'graspgenx.grasp_server', 'graspgenx.samplers',
            'graspgenx.utils.checkpoint_io', 'graspgenx.serving'):
    try:
        importlib.import_module(mod)
    except ImportError as exc:
        missing.append('%s -> %s' % (mod, exc))
print('\n'.join(missing) if missing else 'all imports clean')
sys.exit(1 if missing else 0)
EOF

echo "== 5/6 checkpoints (auto-cloned by first import; ~1.7 GB + grippers) =="
python - <<'EOF'
from graspgenx import get_checkpoints_version_dir
d = get_checkpoints_version_dir()
print('checkpoints at', d)
EOF

echo "== 6/6 smoke test: cylinder cloud -> grasps (prints Orin latency) =="
python - <<'EOF'
import time
import numpy as np
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg

root = str(get_checkpoints_version_dir())
cfg = load_model_cfg(root + '/gen', root + '/dis')
t0 = time.monotonic()
sampler = GraspGenXSampler(cfg, 'robotiq_2f_85',
                           assets_dir=__import__('os').path.expanduser(
                               '~/GraspGenX/assets'))
print('model loaded in %.1f s' % (time.monotonic() - t0))
rng = np.random.default_rng(0)
th = rng.uniform(0, 2 * np.pi, 2000)
z = rng.uniform(0.0, 0.12, 2000)
pc = np.stack([0.03 * np.cos(th), 0.03 * np.sin(th), z], -1).astype('f4')
for run in range(2):   # run 0 pays CUDA warmup; run 1 is the honest number
    t0 = time.monotonic()
    grasps, conf, tags, _ = run_planner_on_object(pc, sampler)
    print('run %d: %d grasps in %.2f s, top conf %s'
          % (run, len(grasps), time.monotonic() - t0,
             np.round(conf[:5], 3) if len(conf) else '[]'))
assert len(grasps) > 0, 'no grasps on a trivial cylinder — investigate'
EOF

echo ""
echo "PASS. The ROS process also needs the ZMQ client libs (system python):"
echo "    pip install --user pyzmq msgpack msgpack-numpy"
echo "Then restart the stack:  zsh ~/RAMMP-Kinova/tools/launch_stack.zsh"

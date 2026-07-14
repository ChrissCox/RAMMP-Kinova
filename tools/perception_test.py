"""End-to-end offline validation of the perception pipeline (no ROS needed):
render scene_cam RGB+depth from the actual sim model, run the REAL detector
code (backends + geometry), and compare every estimated position against the
sim's ground-truth prop positions.

    python3 tools/perception_test.py
    python3 tools/perception_test.py --xml /path/scene_gen3.xml

Bar: every free prop <= 40 mm horizontal error (typically <= 10 mm).
CAVEAT this cannot catch: the offline Renderer's depth mode ignores alpha,
but the LIVE camera pass skips depth writes for transparent geometry — keep
perception-tracked props opaque (see CLAUDE.md).
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ros2_ws', 'src', 'rammp_perception'))
from rammp_perception.backends import ColorBackend            # noqa: E402
from rammp_perception.geometry import CameraModel, mask_to_position  # noqa: E402
from rammp_perception.geometry import size_hint as _size_hint  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--xml', default=os.path.expanduser(
    '~/.ros/mujoco_sim/scene_gen3.xml'))
ap.add_argument('--scene', default=os.path.join(
    REPO, 'ros2_ws', 'src', 'curobo_planner', 'config', 'scene.yaml'))
args = ap.parse_args()

m = mujoco.MjModel.from_xml_path(args.xml)
d = mujoco.MjData(m)
HOME = {'joint_1': 0.0, 'joint_2': 0.262, 'joint_3': 3.142,
        'joint_4': -2.269, 'joint_5': 0.0, 'joint_6': 0.960, 'joint_7': 1.571}
for n, q in HOME.items():
    d.qpos[m.jnt_qposadr[m.joint(n).id]] = q
mujoco.mj_forward(m, d)

W, H = 640, 480
r = mujoco.Renderer(m, height=H, width=W)
r.update_scene(d, camera='scene_cam')
rgb = r.render().copy()
r.enable_depth_rendering()
r.update_scene(d, camera='scene_cam')
depth = r.render().copy()

# camera model straight from scenery.py's scene_cam + the model's fovy
cam = CameraModel([-0.75, 0.0, 1.45], [0.0, -1.0, 0.0, 0.858, 0.0, 0.514])
cam.set_intrinsics_from_fovy(float(m.cam_fovy[m.cam('scene_cam').id]), W, H)

sc = yaml.safe_load(open(args.scene, encoding='utf-8'))
FREE = [o for o in sc['objects'] if o.get('free', False)]
classes = {o['name']: list(o.get('color', [0.8, 0.8, 0.8, 1.0]))[:3]
           for o in FREE}
HINTS = {o['name']: _size_hint(o) for o in FREE}
WORKSPACE = [(-0.55, 0.85), (-0.75, 0.75), (-0.10, 0.75)]
MAX_JUMP = 0.25

truth = {}
for o in FREE:
    for b in range(m.nbody):
        if m.body(b).name == 'obj_' + o['name']:
            truth[o['name']] = np.array(d.xpos[b])
    truth.setdefault(o['name'],
                     np.array([float(v) for v in o['position']]))
last = {o['name']: np.array([float(v) for v in o['position']]) for o in FREE}

cands = {}
for det in ColorBackend(classes).detect(rgb):
    pos, extent = mask_to_position(det.mask, depth, cam,
                                   HINTS.get(det.label))
    if pos is None:
        continue
    if not all(lo <= p <= hi for p, (lo, hi) in zip(pos, WORKSPACE)):
        continue
    cands.setdefault(det.label, []).append(pos)

print('%-18s %8s   estimated position        truth' % ('prop', 'err(mm)'))
worst = 0.0
for label, t in truth.items():
    ps = sorted(cands.get(label, []),
                key=lambda p: np.linalg.norm(p - last[label]))
    if not ps or np.linalg.norm(ps[0] - last[label]) > MAX_JUMP:
        print('%-18s   MISSED (honest: stale YAML pose keeps being used)'
              % label)
        continue
    p = ps[0]
    err = float(np.linalg.norm(p[:2] - t[:2]))   # horizontal: Z is YAML-held
    worst = max(worst, err)
    print('%-18s %8.1f   [%6.3f %6.3f %6.3f]  [%6.3f %6.3f %6.3f]'
          % (label, err * 1000, p[0], p[1], p[2], t[0], t[1], t[2]))
verdict = 'PASS (<40mm)' if worst < 0.04 else 'FAIL'
print('worst horizontal error: %.1f mm - %s' % (worst * 1000, verdict))
sys.exit(0 if worst < 0.04 else 1)

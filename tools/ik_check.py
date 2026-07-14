"""Ground-truth IK feasibility for scene targets + ad-hoc poses, offline.

    python3 tools/ik_check.py                                  # every target
    python3 tools/ik_check.py grasp_x,0.45,0.12,0.12,180,0,0   # + ad-hoc poses
    python3 tools/ik_check.py --xml /path/scene_gen3.xml --scene /path/scene.yaml

For each pose: damped-least-squares IK (position + tool-axis alignment, yaw
free) from 30 random restarts, honoring joint limits; converged solutions are
contact-checked against the scene. This is the reachability truth cuRobo's
IK_FAIL refuses to explain. NOTE: 5-DOF axis check — more permissive than
cuRobo's full 6-DOF goal with tool_spin, so treat failures as definitive and
successes as necessary-but-not-sufficient.
"""
import argparse
import math
import os
import sys

import mujoco
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.expanduser('~/.ros/mujoco_sim/scene_gen3.xml')
DEFAULT_SCENE = os.path.join(REPO, 'ros2_ws', 'src', 'curobo_planner',
                             'config', 'scene.yaml')

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--xml', default=DEFAULT_XML)
ap.add_argument('--scene', default=DEFAULT_SCENE)
ap.add_argument('cases', nargs='*', help='name,x,y,z,roll,pitch,yaw (deg)')
args = ap.parse_args()

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)

ARM = ['joint_%d' % i for i in range(1, 8)]
jids = [model.joint(n).id for n in ARM]
qadr = [model.jnt_qposadr[j] for j in jids]
dofs = [model.jnt_dofadr[j] for j in jids]
limited = [bool(model.jnt_limited[j]) for j in jids]
jrange = [model.jnt_range[j].copy() for j in jids]

base_body = model.body('g_base').id
root = model.jnt_bodyid[model.joint('joint_1').id]
while model.body_parentid[root] != 0:
    root = model.body_parentid[root]


def under(b, r):
    while b != 0:
        if b == r:
            return True
        b = model.body_parentid[b]
    return False


robot_bodies = {b for b in range(model.nbody) if under(b, root)}
pad_bodies = [b for b in range(model.nbody)
              if 'pad' in (model.body(b).name or '') and b in robot_bodies]
assert pad_bodies, 'no pad bodies found'


def set_q(q):
    for a, v in zip(qadr, q):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)


def tool_state():
    """Fingertip-midpoint frame: flange + 0.120*axis (cuRobo's tool_frame)."""
    p_base = data.xpos[base_body].copy()
    pads = np.mean([data.xpos[b] for b in pad_bodies], axis=0)
    axis = pads - p_base
    axis /= np.linalg.norm(axis)
    return p_base + 0.120 * axis, axis


def solve(target_p, target_axis, q0, iters=400):
    q = np.array(q0, float)
    set_q(q)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        p, z = tool_state()
        ep = target_p - p
        ez = np.cross(z, target_axis)
        if np.linalg.norm(ep) < 0.004 and \
           math.degrees(math.asin(min(1.0, np.linalg.norm(ez)))) < 2.0 and \
           float(np.dot(z, target_axis)) > 0:
            return q, True
        lam = 0.12 if np.linalg.norm(ep) > 0.05 else 0.03
        mujoco.mj_jac(model, data, jacp, jacr, p.copy(), base_body)
        J = np.zeros((6, 7))
        for c, d in enumerate(dofs):
            J[:3, c] = jacp[:, d]
            J[3:, c] = jacr[:, d]
        err = np.concatenate([ep, 1.0 * ez])
        if float(np.dot(z, target_axis)) < 0:
            err[3:] = 1.0 * np.cross(z, target_axis + 1e-3)
            if np.linalg.norm(err[3:]) < 1e-6:
                err[3:] = np.array([0.5, 0, 0])
        y = np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        dq = np.clip(J.T @ y, -0.2, 0.2)
        q = q + dq
        for i in range(7):
            if limited[i]:
                q[i] = min(max(q[i], jrange[i][0] + 1e-3), jrange[i][1] - 1e-3)
        set_q(q)
    return q, False


def contacts(q):
    set_q(q)
    out = {}
    for c in range(data.ncon):
        con = data.contact[c]
        depth = -float(con.dist)
        if depth < 0.001:
            continue
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        r1, r2 = b1 in robot_bodies, b2 in robot_bodies
        if not (r1 or r2):
            continue

        def nm(g):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            return n or model.body(model.geom_bodyid[g]).name

        kind = 'SELF' if (r1 and r2) else 'SCENE'
        out['%s: %s vs %s' % (kind, nm(con.geom1), nm(con.geom2))] = \
            max(out.get('%s: %s vs %s' % (kind, nm(con.geom1), nm(con.geom2)),
                        0.0), depth)
    return out


def rot_axis(rpy_deg):
    r, p, y = (math.radians(a) for a in rpy_deg)
    return np.array([
        math.cos(y) * math.sin(p) * math.cos(r) + math.sin(y) * math.sin(r),
        math.sin(y) * math.sin(p) * math.cos(r) - math.cos(y) * math.sin(r),
        math.cos(p) * math.cos(r),
    ])


with open(args.scene, encoding='utf-8') as f:
    scene = yaml.safe_load(f)

rng = np.random.default_rng(0)
HOME = [0.0, 0.262, 3.142, -2.269, 0.0, 0.960, 1.571]
set_q(HOME)
BASELINE = set(contacts(HOME).keys())   # resting mesh interpenetration

cases = [(t['name'], np.array([float(v) for v in t['position']]),
          rot_axis([float(v) for v in t.get('rpy_deg', [180, 0, 0])]))
         for t in scene['targets']]
for spec in args.cases:
    parts = spec.split(',')
    cases.append((parts[0], np.array([float(v) for v in parts[1:4]]),
                  rot_axis([float(v) for v in parts[4:7]])))

for name, tp, ta in cases:
    ok_qs = []
    seeds = [HOME] + [
        [rng.uniform(jrange[i][0], jrange[i][1]) if limited[i]
         else rng.uniform(-math.pi, math.pi) for i in range(7)]
        for _ in range(29)]
    for q0 in seeds:
        q, ok = solve(tp, ta, q0)
        if ok:
            ok_qs.append(q)
    if not ok_qs:
        print('%-16s UNREACHABLE (0/30 IK restarts converged)' % name)
        continue
    best_hits, n_clean = None, 0
    for q in ok_qs:
        hits = {k: v for k, v in contacts(q).items() if k not in BASELINE}
        if not hits:
            n_clean += 1
        if best_hits is None or len(hits) < len(best_hits):
            best_hits = hits
    print('%-16s reachable %d/30, collision-free solutions: %d'
          % (name, len(ok_qs), n_clean))
    if n_clean == 0:
        for k, v in sorted(best_hits.items(), key=lambda kv: -kv[1]):
            print('    best solution still hits  %s  (%.1f mm)' % (k, v * 1000))

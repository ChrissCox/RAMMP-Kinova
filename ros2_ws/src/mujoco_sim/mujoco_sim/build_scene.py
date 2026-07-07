"""Compose the MuJoCo scene: Menagerie Gen3 + Robotiq 2F-85 + the obstacle course.

Generates ONE merged MJCF from:
  * mujoco_menagerie/kinova_gen3/gen3.xml   (joints & position actuators joint_1..7 —
                                             identical names to ros2_kortex)
  * mujoco_menagerie/robotiq_2f85/2f85.xml  (attached per the Gen3 README recipe:
                                             gripper `base` into `bracelet_link` at
                                             pos "0 0 -0.06149039" quat "0 -1 1 0",
                                             excluding `base_mount`)
  * curobo_planner/config/scene.yaml        (obstacles mirrored as static boxes —
                                             the SAME boxes cuRobo plans around;
                                             NOTE MuJoCo box size = HALF extents)

ros2_control mapping fixes baked in (mujoco_ros2_control maps URDF joint names to
MJCF actuator/joint names):
  * actuator `fingers_actuator` -> renamed `robotiq_85_left_knuckle_joint`,
    ctrlrange 0..0.8 rad (was Robotiq-native 0..255; gain rescaled accordingly)
  * joint `left_driver_joint`   -> renamed `robotiq_85_left_knuckle_joint`
  * keyframes removed (they break compilation once the gripper adds joints)

Usage (after cloning the menagerie):
    ros2 run mujoco_sim build_scene --menagerie ~/mujoco_menagerie \
        [--out /path/scene_gen3.xml] [--scene /path/scene.yaml]

Run this ONCE (and again after editing scene.yaml); the bringup launch points
mujoco_ros2_control at the generated file. Uses the mjSpec API (mujoco>=3.1):
if an API detail differs on your installed version, this fails loudly with the
attribute name — report it rather than guessing.
"""

import argparse
import os
import sys

GRIPPER_ATTACH_POS = [0.0, 0.0, -0.06149039]      # from the menagerie Gen3 README
GRIPPER_ATTACH_QUAT = [0.0, -1.0, 1.0, 0.0]       # (w x y z), normalized by MuJoCo
ROS_GRIPPER_JOINT = 'robotiq_85_left_knuckle_joint'  # ros2_kortex's gripper joint
GRIPPER_MAX_RAD = 0.8                              # driver-joint range mapped 0..255
PREFIX = 'g_'                                      # namespace for attached gripper


def _find_scene_yaml(explicit):
    if explicit:
        return explicit
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(
        get_package_share_directory('curobo_planner'), 'config', 'scene.yaml')


def _default_out():
    return os.path.expanduser('~/.ros/mujoco_sim/scene_gen3.xml')


def _delete_keyframes(spec):
    """Remove keyframes across mjSpec API variants; return True on success."""
    keys = getattr(spec, 'keys', None) or getattr(spec, 'keyframes', None)
    if keys is None:
        # by-name fallback (menagerie gen3 ships 'home' and 'retract')
        getter = getattr(spec, 'key', None)
        keys = [k for k in (getter(n) for n in ('home', 'retract'))
                if k is not None] if getter else []
    ok = True
    for k in list(keys):
        try:
            spec.delete(k)
        except AttributeError:
            try:
                k.delete()
            except AttributeError:
                ok = False
    return ok


def _absolutize_meshes(spec, xml_path):
    """Rewrite mesh file refs to absolute paths so the merged XML loads from
    anywhere (attached sub-model assets otherwise resolve against the wrong dir)."""
    base = os.path.dirname(os.path.abspath(xml_path))
    meshdir = getattr(spec, 'meshdir', '') or ''
    for m in getattr(spec, 'meshes', []):
        f = getattr(m, 'file', None)
        if f and not os.path.isabs(f):
            m.file = os.path.normpath(os.path.join(base, meshdir, f))
    for t in getattr(spec, 'textures', []):
        f = getattr(t, 'file', None)
        if f and not os.path.isabs(f):
            texdir = getattr(spec, 'texturedir', '') or ''
            t.file = os.path.normpath(os.path.join(base, texdir, f))


def build(menagerie, scene_yaml, out_path):
    import mujoco
    import yaml

    gen3_xml = os.path.join(menagerie, 'kinova_gen3', 'gen3.xml')
    grip_xml = os.path.join(menagerie, 'robotiq_2f85', '2f85.xml')
    for p in (gen3_xml, grip_xml):
        if not os.path.isfile(p):
            sys.exit('Missing %s — clone github.com/google-deepmind/mujoco_menagerie '
                     'and pass --menagerie <path>' % p)

    spec = mujoco.MjSpec.from_file(gen3_xml)
    grip = mujoco.MjSpec.from_file(grip_xml)

    # The gripper's contact options must WIN the attach merge (the parent's
    # defaults otherwise override them, degrading grasp stability).
    try:
        spec.option.impratio = 10
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    except AttributeError as exc:
        print('WARNING: could not set contact options (%s); grasps may slip.' % exc,
              file=sys.stderr)

    # Resolve mesh/texture paths to absolute BEFORE attach, each against its
    # own source directory.
    _absolutize_meshes(spec, gen3_xml)
    _absolutize_meshes(grip, grip_xml)

    # Keyframes break compilation after the gripper adds joints — drop them.
    if not (_delete_keyframes(spec) and _delete_keyframes(grip)):
        print('WARNING: keyframe deletion incomplete on this mjSpec API; if '
              'compile fails on keyframe size, delete the <keyframe> block '
              'from your menagerie gen3.xml (one-time local edit).',
              file=sys.stderr)

    # -- Attach the gripper's `base` subtree (NOT base_mount) into bracelet_link.
    bracelet = spec.body('bracelet_link')
    if bracelet is None:
        sys.exit("gen3.xml has no body 'bracelet_link' — menagerie layout changed?")
    grip_base = grip.body('base')
    if grip_base is None:
        sys.exit("2f85.xml has no body 'base' — menagerie layout changed?")
    frame = bracelet.add_frame(pos=GRIPPER_ATTACH_POS, quat=GRIPPER_ATTACH_QUAT)
    frame.attach_body(grip_base, PREFIX, '')

    # -- Rename for ros2_control by-name mapping (actuator + state joint).
    act = spec.actuator(PREFIX + 'fingers_actuator')
    if act is None:
        sys.exit('Attached actuator %sfingers_actuator not found — check the '
                 'attach prefix handling on this mujoco version.' % PREFIX)
    act.name = ROS_GRIPPER_JOINT
    # Original: ctrl 0..255 with gain ~0.3137255. Rescale so ctrl is radians.
    scale = 255.0 / GRIPPER_MAX_RAD
    act.ctrlrange = [0.0, GRIPPER_MAX_RAD]
    try:
        act.gainprm[0] = act.gainprm[0] * scale
    except Exception as exc:
        sys.exit('Could not rescale gripper gain (%s) — inspect actuator gainprm.' % exc)
    # NOTE: the driver JOINT is deliberately NOT renamed here — the gripper's
    # equality constraints and tendon reference it by name string, and mjSpec
    # does not update references on rename (compile fails with "unknown
    # element"). The rename happens as a text substitution on the written XML
    # below, which updates definition + all references atomically.
    if spec.joint(PREFIX + 'left_driver_joint') is None:
        sys.exit('Attached joint %sleft_driver_joint not found.' % PREFIX)

    # -- World: floor, light, a viz camera, and the curobo obstacle boxes.
    world = spec.worldbody
    world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[3.0, 3.0, 0.1], rgba=[0.35, 0.38, 0.42, 1.0])
    try:
        world.add_light(pos=[0.0, 0.0, 2.5], dir=[0.0, 0.0, -1.0])
    except Exception:
        pass  # cosmetic only; the viewer's headlight suffices
    world.add_camera(name='viz_cam', pos=[1.6, -1.2, 1.0],
                     xyaxes=[0.6, 0.8, 0.0, -0.35, 0.26, 0.9])

    with open(scene_yaml, 'r') as f:
        scene = yaml.safe_load(f)
    for o in scene.get('obstacles', []):
        dims = [float(v) for v in o['dims']]
        pos = [float(v) for v in o['position']]
        color = [float(v) for v in o.get('color', [0.5, 0.5, 0.5, 1.0])]
        world.add_geom(
            name='obs_' + o['name'], type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[d / 2.0 for d in dims],  # MuJoCo box size = HALF extents!
            pos=pos, rgba=color)

    # -- Compile in-memory to validate the composition itself.
    model = spec.compile()
    print('Compiled OK: %d bodies, %d joints (nq=%d), %d actuators'
          % (model.nbody, model.njnt, model.nq, model.nu))

    # -- Write, then rename the driver joint by TEXT substitution (updates the
    #    joint def + equality + tendon references together).
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    xml = spec.to_xml()
    old = PREFIX + 'left_driver_joint'
    n = xml.count(old)
    if n == 0:
        sys.exit('Expected joint name %r not present in generated XML.' % old)
    xml = xml.replace(old, ROS_GRIPPER_JOINT)
    print('Renamed %s -> %s (%d references)' % (old, ROS_GRIPPER_JOINT, n))
    with open(out_path, 'w') as f:
        f.write(xml)

    # -- Definitive check: reload from disk (exactly what mujoco_ros2_control
    #    does at launch) and verify every name ros2_control will look up.
    m2 = mujoco.MjModel.from_xml_path(out_path)
    for name in ['joint_%d' % i for i in range(1, 8)] + [ROS_GRIPPER_JOINT]:
        if mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            sys.exit('Reload check FAILED: no joint named %r' % name)
    for name in ['joint_%d' % i for i in range(1, 8)] + [ROS_GRIPPER_JOINT]:
        if mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
            sys.exit('Reload check FAILED: no actuator named %r' % name)
    print('Wrote %s (reload-validated)' % out_path)
    print('Point mujoco_bringup.launch.py at it (mujoco_model:=%s)' % out_path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--menagerie', default=os.path.expanduser('~/mujoco_menagerie'),
                    help='path to a clone of google-deepmind/mujoco_menagerie')
    ap.add_argument('--scene', default=None,
                    help='curobo scene.yaml (default: installed curobo_planner config)')
    ap.add_argument('--out', default=_default_out())
    args = ap.parse_args(argv)
    return build(os.path.expanduser(args.menagerie),
                 _find_scene_yaml(args.scene),
                 os.path.expanduser(args.out))


if __name__ == '__main__':
    sys.exit(main())

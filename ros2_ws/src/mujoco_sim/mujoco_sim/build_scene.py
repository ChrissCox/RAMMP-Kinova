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
    joint = spec.joint(PREFIX + 'left_driver_joint')
    if joint is None:
        sys.exit('Attached joint %sleft_driver_joint not found.' % PREFIX)
    joint.name = ROS_GRIPPER_JOINT

    # -- Keyframes: gen3's home/retract have 7 qpos; with the gripper attached
    #    the sizes no longer match and compilation fails. Drop them all.
    try:
        for key in list(spec.keys):
            key.delete()
    except AttributeError:
        try:
            spec.delete_all_keyframes()
        except AttributeError:
            print('WARNING: could not delete keyframes via this mjSpec API; '
                  'if compile fails on keyframe size, remove them from gen3.xml.',
                  file=sys.stderr)

    # -- World: floor, light, a viz camera, and the curobo obstacle boxes.
    world = spec.worldbody
    world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[3.0, 3.0, 0.1], rgba=[0.35, 0.38, 0.42, 1.0])
    world.add_light(pos=[0.0, 0.0, 2.5], dir=[0.0, 0.0, -1.0], directional=True)
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

    # -- Compile to validate, then write the merged XML next to copied assets.
    model = spec.compile()
    print('Compiled OK: %d bodies, %d joints (nq=%d), %d actuators'
          % (model.nbody, model.njnt, model.nq, model.nu))
    # Sanity: every name ros2_control will look up must exist.
    for name in ['joint_%d' % i for i in range(1, 8)] + [ROS_GRIPPER_JOINT]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0:
            sys.exit('Post-compile check FAILED: no actuator named %r' % name)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            sys.exit('Post-compile check FAILED: no joint named %r' % name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(spec.to_xml())
    print('Wrote %s' % out_path)
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

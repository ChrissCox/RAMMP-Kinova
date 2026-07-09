"""Windows mirror: a native MuJoCo 3D window tracking the Jetson's simulation.

Runs on the DEV MACHINE (no ROS install needed):
    pip install mujoco roslibpy
    python -m mujoco_sim.mirror_viewer --host 192.168.1.11 \
        --model .\\scene_gen3.xml --menagerie C:\\path\\to\\mujoco_menagerie

The ARM is driven from /joint_states over rosbridge (launch the Jetson
bringup with mirror:=true). The PROPS run under LOCAL physics: only joint
states cross the network, so a purely kinematic puppet would show props
frozen at their spawn poses forever and the arm would appear to pass through
anything the real sim had nudged. Stepping physics locally lets the mirrored
arm push and knock the local props about like the Jetson's does. It is an
approximation (the two physics runs drift apart over time — press
Backspace in the viewer to reset the local props), and it cannot affect the
real sim. --kinematic restores the old frozen-prop behavior.

Needs the generated scene XML copied from the Jetson (~/.ros/mujoco_sim/
scene_gen3.xml) AND a local menagerie clone for the mesh assets — the XML
references them by path, so regenerate locally if paths differ:
    python -m mujoco_sim.build_scene --menagerie <local clone> --scene scene.yaml --out scene_gen3.xml
"""

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='192.168.1.11', help='Jetson IP (rosbridge)')
    ap.add_argument('--port', type=int, default=9090)
    ap.add_argument('--model', required=True, help='generated scene_gen3.xml')
    ap.add_argument('--kinematic', action='store_true',
                    help='frozen-prop puppet (no local physics)')
    args = ap.parse_args(argv)

    import mujoco
    import mujoco.viewer
    import roslibpy

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # joint name -> qpos address (hinge/slide joints have 1 dof each)
    qpos_addr = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            qpos_addr[name] = model.jnt_qposadr[j]

    latest = {}

    def on_joint_states(msg):
        for name, pos in zip(msg.get('name', []), msg.get('position', [])):
            latest[name] = pos

    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()
    if not client.is_connected:
        sys.exit('Could not reach rosbridge at ws://%s:%d — launch the Jetson '
                 'bringup with mirror:=true' % (args.host, args.port))
    topic = roslibpy.Topic(client, '/joint_states', 'sensor_msgs/JointState',
                           throttle_rate=33)  # ~30 Hz is plenty for a mirror
    topic.subscribe(on_joint_states)
    print('Mirroring /joint_states from ws://%s:%d — close the window to quit.'
          % (args.host, args.port))

    # Physics mode drives the model's own position ACTUATORS (same names as
    # the joints, matching ros2_control): the local arm tracks the mirrored
    # targets exactly like the Jetson's arm tracks its controller — no
    # teleported joints fighting actuator gains, no injected energy.
    act_id = {}
    for a in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if nm:
            act_id[nm] = a

    synced = False

    def drive_arm(physics):
        nonlocal synced
        for name, pos in list(latest.items()):
            if physics:
                a = act_id.get(name)
                if a is not None:
                    data.ctrl[a] = pos
            if not physics or not synced:
                adr = qpos_addr.get(name)
                if adr is not None:
                    data.qpos[adr] = pos
        if latest and not synced:
            synced = True   # initial hard sync so the arm doesn't swing in from zero

    steps = max(1, int(round((1.0 / 60.0) / model.opt.timestep)))
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if args.kinematic:
                    drive_arm(physics=False)
                    mujoco.mj_forward(model, data)
                else:
                    drive_arm(physics=True)
                    for _ in range(steps):
                        mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        topic.unsubscribe()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

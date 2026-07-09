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

    # The mirrored arm is KINEMATIC TRUTH: joints are set from /joint_states
    # every substep — never actuator-driven (an actuator-driven local arm
    # lags and can be BLOCKED by local props, showing the arm in the wrong
    # place entirely; field bug). The drive is rate-bounded so prop contacts
    # see finite velocities instead of teleports, and qvel carries the real
    # motion so pushes on props are physical.
    dof_addr = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and model.jnt_type[j] in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                          int(mujoco.mjtJoint.mjJNT_SLIDE)):
            dof_addr[name] = model.jnt_dofadr[j]
    act_id = {}
    for a in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if nm:
            act_id[nm] = a

    synced = False
    qpin = {}       # our own pinned joint state — data.qpos is perturbed by
                    # mj_step every substep, so tracking against it drifts
    CATCHUP = 4.0   # rad/s cap while converging onto the stream

    def drive_arm(dt):
        nonlocal synced
        for name, pos in list(latest.items()):
            adr = qpos_addr.get(name)
            if adr is None:
                continue
            if not synced or dt is None:
                data.qpos[adr] = pos
                qpin[name] = pos
                continue
            cur = qpin.get(name, pos)
            err = pos - cur
            step = max(-CATCHUP * dt, min(CATCHUP * dt, err))
            nxt = cur + step
            qpin[name] = nxt
            data.qpos[adr] = nxt            # exact kinematic pin
            d = dof_addr.get(name)
            if d is not None:
                data.qvel[d] = step / dt    # real rate: contacts push props
            a = act_id.get(name)
            if a is not None:
                data.ctrl[a] = pos          # actuators agree; no fighting
        if latest and not synced:
            synced = True   # initial hard sync so the arm doesn't swing in from zero

    steps = max(1, int(round((1.0 / 60.0) / model.opt.timestep)))
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if args.kinematic:
                    drive_arm(None)
                    mujoco.mj_forward(model, data)
                else:
                    for _ in range(steps):
                        drive_arm(model.opt.timestep)
                        mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        topic.unsubscribe()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

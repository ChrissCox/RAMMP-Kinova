"""Windows mirror: a native MuJoCo 3D window tracking the Jetson's simulation.

Runs on the DEV MACHINE (no ROS install needed):
    pip install mujoco roslibpy
    python -m mujoco_sim.mirror_viewer --host 192.168.1.11 \
        --model .\\scene_gen3.xml --menagerie C:\\path\\to\\mujoco_menagerie

It is a passive puppet: subscribes to /joint_states over rosbridge (launch the
Jetson bringup with mirror:=true), writes joint positions into a local MjData,
and renders with the interactive viewer. It cannot affect the real sim.

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

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                for name, pos in list(latest.items()):
                    adr = qpos_addr.get(name)
                    if adr is not None:
                        data.qpos[adr] = pos
                mujoco.mj_forward(model, data)  # kinematic puppet: no stepping
                viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        topic.unsubscribe()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

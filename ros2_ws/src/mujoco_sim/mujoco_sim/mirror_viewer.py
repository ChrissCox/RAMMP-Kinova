"""Windows mirror: a native MuJoCo 3D window tracking the Jetson's simulation.

Runs on the DEV MACHINE (no ROS install needed):
    pip install mujoco roslibpy
    python -m mujoco_sim.mirror_viewer --host 192.168.1.11 \
        --model .\\scene_gen3.xml --menagerie C:\\path\\to\\mujoco_menagerie

The ARM is driven from /joint_states over rosbridge (the Jetson bringup
always starts one). The PROPS run under LOCAL physics: only joint
states cross the network, so a purely kinematic puppet would show props
frozen at their spawn poses forever and the arm would appear to pass through
anything the real sim had nudged. Stepping physics locally lets the mirrored
arm push and knock the local props about like the Jetson's does. It is an
approximation (the two physics runs drift apart over time — press
Backspace in the viewer to reset the local props), and it cannot affect the
real sim. --kinematic restores the old frozen-prop behavior.

Needs the generated scene XML copied from the Jetson (~/.ros/mujoco_sim/
scene_gen3.xml) AND a local menagerie clone for the mesh assets — the XML
references them by path, so regenerate locally if paths differ (run from
ros2_ws/src/mujoco_sim so the module imports):
    python -m mujoco_sim.build_scene --menagerie <local clone> \
        --scene ..\\curobo_planner\\config\\scene.yaml --out .\\scene_gen3.xml
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
    ap.add_argument('--camera', metavar='TOPIC', default=None,
                    help="also show a camera topic in an OpenCV window, e.g. "
                         '/rammp_detector/debug_image (what the detector '
                         'sees, detections painted on)')
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
        sys.exit('Could not reach rosbridge at ws://%s:%d — is the Jetson '
                 'bringup running?' % (args.host, args.port))
    topic = roslibpy.Topic(client, '/joint_states', 'sensor_msgs/JointState',
                           throttle_rate=33)  # ~30 Hz is plenty for a mirror
    topic.subscribe(on_joint_states)

    # Perception overlay: what the detectors currently believe, drawn as
    # small cyan spheres (with labels) over the local scene. The local props
    # are an approximation; the spheres are the Jetson's ground truth.
    detections = {}   # label -> (x, y, z), base frame == world frame here

    def on_objects(msg):
        for det in msg.get('detections', []):
            res = det.get('results', [])
            if not res:
                continue
            label = res[0].get('hypothesis', {}).get('class_id', '?')
            p = det.get('bbox', {}).get('center', {}).get('position', {})
            detections[label] = (float(p.get('x', 0.0)), float(p.get('y', 0.0)),
                                 float(p.get('z', 0.0)))

    obj_topic = roslibpy.Topic(client, '/perception/objects',
                               'vision_msgs/Detection3DArray',
                               throttle_rate=200)
    obj_topic.subscribe(on_objects)

    # Optional camera window (--camera): raw rgb8 sensor_msgs/Image over
    # rosbridge (base64), shown via OpenCV. ~2 Hz debug streams only.
    cam_frame = {}
    cam_topic = None
    if args.camera:
        import base64
        import cv2
        import numpy as np

        def on_image(msg):
            try:
                raw = base64.b64decode(msg['data'])
                h, w = int(msg['height']), int(msg['width'])
                img = np.frombuffer(raw, np.uint8)[:h * w * 3].reshape(h, w, 3)
                cam_frame['img'] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as exc:   # keep the mirror alive on a bad frame
                if 'err' not in cam_frame:
                    cam_frame['err'] = True
                    print('camera frame decode failed: %s' % exc)

        cam_topic = roslibpy.Topic(client, args.camera, 'sensor_msgs/Image',
                                   throttle_rate=300)
        cam_topic.subscribe(on_image)

        def show_camera():
            img = cam_frame.get('img')
            if img is not None:
                cv2.imshow(args.camera, img)
                cv2.waitKey(1)
    else:
        def show_camera():
            pass

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

    ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def draw_detections(viewer):
        scn = viewer.user_scn
        scn.ngeom = 0
        for i, (label, p) in enumerate(list(detections.items())):
            if i >= scn.maxgeom:
                break
            g = scn.geoms[i]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                [0.015, 0.0, 0.0], list(p), ident,
                                [0.1, 1.0, 0.9, 0.55])
            g.label = label
            scn.ngeom = i + 1

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
                draw_detections(viewer)
                show_camera()
                viewer.sync()
                time.sleep(1.0 / 60.0)
    finally:
        if cam_topic is not None:
            cam_topic.unsubscribe()
        obj_topic.unsubscribe()
        topic.unsubscribe()
        client.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())

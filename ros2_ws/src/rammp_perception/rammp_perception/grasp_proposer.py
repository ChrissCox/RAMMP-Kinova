"""On-demand GraspGen-X proposals from the live cameras — runs WITH the stack.

    ros2 run rammp_perception grasp_proposer     (started by the bringup)

The neural backend is NVIDIA GraspGen-X (github.com/NVlabs/GraspGenX,
CVPR 2026) running as ITS OWN shipped ZMQ server in ~/graspgen_venv (the
bringup starts it; tools/install_graspgen.zsh sets it up). This node is a
torch-free ZMQ client — GraspGen's numpy/diffusers pins never touch the
ROS process, and an OOM-killed server respawns without taking us down.

    /grasp_proposer/request    std_msgs/String — an object name ("mug") to
                               crop proposals to that object's live position,
                               or "" for the whole workspace
    /grasp_proposer/proposals  std_msgs/String — JSON: ranked grasps in the
                               BASE frame ({p, quat_wxyz, approach, close_axis,
                               width, depth, score}), or {"error": ...}

Conventions: GraspGen-X takes a SEGMENTED object cloud (any frame, meters,
centering done server-side) and returns gripper BASE-LINK poses in the
input-cloud frame: +Z = approach, +X = jaw closing; for robotiq_2f_85 the
fingertips sit +0.136 m along grasp +Z (the gripper config's own number —
NOT the old AnyGrasp center+depth convention). Scores are discriminator
confidences in [0, 1]. GraspGen emits poses+scores ONLY — width is computed
HERE from the cloud extent along the closing axis. Everything is converted
to the base frame; the planner's pad-center fingertip convention
(tool_tip_offset 0.021, tool_spin_deg 90) is the CONSUMER's job — this node
reports what the net saw, nothing more.

Degrades honestly: if the GraspGen server is unreachable the node stays up
and answers every request with a named error — the planner's geometric
synthesizer remains the fallback. AnyGrasp is fully replaced (2026-07-17);
its venv/license stay untouched on the Jetson, git revert restores it.
"""

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from rammp_perception.geometry import CameraModel, quat_to_mat

VOXEL = 0.002           # m — GraspGen's outlier filter (20-NN mean dist
                        # < 14 mm) NEEDS dense surfaces: at the old 4 mm
                        # voxel a scene-range bottle crop (191 pts, ~9 mm
                        # spacing) lost EVERY point to the filter and got
                        # zero grasps (field, 2026-07-17). 2 mm keeps the
                        # 20-NN mean well under the threshold.
MAX_POINTS = 200000     # SHIP cap: applied to the outgoing cloud AFTER
                        # cropping, never before region selection — a
                        # pre-crop cap randomly diluted the 2 mm cloud
                        # back to ~4.5 mm spacing and re-starved the
                        # outlier filter (field, 2026-07-17). Bounds ZMQ
                        # payload (~2.4 MB); the server resamples to 3500.
TIP_DEPTH = 0.136       # m — robotiq_2f_85 fingertip along grasp +Z, from
                        # GraspGen's own gripper config.json ("fingertip":
                        # [0,0,0.136]). Pad-CENTER calibration happens on
                        # the Jetson session (cf. the bottle z-window).
# The graspmoe path ('infer_object') REQUIRES explicit sweep-volume params
# — the server's --default_gripper covers only the plain 'infer' action
# (field, 2026-07-17: "Request is missing 'sweep_volume_params'"). Values
# verbatim from GraspGen's robotiq_2f_85 config.json.
SWEEP_2F85 = {
    'extents_open': [0.085, 0.032, 0.036], 'offset_open': [0.0, 0.0, 0.13],
    'extents_mid': [0.046, 0.032, 0.036], 'offset_mid': [0.0, 0.0, 0.143],
    'gripper_type': 1,           # revolute_2f
    'fingertip_depth': TIP_DEPTH,
}
# island workspace, base frame — matches the detector's honesty gate; the
# scene camera sees the whole kitchen but grasps only happen here
WORKSPACE = [(-0.55, 0.85), (-0.75, 0.75), (-0.10, 0.75)]
CROP_XY = 0.12          # m half-extent around a named object
CROP_Z = (0.015, 0.30)  # m above the object's base, relative to island top


def _mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return [0.25 * s, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


class GraspProposer(Node):

    def __init__(self):
        super().__init__('grasp_proposer')
        self.rgb_topic = self.declare_parameter('rgb_topic', '/d405/color').value
        self.depth_topic = self.declare_parameter('depth_topic', '/d405/depth').value
        self.info_topic = self.declare_parameter(
            'info_topic', '/d405/camera_info').value
        # Same eye-in-hand constants as the d405 detector instance.
        self.attached_frame = self.declare_parameter(
            'camera_attached_frame', 'bracelet_link').value
        self.mount_pos = list(self.declare_parameter(
            'camera_mount_position', [0.0, -0.058, -0.078]).value)
        self.mount_quat = list(self.declare_parameter(
            'camera_mount_quat_wxyz', [0.0, 0.0, 0.0, 1.0]).value)
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        from cv_bridge import CvBridge
        self._bridge = CvBridge()
        self.cam = CameraModel([0, 0, 1], [1, 0, 0, 0, 1, 0])
        self._rgb = self._depth = None
        self._rgb_stamp = self._depth_stamp = None
        self._have_intrinsics = False
        self._live = {}   # label -> [x, y, z] (base frame, from perception)

        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, 2)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 2)
        self.create_subscription(CameraInfo, self.info_topic, self._info_cb, 2)

        # The FIXED scene camera: sees the whole island from a validated
        # pose (same constants as the scene detector; perception_test
        # holds it to <10 mm) — look/grasp perception needs NO arm motion.
        # These MUST match scenery.py's scene_cam definition.
        scene_pos = list(self.declare_parameter(
            'scene_camera_position', [-0.75, 0.0, 1.45]).value)
        scene_axes = list(self.declare_parameter(
            'scene_camera_xyaxes',
            [0.0, -1.0, 0.0, 0.858, 0.0, 0.514]).value)
        self.scene_cam = CameraModel(scene_pos, scene_axes)
        self._scene_rgb = self._scene_depth = None
        self._scene_rgb_stamp = self._scene_depth_stamp = None
        self._scene_pair = None
        self._scene_have_intrinsics = False
        self.create_subscription(
            Image, self.declare_parameter(
                'scene_rgb_topic', '/scene_cam/color').value,
            self._scene_rgb_cb, 2)
        self.create_subscription(
            Image, self.declare_parameter(
                'scene_depth_topic', '/scene_cam/depth').value,
            self._scene_depth_cb, 2)
        self.create_subscription(
            CameraInfo, self.declare_parameter(
                'scene_info_topic', '/scene_cam/camera_info').value,
            self._scene_info_cb, 2)
        try:
            from vision_msgs.msg import Detection3DArray
            self.create_subscription(Detection3DArray, '/perception/objects',
                                     self._objects_cb, 5)
        except ImportError:
            pass
        self.create_subscription(String, '/grasp_proposer/request',
                                 self._request_cb, 1)
        self._pub = self.create_publisher(String, '/grasp_proposer/proposals', 1)
        # The 'look' pipeline: a capture request freezes the current frame
        # (image + cloud + camera pose) and publishes the JPEG for the
        # brain's eyes; a later box request runs inference on that SAME
        # frozen frame, so the brain's box can never drift off the pixels
        # it was drawn on (the arm may move between look and grasp).
        from sensor_msgs.msg import CompressedImage
        self._look_pub = self.create_publisher(
            CompressedImage, '/grasp_proposer/look_image', 1)
        self._capture = None    # dict: name, rgb, pts, uv, stamp, mono

        self.gg_endpoint = self.declare_parameter(
            'graspgen_endpoint', 'tcp://127.0.0.1:5556').value
        self.gg_timeout = float(self.declare_parameter(
            'graspgen_timeout_s', 20.0).value)
        # 'graspmoe' = diffusion grasps UNION swept top-down candidates,
        # all discriminator-scored — the OBB branch natively provides the
        # top-down coverage the old approach_steering plan was for.
        self.gg_planner = self.declare_parameter(
            'graspgen_planner', 'graspmoe').value
        self._sock = None
        self._zmq = None
        self._load_error = 'no contact with the GraspGen server yet'
        self._init_backend()

    # -------------------------------------------------------------- backend
    def _init_backend(self):
        """ZMQ client to GraspGen-X's shipped server. Never fatal: the
        server may still be loading its 1.7 GB of weights when we boot —
        every request retries the connection, and a dead server degrades
        to named errors (the planner's geometric grasps keep working)."""
        try:
            import zmq
            import msgpack_numpy
            msgpack_numpy.patch()   # numpy arrays travel natively
            self._zmq = zmq
        except ImportError as exc:
            self._load_error = ('pyzmq/msgpack-numpy missing in the ROS '
                                'env (pip install pyzmq msgpack '
                                'msgpack-numpy): %s' % exc)
            self.get_logger().error(
                'GraspGen client UNAVAILABLE (%s) — proposals will answer '
                'with this error; geometric grasps still work.'
                % self._load_error)
            return
        # one-shot warmup off the constructor path: health + a tiny dummy
        # inference so the first REAL request doesn't pay the CUDA warmup
        self._warmup_timer = self.create_timer(1.0, self._warmup_once)

    def _warmup_once(self):
        d = self._backend_call({'action': 'health'}, timeout_s=5.0)
        if 'error' in d:
            # KEEP the timer: the server needs ~50 s to load its weights
            # on a fresh boot — a one-shot warmup missed it twice (field,
            # 2026-07-17) and the first real request paid the CUDA warmup.
            self.get_logger().warning(
                'GraspGen server not answering yet (%s) — retrying warmup '
                'in 10 s.' % d['error'])
            self._warmup_timer.timer_period_ns = int(10e9)
            return
        self._warmup_timer.cancel()
        rng = np.random.default_rng(0)
        th = rng.uniform(0, 2 * np.pi, 600)
        dummy = np.stack([0.03 * np.cos(th), 0.03 * np.sin(th),
                          rng.uniform(0.3, 0.4, 600)], axis=-1)
        t0 = time.monotonic()
        d = self._backend_call(
            {'action': 'infer_object',
             'point_cloud': dummy.astype(np.float32),
             'planner': str(self.gg_planner),
             'sweep_volume_params': SWEEP_2F85},
            timeout_s=max(self.gg_timeout, 60.0))
        if 'error' in d:
            self.get_logger().warning('GraspGen warmup failed: %s'
                                      % d['error'])
        else:
            self.get_logger().info(
                'GraspGen-X ready: warmup inference %.1f s, %d grasps'
                % (time.monotonic() - t0,
                   len(d.get('grasps', []))))

    def _backend_call(self, req, timeout_s):
        """One REQ/REP round trip. REQ sockets wedge after an unanswered
        send — on ANY failure the socket is torn down and rebuilt next
        call. Returns the reply dict or {'error': ...}, never raises."""
        if self._zmq is None:
            return {'error': self._load_error}
        import msgpack
        zmq = self._zmq
        try:
            if self._sock is None:
                self._sock = zmq.Context.instance().socket(zmq.REQ)
                self._sock.setsockopt(zmq.LINGER, 0)
                self._sock.connect(self.gg_endpoint)
            self._sock.send(msgpack.packb(req, use_bin_type=True))
            if not self._sock.poll(int(timeout_s * 1000)):
                raise RuntimeError('no reply in %.0f s' % timeout_s)
            rep = msgpack.unpackb(self._sock.recv(), raw=False)
            if not isinstance(rep, dict):
                raise RuntimeError('non-dict reply')
            return rep
        except Exception as exc:
            try:
                if self._sock is not None:
                    self._sock.close()
            finally:
                self._sock = None
            return {'error': 'GraspGen server unreachable at %s (%s)'
                             % (self.gg_endpoint, exc)}

    # -------------------------------------------------------------- callbacks
    def _rgb_cb(self, msg):
        self._rgb = np.asarray(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
        self._rgb_stamp = msg.header.stamp
        self._try_pair()

    def _depth_cb(self, msg):
        d = np.asarray(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
        if d.dtype == np.uint16:
            d = d.astype(np.float32) / 1000.0
        self._depth = d
        self._depth_stamp = msg.header.stamp
        self._try_pair()

    def _try_pair(self):
        """Cache the freshest STAMP-MATCHED color+depth pair continuously —
        sampling both topics at request time raced their arrival and failed
        on ~1 s mismatches whenever a request landed mid-cycle (field)."""
        if self._rgb is None or self._depth is None:
            return
        rs, ds = self._rgb_stamp, self._depth_stamp
        dt = abs(float(rs.sec - ds.sec) + float(rs.nanosec - ds.nanosec) * 1e-9)
        if dt <= 0.05 and self._depth.shape[:2] == self._rgb.shape[:2]:
            self._pair = (self._rgb, self._depth, rs,
                          time.monotonic())

    def _scene_rgb_cb(self, msg):
        self._scene_rgb = np.asarray(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'))
        self._scene_rgb_stamp = msg.header.stamp
        self._try_scene_pair()

    def _scene_depth_cb(self, msg):
        d = np.asarray(
            self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
        if d.dtype == np.uint16:
            d = d.astype(np.float32) / 1000.0
        self._scene_depth = d
        self._scene_depth_stamp = msg.header.stamp
        self._try_scene_pair()

    def _scene_info_cb(self, msg):
        if not self._scene_have_intrinsics:
            self.scene_cam.set_intrinsics_from_info(msg.k)
            self._scene_have_intrinsics = True

    def _try_scene_pair(self):
        if self._scene_rgb is None or self._scene_depth is None:
            return
        rs, ds = self._scene_rgb_stamp, self._scene_depth_stamp
        dt = abs(float(rs.sec - ds.sec) + float(rs.nanosec - ds.nanosec) * 1e-9)
        if dt <= 0.05 and self._scene_depth.shape[:2] == \
                self._scene_rgb.shape[:2]:
            self._scene_pair = (self._scene_rgb, self._scene_depth, rs,
                                time.monotonic())

    def _info_cb(self, msg):
        if not self._have_intrinsics:
            self.cam.set_intrinsics_from_info(msg.k)
            self._have_intrinsics = True

    def _objects_cb(self, msg):
        for det in msg.detections:
            if det.results:
                p = det.bbox.center.position
                self._live[det.results[0].hypothesis.class_id] = \
                    [float(p.x), float(p.y), float(p.z)]

    def _update_camera_pose(self, stamp):
        import rclpy.time
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.attached_frame,
                rclpy.time.Time.from_msg(stamp))
        except Exception:
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        R_bl = quat_to_mat([q.w, q.x, q.y, q.z])
        R_lc = quat_to_mat(self.mount_quat)
        self.cam.set_pose(
            np.array([t.x, t.y, t.z]) + R_bl @ np.array(self.mount_pos),
            R_bl @ R_lc)
        return True

    # ---------------------------------------------------------------- request
    def _fail(self, why):
        self.get_logger().error('proposal request failed: %s' % why)
        self._pub.publish(String(data=json.dumps({'error': why})))

    def _build_cloud(self, cam, rgb, depth, zmin, zmax):
        """Optical-frame cloud + the pixel coords of every surviving point
        (so a 2D box on the image maps onto the downsampled cloud)."""
        h, w = depth.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)
        ok = (z > zmin) & (z < zmax)
        x = (u - cam.cx) / cam.fx * z
        y = (v - cam.cy) / cam.fy * z
        pts = np.stack([x[ok], y[ok], z[ok]], axis=-1).astype(np.float32)
        uv = np.stack([u[ok], v[ok]], axis=-1)
        if pts.shape[0] < 500:
            return None, None
        # voxel downsample only — NO cap here: capping before the region
        # crop starves the object's surface density (the ship cap lives
        # at the send site, after cropping)
        cells = np.floor(pts / VOXEL).astype(np.int64)
        _, keep = np.unique(cells, axis=0, return_index=True)
        keep = np.sort(keep)
        return pts[keep], uv[keep]

    @staticmethod
    def _workspace_filter(cam, pts, uv):
        """Keep only points inside the island workspace box (base frame) —
        the scene camera sees the whole kitchen, and feeding walls and
        floor to the net is how the node OOM-died (NvMap error 12)."""
        pb = cam.p + pts @ cam.R_base_opt.T
        m = ((pb[:, 0] >= WORKSPACE[0][0]) & (pb[:, 0] <= WORKSPACE[0][1])
             & (pb[:, 1] >= WORKSPACE[1][0]) & (pb[:, 1] <= WORKSPACE[1][1])
             & (pb[:, 2] >= WORKSPACE[2][0]) & (pb[:, 2] <= WORKSPACE[2][1]))
        return pts[m], uv[m]

    def _frame(self, source):
        """(cam, pair, zmin, zmax) for a capture source; error string if
        unavailable. 'scene' = the fixed island camera (no TF, no motion);
        'wrist' = the eye-in-hand D405."""
        if source == 'wrist':
            pair = getattr(self, '_pair', None)
            if pair is None:
                return None, 'no stamp-matched D405 pair yet'
            if time.monotonic() - pair[3] > 2.0:
                return None, 'freshest D405 pair is %.1f s old' \
                    % (time.monotonic() - pair[3])
            if not self._update_camera_pose(pair[2]):
                return None, 'TF cannot serve the D405 image stamp yet'
            return (self.cam, pair, 0.05, 1.0), None
        pair = self._scene_pair
        if pair is None:
            return None, 'no stamp-matched scene_cam pair yet'
        if not self._scene_have_intrinsics:
            return None, 'no scene_cam camera_info yet'
        if time.monotonic() - pair[3] > 2.0:
            return None, 'freshest scene_cam pair is %.1f s old' \
                % (time.monotonic() - pair[3])
        return (self.scene_cam, pair, 0.3, 2.5), None

    def _do_capture(self, name, source='scene'):
        """Freeze a frame for the brain's eyes: store image + cloud +
        camera pose together, publish the JPEG. A later box request runs
        on THIS frame — pixels and points can never drift apart. Default
        source is the FIXED scene camera: zero arm motion, stable framing."""
        frame, err = self._frame(source)
        if err:
            return self._fail(err)
        cam, (rgb, depth, rs, age), zmin, zmax = frame
        pts, uv = self._build_cloud(cam, rgb, depth, zmin, zmax)
        if pts is None:
            return self._fail('cloud too small at capture')
        if source == 'scene':
            pts, uv = self._workspace_filter(cam, pts, uv)
            if pts.shape[0] < 300:
                return self._fail('only %d workspace points in the scene '
                                  'capture' % pts.shape[0])
        # ZOOM on the named object: the brain boxes a PART far more
        # reliably when the object fills the frame than when it must
        # first find it in a whole-kitchen image (haiku's boxes wandered
        # across the frame run to run; field, 2026-07-16). The detector
        # already knows WHERE the object is — project its live position
        # into this camera and crop around it. Boxes come back normalized
        # on the CROP and are mapped through (x0, y0, cw, ch) here.
        h, w = depth.shape[:2]
        crop = (0, 0, w, h)
        center = self._live.get(name)
        if center is not None:
            p_opt = cam.R_base_opt.T @ (np.asarray(center) - cam.p)
            if p_opt[2] > 0.05:
                cu = cam.fx * p_opt[0] / p_opt[2] + cam.cx
                cv_ = cam.fy * p_opt[1] / p_opt[2] + cam.cy
                half = max(60, int(0.18 / p_opt[2] * cam.fx))  # ~36 cm fov
                x0 = int(max(0, min(cu - half, w - 2 * half)))
                y0 = int(max(0, min(cv_ - half, h - 2 * half)))
                cw = ch = int(min(2 * half, w - x0, h - y0))
                crop = (x0, y0, cw, ch)
        self._capture = {'name': name, 'rgb': rgb, 'pts': pts, 'uv': uv,
                         'shape': depth.shape[:2], 'crop': crop,
                         'cam_p': cam.p.copy(),
                         'cam_R': cam.R_base_opt.copy(),
                         'mono': time.monotonic()}
        import cv2
        from sensor_msgs.msg import CompressedImage
        x0, y0, cw, ch = crop
        view = rgb[y0:y0 + ch, x0:x0 + cw].copy()
        if cw < 480:    # upscale small crops so the model sees detail
            s = int(np.ceil(480.0 / cw))
            view = cv2.resize(view, (cw * s, ch * s),
                              interpolation=cv2.INTER_NEAREST)
        # Reference grid in the box coordinate system (0-1000): VLMs box
        # far more precisely against visible coordinate anchors.
        gh, gw = view.shape[:2]
        for k in (250, 500, 750):
            xk, yk = int(gw * k / 1000.0), int(gh * k / 1000.0)
            cv2.line(view, (xk, 0), (xk, gh - 1), (255, 255, 255), 1)
            cv2.line(view, (0, yk), (gw - 1, yk), (255, 255, 255), 1)
            cv2.putText(view, str(k), (xk + 3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(view, str(k), (3, yk + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        ok_enc, jpg = cv2.imencode(
            '.jpg', view[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok_enc:
            m = CompressedImage()
            m.format = 'jpeg'
            m.data = jpg.tobytes()
            self._look_pub.publish(m)
        self.get_logger().info(
            'captured look frame for %r (%d cloud pts, crop %s)'
            % (name or 'scene', pts.shape[0], list(crop)))
        self._pub.publish(String(data=json.dumps(
            {'captured': name, 'points': int(pts.shape[0])})))

    def _request_cb(self, msg):
        raw = msg.data.strip()
        req = {}
        if raw.startswith('{'):
            try:
                req = json.loads(raw)
            except ValueError:
                return self._fail('unparseable request JSON')
        else:
            req = {'object': raw}
        if 'capture' in req:
            return self._do_capture(str(req['capture']),
                                    str(req.get('source', 'scene')))
        name = str(req.get('object', '') or '')
        box = req.get('box_2d')    # [ymin, xmin, ymax, xmax], 0-1000 norm
        source = str(req.get('source', 'wrist'))
        if not self._have_intrinsics:
            return self._fail('no camera_info yet — never guess intrinsics')

        t0 = time.monotonic()
        if box is not None:
            # Part-targeted request: use the FROZEN capture frame.
            cap = self._capture
            if cap is None or time.monotonic() - cap['mono'] > 120.0:
                return self._fail('no fresh look capture to apply the box '
                                  'to — call look first')
            pts, uv = cap['pts'], cap['uv']
            cam_p, cam_R = cap['cam_p'], cap['cam_R']
            cx0, cy0, cw, ch = cap.get('crop',
                                       (0, 0, cap['shape'][1],
                                        cap['shape'][0]))
            y0, x0, y1, x1 = [float(b) for b in box]
            # dilate 8% per side: VLM boxes are approximate, and a tight
            # box on a thin part (a handle) can slip off the few voxels
            # that survived downsampling. Boxes are normalized on the
            # CROP the brain saw — map through the crop rect.
            dy, dx = (y1 - y0) * 0.08, (x1 - x0) * 0.08
            px0 = cx0 + (x0 - dx) * cw / 1000.0
            px1 = cx0 + (x1 + dx) * cw / 1000.0
            py0 = cy0 + (y0 - dy) * ch / 1000.0
            py1 = cy0 + (y1 + dy) * ch / 1000.0
            region = ((uv[:, 0] >= px0) & (uv[:, 0] <= px1)
                      & (uv[:, 1] >= py0) & (uv[:, 1] <= py1))
            # 25, not more: a thin part (mug handle) legitimately survives
            # as only ~50 voxels — GraspGen resamples the crop to 3500
            # points with replacement, so a sparse part cloud is workable
            if int(region.sum()) < 25:
                return self._fail(
                    'only %d cloud points inside the box (of %d total in '
                    'the capture) — the part may be too thin for the '
                    'depth/voxel resolution, or the box covers pixels '
                    'outside the 5-100 cm depth gate'
                    % (int(region.sum()), pts.shape[0]))
            # The box's 3D volume in the BASE frame: kept for the
            # planner's wrist-camera retry at the standoff. (AnyGrasp
            # NEEDED it — trained 0.4-0.7 m, scene cam at 1.6 m.
            # GraspGen-X centers the cloud server-side so camera
            # distance shouldn't matter, but the retry contract stays
            # until the scene-view path proves itself on the Jetson.)
            reg_base = cam_p + pts[region] @ cam_R.T
            margin = 0.015
            self._region_base = [
                [round(float(a) - margin, 4) for a in reg_base.min(axis=0)],
                [round(float(a) + margin, 4) for a in reg_base.max(axis=0)]]
        else:
            # Whole-object request on a LIVE frame ('scene' = the fixed
            # island camera, no arm motion; 'wrist' = the orbit-scan path).
            frame, err = self._frame(source)
            if err:
                return self._fail(err)
            cam, (rgb, depth, rs, age), zmin, zmax = frame
            pts, uv = self._build_cloud(cam, rgb, depth, zmin, zmax)
            if pts is None:
                return self._fail('cloud too small from %s' % source)
            n_frame = pts.shape[0]
            if source == 'scene':
                pts, uv = self._workspace_filter(cam, pts, uv)
                if pts.shape[0] < 300:
                    return self._fail('only %d workspace points from the '
                                      'scene camera' % pts.shape[0])
            # named stage counts: sparsity bugs hide without them (a 4 mm
            # voxel once starved GraspGen's outlier filter unseen)
            self.get_logger().info(
                '%s cloud: %d after voxel/cap, %d in workspace'
                % (source, n_frame, pts.shape[0]))
            cam_p, cam_R = cam.p, cam.R_base_opt
            pts_base = cam_p + pts @ cam_R.T
            region = None
            region_base = req.get('region_base')
            if region_base is not None:
                # crop to a base-frame AABB (the brain's box volume,
                # resolved from a scene capture) — takes precedence over
                # the coarse object-position crop
                lo, hi = region_base
                region = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
                if int(region.sum()) < 25:
                    return self._fail(
                        'only %d cloud points in the requested 3D region '
                        'from %s' % (int(region.sum()), source))
            elif name:
                center = self._live.get(name)
                if center is None:
                    return self._fail('no live position for %r — perception '
                                      'has not seen it' % name)
                lo = [center[0] - CROP_XY, center[1] - CROP_XY,
                      -0.07 + CROP_Z[0]]
                hi = [center[0] + CROP_XY, center[1] + CROP_XY,
                      -0.07 + CROP_Z[1]]
                region = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
                # 100, not the old 200: that was AnyGrasp-era calibration.
                # GraspGen's own scene pipeline accepts instances >=100
                # points and resamples to 3500 — a home-view bottle crop
                # measured 186 (field, 2026-07-17) and is perfectly usable.
                if int(region.sum()) < 100:
                    bmin = np.percentile(pts_base, 5, axis=0)
                    bmax = np.percentile(pts_base, 95, axis=0)
                    return self._fail(
                        'only %d cloud points on %r at [%.2f %.2f] — the '
                        'camera sees x %.2f..%.2f y %.2f..%.2f z %.2f..%.2f '
                        'instead'
                        % (region.sum(), name, center[0], center[1],
                           bmin[0], bmax[0], bmin[1], bmax[1],
                           bmin[2], bmax[2]))

        # GraspGen-X takes the SEGMENTED object cloud — the crop IS the
        # input (unlike AnyGrasp's full-cloud + steering-mask). The 25-pt
        # part floor stays honest: the server resamples to 3500 points
        # with replacement, so thin-handle crops are fine.
        cloud = pts[region] if region is not None else pts
        if cloud.shape[0] > MAX_POINTS:
            cloud = cloud[np.random.default_rng(0).choice(
                cloud.shape[0], MAX_POINTS, replace=False)]
        rep = self._backend_call(
            {'action': 'infer_object',
             'point_cloud': np.ascontiguousarray(cloud, dtype=np.float32),
             'planner': str(self.gg_planner),
             'sweep_volume_params': SWEEP_2F85},
            timeout_s=self.gg_timeout)
        gg_err = rep.get('error')
        grasps = None if gg_err else np.asarray(rep.get('grasps', []),
                                                np.float32)
        if gg_err is None and (grasps is None or grasps.size == 0):
            gg_err = 'GraspGen returned no grasps for %r' \
                % (name or 'workspace')
        if gg_err:
            payload = {'error': str(gg_err)}
            if box is not None:
                # still hand back the box's 3D volume: the planner can
                # retry through the wrist camera at the standoff
                payload['region_base'] = self._region_base
            self.get_logger().error('proposal request failed: %s'
                                    % payload['error'])
            return self._pub.publish(String(data=json.dumps(payload)))
        conf = np.asarray(rep.get('confidences', []), np.float32)
        order = np.argsort(-conf)
        out, kept = [], []
        for i in order:
            T = grasps[i]           # (4,4) gripper BASE pose, optical frame
            a_opt = T[:3, 2]        # +Z = approach
            # poor man's NMS (AnyGrasp's gg.nms() equivalent): drop
            # near-duplicates so the planner's 12-probe budget sees
            # DIVERSE candidates, not one grasp ten times
            if any(np.linalg.norm(T[:3, 3] - k[0]) < 0.01
                   and float(np.dot(a_opt, k[1])) > 0.966 for k in kept):
                continue
            kept.append((T[:3, 3].copy(), a_opt.copy()))
            R_b = cam_R @ T[:3, :3]
            p_b = cam_p + cam_R @ T[:3, 3]
            # width is OURS to compute (GraspGen emits poses+scores only):
            # cloud extent along the closing axis near the fingertips,
            # clamped to the 2F-85 stroke
            tip_opt = T[:3, 3] + TIP_DEPTH * a_opt
            near = cloud[np.linalg.norm(cloud - tip_opt, axis=1) < 0.05]
            src = near if near.shape[0] >= 10 else cloud
            proj = src @ T[:3, 0]
            width = float(np.clip(
                np.percentile(proj, 95) - np.percentile(proj, 5) + 0.006,
                0.01, 0.085))
            out.append({
                'p': [round(float(a), 4) for a in p_b],
                'quat_wxyz': [round(float(a), 5)
                              for a in _mat_to_quat_wxyz(R_b)],
                'approach': [round(float(a), 4) for a in R_b[:, 2]],
                'close_axis': [round(float(a), 4) for a in R_b[:, 0]],
                'width': round(width, 4),
                'depth': TIP_DEPTH,
                'score': round(float(conf[i]), 3),
            })
            if len(out) >= 10:
                break
        took = time.monotonic() - t0
        self.get_logger().info(
            'proposals for %r: %d grasps in %.2f s, best score %.2f at '
            '[%.2f %.2f %.2f]' % (name or 'workspace', len(out), took,
                                  out[0]['score'], *out[0]['p']))
        reply = {'object': name, 'took_s': round(took, 2), 'grasps': out}
        if box is not None:
            reply['region_base'] = self._region_base
        self._pub.publish(String(data=json.dumps(reply)))


def main(args=None):
    rclpy.init(args=args)
    node = GraspProposer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

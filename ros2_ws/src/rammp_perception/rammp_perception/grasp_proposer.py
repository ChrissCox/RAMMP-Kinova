"""On-demand AnyGrasp proposals from the live D405 — runs WITH the stack.

    ros2 run rammp_perception grasp_proposer     (started by the bringup)

Loads the AnyGrasp detector ONCE at startup (~10 s, license-checked), then
idles at zero GPU cost until a request arrives:

    /grasp_proposer/request    std_msgs/String — an object name ("mug") to
                               crop proposals to that object's live position,
                               or "" for the whole workspace
    /grasp_proposer/proposals  std_msgs/String — JSON: ranked grasps in the
                               BASE frame ({p, quat_wxyz, approach, close_axis,
                               width, score}), or {"error": ...}

Conventions: AnyGrasp returns grasp CENTERS in the optical camera frame with
rotation columns X=approach, Y=jaw-closing. Everything is converted to the
base frame here; converting to the planner's pad-center fingertip convention
(tool_tip_offset 0.021, tool_spin_deg 90) is the CONSUMER's job — this node
reports what the net saw, nothing more.

Degrades honestly: if gsnet / the license / the venv is unavailable the node
stays up and answers every request with a named error — the planner's
geometric synthesizer remains the fallback (a license drift, issue #164,
must degrade the stack, never halt it).

The AnyGrasp runtime lives in ~/anygrasp_venv (--system-site-packages) and
~/anygrasp_sdk; both are sys.path-injected here so the node itself runs as a
normal ros2 entry point. cwd moves to grasp_detection/ because the license
check resolves ./license relative to it.
"""

import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from rammp_perception.geometry import CameraModel, quat_to_mat

VENV_SITE = os.path.expanduser(
    '~/anygrasp_venv/lib/python3.10/site-packages')
SDK_DIR = os.path.expanduser('~/anygrasp_sdk/grasp_detection')
CHECKPOINT = os.path.join(SDK_DIR, 'log', 'checkpoint_detection.tar')
VOXEL = 0.004           # m — the D405 cloud must be downsampled (a raw
                        # ~1M-point cloud triggered a >30 GB alloc, SDK #29)
MAX_POINTS = 120000
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
        try:
            from vision_msgs.msg import Detection3DArray
            self.create_subscription(Detection3DArray, '/perception/objects',
                                     self._objects_cb, 5)
        except ImportError:
            pass
        self.create_subscription(String, '/grasp_proposer/request',
                                 self._request_cb, 1)
        self._pub = self.create_publisher(String, '/grasp_proposer/proposals', 1)

        self._detector = None
        self._load_error = 'not loaded yet'
        self._load_detector()

    # -------------------------------------------------------------- lifecycle
    def _load_detector(self):
        t0 = time.monotonic()
        try:
            if not os.path.isdir(VENV_SITE):
                raise RuntimeError('anygrasp venv missing at %s' % VENV_SITE)
            # addsitedir, not sys.path.insert: pointnet2 lives in the venv
            # as an EGG, reachable only through easy-install.pth processing
            import site
            site.addsitedir(VENV_SITE)
            if SDK_DIR not in sys.path:
                sys.path.insert(0, SDK_DIR)
            os.chdir(SDK_DIR)   # the license check resolves ./license
            from types import SimpleNamespace
            from gsnet import create_detector
            cfgs = SimpleNamespace(checkpoint_path=CHECKPOINT,
                                   max_gripper_width=0.085,
                                   gripper_height=0.03)
            det = create_detector(cfgs)
            if not det:
                raise RuntimeError('create_detector returned falsy — '
                                   'license failed? (gsnet logs above)')
            self._detector = det
            self.get_logger().info(
                'AnyGrasp detector ready in %.1f s (license passed)'
                % (time.monotonic() - t0))
        except Exception as exc:
            self._load_error = str(exc)
            # one loud line, then degrade: the geometric synthesizer in the
            # planner keeps working without us
            self.get_logger().error(
                'AnyGrasp UNAVAILABLE (%s) — proposals will answer with '
                'this error; the planner\'s geometric grasps still work.'
                % exc)

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

    def _request_cb(self, msg):
        name = msg.data.strip()
        if self._detector is None:
            return self._fail('AnyGrasp unavailable: %s' % self._load_error)
        if getattr(self, '_pair', None) is None:
            return self._fail('no stamp-matched D405 pair yet')
        if not self._have_intrinsics:
            return self._fail('no camera_info yet — never guess intrinsics')
        rgb, depth, rs, age = self._pair
        if time.monotonic() - age > 2.0:
            return self._fail('freshest matched color/depth pair is %.1f s '
                              'old — camera stalled?'
                              % (time.monotonic() - age))
        if not self._update_camera_pose(rs):
            return self._fail('TF cannot serve the image stamp yet')

        t0 = time.monotonic()
        # cloud in the OPTICAL frame (x right, y down, z forward) — the
        # convention AnyGrasp was trained on
        h, w = depth.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)
        ok = (z > 0.05) & (z < 1.0)
        x = (u - self.cam.cx) / self.cam.fx * z
        y = (v - self.cam.cy) / self.cam.fy * z
        pts = np.stack([x[ok], y[ok], z[ok]], axis=-1).astype(np.float32)
        if pts.shape[0] < 500:
            return self._fail('cloud too small (%d pts in 5-100 cm)'
                              % pts.shape[0])
        # voxel downsample (numpy hash — SDK #29: never feed a raw cloud)
        cells = np.floor(pts / VOXEL).astype(np.int64)
        _, keep = np.unique(cells, axis=0, return_index=True)
        if keep.shape[0] > MAX_POINTS:
            keep = np.random.default_rng(0).choice(
                keep, MAX_POINTS, replace=False)
        pts = pts[np.sort(keep)]

        # base-frame positions per point (for the object crop)
        pts_base = self.cam.p + pts @ self.cam.R_base_opt.T
        region = None
        if name:
            center = self._live.get(name)
            if center is None:
                return self._fail('no live position for %r — perception has '
                                  'not seen it' % name)
            lo = [center[0] - CROP_XY, center[1] - CROP_XY,
                  -0.07 + CROP_Z[0]]
            hi = [center[0] + CROP_XY, center[1] + CROP_XY,
                  -0.07 + CROP_Z[1]]
            region = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
            if int(region.sum()) < 200:
                bmin = np.percentile(pts_base, 5, axis=0)
                bmax = np.percentile(pts_base, 95, axis=0)
                return self._fail(
                    'only %d cloud points on %r at [%.2f %.2f] — the camera '
                    'sees x %.2f..%.2f y %.2f..%.2f z %.2f..%.2f instead'
                    % (region.sum(), name, center[0], center[1],
                       bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2]))

        gg = self._detector.get_grasp(pts, {
            'dense_grasp': False,
            'collision_detection': True,
            'region_steering': region,
            'approach_steering': None,
            'approach_thresh': float(np.pi),
        })
        if gg is None or len(gg) == 0:
            return self._fail('AnyGrasp returned no grasps for %r'
                              % (name or 'workspace'))
        gg = gg.nms().sort_by_score()
        out = []
        for g in gg[:10]:
            R_b = self.cam.R_base_opt @ g.rotation_matrix
            p_b = self.cam.p + self.cam.R_base_opt @ g.translation
            out.append({
                'p': [round(float(a), 4) for a in p_b],
                'quat_wxyz': [round(float(a), 5)
                              for a in _mat_to_quat_wxyz(R_b)],
                'approach': [round(float(a), 4) for a in R_b[:, 0]],
                'close_axis': [round(float(a), 4) for a in R_b[:, 1]],
                'width': round(float(g.width), 4),
                'depth': round(float(g.depth), 4),
                'score': round(float(g.score), 3),
            })
        took = time.monotonic() - t0
        self.get_logger().info(
            'proposals for %r: %d grasps in %.2f s, best score %.2f at '
            '[%.2f %.2f %.2f]' % (name or 'workspace', len(out), took,
                                  out[0]['score'], *out[0]['p']))
        self._pub.publish(String(data=json.dumps(
            {'object': name, 'took_s': round(took, 2), 'grasps': out})))


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

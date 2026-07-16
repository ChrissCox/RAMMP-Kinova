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
MAX_POINTS = 60000      # 120k of whole-kitchen scene cloud OOM-killed the
                        # node (NvMap error 12) — workspace-crop + this cap
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
        # voxel downsample (numpy hash — SDK #29: never feed a raw cloud)
        cells = np.floor(pts / VOXEL).astype(np.int64)
        _, keep = np.unique(cells, axis=0, return_index=True)
        if keep.shape[0] > MAX_POINTS:
            keep = np.random.default_rng(0).choice(
                keep, MAX_POINTS, replace=False)
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
        view = rgb[y0:y0 + ch, x0:x0 + cw]
        if cw < 480:    # upscale small crops so the model sees detail
            s = int(np.ceil(480.0 / cw))
            view = cv2.resize(view, (cw * s, ch * s),
                              interpolation=cv2.INTER_NEAREST)
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
        if self._detector is None:
            return self._fail('AnyGrasp unavailable: %s' % self._load_error)
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
            # as only ~50 voxels — the full cloud still gives the net its
            # context, the box only steers where proposals may land
            if int(region.sum()) < 25:
                return self._fail(
                    'only %d cloud points inside the box (of %d total in '
                    'the capture) — the part may be too thin for the '
                    'depth/voxel resolution, or the box covers pixels '
                    'outside the 5-100 cm depth gate'
                    % (int(region.sum()), pts.shape[0]))
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
            if source == 'scene':
                pts, uv = self._workspace_filter(cam, pts, uv)
                if pts.shape[0] < 300:
                    return self._fail('only %d workspace points from the '
                                      'scene camera' % pts.shape[0])
            cam_p, cam_R = cam.p, cam.R_base_opt
            pts_base = cam_p + pts @ cam_R.T
            region = None
            if name:
                center = self._live.get(name)
                if center is None:
                    return self._fail('no live position for %r — perception '
                                      'has not seen it' % name)
                lo = [center[0] - CROP_XY, center[1] - CROP_XY,
                      -0.07 + CROP_Z[0]]
                hi = [center[0] + CROP_XY, center[1] + CROP_XY,
                      -0.07 + CROP_Z[1]]
                region = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
                if int(region.sum()) < 200:
                    bmin = np.percentile(pts_base, 5, axis=0)
                    bmax = np.percentile(pts_base, 95, axis=0)
                    return self._fail(
                        'only %d cloud points on %r at [%.2f %.2f] — the '
                        'camera sees x %.2f..%.2f y %.2f..%.2f z %.2f..%.2f '
                        'instead'
                        % (region.sum(), name, center[0], center[1],
                           bmin[0], bmax[0], bmin[1], bmax[1],
                           bmin[2], bmax[2]))

        # dense_grasp=True, Jake-style: coverage over shyness — the
        # downstream reachability gates do the quality filtering.
        gg = self._detector.get_grasp(pts, {
            'dense_grasp': True,
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
            R_b = cam_R @ g.rotation_matrix
            p_b = cam_p + cam_R @ g.translation
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

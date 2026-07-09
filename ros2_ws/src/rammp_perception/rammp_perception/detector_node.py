"""Continuous scene perception: scene_cam RGB-D -> 3D object detections.

Subscribes to the sim's scene camera (mujoco_ros2_control publishes color,
depth, and camera_info), runs a detection backend at a duty-cycled rate, and
publishes:

  /perception/objects   vision_msgs/Detection3DArray  (base-frame positions)
  /perception/markers   visualization_msgs/MarkerArray (Foxglove: spheres+labels)

The planner subscribes to /perception/objects and OVERRIDES its scene props'
poses with fresh detections — a knocked-over bottle is finally avoided where
it actually lies, not where scene.yaml says it stood.

Backends (see backends.py): 'color' (sim bring-up: matches the props' known
colors, zero ML deps) and 'owlvit' (real open-vocabulary via transformers).
For NanoOWL (Jetson TensorRT), run its own ROS node and feed this node via
external 2D detections — planned for the phase-2 wiring.

The camera pose parameters MUST match scenery.py's scene_cam. Duplicate-label
ambiguity (mug and apple are both red to the color backend) is resolved by
position continuity: each label keeps the candidate nearest its last-known
position (seeded from scene.yaml).
"""

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

from rammp_perception.backends import make_backend
from rammp_perception.geometry import CameraModel, mask_to_position


def _find_scene_yaml(explicit):
    if explicit:
        return explicit
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('curobo_planner'),
                        'config', 'scene.yaml')


class Detector(Node):

    def __init__(self):
        super().__init__('rammp_detector')
        self.rgb_topic = self.declare_parameter('rgb_topic', '/scene_cam/color').value
        self.depth_topic = self.declare_parameter('depth_topic', '/scene_cam/depth').value
        self.info_topic = self.declare_parameter('info_topic', '/scene_cam/camera_info').value
        self.backend_name = self.declare_parameter('backend', 'color').value
        self.rate = float(self.declare_parameter('rate', 2.0).value)
        scene_file = self.declare_parameter('scene_file', '').value
        # FIXED camera: these MUST match scenery.py's scene_cam definition.
        cam_pos = list(self.declare_parameter(
            'camera_position', [-0.75, 0.0, 1.45]).value)
        cam_axes = list(self.declare_parameter(
            'camera_xyaxes', [0.0, -1.0, 0.0, 0.858, 0.0, 0.514]).value)
        # MOVING (eye-in-hand) camera: set camera_attached_frame to the TF
        # frame the camera rides on (e.g. bracelet_link for the D405); the
        # mount pos/quat are the camera's constants in that frame and MUST
        # match build_scene's d405 definition. The pose is then looked up
        # from TF every tick instead of using the fixed values above.
        self.attached_frame = self.declare_parameter(
            'camera_attached_frame', '').value
        self.mount_pos = list(self.declare_parameter(
            'camera_mount_position', [0.0, -0.058, -0.078]).value)
        self.mount_quat = list(self.declare_parameter(
            'camera_mount_quat_wxyz', [0.0, 0.0, 0.0, 1.0]).value)
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self._tf_buffer = None
        if self.attached_frame:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        import yaml
        with open(_find_scene_yaml(scene_file)) as f:
            scene = yaml.safe_load(f)
        # Track FREE props only: statics (door, posts, board) never move —
        # YAML remains their truth, and excluding them kills wood-vs-wood
        # and decor cross-matches.
        objs = [o for o in scene.get('objects', []) if o.get('free', False)]
        self.classes = {o['name']: list(o.get('color', [0.8, 0.8, 0.8, 1.0]))[:3]
                        for o in objs}
        self.last_pos = {o['name']: np.asarray([float(v) for v in o['position']])
                         for o in objs}
        # Honesty gates: a detection farther than this from the label's last
        # known position is REJECTED (a stale pose beats a wrong one), and
        # anything outside the island workspace box is background noise
        # (white arm links / beige wall match prop colors otherwise).
        self.max_jump = float(self.declare_parameter('max_jump', 0.25).value)
        self.workspace = [(-0.55, 0.85), (-0.75, 0.75), (-0.10, 0.75)]

        self.cam = CameraModel(cam_pos, cam_axes)
        self.backend = make_backend(self.backend_name, self.classes,
                                    list(self.classes.keys()))

        self._rgb = None
        self._depth = None
        from cv_bridge import CvBridge
        self._bridge = CvBridge()
        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, 2)
        self.create_subscription(Image, self.depth_topic, self._depth_cb, 2)
        self.create_subscription(CameraInfo, self.info_topic, self._info_cb, 2)

        self._pub = self.create_publisher(Detection3DArray, '/perception/objects', 5)
        self._marker_pub = self.create_publisher(MarkerArray, '/perception/markers', 2)
        self.create_timer(1.0 / max(self.rate, 0.1), self._tick)
        self.get_logger().info(
            'rammp perception up: backend=%s, %d classes (%s), %.1f Hz'
            % (self.backend_name, len(self.classes),
               ', '.join(self.classes), self.rate))

    # ------------------------------------------------------------- callbacks
    def _rgb_cb(self, msg):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self._rgb = np.asarray(img)

    def _depth_cb(self, msg):
        d = np.asarray(self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
        if d.dtype == np.uint16:
            d = d.astype(np.float32) / 1000.0
        self._depth = d

    def _info_cb(self, msg):
        if self.cam.fx is None:
            self.cam.set_intrinsics_from_info(msg.k)
            self.get_logger().info('camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f'
                                   % (self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy))

    def _update_camera_pose(self):
        """Eye-in-hand: base<-attached_frame from TF, composed with the
        fixed mount transform. Returns False when TF isn't available yet."""
        import rclpy.time
        from rammp_perception.geometry import quat_to_mat
        try:
            tf = self._tf_buffer.lookup_transform(
                self.base_frame, self.attached_frame, rclpy.time.Time())
        except Exception:
            return False
        t = tf.transform.translation
        q = tf.transform.rotation
        R_bl = quat_to_mat([q.w, q.x, q.y, q.z])   # base <- link
        p_bl = np.array([t.x, t.y, t.z])
        R_lc = quat_to_mat(self.mount_quat)        # link <- mj-camera
        p_lc = np.array(self.mount_pos)
        self.cam.set_pose(p_bl + R_bl @ p_lc, R_bl @ R_lc)
        return True

    # ------------------------------------------------------------------ tick
    def _tick(self):
        if self._rgb is None or self._depth is None:
            return
        if self.cam.fx is None:
            # camera_info not seen yet: derive from the sim default fovy
            self.cam.set_intrinsics_from_fovy(45.0, self._rgb.shape[1],
                                              self._rgb.shape[0])
            self.get_logger().warning('no camera_info yet — using fovy=45 intrinsics')
        rgb, depth = self._rgb, self._depth
        if depth.shape[:2] != rgb.shape[:2]:
            return
        if self.attached_frame and not self._update_camera_pose():
            return   # no TF yet — skip this tick rather than lie

        candidates = {}   # label -> list of (pos, extent, score)
        for det in self.backend.detect(rgb):
            pos, extent = mask_to_position(det.mask, depth, self.cam)
            if pos is None:
                continue
            if not all(lo <= p <= hi for p, (lo, hi)
                       in zip(pos, self.workspace)):
                continue   # background (wall/floor/arm base) — not a prop
            candidates.setdefault(det.label, []).append((pos, extent, det.score))

        arr = Detection3DArray()
        arr.header.frame_id = self.base_frame
        arr.header.stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = self.base_frame
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        mid = 0
        for label, cands in candidates.items():
            # position continuity beats score when a color/class is ambiguous
            ref = self.last_pos.get(label)
            if ref is not None:
                cands.sort(key=lambda c: float(np.linalg.norm(c[0] - ref)))
                if float(np.linalg.norm(cands[0][0] - ref)) > self.max_jump:
                    continue   # nothing near where this prop should be: MISS
            pos, extent, score = cands[0]
            self.last_pos[label] = pos
            det = Detection3D()
            det.header = arr.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)
            det.bbox.center.position.x = float(pos[0])
            det.bbox.center.position.y = float(pos[1])
            det.bbox.center.position.z = float(pos[2])
            det.bbox.size = Vector3(x=float(extent[0]), y=float(extent[1]),
                                    z=float(min(extent)))
            arr.detections.append(det)

            m = Marker()
            m.header.frame_id = self.base_frame
            m.ns = 'perception'
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=0.04, y=0.04, z=0.04)
            m.color = ColorRGBA(r=0.1, g=1.0, b=0.9, a=0.9)
            markers.markers.append(m)
            t = Marker()
            t.header.frame_id = self.base_frame
            t.ns = 'perception_labels'
            t.id = mid
            mid += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(x=float(pos[0]), y=float(pos[1]),
                                    z=float(pos[2]) + 0.07)
            t.pose.orientation.w = 1.0
            t.scale = Vector3(x=0.0, y=0.0, z=0.035)
            t.color = ColorRGBA(r=0.1, g=1.0, b=0.9, a=0.9)
            t.text = label
            markers.markers.append(t)

        self._pub.publish(arr)
        self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
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

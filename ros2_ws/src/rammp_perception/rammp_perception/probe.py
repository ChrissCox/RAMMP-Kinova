"""One-shot perception diagnostics: inspect what the camera topics actually
deliver and where detections die. Run alongside the bringup:

    ros2 run rammp_perception probe
    ros2 run rammp_perception probe --ros-args -p rgb_topic:=/d405/color ...

Prints image encodings/shapes, depth statistics, intrinsics, raw backend
blobs, and every candidate's 3D position WITH the gate verdicts — so
'detections: []' becomes a named cause (depth scale, image flip, color
mismatch, gate rejection...).
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from rammp_perception.backends import ColorBackend
from rammp_perception.detector_node import _find_scene_yaml
from rammp_perception.geometry import CameraModel, mask_to_position


class Probe(Node):

    def __init__(self):
        super().__init__('rammp_probe')
        self.rgb_topic = self.declare_parameter('rgb_topic', '/scene_cam/color').value
        self.depth_topic = self.declare_parameter('depth_topic', '/scene_cam/depth').value
        self.info_topic = self.declare_parameter('info_topic', '/scene_cam/camera_info').value
        cam_pos = list(self.declare_parameter(
            'camera_position', [-0.75, 0.0, 1.45]).value)
        cam_axes = list(self.declare_parameter(
            'camera_xyaxes', [0.0, -1.0, 0.0, 0.858, 0.0, 0.514]).value)
        self.cam = CameraModel(cam_pos, cam_axes)
        self.rgb = self.depth = self.info = None
        self.rgb_enc = self.depth_enc = None
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.create_subscription(Image, self.rgb_topic, self._rgb, 2)
        self.create_subscription(Image, self.depth_topic, self._depth, 2)
        self.create_subscription(CameraInfo, self.info_topic, self._info, 2)

    def _rgb(self, m):
        self.rgb_enc = m.encoding
        self.rgb = np.asarray(self.bridge.imgmsg_to_cv2(m, desired_encoding='rgb8'))

    def _depth(self, m):
        self.depth_enc = m.encoding
        d = np.asarray(self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough'))
        self.depth = d

    def _info(self, m):
        self.info = m


def main(args=None):
    import yaml
    rclpy.init(args=args)
    n = Probe()
    print('waiting for one frame of each topic...')
    for _ in range(300):
        rclpy.spin_once(n, timeout_sec=0.1)
        if n.rgb is not None and n.depth is not None and n.info is not None:
            break
    if n.rgb is None or n.depth is None:
        print('TIMEOUT: rgb=%s depth=%s info=%s — topics not delivering'
              % (n.rgb is not None, n.depth is not None, n.info is not None))
        return 1

    rgb, depth = n.rgb, n.depth
    print('\n--- RGB: %s  shape=%s dtype=%s' % (n.rgb_enc, rgb.shape, rgb.dtype))
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        print('    !! image is %dx%d — the camera is publishing degenerate frames.' % (w, h))
        print('    !! (MJCF camera `resolution` defaults to 1x1: rebuild the scene'
              ' with a build_scene that sets it.)')
        return 1
    print('    center pixel rgb=%s   top-left=%s   bottom-left=%s'
          % (rgb[h // 2, w // 2].tolist(), rgb[5, 5].tolist(), rgb[-5, 5].tolist()))
    print('--- DEPTH: %s  shape=%s dtype=%s' % (n.depth_enc, depth.shape, depth.dtype))
    dd = depth.astype(float).ravel()
    finite = dd[np.isfinite(dd)]
    print('    min=%.4f max=%.4f median=%.4f finite=%.0f%%'
          % (finite.min(), finite.max(), np.median(finite),
              100.0 * len(finite) / len(dd)))
    if np.median(finite) > 50:
        print('    !! median depth > 50 — looks like MILLIMETRES, node expects metres')
    if finite.max() <= 1.001:
        print('    !! depth <= 1.0 everywhere — looks like a raw NDC/z-buffer, not metres')
    if n.info is not None:
        k = n.info.k
        print('--- CAMERA_INFO: fx=%.1f fy=%.1f cx=%.1f cy=%.1f  (%dx%d)'
              % (k[0], k[4], k[2], k[5], n.info.width, n.info.height))
        n.cam.set_intrinsics_from_info(k)
    else:
        n.cam.set_intrinsics_from_fovy(58.0, w, h)
        print('--- CAMERA_INFO: none — fovy fallback')

    scene = yaml.safe_load(open(_find_scene_yaml('')))
    objs = [o for o in scene.get('objects', []) if o.get('free', False)]
    classes = {o['name']: list(o.get('color'))[:3] for o in objs}
    yaml_pos = {o['name']: np.array([float(v) for v in o['position']]) for o in objs}
    dets = ColorBackend(classes).detect(rgb)
    print('\n--- BACKEND: %d raw blobs' % len(dets))
    if not dets:
        print('    (no color matches at all: image colors differ from scene.yaml,'
              ' or the image is not what we think it is)')
    ws = [(-0.55, 0.85), (-0.75, 0.75), (-0.10, 0.75)]
    dm = depth.astype(np.float32)
    if depth.dtype == np.uint16 or np.median(finite) > 50:
        dm = dm / 1000.0
    for det in dets:
        pos, ext = mask_to_position(det.mask, dm, n.cam)
        if pos is None:
            print('    %-12s %6dpx -> NO DEPTH inside mask' % (det.label, det.mask.sum()))
            continue
        in_ws = all(lo <= p <= hi for p, (lo, hi) in zip(pos, ws))
        jump = float(np.linalg.norm(pos - yaml_pos.get(det.label, pos)))
        verdict = 'OK' if (in_ws and jump < 0.25) else (
            'GATED: outside workspace' if not in_ws else 'GATED: %.2fm from YAML pose' % jump)
        print('    %-12s %6dpx -> [%6.3f %6.3f %6.3f]  %s'
              % (det.label, det.mask.sum(), pos[0], pos[1], pos[2], verdict))
    return 0


if __name__ == '__main__':
    main()

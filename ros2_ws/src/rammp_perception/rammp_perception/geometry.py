"""Detection + depth -> 3D positions in the robot base frame.

Deliberately ROS-free (numpy only) so the whole 2D->3D path is testable
against locally-rendered frames — camera-convention bugs are caught by an
offline test comparing estimates to ground-truth prop poses, not by a robot
smacking something.

Conventions handled here:
  * MuJoCo camera frames are X-right / Y-up / looking down -Z.
    ROS optical frames are X-right / Y-down / Z-forward.
    R_MJ2OPT maps optical vectors into the MuJoCo camera frame.
  * Unprojection uses pinhole intrinsics (fx, fy, cx, cy). From a
    CameraInfo K matrix when available, else derived from fovy + size.
"""

import math

import numpy as np

# optical -> mujoco-camera basis change (columns = optical axes in mj-cam frame)
R_MJ2OPT = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])


def intrinsics_from_fovy(fovy_deg, width, height):
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return fy, fy, width / 2.0, height / 2.0   # fx = fy for square pixels


def quat_to_mat(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class CameraModel:
    """Camera intrinsics + pose in the base frame.

    Fixed cameras: construct with (position, xyaxes) copied from scenery.py.
    Moving (eye-in-hand) cameras: construct with anything, then call
    set_pose(p, R_base_mjcam) per frame from TF/FK — R is the MuJoCo-camera
    frame (X-right, Y-up, looking -Z) in the base frame.
    """

    def __init__(self, position, xyaxes, fx=None, fy=None, cx=None, cy=None):
        self.p = np.asarray(position, float)
        x = np.asarray(xyaxes[:3], float)
        y = np.asarray(xyaxes[3:], float)
        x = x / np.linalg.norm(x)
        y = y - x * float(np.dot(y, x))
        y = y / np.linalg.norm(y)
        z = np.cross(x, y)
        self.R_base_mjcam = np.column_stack([x, y, z])
        self.R_base_opt = self.R_base_mjcam @ R_MJ2OPT
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def set_pose(self, position, R_base_mjcam):
        self.p = np.asarray(position, float)
        self.R_base_mjcam = np.asarray(R_base_mjcam, float)
        self.R_base_opt = self.R_base_mjcam @ R_MJ2OPT

    def set_intrinsics_from_info(self, k_matrix):
        """k_matrix: the 9-float K from sensor_msgs/CameraInfo."""
        self.fx, self.fy = float(k_matrix[0]), float(k_matrix[4])
        self.cx, self.cy = float(k_matrix[2]), float(k_matrix[5])

    def set_intrinsics_from_fovy(self, fovy_deg, width, height):
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fovy(
            fovy_deg, width, height)

    def unproject(self, u, v, depth):
        """Pixel (u, v) at `depth` metres -> 3D point in the BASE frame."""
        x = (u - self.cx) / self.fx * depth
        y = (v - self.cy) / self.fy * depth
        p_opt = np.array([x, y, depth])
        return self.p + self.R_base_opt @ p_opt


def mask_to_position(mask, depth_img, cam):
    """Binary mask + depth image -> (base-frame position, extent metres).

    Position: unprojected mask centroid at the mask's MEDIAN depth (median
    rejects background bleed at the silhouette edges). Extent: the mask's
    pixel bounding box scaled by depth — a coarse but honest size estimate.
    Returns (None, None) for empty/invalid masks.
    """
    vs, us = np.nonzero(mask)
    if len(us) < 8:
        return None, None
    d = depth_img[vs, us]
    d = d[np.isfinite(d) & (d > 0.05) & (d < 8.0)]
    if len(d) < 8:
        return None, None
    depth = float(np.median(d))
    u, v = float(us.mean()), float(vs.mean())
    pos = cam.unproject(u, v, depth)
    px_w = float(us.max() - us.min() + 1)
    px_h = float(vs.max() - vs.min() + 1)
    extent = (px_w / cam.fx * depth, px_h / cam.fy * depth)
    return pos, extent

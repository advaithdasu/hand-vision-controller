"""Camera-space hand pose -> robot-space end-effector target.

Position: the palm's (x, y) in the mirrored image maps to the robot's
(y, z); the hand's apparent size maps to depth along robot x, so pulling
your hand toward the camera pulls the arm back toward its base.

Orientation: the palm's rotation *relative to a captured reference pose*
is remapped into robot axes and applied on top of the default tool-down
pose. Relative mapping means the operator can re-zero at any comfortable
wrist posture instead of matching an absolute convention.
"""

from __future__ import annotations

import numpy as np

from .config import MappingConfig
from .transforms import (
    matrix_from_quat,
    quat_angle_between,
    quat_from_matrix,
    quat_slerp,
)

# Default end-effector orientation: tool axis pointing straight down
# (rotation of pi about world y), natural for tabletop manipulation and
# reachable with zero wrist roll on this arm's yaw/pitch/pitch wrist path.
TOOL_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
TOOL_DOWN_MAT = matrix_from_quat(TOOL_DOWN_QUAT)

# Camera axes -> robot axes (proper rotation, det = +1):
#   camera x (right on the mirrored image) -> robot -y
#   camera y (down on the image)           -> robot -z
#   camera z (into the scene)              -> robot +x
CAM_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def _remap(v: float, src: tuple, dst: tuple) -> float:
    t = (v - src[0]) / (src[1] - src[0])
    return dst[0] + np.clip(t, 0.0, 1.0) * (dst[1] - dst[0])


class HandToRobotMapper:
    def __init__(self, cfg: MappingConfig):
        self.cfg = cfg
        self.scale_ref: float | None = None
        self.rot_ref: np.ndarray | None = None  # palm rotation at calibration

    def calibrate(self, hand_scale: float, palm_rot: np.ndarray) -> None:
        """Capture the current hand pose as the neutral reference."""
        self.scale_ref = hand_scale
        self.rot_ref = palm_rot.copy()

    # ---------------- position ----------------

    def map_position(self, palm_xy: np.ndarray, hand_scale: float) -> np.ndarray:
        """(normalized image x, y) + apparent hand size -> robot xyz target."""
        ws = self.cfg.workspace
        # Image is mirrored, so image-right = operator-right = robot -y as
        # seen from behind the arm; both views then move the same way.
        y = _remap(palm_xy[0], self.cfg.img_x_range, (ws.y[1], ws.y[0]))
        z = _remap(palm_xy[1], self.cfg.img_y_range, (ws.z[1], ws.z[0]))
        # Depth window is relative to the calibrated neutral hand size, so
        # hand size and seating distance don't pin the arm at one end.
        ref = self.scale_ref if self.scale_ref is not None else self.cfg.default_hand_scale
        near = ref * self.cfg.depth_near_ratio
        far = ref * self.cfg.depth_far_ratio
        x = _remap(hand_scale, (near, far), ws.x)
        return ws.clamp(np.array([x, y, z]))

    # ---------------- orientation ----------------

    def map_orientation(self, palm_rot: np.ndarray) -> np.ndarray:
        """Palm rotation (camera frame) -> end-effector target quaternion."""
        if self.rot_ref is None:
            return TOOL_DOWN_QUAT.copy()
        # Rotation of the palm since calibration, expressed in camera axes,
        # then conjugated into robot axes.
        r_rel_cam = palm_rot @ self.rot_ref.T
        r_rel_robot = CAM_TO_ROBOT @ r_rel_cam @ CAM_TO_ROBOT.T
        q_target = quat_from_matrix(r_rel_robot @ TOOL_DOWN_MAT)
        return self._clamp_tilt(q_target)

    def _clamp_tilt(self, q: np.ndarray) -> np.ndarray:
        """Limit deviation from the default pose to max_tilt_rad."""
        ang = quat_angle_between(TOOL_DOWN_QUAT, q)
        if ang <= self.cfg.max_tilt_rad:
            return q
        return quat_slerp(TOOL_DOWN_QUAT, q, self.cfg.max_tilt_rad / ang)

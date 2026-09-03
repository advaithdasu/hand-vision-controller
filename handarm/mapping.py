"""Camera-space hand pose -> robot-space end-effector target.

Position: the palm's (x, y) in the mirrored image maps to the robot's
(y, z); the hand's apparent size maps to depth along robot x, so pulling
your hand toward the camera pulls the arm back toward its base. A position
offset lets the operator clutch: after a fist-freeze the arm resumes from
where it stopped, not from wherever the hand now happens to be.

Orientation: the palm's rotation *relative to a reference pose* is
remapped into robot axes and applied on top of the default tool-down pose.
The reference is captured at calibration and re-anchored on clutch release
so the wrist, too, resumes from its frozen pose.
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
        self.pos_offset = np.zeros(3)           # clutch rebasing, robot frame

    def calibrate(self, hand_scale: float, palm_rot: np.ndarray) -> None:
        """Capture the current hand pose as the neutral reference."""
        # Floor the reference so a degenerate detection (scale 0) can't
        # collapse the depth window into a divide-by-zero.
        self.scale_ref = max(float(hand_scale), 1e-4)
        self.rot_ref = palm_rot.copy()
        self.pos_offset = np.zeros(3)

    # ---------------- position ----------------

    def _map_position_absolute(self, palm_xy: np.ndarray, hand_scale: float) -> np.ndarray:
        """Absolute map, before the clutch offset and workspace clamp."""
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
        return np.array([x, y, z])

    def map_position(self, palm_xy: np.ndarray, hand_scale: float) -> np.ndarray:
        """(normalized image x, y) + apparent hand size -> robot xyz target."""
        p = self._map_position_absolute(palm_xy, hand_scale) + self.pos_offset
        return self.cfg.workspace.clamp(p)

    def rebase_position(self, palm_xy: np.ndarray, hand_scale: float,
                        target: np.ndarray) -> None:
        """Re-anchor the position map so the current hand pose maps exactly
        to `target` — called when a clutch releases, so the arm continues
        from where it froze instead of swinging to the hand's new spot."""
        self.pos_offset = np.asarray(target, dtype=float) - self._map_position_absolute(
            palm_xy, hand_scale
        )

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

    def rebase_orientation(self, palm_rot: np.ndarray, target: np.ndarray) -> None:
        """Re-anchor the orientation reference so the current palm rotation
        maps exactly to `target` (the orientation counterpart of
        rebase_position). Inverts map_orientation: with R_rel,robot = T D⁻¹
        the camera-frame relative rotation is Mᵀ R_rel,robot M, and the
        reference that yields it from the current palm rotation P is
        R_rel,camᵀ P."""
        r_rel_robot = matrix_from_quat(target) @ TOOL_DOWN_MAT.T
        r_rel_cam = CAM_TO_ROBOT.T @ r_rel_robot @ CAM_TO_ROBOT
        self.rot_ref = r_rel_cam.T @ palm_rot

    def _clamp_tilt(self, q: np.ndarray) -> np.ndarray:
        """Limit deviation from the default pose to max_tilt_rad."""
        ang = quat_angle_between(TOOL_DOWN_QUAT, q)
        if ang <= self.cfg.max_tilt_rad:
            return q
        return quat_slerp(TOOL_DOWN_QUAT, q, self.cfg.max_tilt_rad / ang)

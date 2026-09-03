"""Central configuration for the teleoperation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS_DIR / "scene.xml"
HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
TCP_SITE = "tcp"

# Comfortable elbow-up home configuration.
HOME_Q = np.array([0.0, 0.55, 0.85, 0.0, 0.75, 0.0])


@dataclass
class WorkspaceBox:
    """Axis-aligned reachable box for the end effector target, robot frame."""

    x: tuple = (0.26, 0.60)
    y: tuple = (-0.34, 0.34)
    z: tuple = (0.05, 0.55)

    def clamp(self, p: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(p[0], *self.x),
                np.clip(p[1], *self.y),
                np.clip(p[2], *self.z),
            ]
        )


@dataclass
class MappingConfig:
    workspace: WorkspaceBox = field(default_factory=WorkspaceBox)
    # Active region of the (normalized) image that maps onto the workspace.
    img_x_range: tuple = (0.15, 0.85)
    img_y_range: tuple = (0.15, 0.85)
    # Depth from apparent hand size (pose_features.apparent_scale, image
    # height per meter), as ratios of the calibrated neutral size. The
    # default only matters before the first calibration.
    default_hand_scale: float = 1.9
    depth_near_ratio: float = 1.6     # hand this much bigger -> arm retracted
    depth_far_ratio: float = 0.6      # hand this much smaller -> arm extended
    # Maximum end-effector tilt away from the default (tool-down) pose.
    max_tilt_rad: float = 1.2


@dataclass
class IKConfig:
    max_iters: int = 12
    pos_tol: float = 1.5e-3          # m
    ori_tol: float = 1.5e-2          # rad
    ori_weight: float = 0.35         # orientation error weight vs position
    base_damping: float = 0.02       # always-on lambda
    singular_damping: float = 0.20   # extra lambda ramped in near singularity
    manip_threshold: float = 0.006   # manipulability below which damping ramps
                                     # (home pose measures ~0.022)
    max_step: float = 0.35           # rad, per-joint per-iteration clamp
    max_pos_err: float = 0.12        # m, clamp translational error per iter


@dataclass
class ControlConfig:
    # One Euro filter for the hand target position. beta is in Hz per
    # (m/s): hovering gets the 1 Hz floor (no jitter), a 1 m/s sweep opens
    # the cutoff to 3.5 Hz (~45 ms lag instead of ~160 ms).
    pos_min_cutoff: float = 1.0      # Hz
    pos_beta: float = 2.5
    pos_d_cutoff: float = 1.0
    # Same for orientation; beta in Hz per (rad/s).
    ori_cutoff: float = 2.0
    ori_beta: float = 1.0
    ori_d_cutoff: float = 1.0
    pinch_cutoff: float = 8.0
    # Joint-space velocity limit applied to commanded targets.
    max_joint_vel: float = 3.5       # rad/s
    # Pinch hysteresis (ratio of thumb-index distance to hand scale).
    pinch_close: float = 0.32
    pinch_open: float = 0.45
    # Pinch ratio above pinch_open at which the aperture reads fully open.
    aperture_margin: float = 0.25
    # Frames without a detection before the arm holds position.
    hold_after_lost_frames: int = 5
    # Gesture debouncing: a fist must persist this long before it clutches,
    # an open hand this long before it releases, so a single misdetected
    # frame can neither freeze the arm nor drop it back into tracking.
    clutch_engage_sec: float = 0.12
    clutch_release_sec: float = 0.08
    # Auto-calibration gate: the hand must be fully inside the frame and
    # nearly still for this long before its pose is taken as neutral.
    calib_settle_sec: float = 0.5
    calib_max_speed: float = 0.25    # image heights per second
    calib_edge_margin: float = 0.03  # normalized image units


@dataclass
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    sim_render_size: tuple = (640, 640)   # (width, height) of the sim panel
    display_height: int = 640
    orientation_control: bool = True
    mirror: bool = True
    recordings_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "recordings"
    )
    mapping: MappingConfig = field(default_factory=MappingConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    control: ControlConfig = field(default_factory=ControlConfig)

"""App-level pipeline tests with synthetic hand observations (no camera).

Exercises TeleopApp.process_hand + solve_and_command end to end: gesture
handling, calibration, target mapping, IK, and the physics response.
"""

import time

import numpy as np
import pytest

from handarm.app import TeleopApp
from handarm.config import AppConfig
from handarm.transforms import quat_angle_between

from .test_pose_features import (
    rot_x,
    synthetic_fist,
    synthetic_open_hand,
    synthetic_pinch,
    synthetic_tucked_fist,
)

ASPECT = AppConfig().camera_width / AppConfig().camera_height


class FakeObs:
    """Duck-typed HandObservation (avoids importing mediapipe in tests)."""

    def __init__(self, image_landmarks, world_landmarks):
        self.image_landmarks = image_landmarks
        self.world_landmarks = world_landmarks
        self.handedness = "Right"


def obs_at(image_xy, k=2.0, world=None):
    """A hand of the given shape whose palm center sits at image_xy
    (normalized), projected orthographically with scale k (∝ 1/depth).
    World landmarks are the metric shape itself, as MediaPipe reports."""
    world = synthetic_open_hand() if world is None else world
    palm = world[[0, 5, 9, 13, 17], :2].mean(axis=0)
    img = world.copy()
    img[:, 0] = image_xy[0] + (world[:, 0] - palm[0]) * k / ASPECT
    img[:, 1] = image_xy[1] + (world[:, 1] - palm[1]) * k
    return FakeObs(img, world)


@pytest.fixture
def app():
    return TeleopApp(AppConfig())


DT = 1 / 30


def drive(app, obs, seconds):
    for _ in range(round(seconds / DT)):
        app.tick(obs, DT)
        app.sim.step(DT)


def settle(app):
    """Hold an open hand still at the image center until calibration locks on."""
    drive(app, obs_at([0.5, 0.5]), 0.8)
    assert app.calibrated


def test_hand_position_moves_arm(app):
    settle(app)
    p_center = app.target_pos.copy()

    drive(app, obs_at([0.2, 0.5]), 1.0)  # move hand left in mirrored image
    assert app.target_pos[1] > p_center[1] + 0.05  # robot y increases

    # And the physical arm actually tracked the target.
    tcp, _ = app.kin.fk(app.sim.arm_q())
    assert np.linalg.norm(tcp - app.target_pos) < 0.03


def test_hand_height_maps_to_z(app):
    settle(app)
    z0 = app.target_pos[2]
    drive(app, obs_at([0.5, 0.25]), 1.0)  # raise hand
    assert app.target_pos[2] > z0 + 0.05


def test_hand_depth_maps_to_x(app):
    settle(app)
    x0 = app.target_pos[0]
    drive(app, obs_at([0.5, 0.5], k=2.6), 1.0)  # hand closer -> bigger
    assert app.target_pos[0] < x0 - 0.05  # arm retracts


def test_palm_tilt_does_not_change_depth(app):
    """Regression: the wrist-to-knuckle image length used as the depth
    proxy shrank with palm pitch, so tilting the wrist extended the arm."""
    settle(app)
    drive(app, obs_at([0.5, 0.5]), 0.5)
    x_flat = app.target_pos[0]
    pitched = synthetic_open_hand() @ rot_x(0.7).T
    drive(app, obs_at([0.5, 0.5], world=pitched), 1.0)
    assert abs(app.target_pos[0] - x_flat) < 0.01
    # The wrist did follow the tilt.
    _, quat = app.kin.fk(app.q_cmd)
    assert quat_angle_between(app.target_quat, quat) < 0.2


def test_pinch_closes_gripper(app):
    drive(app, obs_at([0.5, 0.5]), 0.3)
    assert not app.gripping
    assert app.grip_opening > 0.5

    drive(app, obs_at([0.5, 0.5], world=synthetic_pinch()), 0.5)
    assert app.gripping
    assert app.grip_opening < 0.2


def test_tucked_thumb_fist_clutches_without_gripping(app):
    settle(app)
    drive(app, obs_at([0.5, 0.5], world=synthetic_tucked_fist()), 0.5)
    assert app.clutched
    assert not app.gripping
    assert app.grip_opening == 1.0


def test_fist_freezes_tracking(app):
    settle(app)
    drive(app, obs_at([0.5, 0.5], world=synthetic_fist()), 0.3)
    assert app.frozen
    p0 = app.target_pos.copy()

    drive(app, obs_at([0.8, 0.8], world=synthetic_fist()), 0.5)
    assert np.allclose(app.target_pos, p0)  # target did not follow the fist

    # Opening the hand releases the clutch; tracking resumes from the
    # frozen pose (not the hand's new spot) and then follows the hand.
    drive(app, obs_at([0.8, 0.8]), 0.5)
    assert not app.frozen
    assert np.allclose(app.target_pos, p0, atol=1e-6)
    drive(app, obs_at([0.6, 0.5]), 0.5)
    assert not np.allclose(app.target_pos, p0, atol=0.02)


def test_clutch_debounces_single_frame_glitches(app):
    settle(app)
    app.tick(obs_at([0.5, 0.5], world=synthetic_fist()), DT)
    assert not app.clutched
    drive(app, obs_at([0.5, 0.5], world=synthetic_fist()), 0.3)
    assert app.clutched
    app.tick(obs_at([0.5, 0.5]), DT)
    assert app.clutched


def test_clutch_does_not_drop_held_object(app):
    """Regression: fist-clutching while pinch-holding must not open the
    gripper (is_fist used to require pinch_ratio > pinch_open, which
    flipped the hysteresis and released the grasp)."""
    settle(app)
    drive(app, obs_at([0.5, 0.5], world=synthetic_pinch()), 0.5)
    assert app.gripping and app.grip_opening < 0.2

    drive(app, obs_at([0.5, 0.5], world=synthetic_fist()), 0.5)  # clutch engaged
    assert app.frozen
    assert app.grip_opening < 0.2, "gripper opened while clutched"

    drive(app, obs_at([0.5, 0.5]), 0.5)  # release clutch
    assert not app.frozen
    assert app.grip_latched
    assert app.grip_opening < 0.2, "grasp dropped on clutch release"

    # Re-pinching re-arms normal gripper tracking.
    drive(app, obs_at([0.5, 0.5], world=synthetic_pinch()), 0.5)
    assert not app.grip_latched and app.gripping
    drive(app, obs_at([0.5, 0.5]), 0.5)
    assert app.grip_opening > 0.5  # now an open hand releases as usual


def test_clutch_resumes_from_frozen_pose(app):
    """The point of a clutch: reposition the hand without moving the arm."""
    settle(app)
    drive(app, obs_at([0.35, 0.5]), 1.0)
    drive(app, obs_at([0.35, 0.5], world=synthetic_fist()), 0.3)
    assert app.clutched
    frozen = app.target_pos.copy()
    drive(app, obs_at([0.75, 0.65], k=2.4, world=synthetic_fist()), 0.3)
    assert np.allclose(app.target_pos, frozen)

    away = app.mapper.map_position(np.array([0.75, 0.65]), 2.4)
    assert np.linalg.norm(away - frozen) > 0.15  # an absolute map would lurch here
    drive(app, obs_at([0.75, 0.65], k=2.4), 0.5)
    assert not app.clutched
    assert np.allclose(app.target_pos, frozen, atol=1e-6)

    drive(app, obs_at([0.7, 0.65], k=2.4), 1.0)  # hand left -> robot +y
    assert app.target_pos[1] > frozen[1] + 0.03


def test_manual_freeze_survives_open_hand(app):
    """Regression: the 'f' freeze used to be undone by the open-hand
    unfreeze rule on the very next frame."""
    settle(app)
    app.manual_freeze = True  # what handle_key('f') does
    p0 = app.target_pos.copy()
    drive(app, obs_at([0.8, 0.7]), 0.5)  # open hand, moving
    assert app.frozen
    assert np.allclose(app.target_pos, p0)
    app.manual_freeze = False
    drive(app, obs_at([0.8, 0.7]), 0.5)
    assert np.allclose(app.target_pos, p0, atol=1e-6)  # rebased, no lurch


def test_calibration_waits_for_a_still_hand(app):
    cfg = app.cfg.control
    for i in range(20):  # sweeping in from the edge
        app.tick(obs_at([-0.1 + i * 0.03, 0.5]), DT)
    assert not app.calibrated and app.calib_progress == 0.0
    drive(app, obs_at([0.5, 0.5]), cfg.calib_settle_sec * 0.6)
    assert not app.calibrated and app.calib_progress > 0.4
    drive(app, obs_at([0.5, 0.5]), cfg.calib_settle_sec * 0.6)
    assert app.calibrated


def test_calibration_request_is_immediate_with_a_hand(app):
    app.tick(obs_at([0.6, 0.4]), DT)
    assert not app.calibrated
    app.request_calibration()  # what handle_key('c') does
    assert app.calibrated and app.mapper.scale_ref is not None
    app.tick(None, DT)
    app.request_calibration()  # no hand: re-arm the gate instead
    assert not app.calibrated


def test_lost_hand_holds_position(app):
    settle(app)
    app.tick(obs_at([0.75, 0.3]), DT)  # target jumps away; arm still moving
    q_tracking = app.q_cmd.copy()
    for _ in range(app.cfg.control.hold_after_lost_frames - 1):
        app.tick(None, DT)
    assert not np.allclose(app.q_cmd, q_tracking)  # still converging in the grace window
    app.tick(None, DT)
    q_held = app.q_cmd.copy()
    for _ in range(10):
        app.tick(None, DT)
    assert np.array_equal(app.q_cmd, q_held)


def test_fast_sweep_tracks_with_little_lag(app):
    settle(app)
    drive(app, obs_at([0.35, 0.5]), 1.0)
    # Sweep 0.3 of the image width in 0.4 s.
    seconds = 0.4
    n = round(seconds / DT)
    x = 0.35
    for i in range(1, n + 1):
        x = 0.35 + 0.3 * i / n
        app.tick(obs_at([x, 0.5]), DT)
    ideal = app.mapper.map_position(np.array([x, 0.5]), 2.0)
    # Image units/s -> m/s: the 0.7-wide active image strip spans the
    # 0.68 m workspace in y (~0.7 m/s here).
    speed = (0.3 / seconds) * (0.68 / 0.7)
    lag_sec = abs(ideal[1] - app.target_pos[1]) / speed
    assert lag_sec < 0.09


def test_realtime_budget(app):
    """One control tick (IK + 1/30 s physics) must fit a 30 fps budget."""
    settle(app)  # warm up
    t0 = time.perf_counter()
    n = 60
    for _ in range(n):
        app.tick(obs_at([0.55, 0.45]), DT)
        app.sim.step(DT)
    per_tick_ms = (time.perf_counter() - t0) / n * 1000
    assert per_tick_ms < 15, f"control tick {per_tick_ms:.1f} ms exceeds budget"

"""App-level pipeline tests with synthetic hand observations (no camera).

Exercises TeleopApp.process_hand + solve_and_command end to end: gesture
handling, calibration, target mapping, IK, and the physics response.
"""

import numpy as np
import pytest

from handarm.app import TeleopApp
from handarm.config import AppConfig

from .test_pose_features import synthetic_fist, synthetic_open_hand, synthetic_pinch


class FakeObs:
    """Duck-typed HandObservation (avoids importing mediapipe in tests)."""

    def __init__(self, image_landmarks, world_landmarks):
        self.image_landmarks = image_landmarks
        self.world_landmarks = world_landmarks
        self.handedness = "Right"


def obs_at(image_xy, scale=0.18):
    """An open hand whose palm center sits at image_xy (normalized)."""
    world = synthetic_open_hand()
    img = world / (np.linalg.norm(world[9]) + 1e-9) * scale  # middle MCP dist = scale
    img[:, :2] += np.asarray(image_xy) - img[[0, 5, 9, 13, 17], :2].mean(axis=0)
    return FakeObs(img, world)


@pytest.fixture
def app():
    return TeleopApp(AppConfig())


DT = 1 / 30


def drive(app, obs, seconds):
    for _ in range(int(seconds / DT)):
        app.process_hand(obs, DT)
        app.solve_and_command(DT)
        app.sim.step(DT)


def test_hand_position_moves_arm(app):
    drive(app, obs_at([0.5, 0.5]), 0.2)  # calibrate at center
    p_center = app.target_pos.copy()

    drive(app, obs_at([0.2, 0.5]), 1.0)  # move hand left in mirrored image
    assert app.target_pos[1] > p_center[1] + 0.05  # robot y increases

    # And the physical arm actually tracked the target.
    tcp, _ = app.kin.fk(app.sim.arm_q())
    assert np.linalg.norm(tcp - app.target_pos) < 0.03


def test_hand_height_maps_to_z(app):
    drive(app, obs_at([0.5, 0.5]), 0.2)
    z0 = app.target_pos[2]
    drive(app, obs_at([0.5, 0.25]), 1.0)  # raise hand
    assert app.target_pos[2] > z0 + 0.05


def test_pinch_closes_gripper(app):
    open_obs = FakeObs(obs_at([0.5, 0.5]).image_landmarks, synthetic_open_hand())
    drive(app, open_obs, 0.3)
    assert not app.gripping
    assert app.grip_opening > 0.5

    pinch_obs = FakeObs(obs_at([0.5, 0.5]).image_landmarks, synthetic_pinch())
    drive(app, pinch_obs, 0.5)
    assert app.gripping
    assert app.grip_opening < 0.2


def test_fist_freezes_tracking(app):
    drive(app, obs_at([0.5, 0.5]), 0.3)
    p0 = app.target_pos.copy()

    fist_img = obs_at([0.8, 0.8]).image_landmarks
    drive(app, FakeObs(fist_img, synthetic_fist()), 0.5)
    assert app.frozen
    assert np.allclose(app.target_pos, p0)  # target did not follow the fist

    # Opening the hand releases the clutch and tracking resumes.
    drive(app, obs_at([0.8, 0.5]), 0.5)
    assert not app.frozen
    assert not np.allclose(app.target_pos, p0)


def test_clutch_does_not_drop_held_object(app):
    """Regression: fist-clutching while pinch-holding must not open the
    gripper (is_fist requires pinch_ratio > pinch_open, which used to flip
    the hysteresis and release the grasp)."""
    img = obs_at([0.5, 0.5]).image_landmarks
    drive(app, FakeObs(img, synthetic_pinch()), 0.5)
    assert app.gripping and app.grip_opening < 0.2

    drive(app, FakeObs(img, synthetic_fist()), 0.5)  # clutch engaged
    assert app.frozen
    assert app.grip_opening < 0.2, "gripper opened while clutched"

    drive(app, FakeObs(img, synthetic_open_hand()), 0.5)  # release clutch
    assert not app.frozen
    assert app.grip_latched
    assert app.grip_opening < 0.2, "grasp dropped on clutch release"

    # Re-pinching re-arms normal gripper tracking.
    drive(app, FakeObs(img, synthetic_pinch()), 0.5)
    assert not app.grip_latched and app.gripping
    drive(app, FakeObs(img, synthetic_open_hand()), 0.5)
    assert app.grip_opening > 0.5  # now an open hand releases as usual


def test_manual_freeze_survives_open_hand(app):
    """Regression: the 'f' freeze used to be undone by the open-hand
    unfreeze rule on the very next frame."""
    drive(app, obs_at([0.5, 0.5]), 0.3)
    app.manual_freeze = True  # what handle_key('f') does
    p0 = app.target_pos.copy()
    drive(app, obs_at([0.8, 0.7]), 0.5)  # open hand, moving
    assert app.frozen
    assert np.allclose(app.target_pos, p0)


def test_hand_scale_invariant_to_inplane_rotation(app):
    """Regression: raw normalized landmarks are anisotropic (x/width,
    y/height), so in-plane rotation used to masquerade as depth motion."""
    lm = synthetic_open_hand()
    upright = lm / 3.0  # wrist->MCP mostly along image y
    # Physically rotate 90 degrees in pixel space, then re-normalize:
    # pixel-x becomes pixel-y and vice versa, so normalized x picks up a
    # 1/aspect factor and normalized y an aspect factor.
    rot90 = upright.copy()
    rot90[:, [0, 1]] = rot90[:, [1, 0]] * np.array(
        [1 / app.frame_aspect, app.frame_aspect]
    )

    s_up = app._hand_scale(upright)
    s_rot = app._hand_scale(rot90)
    assert s_up == pytest.approx(s_rot, rel=1e-6)


def test_realtime_budget(app):
    """One control tick (IK + 1/30 s physics) must fit a 30 fps budget."""
    import time

    drive(app, obs_at([0.5, 0.5]), 0.2)  # warm up
    t0 = time.perf_counter()
    n = 60
    for _ in range(n):
        app.process_hand(obs_at([0.55, 0.45]), DT)
        app.solve_and_command(DT)
        app.sim.step(DT)
    per_tick_ms = (time.perf_counter() - t0) / n * 1000
    assert per_tick_ms < 15, f"control tick {per_tick_ms:.1f} ms exceeds budget"

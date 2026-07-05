import numpy as np
import pytest

from handarm.config import MappingConfig
from handarm.mapping import CAM_TO_ROBOT, TOOL_DOWN_QUAT, HandToRobotMapper
from handarm.transforms import matrix_from_quat, quat_angle_between


@pytest.fixture
def mapper():
    return HandToRobotMapper(MappingConfig())


def test_cam_to_robot_is_proper_rotation():
    assert np.allclose(CAM_TO_ROBOT @ CAM_TO_ROBOT.T, np.eye(3))
    assert np.isclose(np.linalg.det(CAM_TO_ROBOT), 1.0)


def test_positions_stay_in_workspace(mapper):
    ws = mapper.cfg.workspace
    rng = np.random.default_rng(3)
    for _ in range(200):
        xy = rng.random(2) * 1.4 - 0.2       # includes out-of-frame values
        scale = rng.random() * 0.5 + 0.01
        p = mapper.map_position(xy, scale)
        assert ws.x[0] <= p[0] <= ws.x[1]
        assert ws.y[0] <= p[1] <= ws.y[1]
        assert ws.z[0] <= p[2] <= ws.z[1]


def test_position_directions(mapper):
    center = np.array([0.5, 0.5])
    mid_scale = 0.19
    p0 = mapper.map_position(center, mid_scale)
    # Hand moves right in the mirrored image -> robot y decreases.
    p_right = mapper.map_position(center + [0.2, 0], mid_scale)
    assert p_right[1] < p0[1]
    # Hand moves up in the image (y decreases) -> robot z increases.
    p_up = mapper.map_position(center - [0, 0.2], mid_scale)
    assert p_up[2] > p0[2]
    # Hand approaches the camera (bigger) -> arm retracts (x decreases).
    p_near = mapper.map_position(center, mid_scale + 0.06)
    assert p_near[0] < p0[0]


def test_depth_window_follows_calibration(mapper):
    """The calibrated neutral hand size recenters the depth axis."""
    center = np.array([0.5, 0.5])
    small_ref, big_ref = 0.10, 0.30

    mapper.calibrate(small_ref, np.eye(3))
    x_small_hand = mapper.map_position(center, small_ref)[0]
    mapper.calibrate(big_ref, np.eye(3))
    x_big_hand = mapper.map_position(center, big_ref)[0]
    # Both operators hold their hand at their own neutral distance and the
    # arm sits at the same mid-workspace depth, not pinned at an edge.
    assert x_small_hand == pytest.approx(x_big_hand, abs=1e-9)
    ws = mapper.cfg.workspace
    assert ws.x[0] < x_small_hand < ws.x[1]

    # Depth motion still works around the calibrated reference.
    assert mapper.map_position(center, big_ref * 1.5)[0] < x_big_hand
    assert mapper.map_position(center, big_ref * 0.7)[0] > x_big_hand


def test_orientation_identity_before_calibration(mapper):
    q = mapper.map_orientation(np.eye(3))
    assert quat_angle_between(q, TOOL_DOWN_QUAT) < 1e-9


def test_orientation_neutral_after_calibration(mapper):
    R = matrix_from_quat(np.array([0.3, -0.2, 0.5, 0.79]))
    mapper.calibrate(0.2, R)
    q = mapper.map_orientation(R)  # same pose as calibration -> neutral
    assert quat_angle_between(q, TOOL_DOWN_QUAT) < 1e-9


def test_orientation_tilt_clamped(mapper):
    mapper.calibrate(0.2, np.eye(3))
    # A 180-degree flip of the palm must be clamped to max_tilt_rad.
    flip = np.diag([1.0, -1.0, -1.0])
    q = mapper.map_orientation(flip)
    assert quat_angle_between(q, TOOL_DOWN_QUAT) <= mapper.cfg.max_tilt_rad + 1e-6


def test_orientation_rotation_magnitude_preserved(mapper):
    mapper.calibrate(0.2, np.eye(3))
    ang = 0.5
    tilt = np.array(
        [
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)],
        ]
    )
    q = mapper.map_orientation(tilt)
    assert quat_angle_between(q, TOOL_DOWN_QUAT) == pytest.approx(ang, abs=1e-6)

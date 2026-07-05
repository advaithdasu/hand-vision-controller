import numpy as np
import pytest

from handarm import transforms as tr


def rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_matrix_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rand_quat(rng)
        m = tr.matrix_from_quat(q)
        q2 = tr.quat_from_matrix(m)
        # q and -q are the same rotation
        assert min(np.linalg.norm(q - q2), np.linalg.norm(q + q2)) < 1e-9


def test_matrix_is_rotation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = tr.matrix_from_quat(rand_quat(rng))
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(m) == pytest.approx(1.0)


def test_axis_angle_identity():
    assert np.allclose(tr.quat_to_axis_angle(np.array([0, 0, 0, 1.0])), 0)


def test_axis_angle_known_rotation():
    # 90 degrees about z
    q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    aa = tr.quat_to_axis_angle(q)
    assert np.allclose(aa, [0, 0, np.pi / 2], atol=1e-9)


def test_orientation_error_zero_for_same():
    rng = np.random.default_rng(2)
    q = rand_quat(rng)
    assert np.linalg.norm(tr.orientation_error(q, q)) < 1e-9
    # sign-flipped quaternion is the same orientation
    assert np.linalg.norm(tr.orientation_error(q, -q)) < 1e-9


def test_slerp_endpoints_and_midpoint():
    a = np.array([0, 0, 0, 1.0])
    b = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg z
    assert np.allclose(tr.quat_slerp(a, b, 0.0), a)
    assert np.allclose(np.abs(tr.quat_slerp(a, b, 1.0)), np.abs(b), atol=1e-9)
    mid = tr.quat_slerp(a, b, 0.5)
    assert tr.quat_angle_between(a, mid) == pytest.approx(np.pi / 4, abs=1e-6)


def test_quat_multiply_composition():
    qz = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    q_full = tr.quat_multiply(qz, qz)  # two 90s = 180 about z
    assert tr.quat_angle_between(np.array([0, 0, 0, 1.0]), q_full) == pytest.approx(np.pi, abs=1e-6)

import numpy as np
import pytest

from handarm import pose_features as pf


def synthetic_open_hand():
    """Flat right hand, palm facing -z, fingers along +y, in 'meters'."""
    lm = np.zeros((21, 3))
    lm[pf.WRIST] = [0, 0, 0]
    # Thumb angled off to +x.
    lm[1], lm[2], lm[3], lm[4] = [0.02, 0.01, 0], [0.045, 0.03, 0], [0.06, 0.05, 0], [0.07, 0.065, 0]
    # Finger columns: (mcp x, spacing along y).
    for x, (mcp, pip, tip) in zip(
        (0.03, 0.01, -0.01, -0.03),
        ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)),
    ):
        lm[mcp] = [x, 0.09, 0]
        lm[pip] = [x, 0.13, 0]
        lm[mcp + 2] = [x, 0.155, 0]  # dip
        lm[tip] = [x, 0.175, 0]
    return lm


def synthetic_fist():
    """All fingers curled back toward the wrist."""
    lm = synthetic_open_hand()
    for mcp, pip, tip in pf.FINGERS.values():
        base = lm[mcp]
        lm[pip] = base + [0, 0.02, -0.02]
        lm[pip + 1] = base + [0, 0.01, -0.035]
        lm[tip] = base + [0, -0.005, -0.03]  # tip folded to the palm
    # Thumb wrapped across, not near the index tip.
    lm[pf.THUMB_TIP] = [0.0, 0.05, -0.03]
    return lm


def synthetic_tucked_fist():
    """Fist with the thumb wrapped over the curled index: the thumb tip
    lands next to the index tip, so thumb-index distance looks like a pinch."""
    lm = synthetic_fist()
    lm[pf.THUMB_TIP] = lm[pf.INDEX_TIP] + [0.012, 0.005, -0.008]
    return lm


def synthetic_pinch():
    """Open hand but thumb and index tips touching."""
    lm = synthetic_open_hand()
    meeting_point = np.array([0.03, 0.12, -0.01])
    lm[pf.THUMB_TIP] = meeting_point + [0.001, 0, 0]
    lm[pf.INDEX_TIP] = meeting_point
    lm[pf.INDEX_PIP] = [0.03, 0.115, 0]
    return lm


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def project(world, k, aspect):
    """Orthographic pinhole stand-in: world landmarks (meters, camera axes)
    scaled by k = focal / depth into normalized image coordinates."""
    img = world.copy()
    img[:, 0] = (0.5 + world[:, 0] * k) / aspect
    img[:, 1] = 0.5 + world[:, 1] * k
    return img


def test_hand_scale_positive():
    assert pf.hand_scale(synthetic_open_hand()) > 0.05


def test_palm_frame_orthonormal():
    R = pf.palm_frame(synthetic_open_hand())
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_palm_frame_forward_axis():
    R = pf.palm_frame(synthetic_open_hand())
    # Forward axis should point wrist -> middle MCP, i.e. mostly +y here.
    assert R[1, 0] > 0.9


def test_open_hand_fingers_extended():
    lm = synthetic_open_hand()
    assert pf.count_extended(lm) == 4
    assert not pf.is_fist(lm)


def test_fist_detected():
    lm = synthetic_fist()
    assert pf.count_extended(lm) == 0
    assert pf.is_fist(lm)


def test_tucked_thumb_fist_detected():
    lm = synthetic_tucked_fist()
    assert pf.pinch_ratio(lm) < 0.32  # would read as a pinch by thumb-index distance
    assert pf.is_fist(lm)


def test_pinch_ratio_discriminates():
    assert pf.pinch_ratio(synthetic_pinch()) < 0.15
    assert pf.pinch_ratio(synthetic_open_hand()) > 0.5


def test_pinch_is_not_fist():
    assert not pf.is_fist(synthetic_pinch())


def test_palm_center_between_wrist_and_knuckles():
    lm = synthetic_open_hand()
    c = pf.palm_center(lm)
    assert 0 < c[1] < 0.09


def test_fully_in_frame():
    lm = synthetic_open_hand() + [0.5, 0.4, 0]
    assert pf.fully_in_frame(lm, 0.03)
    lm[pf.INDEX_TIP, 1] = 0.99
    assert not pf.fully_in_frame(lm, 0.03)


def test_apparent_scale_invariant_to_palm_tilt():
    """The raw knuckle length shrinks with cos(pitch); the image/world ratio
    does not, so tilting the wrist no longer reads as reaching."""
    aspect = 16 / 9
    base = synthetic_open_hand()
    ref = pf.apparent_scale(project(base, 2.0, aspect), base, aspect)
    for R in (rot_x(0.6), rot_x(-0.8), rot_y(0.7), rot_z(1.2), rot_x(0.5) @ rot_y(0.4)):
        tilted = base @ R.T
        assert pf.apparent_scale(project(tilted, 2.0, aspect), tilted, aspect) == pytest.approx(
            ref, rel=1e-6
        )
    pitched = base @ rot_x(0.8).T
    raw_flat = pf.hand_scale(project(base, 2.0, aspect) * [aspect, 1, 1])
    raw_pitched = pf.hand_scale(project(pitched, 2.0, aspect) * [aspect, 1, 1])
    assert raw_pitched < 0.8 * raw_flat


def test_apparent_scale_tracks_inverse_depth():
    base = synthetic_open_hand()
    near = pf.apparent_scale(project(base, 3.0, 1.0), base, 1.0)
    far = pf.apparent_scale(project(base, 1.5, 1.0), base, 1.0)
    assert near / far == pytest.approx(2.0, rel=1e-9)


def test_apparent_scale_finite_with_no_image_extent():
    base = synthetic_open_hand()
    along_z = np.zeros_like(base)
    along_z[:, 2] = base[:, 0] + base[:, 1]
    s = pf.apparent_scale(project(along_z, 2.0, 1.0), along_z, 1.0)
    assert np.isfinite(s) and s == 0.0

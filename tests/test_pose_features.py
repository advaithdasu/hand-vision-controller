import numpy as np

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


def synthetic_pinch():
    """Open hand but thumb and index tips touching."""
    lm = synthetic_open_hand()
    meeting_point = np.array([0.03, 0.12, -0.01])
    lm[pf.THUMB_TIP] = meeting_point + [0.001, 0, 0]
    lm[pf.INDEX_TIP] = meeting_point
    lm[pf.INDEX_PIP] = [0.03, 0.115, 0]
    return lm


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


def test_pinch_ratio_discriminates():
    assert pf.pinch_ratio(synthetic_pinch()) < 0.15
    assert pf.pinch_ratio(synthetic_open_hand()) > 0.5


def test_pinch_is_not_fist():
    assert not pf.is_fist(synthetic_pinch())


def test_palm_center_between_wrist_and_knuckles():
    lm = synthetic_open_hand()
    c = pf.palm_center(lm)
    assert 0 < c[1] < 0.09

"""Pure-numpy geometry extracted from MediaPipe's 21 hand landmarks.

All functions take a (21, 3) array of landmark coordinates. They work on
either normalized-image landmarks or metric world landmarks; docstrings note
which is expected. Keeping this module free of MediaPipe imports makes the
geometry unit-testable with synthetic hands.
"""

from __future__ import annotations

import numpy as np

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

# Skeleton edges for drawing the overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def hand_scale(lm: np.ndarray) -> float:
    """Characteristic hand size: wrist to middle-finger MCP distance."""
    return float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))


def palm_center(lm: np.ndarray) -> np.ndarray:
    """Centroid of the wrist and the four finger MCP knuckles."""
    idx = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return lm[idx].mean(axis=0)


def palm_frame(lm: np.ndarray) -> np.ndarray:
    """Orthonormal palm frame as a 3x3 rotation matrix (columns = axes).

    Column 0: forward, wrist toward middle knuckle.
    Column 1: lateral, pinky side toward index side.
    Column 2: palm normal (right-handed with the above).
    Use metric world landmarks for a rotation that tracks true hand tilt.
    """
    forward = lm[MIDDLE_MCP] - lm[WRIST]
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    lateral = lm[INDEX_MCP] - lm[PINKY_MCP]
    normal = np.cross(forward, lateral)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    lateral = np.cross(normal, forward)
    return np.column_stack([forward, lateral, normal])


def pinch_ratio(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance, normalized by hand scale.

    Small (< ~0.3) means the operator is pinching.
    """
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / (hand_scale(lm) + 1e-9)


def finger_extensions(lm: np.ndarray) -> dict:
    """Per-finger extension ratio: tip-to-wrist over MCP-to-wrist distance.

    Extended fingers read ~1.7-2.1, curled fingers ~0.7-1.2.
    """
    wrist = lm[WRIST]
    out = {}
    for name, (mcp, _pip, tip) in FINGERS.items():
        base = np.linalg.norm(lm[mcp] - wrist) + 1e-9
        out[name] = float(np.linalg.norm(lm[tip] - wrist) / base)
    return out


def count_extended_from(ext: dict, threshold: float = 1.4) -> int:
    return sum(1 for r in ext.values() if r > threshold)


def count_extended(lm: np.ndarray, threshold: float = 1.4) -> int:
    return count_extended_from(finger_extensions(lm), threshold)


def is_fist_from(ext: dict, pinch: float) -> bool:
    """Fist from precomputed extension ratios and pinch ratio.

    The pinch check keeps a pinch (index curled toward thumb) from being
    misread as a fist when the other fingers relax.
    """
    curled = sum(1 for name in ("middle", "ring", "pinky") if ext[name] < 1.25)
    return curled == 3 and ext["index"] < 1.25 and pinch > 0.5


def is_fist(lm: np.ndarray) -> bool:
    return is_fist_from(finger_extensions(lm), pinch_ratio(lm))

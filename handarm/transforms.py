"""Minimal quaternion / rotation utilities.

Quaternions are [x, y, z, w] (PyBullet convention). Rotation matrices are
3x3 numpy arrays whose columns are the frame axes expressed in the parent.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b: rotate by b first, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Return the rotation vector (axis * angle) for a unit quaternion."""
    q = quat_normalize(q)
    if q[3] < 0:  # shortest arc
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[3])
    return q[:3] / s * angle


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Shepperd's method: rotation matrix -> quaternion [x, y, z, w]."""
    m = np.asarray(m, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return quat_normalize(
            np.array(
                [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
            )
        )
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return quat_normalize(np.array(q))


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return quat_normalize(np.sin((1 - t) * theta) / s * a + np.sin(t * theta) / s * b)


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute rotation angle (radians) between two orientations."""
    return float(np.linalg.norm(orientation_error(b, a)))


def orientation_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector that takes q_current to q_target."""
    return quat_to_axis_angle(quat_multiply(q_target, quat_conjugate(q_current)))

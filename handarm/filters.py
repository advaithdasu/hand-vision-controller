"""Signal smoothing: One Euro filters for position and orientation.

The One Euro filter (Casiez et al., 2012) adapts its cutoff with signal
speed: heavy smoothing when the hand hovers (kills jitter), light smoothing
during fast moves (kills lag). It is the standard choice for pointing and
tracking interfaces.
"""

from __future__ import annotations

import numpy as np

from .transforms import quat_angle_between, quat_normalize, quat_slerp


def _smoothing_factor(dt: float, cutoff: float) -> float:
    r = 2.0 * np.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """Vector-valued One Euro filter."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()
        # A zero/negative dt (coarsened clocks) must not wipe filter memory;
        # hold the previous estimate instead of passing raw jitter through.
        if dt <= 0:
            return self._x.copy()

        # Derivative estimate, low-passed.
        dx = (x - self._x) / dt
        a_d = _smoothing_factor(dt, self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        # Speed-adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _smoothing_factor(dt, cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class QuaternionLowPass:
    """One Euro filter on orientation: slerp toward the target with a cutoff
    that opens up with angular speed. beta = 0 degrades to a plain
    first-order low-pass."""

    def __init__(self, cutoff: float = 3.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.cutoff = cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._q = None
        self._rate = 0.0  # low-passed angular speed of the input, rad/s

    def reset(self) -> None:
        self._q = None
        self._rate = 0.0

    def __call__(self, q: np.ndarray, dt: float) -> np.ndarray:
        q = quat_normalize(q)
        if self._q is None:
            self._q = q.copy()
            return q.copy()
        if dt <= 0:
            return self._q.copy()
        a_d = _smoothing_factor(dt, self.d_cutoff)
        self._rate = a_d * (quat_angle_between(self._q, q) / dt) + (1 - a_d) * self._rate
        a = _smoothing_factor(dt, self.cutoff + self.beta * self._rate)
        self._q = quat_slerp(self._q, q, a)
        return self._q.copy()


class ScalarLowPass:
    """Simple exponential moving average with a time-constant cutoff."""

    def __init__(self, cutoff: float = 5.0):
        self.cutoff = cutoff
        self._y = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, y: float, dt: float) -> float:
        if self._y is None:
            self._y = float(y)
            return self._y
        if dt <= 0:
            return self._y
        a = _smoothing_factor(dt, self.cutoff)
        self._y = a * float(y) + (1 - a) * self._y
        return self._y

import numpy as np

from handarm.filters import OneEuroFilter, QuaternionLowPass
from handarm.transforms import quat_angle_between


def test_one_euro_reduces_jitter():
    """Filtered noise around a constant should have much lower variance."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)
    dt = 1 / 30
    raw = 0.5 + rng.normal(0, 0.01, size=(300, 3))
    out = np.array([f(x, dt) for x in raw])
    tail_raw = raw[100:]
    tail_out = out[100:]
    assert tail_out.std() < 0.5 * tail_raw.std()


def test_one_euro_tracks_fast_motion():
    """With speed-adaptive cutoff, a fast ramp is followed with small lag."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    dt = 1 / 30
    xs = np.linspace(0, 1, 60)  # moves 1 unit in 2 s
    out = [f(np.array([x]), dt)[0] for x in xs]
    assert abs(out[-1] - xs[-1]) < 0.1


def test_one_euro_first_sample_passthrough():
    f = OneEuroFilter()
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(f(x, 1 / 30), x)


def test_quaternion_lowpass_converges():
    f = QuaternionLowPass(cutoff=3.0)
    a = np.array([0, 0, 0, 1.0])
    b = np.array([0, 0, np.sin(0.4), np.cos(0.4)])
    q = f(a, 1 / 30)
    for _ in range(120):
        q = f(b, 1 / 30)
    assert quat_angle_between(q, b) < 0.01


def test_quaternion_one_euro_opens_cutoff_with_speed():
    """A steady sweep: the adaptive filter trails the input by less than
    the fixed-cutoff one."""
    fixed = QuaternionLowPass(cutoff=2.0, beta=0.0)
    adaptive = QuaternionLowPass(cutoff=2.0, beta=1.0)
    dt = 1 / 30
    for i in range(90):
        t = i * dt
        q = np.array([0, 0, np.sin(t), np.cos(t)])  # 2 rad/s about z
        lag_fixed = quat_angle_between(fixed(q, dt), q)
        lag_adaptive = quat_angle_between(adaptive(q, dt), q)
    assert lag_adaptive < 0.7 * lag_fixed
    assert lag_adaptive < 0.1


def test_quaternion_lowpass_smooths():
    """One step toward a new target must move only part of the way."""
    f = QuaternionLowPass(cutoff=3.0)
    a = np.array([0, 0, 0, 1.0])
    b = np.array([0, 0, np.sin(0.5), np.cos(0.5)])  # 1 rad away
    f(a, 1 / 30)
    q = f(b, 1 / 30)
    moved = quat_angle_between(a, q)
    assert 0.05 < moved < 0.8

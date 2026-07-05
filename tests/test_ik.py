import mujoco
import numpy as np
import pytest

from handarm.config import HOME_Q, SCENE_XML, IKConfig
from handarm.ik import DLSSolver
from handarm.kinematics import ArmKinematics
from handarm.transforms import quat_angle_between


@pytest.fixture(scope="module")
def kin():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    return ArmKinematics(model)


@pytest.fixture(scope="module")
def solver(kin):
    return DLSSolver(kin, IKConfig())


def reachable_poses(kin, n, seed):
    """Sample targets via FK of random interior joint configurations."""
    rng = np.random.default_rng(seed)
    poses = []
    while len(poses) < n:
        # Stay away from the extreme ends of each joint range.
        q = kin.lower + (0.15 + 0.7 * rng.random(kin.ndof)) * (kin.upper - kin.lower)
        pos, quat = kin.fk(q)
        if pos[2] > 0.05:  # above the floor
            poses.append((pos, quat))
    return poses


def test_fk_home_pose_sane(kin):
    pos, _ = kin.fk(HOME_Q)
    # Home should be in front of the base, at a workable height.
    assert 0.2 < pos[0] < 0.7
    assert abs(pos[1]) < 0.05
    assert 0.1 < pos[2] < 0.7


def test_ik_reaches_reachable_targets(kin, solver):
    """Cold-start solves with restarts: targets sampled across the entire
    joint space, including behind and above the arm."""
    solved = 0
    for pos, quat in reachable_poses(kin, 25, seed=42):
        res = solver.solve(HOME_Q, pos, quat, restarts=12)
        if res.pos_err < 5e-3 and res.ori_err < np.radians(3):
            solved += 1
    assert solved >= 23, f"only {solved}/25 targets solved"


def test_ik_tracks_smooth_path(kin, solver):
    """Continuous tracking (the actual use case): small target steps."""
    q = HOME_Q.copy()
    pos0, quat0 = kin.fk(HOME_Q)
    waypoints = [pos0 + np.array([0.0, dy, 0.0]) for dy in np.linspace(0, 0.25, 40)]
    for target in waypoints:
        res = solver.solve(q, target, quat0)
        q = res.q
        assert res.pos_err < 3e-3, f"lost tracking at {target}, err {res.pos_err}"


def test_ik_respects_joint_limits(kin, solver):
    for pos, quat in reachable_poses(kin, 10, seed=7):
        res = solver.solve(HOME_Q, pos, quat)
        assert np.all(res.q >= kin.lower - 1e-9)
        assert np.all(res.q <= kin.upper + 1e-9)


def test_ik_stable_at_unreachable_target(kin, solver):
    """A target far outside the workspace must not blow up the solution."""
    res = solver.solve(HOME_Q, np.array([2.0, 0.0, 0.3]), np.array([1.0, 0, 0, 0]))
    assert np.all(np.isfinite(res.q))
    assert np.all(res.q >= kin.lower - 1e-9) and np.all(res.q <= kin.upper + 1e-9)
    # The arm should stretch toward the target rather than fold up.
    pos, _ = kin.fk(res.q)
    assert pos[0] > 0.4


def test_ik_stable_near_singularity(kin, solver):
    """Fully-stretched configuration is singular; steps must stay bounded."""
    q_stretched = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0, 0.0])  # arm straight out
    pos, quat = kin.fk(q_stretched)
    # Ask for a pose slightly beyond full extension.
    res = solver.solve(q_stretched, pos + np.array([0.05, 0, 0]), quat)
    assert np.all(np.isfinite(res.q))
    step = np.abs(res.q - q_stretched)
    assert np.all(step < 1.5), f"joint jump too large near singularity: {step}"


def test_ik_converges_quickly_when_warm(kin, solver):
    """Warm-started solves (the per-frame case) should converge fast."""
    pos, quat = kin.fk(HOME_Q)
    res = solver.solve(HOME_Q, pos + np.array([0.01, 0.01, 0.0]), quat)
    assert res.converged
    assert res.iters <= 5

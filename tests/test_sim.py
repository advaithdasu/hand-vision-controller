import numpy as np
import pytest

from handarm.config import HOME_Q, IKConfig
from handarm.ik import DLSSolver
from handarm.kinematics import ArmKinematics
from handarm.sim import ArmSim


@pytest.fixture(scope="module")
def sim():
    return ArmSim(render_size=(160, 160))


def advance(sim, seconds):
    """Advance physics in real-time-sized chunks (step() caps catch-up
    at 0.1 s per call by design, to avoid spiral-of-death after stalls)."""
    for _ in range(int(seconds / 0.05)):
        sim.step(0.05)


def test_sim_loads_and_renders(sim):
    px = sim.render()
    assert px.shape == (160, 160, 3)
    assert px.std() > 1.0  # not a blank frame


def test_arm_settles_at_home(sim):
    sim.reset()
    advance(sim, 1.0)  # one second of physics
    assert np.linalg.norm(sim.arm_q() - HOME_Q) < 0.05


def test_arm_tracks_joint_command(sim):
    sim.reset()
    target = HOME_Q + np.array([0.4, -0.15, 0.2, 0.3, -0.2, 0.5])
    sim.set_joint_targets(target)
    advance(sim, 1.5)
    err = np.abs(sim.arm_q() - target)
    assert np.all(err < 0.06), f"joint tracking error {err}"


def test_end_effector_follows_ik_pipeline(sim):
    """Integration: command a Cartesian shift via IK, arm should get there."""
    sim.reset()
    kin = ArmKinematics(sim.model)
    solver = DLSSolver(kin, IKConfig())
    pos0, quat0 = kin.fk(HOME_Q)
    target = pos0 + np.array([0.0, 0.15, -0.1])
    res = solver.solve(HOME_Q, target, quat0)
    assert res.converged
    sim.set_joint_targets(res.q)
    advance(sim, 1.5)
    pos_now, _ = kin.fk(sim.arm_q())
    assert np.linalg.norm(pos_now - target) < 0.01


def test_gripper_opens_and_closes(sim):
    sim.reset()
    sim.set_gripper(0.0)
    advance(sim, 1.0)
    fl = sim.data.joint("finger_left").qpos[0]
    assert fl < 0.005
    sim.set_gripper(1.0)
    advance(sim, 1.0)
    fl = sim.data.joint("finger_left").qpos[0]
    assert fl > 0.030


def test_step_locks_to_wall_time(sim):
    t0 = sim.data.time
    advance(sim, 0.5)
    assert sim.data.time - t0 == pytest.approx(0.5, abs=sim.timestep * 12)


def test_step_caps_catchup_after_stall(sim):
    """A huge wall-clock gap must not trigger an unbounded physics burst."""
    t0 = sim.data.time
    sim.step(5.0)
    assert sim.data.time - t0 <= 0.1 + sim.timestep

"""Headless end-to-end task test: grasp a cube and lift it.

Drives the same IK -> joint command -> physics path the live teleop loop
uses, with a scripted Cartesian trajectory instead of a hand.
"""

import numpy as np
import pytest

from handarm.config import HOME_Q, ControlConfig, IKConfig
from handarm.ik import DLSSolver
from handarm.kinematics import ArmKinematics
from handarm.mapping import TOOL_DOWN_QUAT
from handarm.sim import ArmSim


@pytest.fixture(scope="module")
def rig():
    sim = ArmSim(render_size=(160, 160))
    kin = ArmKinematics(sim.model)
    solver = DLSSolver(kin, IKConfig())
    return sim, kin, solver


def goto(sim, kin, solver, q_cmd, target, seconds=1.2, grip=None):
    """Track a Cartesian target the way the app does: warm-started IK each
    control tick, rate-limited joint targets, physics at 500 Hz."""
    dt = 1 / 30
    max_dq = ControlConfig().max_joint_vel * dt  # same limit as the live loop
    for _ in range(int(seconds / dt)):
        res = solver.solve(q_cmd, np.asarray(target, float), TOOL_DOWN_QUAT)
        q_cmd = q_cmd + np.clip(res.q - q_cmd, -max_dq, max_dq)
        sim.set_joint_targets(q_cmd)
        if grip is not None:
            sim.set_gripper(grip)
        sim.step(dt)
    return q_cmd


def cube_pos(sim):
    return sim.data.body("cube_orange").xpos.copy()


def test_pick_and_place(rig):
    sim, kin, solver = rig
    sim.reset()
    q = HOME_Q.copy()
    cube0 = cube_pos(sim)
    assert cube0[2] < 0.05  # starts on the floor

    hover = cube0 + [0, 0, 0.15]
    grasp = cube0 + [0, 0, 0.015]  # tcp just above cube center

    q = goto(sim, kin, solver, q, hover, seconds=1.5, grip=1.0)
    q = goto(sim, kin, solver, q, grasp, seconds=1.2, grip=1.0)

    tcp_now, _ = kin.fk(sim.arm_q())
    assert np.linalg.norm(tcp_now[:2] - cube0[:2]) < 0.02, "not above the cube"

    q = goto(sim, kin, solver, q, grasp, seconds=0.8, grip=0.0)   # close
    q = goto(sim, kin, solver, q, hover, seconds=1.5, grip=0.0)   # lift

    lifted = cube_pos(sim)
    assert lifted[2] > 0.10, f"cube not lifted, z={lifted[2]:.3f}"

    # Carry sideways and release over a drop point.
    drop = np.array([0.36, 0.20, 0.18])
    q = goto(sim, kin, solver, q, drop, seconds=1.8, grip=0.0)
    assert cube_pos(sim)[2] > 0.08, "cube dropped during transport"

    q = goto(sim, kin, solver, q, drop, seconds=1.0, grip=1.0)    # open
    settled = cube_pos(sim)
    assert settled[2] < 0.06, "cube did not fall after release"
    assert np.linalg.norm(settled[:2] - drop[:2]) < 0.12, "cube landed far from drop point"

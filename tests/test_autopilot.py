"""Headless autopilot test: the scripted pick-and-place must succeed in
MuJoCo through the live TeleopApp -> IK -> physics path, with a physical
(friction) grasp."""

import numpy as np
import pytest

from handarm.app import TeleopApp
from handarm.autopilot import (
    DROP_SLOTS,
    TRAY_CENTER,
    Autopilot,
    SimPlant,
    plan_pick_and_place,
)
from handarm.config import HOME_Q, AppConfig
from handarm.mapping import TOOL_DOWN_QUAT

TRAY_INNER = 0.078  # half extent inside the walls
CUBE_HALF = 0.02
CUBE_STARTS = [np.array([0.42, -0.12, 0.02]), np.array([0.50, 0.06, 0.02])]


@pytest.fixture(scope="module")
def app():
    return TeleopApp(AppConfig(sim_render_size=(160, 160)))


def run_until(pilot, app, done, max_seconds, dt=1 / 30):
    t = 0.0
    while not done() and t < max_seconds:
        pilot.tick(dt)
        app.sim.step(dt)
        t += dt
    return t


def test_waypoints_are_reachable(app):
    for cube, slot in zip(CUBE_STARTS, DROP_SLOTS):
        q = HOME_Q.copy()
        for seg in plan_pick_and_place(cube, slot):
            res = app.solver.solve(q, seg.pos, TOOL_DOWN_QUAT)
            for _ in range(8):
                if res.converged:
                    break
                res = app.solver.solve(res.q, seg.pos, TOOL_DOWN_QUAT)
            assert res.pos_err < 3e-3, seg.label
            q = res.q


def test_autopilot_places_both_cubes(app):
    app.sim.reset()
    app.reset_control()
    plant = SimPlant(app.sim, app.kin)
    pilot = Autopilot(app, plant)

    t = run_until(pilot, app, lambda: pilot.round_complete, max_seconds=45)
    assert pilot.round_complete, f"did not finish (status: {pilot.status})"
    assert t < 30, f"took {t:.1f} s"
    assert not plant.holding

    for i, (c, slot) in enumerate(zip(plant.cube_positions(), DROP_SLOTS)):
        assert abs(c[0] - TRAY_CENTER[0]) < TRAY_INNER - CUBE_HALF, f"cube {i} x={c[0]:.3f}"
        assert abs(c[1] - TRAY_CENTER[1]) < TRAY_INNER - CUBE_HALF, f"cube {i} y={c[1]:.3f}"
        assert c[2] < 0.06, f"cube {i} not resting, z={c[2]:.3f}"
        assert np.linalg.norm(c[:2] - slot[:2]) < 0.04, f"cube {i} far from slot"


def test_freeze_pauses_the_script(app):
    app.sim.reset()
    app.reset_control()
    pilot = Autopilot(app, SimPlant(app.sim, app.kin))
    run_until(pilot, app, lambda: pilot.status == "reaching for the cube", 5)
    pilot.tick(1 / 30)
    before = app.target_pos.copy()
    app.manual_freeze = True
    for _ in range(30):
        pilot.tick(1 / 30)
    assert np.allclose(app.target_pos, before)
    app.manual_freeze = False
    pilot.tick(1 / 30)
    assert not np.allclose(app.target_pos, before)

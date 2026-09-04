"""Autopilot: a scripted pick-and-place that drives the arm with no camera.

Mirrors web/src/autopilot.ts. The script feeds Cartesian targets into the
same path a hand does — TeleopApp.solve_and_command (warm-started DLS IK,
joint-rate limiting) and the MuJoCo physics — so what plays is the real
control stack, not a canned animation. Grasping here is physical (friction
between the fingers and the cube), which is why the verify-after-lift
retry exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mapping import TOOL_DOWN_QUAT
from .transforms import quat_slerp

TRAY_CENTER = np.array([0.36, 0.26, 0.0])
# Drop slots inside the tray, one per cube, spread along its x axis.
DROP_SLOTS = [
    TRAY_CENTER + np.array([-0.035, 0.0, 0.0]),
    TRAY_CENTER + np.array([0.035, 0.0, 0.0]),
]
CUBE_BODIES = ("cube_orange", "cube_blue")

HOVER_Z = 0.15        # above the cube center before descending
GRASP_DZ = 0.015      # tcp just above the cube center
TRAY_HOVER_Z = 0.20
TRAY_DROP_Z = 0.10    # low enough that the cube can't bounce out

PLACED_RADIUS = 0.06  # a cube closer than this to its slot counts as placed
MAX_ATTEMPTS = 3      # per cube before moving on
START_PAUSE = 1.0     # seconds to rest before the first pick


@dataclass
class Segment:
    pos: np.ndarray          # tcp target at the end of the segment
    grip: float              # gripper opening to hold through it (0..1)
    duration: float          # seconds
    label: str
    verify_grasp: bool = False  # after this, the cube must be held or retry


def plan_pick_and_place(cube: np.ndarray, slot: np.ndarray) -> list[Segment]:
    """Waypoints for picking `cube` and dropping it at `slot`."""
    cube = np.asarray(cube, float)
    hover = cube + [0, 0, HOVER_Z]
    grasp = cube + [0, 0, GRASP_DZ]
    above = np.array([slot[0], slot[1], TRAY_HOVER_Z])
    drop = np.array([slot[0], slot[1], TRAY_DROP_Z])
    return [
        Segment(hover, 1.0, 1.4, "reaching for the cube"),
        Segment(grasp, 1.0, 1.0, "descending"),
        Segment(grasp, 0.0, 0.6, "closing the gripper"),
        Segment(hover, 0.0, 1.0, "lifting", verify_grasp=True),
        Segment(above, 0.0, 1.6, "carrying to the tray"),
        Segment(drop, 0.0, 0.8, "lowering"),
        Segment(drop, 1.0, 0.5, "releasing"),
        Segment(above, 1.0, 0.8, "retreating"),
    ]


def smoothstep(t: float) -> float:
    u = min(max(t, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class SimPlant:
    """What the script needs from the MuJoCo scene: cube poses and whether
    one is in the gripper (inferred: near the tcp and off the floor)."""

    def __init__(self, sim, kin):
        self.sim = sim
        self.kin = kin

    def cube_positions(self) -> list[np.ndarray]:
        return [self.sim.data.body(n).xpos.copy() for n in CUBE_BODIES]

    @property
    def holding(self) -> bool:
        tcp, _ = self.kin.fk(self.sim.arm_q())
        return any(
            np.linalg.norm(c - tcp) < 0.05 and c[2] > 0.06
            for c in self.cube_positions()
        )


@dataclass
class _Pick:
    cube: int
    attempt: int
    segs: list[Segment]
    start: np.ndarray
    start_quat: np.ndarray  # eased to tool-down like the position
    i: int = 0
    t: float = 0.0


@dataclass
class _Pause:
    t: float


@dataclass
class Autopilot:
    """Drives `app` (a TeleopApp) through `plant` with no hand input."""

    app: object
    plant: object
    phase: object = field(default_factory=lambda: _Pause(START_PAUSE))
    skipped: set = field(default_factory=set)
    round_complete: bool = False
    status: str = "starting"

    def restart(self) -> None:
        """Start over after the scene has been reset."""
        self.phase = _Pause(0.3)
        self.skipped.clear()
        self.round_complete = False
        self.status = "starting"

    def _next_cube(self) -> int:
        cubes = self.plant.cube_positions()
        for i, (c, slot) in enumerate(zip(cubes, DROP_SLOTS)):
            if i in self.skipped:
                continue
            if np.linalg.norm(c[:2] - slot[:2]) > PLACED_RADIUS:
                return i
        return -1

    def _begin_pick(self, cube: int, attempt: int) -> None:
        p = self.plant.cube_positions()[cube]
        self.phase = _Pick(cube, attempt, plan_pick_and_place(p, DROP_SLOTS[cube]),
                           np.array(self.app.target_pos, float),
                           np.array(self.app.target_quat, float))

    def tick(self, dt: float) -> None:
        """Advance the script and run IK -> command. Physics is stepped by
        the caller, as in the live loop. A freeze pauses the script."""
        if not self.app.frozen:
            self._advance(dt)
        self.app.tick_scripted(dt)

    def _advance(self, dt: float) -> None:
        ph = self.phase
        if isinstance(ph, _Pause):
            ph.t -= dt
            self.status = "resting"
            if ph.t > 0:
                return
            nxt = self._next_cube()
            if nxt < 0:
                self.round_complete = True
                self.status = "done"
                ph.t = float("inf")  # park until restart()
                return
            self._begin_pick(nxt, 1)
            return

        seg = ph.segs[ph.i]
        ph.t += dt
        a = smoothstep(ph.t / seg.duration)
        self.app.target_pos = ph.start + (seg.pos - ph.start) * a
        # The home pose is not tool-down; snapping the wrist there at the
        # edge of reach would ask IK for an unreachable pose on tick one.
        self.app.target_quat = quat_slerp(ph.start_quat, TOOL_DOWN_QUAT, a)
        grip_from = ph.segs[ph.i - 1].grip if ph.i > 0 else 1.0
        self.app.grip_opening = grip_from + (seg.grip - grip_from) * a
        self.status = seg.label
        if ph.t < seg.duration:
            return

        ph.start = seg.pos.copy()
        ph.start_quat = TOOL_DOWN_QUAT.copy()
        ph.t = 0.0
        ph.i += 1
        if seg.verify_grasp and not self.plant.holding:
            if ph.attempt >= MAX_ATTEMPTS:
                self.skipped.add(ph.cube)
                self.phase = _Pause(0.3)
            else:
                self._begin_pick(ph.cube, ph.attempt + 1)
            return
        if ph.i >= len(ph.segs):
            self.phase = _Pause(0.4)

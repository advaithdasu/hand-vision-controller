"""MuJoCo simulation of the arm and the pick-and-place scene."""

from __future__ import annotations

import mujoco
import numpy as np

from .config import ARM_JOINTS, HOME_Q, SCENE_XML


class ArmSim:
    """Owns the live MjModel/MjData, actuates the arm, renders the scene."""

    GRIP_RANGE = (0.0, 0.035)  # prismatic finger travel, meters

    def __init__(self, render_size=(640, 640)):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.timestep = self.model.opt.timestep

        self.qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.act_ids = np.array([self.model.actuator(f"a{i+1}").id for i in range(6)])
        self.grip_act_ids = np.array(
            [self.model.actuator("grip_left").id, self.model.actuator("grip_right").id]
        )
        self.target_mocap_id = self.model.body("target").mocapid[0]
        self._cam_id = self.model.camera("demo").id

        w, h = render_size
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._sim_time_debt = 0.0

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_adr] = HOME_Q
        self.data.ctrl[self.act_ids] = HOME_Q
        self.set_gripper(1.0)
        mujoco.mj_forward(self.model, self.data)

    # ---------------- commands ----------------

    def set_joint_targets(self, q: np.ndarray) -> None:
        self.data.ctrl[self.act_ids] = q

    def set_gripper(self, opening: float) -> None:
        """opening in [0, 1]: 0 = fully closed, 1 = fully open."""
        lo, hi = self.GRIP_RANGE
        self.data.ctrl[self.grip_act_ids] = lo + float(np.clip(opening, 0, 1)) * (hi - lo)

    def set_target_marker(self, pos: np.ndarray) -> None:
        self.data.mocap_pos[self.target_mocap_id] = pos

    # ---------------- stepping / state ----------------

    def step(self, wall_dt: float) -> int:
        """Advance physics to keep sim time locked to wall time."""
        self._sim_time_debt += min(wall_dt, 0.1)  # cap catch-up after stalls
        n = int(self._sim_time_debt / self.timestep)
        self._sim_time_debt -= n * self.timestep
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return n

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    # ---------------- rendering ----------------

    def render(self) -> np.ndarray:
        """RGB image of the scene from the demo camera."""
        self.renderer.update_scene(self.data, camera=self._cam_id)
        return self.renderer.render()

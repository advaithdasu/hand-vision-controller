"""Forward kinematics and Jacobians for the arm, backed by MuJoCo.

Uses a private scratch MjData so IK iterations never touch the live
simulation state.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .config import ARM_JOINTS, TCP_SITE
from .transforms import quat_from_matrix


class ArmKinematics:
    def __init__(self, model: mujoco.MjModel, joint_names=ARM_JOINTS, site_name=TCP_SITE):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch buffer, never stepped
        self.site_id = model.site(site_name).id
        self.qpos_adr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        self.dof_adr = np.array([model.joint(n).dofadr[0] for n in joint_names])
        ranges = np.array([model.joint(n).range for n in joint_names])
        self.lower = ranges[:, 0]
        self.upper = ranges[:, 1]
        self.ndof = len(joint_names)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)

    def _sync(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _site_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, quat_from_matrix(rot)

    def _site_jacobian(self) -> np.ndarray:
        mujoco.mj_comPos(self.model, self.data)  # required before mj_jacSite
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])

    def fk(self, q: np.ndarray):
        """Joint angles -> (tcp position, tcp quaternion [x,y,z,w])."""
        self._sync(q)
        return self._site_pose()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """6xN geometric Jacobian of the tcp site (world frame, rows =
        [linear; angular]), columns restricted to the six arm joints."""
        self._sync(q)
        return self._site_jacobian()

    def fk_and_jacobian(self, q: np.ndarray):
        """Pose and Jacobian from a single kinematics pass — the per-IK-
        iteration hot path."""
        self._sync(q)
        pos, quat = self._site_pose()
        return pos, quat, self._site_jacobian()

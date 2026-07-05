"""Damped least squares (Levenberg-Marquardt) inverse kinematics.

Solves for the 6 arm joint angles that place the tool center point at a
target pose. Three practical concerns drive the design:

* Singularities. Near a singular configuration the Jacobian loses rank and
  a plain pseudo-inverse commands huge joint velocities. We monitor the
  manipulability measure w = sqrt(det(J J^T)) and ramp extra damping in as
  w collapses, trading tracking accuracy for bounded, stable steps.

* Joint limits. Every iterate is clamped to the URDF/MJCF joint ranges, so
  the solver only ever explores feasible configurations.

* Continuity. The solver warm-starts from the current configuration and
  clamps both the task-space error and the per-iteration joint step, so
  consecutive frames produce nearby solutions instead of branch flips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import IKConfig
from .transforms import orientation_error


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float   # meters
    ori_err: float   # radians
    iters: int
    converged: bool
    manipulability: float


class DLSSolver:
    def __init__(self, kin, cfg: IKConfig | None = None):
        """kin: object with fk(q), jacobian(q), clamp(q), lower/upper."""
        self.kin = kin
        self.cfg = cfg or IKConfig()

    def solve(self, q_init: np.ndarray, target_pos: np.ndarray,
              target_quat: np.ndarray, restarts: int = 0) -> IKResult:
        """Solve IK warm-started from q_init.

        restarts > 0 adds cold-start attempts (base yaw aimed at the
        target, then random seeds) with a larger iteration budget; used
        for one-shot solves, not the per-frame tracking path.
        """
        best = self._descend(q_init, target_pos, target_quat, self.cfg.max_iters)
        if best.converged or restarts <= 0:
            return best

        seeds = [self._aimed_seed(q_init, target_pos)]
        rng = np.random.default_rng(0)
        while len(seeds) < restarts:
            seeds.append(self.kin.lower + rng.random(self.kin.ndof)
                         * (self.kin.upper - self.kin.lower))
        for seed in seeds:
            res = self._descend(seed, target_pos, target_quat, self.cfg.max_iters * 4)
            if res.converged:
                return res
            if res.pos_err + res.ori_err < best.pos_err + best.ori_err:
                best = res
        return best

    def _aimed_seed(self, q_init: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        q = np.asarray(q_init, dtype=float).copy()
        q[0] = np.arctan2(target_pos[1], target_pos[0])
        return q

    def _descend(self, q_init: np.ndarray, target_pos: np.ndarray,
                 target_quat: np.ndarray, max_iters: int) -> IKResult:
        cfg = self.cfg
        q = self.kin.clamp(np.asarray(q_init, dtype=float).copy())
        pos_err = ori_err = np.inf
        manip = 0.0
        it = 0

        for it in range(1, max_iters + 1):
            pos, quat, J = self.kin.fk_and_jacobian(q)
            # Manipulability (Nakamura & Hanafusa), measured on the raw
            # Jacobian before task weighting so the damping threshold has
            # a fixed geometric meaning.
            manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

            e_pos = target_pos - pos
            e_ori = orientation_error(target_quat, quat)
            pos_err = float(np.linalg.norm(e_pos))
            ori_err = float(np.linalg.norm(e_ori))
            if pos_err < cfg.pos_tol and ori_err < cfg.ori_tol:
                return IKResult(q, pos_err, ori_err, it, True, manip)

            # Clamp the translational error so distant targets pull the
            # arm smoothly instead of demanding one giant step.
            if pos_err > cfg.max_pos_err:
                e_pos = e_pos * (cfg.max_pos_err / pos_err)
            e = np.concatenate([e_pos, cfg.ori_weight * e_ori])

            J = np.vstack([J[:3], cfg.ori_weight * J[3:]])
            JJT = J @ J.T
            lam = cfg.base_damping
            if manip < cfg.manip_threshold:
                lam += cfg.singular_damping * (1.0 - manip / cfg.manip_threshold)

            dq = J.T @ np.linalg.solve(JJT + (lam ** 2) * np.eye(6), e)
            dq = np.clip(dq, -cfg.max_step, cfg.max_step)
            q = self.kin.clamp(q + dq)

        pos, quat = self.kin.fk(q)
        pos_err = float(np.linalg.norm(target_pos - pos))
        ori_err = float(np.linalg.norm(orientation_error(target_quat, quat)))
        converged = pos_err < cfg.pos_tol and ori_err < cfg.ori_tol
        return IKResult(q, pos_err, ori_err, it, converged, manip)

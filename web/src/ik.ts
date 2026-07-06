/**
 * Damped least squares (Levenberg-Marquardt) inverse kinematics —
 * TypeScript port of handarm/ik.py.
 *
 * dq = J^T (J J^T + lambda^2 I)^-1 e, with manipulability-adaptive
 * damping near singularities, joint-limit clamping on every iterate, and
 * clamped error/step magnitudes for frame-to-frame continuity.
 */

import { IK } from "./config";
import { clampJoints, fk, jacobian } from "./kinematics";
import { orientationError, type Quat, type Vec3, vecNorm } from "./transforms";

export interface IKResult {
  q: number[];
  posErr: number; // meters
  oriErr: number; // radians
  iters: number;
  converged: boolean;
  manipulability: number;
}

/**
 * Solve A x = b for a 6x6 system via Gaussian elimination with partial
 * pivoting. Also returns det(A) from the pivot product.
 */
export function solveLinear6(
  A: number[][],
  b: number[],
): { x: number[]; det: number } {
  const n = 6;
  const M = A.map((row, i) => [...row, b[i]]);
  let det = 1;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (pivot !== col) {
      [M[col], M[pivot]] = [M[pivot], M[col]];
      det = -det;
    }
    const p = M[col][col];
    det *= p;
    if (Math.abs(p) < 1e-14) return { x: new Array(n).fill(0), det: 0 };
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / p;
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let c = r + 1; c < n; c++) s -= M[r][c] * x[c];
    x[r] = s / M[r][r];
  }
  return { x, det };
}

function matMulT(J: number[][]): number[][] {
  // J (6x6) times its transpose.
  const out: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++) {
      let s = 0;
      for (let k = 0; k < 6; k++) s += J[i][k] * J[j][k];
      out[i][j] = s;
    }
  return out;
}

function detOnly(A: number[][]): number {
  return solveLinear6(A, new Array(6).fill(0)).det;
}

function descend(
  qInit: number[],
  targetPos: Vec3,
  targetQuat: Quat,
  maxIters: number,
): IKResult {
  let q = clampJoints([...qInit]);
  let posErr = Infinity;
  let oriErr = Infinity;
  let manip = 0;
  let it = 0;

  for (it = 1; it <= maxIters; it++) {
    const f = fk(q);
    const J = jacobian(q, f);
    // Manipulability (Nakamura & Hanafusa) on the raw Jacobian so the
    // damping threshold has a fixed geometric meaning.
    manip = Math.sqrt(Math.max(detOnly(matMulT(J)), 0));

    let ePos: Vec3 = [
      targetPos[0] - f.pos[0],
      targetPos[1] - f.pos[1],
      targetPos[2] - f.pos[2],
    ];
    const eOri = orientationError(targetQuat, f.quat);
    posErr = vecNorm(ePos);
    oriErr = vecNorm(eOri);
    if (posErr < IK.posTol && oriErr < IK.oriTol) {
      return { q, posErr, oriErr, iters: it, converged: true, manipulability: manip };
    }

    // Clamp the translational error so distant targets pull smoothly.
    if (posErr > IK.maxPosErr) {
      const s = IK.maxPosErr / posErr;
      ePos = [ePos[0] * s, ePos[1] * s, ePos[2] * s];
    }
    const e = [...ePos, ...eOri.map((v) => v * IK.oriWeight)];

    // Weight the orientation rows.
    const Jw = J.map((row, r) =>
      r < 3 ? [...row] : row.map((v) => v * IK.oriWeight),
    );
    const JJT = matMulT(Jw);
    let lam = IK.baseDamping;
    if (manip < IK.manipThreshold) {
      lam += IK.singularDamping * (1 - manip / IK.manipThreshold);
    }
    for (let i = 0; i < 6; i++) JJT[i][i] += lam * lam;

    const { x: y } = solveLinear6(JJT, e);
    // dq = Jw^T y, clamped per joint.
    const dq = new Array(6).fill(0);
    for (let i = 0; i < 6; i++) {
      let s = 0;
      for (let r = 0; r < 6; r++) s += Jw[r][i] * y[r];
      dq[i] = Math.min(Math.max(s, -IK.maxStep), IK.maxStep);
    }
    q = clampJoints(q.map((v, i) => v + dq[i]));
  }

  const f = fk(q);
  posErr = vecNorm([
    targetPos[0] - f.pos[0],
    targetPos[1] - f.pos[1],
    targetPos[2] - f.pos[2],
  ]);
  oriErr = vecNorm(orientationError(targetQuat, f.quat));
  const converged = posErr < IK.posTol && oriErr < IK.oriTol;
  return { q, posErr, oriErr, iters: it - 1, converged, manipulability: manip };
}

/** Warm-started per-frame solve (the tracking path). */
export function solveIK(
  qInit: number[],
  targetPos: Vec3,
  targetQuat: Quat,
): IKResult {
  return descend(qInit, targetPos, targetQuat, IK.maxIters);
}

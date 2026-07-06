import { describe, expect, it } from "vitest";

import { HOME_Q } from "../src/config";
import { solveIK } from "../src/ik";
import { fk, JOINT_LOWER, JOINT_UPPER } from "../src/kinematics";
import { TOOL_DOWN_QUAT } from "../src/mapping";
import { type Vec3 } from "../src/transforms";

describe("DLS IK", () => {
  it("converges quickly when warm-started near the target", () => {
    const f = fk(HOME_Q);
    const target: Vec3 = [f.pos[0] + 0.01, f.pos[1] + 0.01, f.pos[2]];
    const res = solveIK(HOME_Q, target, f.quat);
    expect(res.converged).toBe(true);
    expect(res.iters).toBeLessThanOrEqual(5);
  });

  it("tracks a smooth path without losing lock", () => {
    let q = [...HOME_Q];
    const f0 = fk(HOME_Q);
    for (let i = 0; i <= 40; i++) {
      const dy = (0.25 * i) / 40;
      const target: Vec3 = [f0.pos[0], f0.pos[1] + dy, f0.pos[2]];
      const res = solveIK(q, target, f0.quat);
      q = res.q;
      expect(res.posErr).toBeLessThan(3e-3);
    }
  });

  it("reaches a tool-down pose over the workspace", () => {
    const target: Vec3 = [0.45, 0.1, 0.2];
    let q = [...HOME_Q];
    // A few warm-started rounds, as consecutive frames would do.
    for (let i = 0; i < 4; i++) q = solveIK(q, target, TOOL_DOWN_QUAT).q;
    const res = solveIK(q, target, TOOL_DOWN_QUAT);
    expect(res.posErr).toBeLessThan(2e-3);
    expect(res.oriErr).toBeLessThan(0.02);
  });

  it("respects joint limits", () => {
    const res = solveIK(HOME_Q, [0.5, 0.3, 0.1], TOOL_DOWN_QUAT);
    res.q.forEach((v, i) => {
      expect(v).toBeGreaterThanOrEqual(JOINT_LOWER[i] - 1e-9);
      expect(v).toBeLessThanOrEqual(JOINT_UPPER[i] + 1e-9);
    });
  });

  it("stays bounded at an unreachable target", () => {
    const res = solveIK(HOME_Q, [2.0, 0, 0.3], TOOL_DOWN_QUAT);
    expect(res.q.every((v) => Number.isFinite(v))).toBe(true);
    const f = fk(res.q);
    expect(f.pos[0]).toBeGreaterThan(0.4); // stretched toward the target
  });

  it("takes bounded steps near the stretched-out singularity", () => {
    const qStretched = [0, Math.PI / 2, 0, 0, 0, 0];
    const f = fk(qStretched);
    const res = solveIK(
      qStretched,
      [f.pos[0] + 0.05, f.pos[1], f.pos[2]],
      f.quat,
    );
    res.q.forEach((v, i) => {
      expect(Math.abs(v - qStretched[i])).toBeLessThan(1.5);
    });
  });
});

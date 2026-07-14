import { describe, expect, it } from "vitest";

import {
  matrixFromQuat,
  type Quat,
  quatAngleBetween,
  quatFromMatrix,
  quatNormalize,
  quatToAxisAngle,
  vecNorm,
} from "../src/transforms";

describe("quatFromMatrix", () => {
  it("round-trips with matrixFromQuat through all four Shepperd branches", () => {
    const cases: Quat[] = [
      quatNormalize([0.1, 0.2, 0.3, 0.9]), // trace > 0 branch
      [1, 0, 0, 0], // 180 deg about x: m00 dominant branch
      [0, 1, 0, 0], // 180 deg about y: m11 dominant branch
      [0, 0, 1, 0], // 180 deg about z: m22 dominant branch
      quatNormalize([0.7, 0.7, 0.1, 0.05]), // near-180, mixed axis
    ];
    for (const q of cases) {
      const back = quatFromMatrix(matrixFromQuat(q));
      expect(quatAngleBetween(back, q)).toBeLessThan(1e-6);
    }
  });
});

describe("quatToAxisAngle", () => {
  it("returns zero at identity and the shortest arc for double-cover inputs", () => {
    expect(quatToAxisAngle([0, 0, 0, 1])).toEqual([0, 0, 0]);

    const q: Quat = quatNormalize([0, 0, Math.sin(0.4), Math.cos(0.4)]);
    const neg: Quat = [-q[0], -q[1], -q[2], -q[3]];
    const rv = quatToAxisAngle(q);
    const rvNeg = quatToAxisAngle(neg);
    // -q is the same rotation; both must give the same (shortest) vector.
    for (let i = 0; i < 3; i++) expect(rvNeg[i]).toBeCloseTo(rv[i], 9);
    expect(vecNorm(rv)).toBeCloseTo(0.8, 9); // angle = 2 * 0.4
  });
});

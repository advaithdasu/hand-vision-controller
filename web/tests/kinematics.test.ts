import { describe, expect, it } from "vitest";

import { fk, jacobian } from "../src/kinematics";
import { solveLinear6 } from "../src/ik";
import { quatAngleBetween, type Quat } from "../src/transforms";

/**
 * Reference poses generated from the Python + MuJoCo implementation
 * (ArmKinematics.fk / .jacobian on assets/scene.xml). The TS chain must
 * reproduce MuJoCo's site FK exactly for solutions to be interchangeable.
 */
const MUJOCO_REFERENCE: {
  q: number[];
  pos: number[];
  quat: number[];
  manip: number;
}[] = [
  {
    q: [0, 0.55, 0.85, 0, 0.75, 0],
    pos: [0.59619939, 0.0, 0.40086994],
    quat: [0.0, 0.87959015, 0.0, 0.47573224],
    manip: 0.02201705,
  },
  { q: [0, 0, 0, 0, 0, 0], pos: [0, 0, 0.95], quat: [0, 0, 0, 1], manip: 0 },
  {
    q: [0.740565, 1.525301, 1.345346, -3.453156, -0.835305, 4.694222],
    pos: [0.24787044, 0.17731047, -0.23870705],
    quat: [0.2534692, 0.9238191, -0.28424588, -0.03893464],
    manip: 0.02572021,
  },
  {
    q: [-2.928829, 1.233517, 1.449699, -0.402942, -0.823324, -2.784392],
    pos: [-0.54274169, -0.16432591, -0.04849525],
    quat: [0.13960641, -0.79752625, 0.19953548, 0.55194884],
    manip: 0.0284555,
  },
  {
    q: [-1.451172, -0.210907, 0.022196, 0.672269, 2.071191, 3.677707],
    pos: [0.08695915, 0.00911334, 0.72305292],
    quat: [0.66633856, -0.47668696, 0.56083766, 0.11926264],
    manip: 0.00021905,
  },
  {
    q: [0.723301, 1.877607, -1.389294, -4.269911, 0.470416, -5.731007],
    pos: [0.30962865, 0.36080511, 0.50368265],
    quat: [-0.01354053, 0.24635857, 0.96138482, -0.1219153],
    manip: 0.01788699,
  },
];

function manipulability(q: number[]): number {
  const J = jacobian(q);
  const JJT = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: 6 }, (_, j) => {
      let s = 0;
      for (let k = 0; k < 6; k++) s += J[i][k] * J[j][k];
      return s;
    }),
  );
  return Math.sqrt(Math.max(solveLinear6(JJT, new Array(6).fill(0)).det, 0));
}

describe("FK parity with MuJoCo", () => {
  it.each(MUJOCO_REFERENCE)("matches reference pose for q=$q", (ref) => {
    const f = fk(ref.q);
    for (let i = 0; i < 3; i++) {
      expect(f.pos[i]).toBeCloseTo(ref.pos[i], 6);
    }
    expect(quatAngleBetween(f.quat, ref.quat as Quat)).toBeLessThan(1e-6);
    expect(manipulability(ref.q)).toBeCloseTo(ref.manip, 6);
  });
});

describe("jacobian", () => {
  it("matches finite differences of FK", () => {
    const q = [0.3, 0.7, -0.5, 0.4, 0.6, -0.2];
    const J = jacobian(q);
    const h = 1e-6;
    for (let i = 0; i < 6; i++) {
      const qp = [...q];
      qp[i] += h;
      const qm = [...q];
      qm[i] -= h;
      const fp = fk(qp);
      const fm = fk(qm);
      for (let r = 0; r < 3; r++) {
        const num = (fp.pos[r] - fm.pos[r]) / (2 * h);
        expect(J[r][i]).toBeCloseTo(num, 5);
      }
    }
  });
});

describe("solveLinear6", () => {
  it("solves a known system and returns its determinant", () => {
    const A: number[][] = Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 6 }, (_, j): number =>
        i === j ? 2 : i + j === 5 ? 0.5 : 0,
      ),
    );
    const xTrue = [1, -2, 3, 0.5, -0.25, 4];
    const b = A.map((row) => row.reduce((s, v, j) => s + v * xTrue[j], 0));
    const { x, det } = solveLinear6(A, b);
    x.forEach((v, i) => expect(v).toBeCloseTo(xTrue[i], 9));
    expect(det).not.toBe(0);
  });
});

import { describe, expect, it } from "vitest";

import { OneEuroFilter, QuaternionLowPass, ScalarLowPass, SlopFilter } from "../src/filters";
import { type Quat, quatAngleBetween } from "../src/transforms";

const DT = 1 / 30;

describe("OneEuroFilter", () => {
  it("passes the first sample through", () => {
    const f = new OneEuroFilter();
    expect(f.apply([1, 2, 3], DT)).toEqual([1, 2, 3]);
  });

  it("reduces jitter around a constant", () => {
    const f = new OneEuroFilter(1.0, 0.01);
    // Deterministic pseudo-noise.
    const noise = (i: number) => 0.01 * Math.sin(i * 12.9898) * Math.cos(i * 78.233);
    const raw: number[] = [];
    const out: number[] = [];
    for (let i = 0; i < 300; i++) {
      const x = 0.5 + noise(i);
      raw.push(x);
      out.push(f.apply([x], DT)[0]);
    }
    const std = (a: number[]) => {
      const t = a.slice(100);
      const mean = t.reduce((s, v) => s + v, 0) / t.length;
      return Math.sqrt(t.reduce((s, v) => s + (v - mean) ** 2, 0) / t.length);
    };
    expect(std(out)).toBeLessThan(0.5 * std(raw));
  });

  it("tracks fast motion with small lag", () => {
    const f = new OneEuroFilter(1.0, 0.5);
    let last = 0;
    for (let i = 0; i < 60; i++) {
      last = f.apply([i / 59], DT)[0];
    }
    expect(Math.abs(last - 1)).toBeLessThan(0.1);
  });
});

describe("QuaternionLowPass", () => {
  it("opens its cutoff with angular speed when beta > 0", () => {
    // A steady 2 rad/s sweep: the adaptive filter must trail the input by
    // less than the fixed-cutoff one.
    const fixed = new QuaternionLowPass(2.0, 0);
    const adaptive = new QuaternionLowPass(2.0, 1.0);
    const sweep = (t: number): Quat => [0, 0, Math.sin(t), Math.cos(t)]; // 2t rad about z
    let lagFixed = 0;
    let lagAdaptive = 0;
    for (let i = 0; i < 90; i++) {
      const q = sweep(i * DT);
      lagFixed = quatAngleBetween(fixed.apply(q, DT), q);
      lagAdaptive = quatAngleBetween(adaptive.apply(q, DT), q);
    }
    expect(lagAdaptive).toBeLessThan(0.7 * lagFixed);
    expect(lagAdaptive).toBeLessThan(0.1);
  });

  it("converges to a held target and smooths steps", () => {
    const f = new QuaternionLowPass(3.0);
    const a: Quat = [0, 0, 0, 1];
    const b: Quat = [0, 0, Math.sin(0.25), Math.cos(0.25)];
    f.apply(a, DT);
    let q = f.apply(b, DT);
    const firstStep = quatAngleBetween(a, q);
    expect(firstStep).toBeGreaterThan(0.02);
    expect(firstStep).toBeLessThan(0.4); // partial move only
    for (let i = 0; i < 120; i++) q = f.apply(b, DT);
    expect(quatAngleBetween(q, b)).toBeLessThan(0.01);
  });
});

describe("ScalarLowPass", () => {
  it("passes the first sample through and converges to a held value", () => {
    const f = new ScalarLowPass(8.0);
    expect(f.apply(0.5, DT)).toBe(0.5);
    let y = 0.5;
    for (let i = 0; i < 120; i++) y = f.apply(1.0, DT);
    expect(y).toBeCloseTo(1.0, 3);
  });
});

describe("dt <= 0 handling", () => {
  it("holds the previous estimate instead of resetting to the raw sample", () => {
    const oneEuro = new OneEuroFilter(1.0, 0.01);
    oneEuro.apply([0, 0, 0], DT);
    const smoothed = oneEuro.apply([0.1, 0, 0], DT);
    expect(oneEuro.apply([100, 100, 100], 0)).toEqual(smoothed);

    const scalar = new ScalarLowPass(8.0);
    scalar.apply(0, DT);
    const s = scalar.apply(0.2, DT);
    expect(scalar.apply(50, 0)).toBe(s);

    const quat = new QuaternionLowPass(3.0);
    const q0: Quat = [0, 0, 0, 1];
    const q90: Quat = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
    quat.apply(q0, DT);
    const qs = quat.apply(q90, DT);
    expect(quatAngleBetween(quat.apply(q90, 0), qs)).toBeLessThan(1e-12);
  });
});

describe("SlopFilter", () => {
  it("absorbs wobble smaller than the slop, anywhere in the range", () => {
    const f = new SlopFilter(0.02);
    expect(f.apply(0.4)).toBe(0.4); // first sample passes through
    for (const v of [0.41, 0.39, 0.415, 0.385]) expect(f.apply(v)).toBe(0.4);
    // And again after the signal has moved somewhere else entirely.
    const held = f.apply(1.0); // 0.98: a big move leaves one slop of lag
    for (const v of [0.99, 0.97, 0.985]) expect(f.apply(v)).toBe(held);
  });

  it("follows real motion, one slop behind", () => {
    const f = new SlopFilter(0.02);
    f.apply(0);
    expect(f.apply(0.5)).toBeCloseTo(0.48, 9);
    expect(f.apply(1.0)).toBeCloseTo(0.98, 9);
    // Reversing costs twice the slop before the output moves again.
    expect(f.apply(0.97)).toBeCloseTo(0.98, 9);
    expect(f.apply(0.9)).toBeCloseTo(0.92, 9);
  });

  it("is a pass-through with zero slop", () => {
    const f = new SlopFilter();
    for (const v of [0.3, -1, 7]) expect(f.apply(v)).toBe(v);
  });
});

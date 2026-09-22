import { describe, expect, it } from "vitest";

import {
  apparentScale,
  FOCAL_HEIGHTS,
  countExtendedFrom,
  fingerExtensions,
  fullyInFrame,
  handScale,
  isFistFrom,
  palmFrame,
  pinchRatio,
} from "../src/poseFeatures";
import { matMul3, matTranspose3, identity3, matVec3, rotX, rotY, rotZ } from "../src/transforms";
import {
  projectPinhole,
  syntheticFist,
  syntheticOpenHand,
  syntheticPinch,
  syntheticTuckedFist,
} from "./synthetic";

describe("pose features", () => {
  it("palm frame is orthonormal", () => {
    const R = palmFrame(syntheticOpenHand());
    const RRt = matMul3(R, matTranspose3(R));
    const I = identity3();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(RRt[i][j]).toBeCloseTo(I[i][j], 6);
  });

  it("detects open hand, fist and pinch", () => {
    const open = syntheticOpenHand();
    expect(countExtendedFrom(fingerExtensions(open))).toBe(4);
    expect(isFistFrom(fingerExtensions(open))).toBe(false);

    const fist = syntheticFist();
    expect(countExtendedFrom(fingerExtensions(fist))).toBe(0);
    expect(isFistFrom(fingerExtensions(fist))).toBe(true);

    const pinch = syntheticPinch();
    expect(pinchRatio(pinch)).toBeLessThan(0.15);
    expect(isFistFrom(fingerExtensions(pinch))).toBe(false);
  });

  it("recognizes a fist with the thumb tucked over the fingers", () => {
    const fist = syntheticTuckedFist();
    // The thumb-index distance reads as a pinch...
    expect(pinchRatio(fist)).toBeLessThan(0.32);
    // ...but the four curled fingers still make it a fist.
    expect(isFistFrom(fingerExtensions(fist))).toBe(true);
  });

  it("hand scale is positive and sane", () => {
    expect(handScale(syntheticOpenHand())).toBeGreaterThan(0.05);
  });

  it("checks that every landmark is inside the frame margin", () => {
    const lm = syntheticOpenHand().map(
      (p): [number, number, number] => [0.5 + p[0], 0.4 + p[1], p[2]],
    );
    expect(fullyInFrame(lm, 0.03)).toBe(true);
    lm[8][1] = 0.99;
    expect(fullyInFrame(lm, 0.03)).toBe(false);
  });
});

describe("apparentScale", () => {
  const project = projectPinhole;
  /** Apparent size the projection is built to produce. */
  const expected = (depth: number): number => FOCAL_HEIGHTS / depth;

  it("reads the same wherever in the frame the hand sits", () => {
    // The coupling case: under perspective a *tilted* palm's segments
    // project longer or shorter depending on where in the frame they
    // are. Uncorrected, the same hand read 1.09 at the top of the frame
    // and 0.76 at the bottom — a 42% swing in apparent depth that the
    // arm followed, so raising a hand also pushed it away.
    const aspect = 16 / 9;
    const depth = 0.55;
    const half = (0.5 * depth) / FOCAL_HEIGHTS; // frame edge, in meters
    for (const tilt of [0, 0.3, 0.6]) {
      const hand = syntheticOpenHand().map((p) => matVec3(matMul3(rotX(tilt), rotY(tilt)), p));
      const seen = ([
        [0, 0], [0, -half], [0, half], [half * aspect, 0], [-half * aspect, half],
      ] as [number, number][]).map((at) =>
        apparentScale(project(hand, depth, aspect, at), hand, aspect),
      );
      const spread = (Math.max(...seen) - Math.min(...seen)) / expected(depth);
      expect(spread).toBeLessThan(0.02);
      // Finite palm thickness leaves a small tilt-dependent bias, which
      // calibration absorbs as long as it doesn't track hand position.
      for (const s of seen) expect(Math.abs(s / expected(depth) - 1)).toBeLessThan(0.08);
    }
  });

  it("is invariant to palm pitch and yaw, unlike the raw knuckle length", () => {
    const aspect = 16 / 9;
    const base = syntheticOpenHand();
    const flat = project(base, 0.7, aspect);
    const ref = apparentScale(flat, base, aspect);
    const rawRef = handScale(flat.map((p): [number, number, number] => [p[0] * aspect, p[1], 0]));
    for (const R of [rotX(0.6), rotX(-0.8), rotY(0.7), rotZ(1.2), matMul3(rotX(0.5), rotY(0.4))]) {
      const tilted = base.map((p) => matVec3(R, p));
      const img = project(tilted, 0.7, aspect);
      expect(Math.abs(apparentScale(img, tilted, aspect) / ref - 1)).toBeLessThan(0.08);
    }
    // The naive proxy shrinks with cos(pitch): tilting reads as depth.
    const pitched = base.map((p) => matVec3(rotX(0.8), p));
    const rawPitched = handScale(
      project(pitched, 0.7, aspect).map((p): [number, number, number] => [p[0] * aspect, p[1], 0]),
    );
    expect(rawPitched).toBeLessThan(0.85 * rawRef);
  });

  it("scales with 1 / depth", () => {
    const base = syntheticOpenHand();
    const near = apparentScale(project(base, 0.5, 1), base, 1);
    const far = apparentScale(project(base, 1.0, 1), base, 1);
    expect(near / far).toBeCloseTo(2.0, 6);
  });

  it("stays finite when the palm has no image-plane extent", () => {
    // Every palm segment along the optical axis: the projected extent is
    // zero, and without the foreshortening floor this would be 0/0.
    const base = syntheticOpenHand();
    const alongZ = base.map((p): [number, number, number] => [0, 0, p[0] + p[1]]);
    const s = apparentScale(project(alongZ, 1.0, 1), alongZ, 1);
    expect(Number.isFinite(s)).toBe(true);
    expect(s).toBe(0);
  });
});

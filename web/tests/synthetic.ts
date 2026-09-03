/**
 * Synthetic hands shared by the test suites (mirrors the Python tests).
 * Kept out of the .test files: importing one of those would re-register
 * its tests inside the importer.
 */

import { type Landmarks } from "../src/poseFeatures";

/** Same synthetic hands as the Python tests. */
export function syntheticOpenHand(): Landmarks {
  const lm: Landmarks = Array.from({ length: 21 }, () => [0, 0, 0]);
  lm[1] = [0.02, 0.01, 0];
  lm[2] = [0.045, 0.03, 0];
  lm[3] = [0.06, 0.05, 0];
  lm[4] = [0.07, 0.065, 0];
  const cols: [number, [number, number, number]][] = [
    [0.03, [5, 6, 8]],
    [0.01, [9, 10, 12]],
    [-0.01, [13, 14, 16]],
    [-0.03, [17, 18, 20]],
  ];
  for (const [x, [mcp, pip, tip]] of cols) {
    lm[mcp] = [x, 0.09, 0];
    lm[pip] = [x, 0.13, 0];
    lm[mcp + 2] = [x, 0.155, 0];
    lm[tip] = [x, 0.175, 0];
  }
  return lm;
}

export function syntheticFist(): Landmarks {
  const lm = syntheticOpenHand();
  for (const [mcp, pip, tip] of [[5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]) {
    const base = lm[mcp];
    lm[pip] = [base[0], base[1] + 0.02, base[2] - 0.02];
    lm[pip + 1] = [base[0], base[1] + 0.01, base[2] - 0.035];
    lm[tip] = [base[0], base[1] - 0.005, base[2] - 0.03];
  }
  lm[4] = [0, 0.05, -0.03];
  return lm;
}

/** Fist with the thumb wrapped over the curled index: thumb tip lands
 * right next to the index tip, so thumb-index distance looks like a pinch. */
export function syntheticTuckedFist(): Landmarks {
  const lm = syntheticFist();
  const idx = lm[8];
  lm[4] = [idx[0] + 0.012, idx[1] + 0.005, idx[2] - 0.008];
  return lm;
}

export function syntheticPinch(): Landmarks {
  const lm = syntheticOpenHand();
  lm[4] = [0.031, 0.12, -0.01];
  lm[8] = [0.03, 0.12, -0.01];
  lm[6] = [0.03, 0.115, 0];
  return lm;
}

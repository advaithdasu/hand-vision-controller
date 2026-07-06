/**
 * Hand-landmark geometry — TypeScript port of handarm/pose_features.py.
 * All functions take a 21-element array of [x, y, z] landmark positions.
 */

import {
  type Mat3,
  type Vec3,
  vecCross,
  vecNorm,
  vecNormalize,
  vecSub,
} from "./transforms";

export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_TIP = 8;
export const MIDDLE_MCP = 9, MIDDLE_TIP = 12;
export const RING_MCP = 13, RING_TIP = 16;
export const PINKY_MCP = 17, PINKY_TIP = 20;

export type Landmarks = Vec3[]; // length 21

export const FINGERS: Record<string, [number, number, number]> = {
  index: [INDEX_MCP, INDEX_PIP, INDEX_TIP],
  middle: [MIDDLE_MCP, 10, MIDDLE_TIP],
  ring: [RING_MCP, 14, RING_TIP],
  pinky: [PINKY_MCP, 18, PINKY_TIP],
};

/** Skeleton edges for drawing the overlay. */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Characteristic hand size: wrist to middle-finger MCP distance. */
export function handScale(lm: Landmarks): number {
  return vecNorm(vecSub(lm[MIDDLE_MCP], lm[WRIST]));
}

/** Centroid of the wrist and the four finger MCP knuckles. */
export function palmCenter(lm: Landmarks): Vec3 {
  const idx = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];
  const c: Vec3 = [0, 0, 0];
  for (const i of idx) {
    c[0] += lm[i][0] / idx.length;
    c[1] += lm[i][1] / idx.length;
    c[2] += lm[i][2] / idx.length;
  }
  return c;
}

/**
 * Orthonormal palm frame as a 3x3 rotation matrix (columns = axes):
 * forward (wrist -> middle MCP), lateral (pinky -> index), palm normal.
 */
export function palmFrame(lm: Landmarks): Mat3 {
  const forward = vecNormalize(vecSub(lm[MIDDLE_MCP], lm[WRIST]));
  const lateralRaw = vecSub(lm[INDEX_MCP], lm[PINKY_MCP]);
  const normal = vecNormalize(vecCross(forward, lateralRaw));
  const lateral = vecCross(normal, forward);
  return [
    [forward[0], lateral[0], normal[0]],
    [forward[1], lateral[1], normal[1]],
    [forward[2], lateral[2], normal[2]],
  ];
}

/** Thumb-tip to index-tip distance, normalized by hand scale. */
export function pinchRatio(lm: Landmarks): number {
  return vecNorm(vecSub(lm[THUMB_TIP], lm[INDEX_TIP])) / (handScale(lm) + 1e-9);
}

/** Per-finger extension ratio: tip-to-wrist over MCP-to-wrist distance. */
export function fingerExtensions(lm: Landmarks): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [name, [mcp, , tip]] of Object.entries(FINGERS)) {
    const base = vecNorm(vecSub(lm[mcp], lm[WRIST])) + 1e-9;
    out[name] = vecNorm(vecSub(lm[tip], lm[WRIST])) / base;
  }
  return out;
}

export function countExtendedFrom(
  ext: Record<string, number>,
  threshold = 1.4,
): number {
  return Object.values(ext).filter((r) => r > threshold).length;
}

/**
 * Fist from precomputed extension ratios and pinch ratio. The pinch check
 * keeps a pinch from being misread as a fist when the other fingers relax.
 */
export function isFistFrom(ext: Record<string, number>, pinch: number): boolean {
  const curled = ["middle", "ring", "pinky"].filter((n) => ext[n] < 1.25).length;
  return curled === 3 && ext["index"] < 1.25 && pinch > 0.5;
}

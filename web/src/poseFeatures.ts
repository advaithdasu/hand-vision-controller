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

/**
 * Rigid palm segments used for the depth estimate. Four segments spanning
 * both palm axes so no single tilt direction can collapse the estimate.
 */
const PALM_SEGMENTS: [number, number][] = [
  [WRIST, MIDDLE_MCP],
  [INDEX_MCP, PINKY_MCP],
  [WRIST, INDEX_MCP],
  [WRIST, PINKY_MCP],
];

/**
 * Below this fraction of the palm's true extent, the image-plane
 * projection is too foreshortened to trust; the estimate saturates
 * instead of blowing up as the palm turns edge-on.
 */
const FORESHORTEN_FLOOR = 0.3;

/**
 * Apparent hand size that is invariant to palm orientation.
 *
 * The naive proxy (wrist-to-knuckle length in the image) shrinks when the
 * palm pitches toward the camera, so tilting the wrist masquerades as
 * reaching. MediaPipe also returns metric, camera-aligned world landmarks
 * for the same hand; the ratio of the image-plane palm extent to the
 * world-space palm extent *projected onto the same plane* cancels the
 * foreshortening and is proportional to 1 / depth.
 *
 * Image x is scaled by the frame aspect so the measure is isotropic (in
 * units of image height per meter).
 */
export function apparentScale(
  imageLm: Landmarks,
  worldLm: Landmarks,
  frameAspect: number,
): number {
  let imgSq = 0;
  let projSq = 0;
  let fullSq = 0;
  for (const [a, b] of PALM_SEGMENTS) {
    const dx = (imageLm[b][0] - imageLm[a][0]) * frameAspect;
    const dy = imageLm[b][1] - imageLm[a][1];
    imgSq += dx * dx + dy * dy;
    const w = vecSub(worldLm[b], worldLm[a]);
    projSq += w[0] * w[0] + w[1] * w[1];
    fullSq += w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  }
  const floorSq = FORESHORTEN_FLOOR * FORESHORTEN_FLOOR * fullSq;
  return Math.sqrt(imgSq / (Math.max(projSq, floorSq) + 1e-12));
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

/** True if every landmark lies inside the image with the given margin. */
export function fullyInFrame(lm: Landmarks, margin: number): boolean {
  for (const p of lm) {
    if (p[0] < margin || p[0] > 1 - margin) return false;
    if (p[1] < margin || p[1] > 1 - margin) return false;
  }
  return true;
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

/** Extension ratio above which a finger counts as extended. */
export const EXTENDED_MIN = 1.4;
/**
 * Extension ratio below which a finger counts as curled into the palm.
 * A pinching index finger reads ~1.3 (its tip meets the thumb out in
 * front of the palm), a fisted one ~0.9-1.1.
 */
export const CURLED_MAX = 1.2;

export function countExtendedFrom(
  ext: Record<string, number>,
  threshold = EXTENDED_MIN,
): number {
  return Object.values(ext).filter((r) => r > threshold).length;
}

/**
 * Fist from precomputed extension ratios: all four fingers curled into the
 * palm. The thumb is deliberately ignored — a thumb tucked over the
 * fingers sits close to the index tip and would otherwise read as a
 * pinch, closing the gripper when the operator meant to clutch.
 */
export function isFistFrom(ext: Record<string, number>): boolean {
  return Object.keys(FINGERS).every((n) => ext[n] < CURLED_MAX);
}

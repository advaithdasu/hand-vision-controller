/** Central configuration — TypeScript port of handarm/config.py. */

export const HOME_Q = [0, 0.55, 0.85, 0, 0.75, 0];

export interface WorkspaceBox {
  x: [number, number];
  y: [number, number];
  z: [number, number];
}

export const WORKSPACE: WorkspaceBox = {
  x: [0.26, 0.6],
  y: [-0.34, 0.34],
  z: [0.05, 0.55],
};

export const MAPPING = {
  imgXRange: [0.15, 0.85] as [number, number],
  imgYRange: [0.15, 0.85] as [number, number],
  // Depth from apparent hand size, as ratios of the calibrated neutral size.
  defaultHandScale: 0.17,
  depthNearRatio: 1.6,
  depthFarRatio: 0.6,
  maxTiltRad: 1.2,
};

export const IK = {
  maxIters: 12,
  posTol: 1.5e-3,
  oriTol: 1.5e-2,
  oriWeight: 0.35,
  baseDamping: 0.02,
  singularDamping: 0.2,
  manipThreshold: 0.006,
  maxStep: 0.35,
  maxPosErr: 0.12,
};

export const CONTROL = {
  posMinCutoff: 1.2,
  posBeta: 0.015,
  posDCutoff: 1.0,
  oriCutoff: 3.0,
  pinchCutoff: 8.0,
  maxJointVel: 2.5,
  pinchClose: 0.32,
  pinchOpen: 0.45,
  apertureMargin: 0.25,
  holdAfterLostSec: 0.15,
};

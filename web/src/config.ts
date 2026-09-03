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
  // Depth from apparent hand size (poseFeatures.apparentScale, image
  // height per meter), as ratios of the calibrated neutral size.
  // The default only matters before the first calibration.
  defaultHandScale: 1.9,
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
  // One Euro filter on the position target. beta is in Hz per (m/s):
  // hovering gets the 1 Hz floor (no jitter), a 1 m/s sweep opens the
  // cutoff to 3.5 Hz (~45 ms lag instead of ~160 ms).
  posMinCutoff: 1.0,
  posBeta: 2.5,
  posDCutoff: 1.0,
  // Same for orientation; beta in Hz per (rad/s).
  oriMinCutoff: 2.0,
  oriBeta: 1.0,
  oriDCutoff: 1.0,
  pinchCutoff: 8.0,
  maxJointVel: 3.5,
  pinchClose: 0.32,
  pinchOpen: 0.45,
  apertureMargin: 0.25,
  holdAfterLostSec: 0.15,
  // Gesture debouncing: a fist must persist this long before it clutches,
  // an open hand this long before it releases, so a single misdetected
  // frame can neither freeze the arm nor drop it back into tracking.
  clutchEngageSec: 0.12,
  clutchReleaseSec: 0.08,
  // Auto-calibration gate: the hand must be fully inside the frame and
  // nearly still for this long before its pose is taken as neutral.
  calibSettleSec: 0.5,
  calibMaxSpeed: 0.25, // image heights per second
  calibEdgeMargin: 0.03, // normalized image units
};

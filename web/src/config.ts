/**
 * Central configuration — TypeScript port of handarm/config.py, plus the
 * scene layout that mirrors assets/scene.xml.
 */

import { type Vec3 } from "./transforms";

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

const CUBE_HALF = 0.02;

/** Shared so the renderer, the scripted demo, and the tests can't drift. */
export const SCENE = {
  cubeHalf: CUBE_HALF,
  cubeStarts: [
    [0.42, -0.12, CUBE_HALF],
    [0.5, 0.06, CUBE_HALF],
  ] as Vec3[],
  trayCenter: [0.36, 0.26, 0] as Vec3,
  /** Half extent of the tray floor inside its walls. */
  trayInner: 0.078,
  /** tcp-to-cube-center distance at which a closing gripper grasps. */
  graspRadius: 0.055,
  /** Gripper opening below which a grasp engages. */
  graspClosedAperture: 0.25,
};

export const MAPPING = {
  /**
   * Lateral gain: metres of tool travel per *image height* of hand
   * travel. Image x is divided by the frame aspect first, so the gain is
   * isotropic — 10 cm of hand motion moves the tool the same distance
   * sideways as upward, instead of sideways feeling ~2x sluggish on a
   * 16:9 frame. At 1.2, sweeping the full workspace (0.68 m wide, 0.50 m
   * tall) costs 0.57 / 0.42 frame heights, so the hand stays well inside
   * the camera's view. Raise it for a twitchier arm, lower it for finer
   * placement.
   */
  posGain: 1.2,
  // Depth from apparent hand size (poseFeatures.apparentScale, image
  // height per meter). Apparent size is proportional to 1 / distance, so
  // the map inverts it first and works in *distance* — otherwise the
  // near half of the reach eats most of the hand travel and the far half
  // feels dead. The window is a +-20% change in hand distance: about
  // +-11 cm at a 55 cm neutral, covering the arm's 34 cm of reach.
  // Near maps to full reach and far to the retracted end: the hand
  // coming toward the camera pushes the arm out (see mapping.ts).
  defaultHandScale: 1.9,
  depthNearDist: 0.8,
  depthFarDist: 1.2,
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
  // Depth (robot x) is filtered on its own, not as part of the position
  // vector: the shared One Euro speed term let a fast *lateral* sweep
  // open the cutoff for depth too, passing the noisiest axis straight
  // through at the worst moment. Its own filter is slower than the
  // lateral one, which suits a cue recovered from apparent hand size.
  depthMinCutoff: 0.7,
  depthBeta: 3.0,
  depthDCutoff: 1.0,
  // Dead zone the depth target must escape before the arm follows, in
  // metres of reach. Raising a hand also swings it slightly toward the
  // camera (the forearm pivots on an arc), so pure vertical intent
  // leaks a little depth; this absorbs that without blunting a
  // deliberate push, which clears it in the first centimetre or two of
  // hand travel.
  depthSlop: 0.02,
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

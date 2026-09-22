import { describe, expect, it } from "vitest";

import { type ArmPlant, TeleopController } from "../src/app";
import { CONTROL, MAPPING } from "../src/config";
import { type HandObservation } from "../src/handTracker";
import { fk } from "../src/kinematics";
import { type Landmarks } from "../src/poseFeatures";
import { TOOL_DOWN_QUAT } from "../src/mapping";
import { matVec3, quatAngleBetween, rotX, type Vec3, vecNorm, vecSub } from "../src/transforms";
import {
  projectPinhole,
  syntheticFist,
  syntheticOpenHand,
  syntheticPinch,
  syntheticTuckedFist,
} from "./synthetic";

const DT = 1 / 30;

function stubScene(): ArmPlant {
  return {
    setJointTargets: () => {},
    setGripper: () => {},
    setTargetMarker: () => {},
    step: () => {},
  };
}

/**
 * A hand of the given shape whose palm center sits at `xy` in the image,
 * projected orthographically with scale k (∝ 1/depth); world landmarks
 * are the metric shape itself, as MediaPipe would report them.
 */
function obsAt(
  xy: [number, number],
  k = 2.0,
  shape: Landmarks = syntheticOpenHand(),
): HandObservation {
  const palmIdx = [0, 5, 9, 13, 17];
  const cx = palmIdx.reduce((s, i) => s + shape[i][0], 0) / palmIdx.length;
  const cy = palmIdx.reduce((s, i) => s + shape[i][1], 0) / palmIdx.length;
  const imageLandmarks = shape.map((p): [number, number, number] => [
    xy[0] + (p[0] - cx) * k,
    xy[1] + (p[1] - cy) * k,
    p[2],
  ]);
  return { imageLandmarks, worldLandmarks: shape };
}

function make(): TeleopController {
  const ctl = new TeleopController(stubScene());
  ctl.frameAspect = 1; // square pixels keep the synthetic projection exact
  return ctl;
}

function run(ctl: TeleopController, obs: HandObservation | null, seconds: number): void {
  for (let i = 0; i < Math.round(seconds / DT); i++) ctl.tick(obs, DT);
}

/** Hold an open hand still at the image center until calibration locks on. */
function settle(ctl: TeleopController): void {
  run(ctl, obsAt([0.5, 0.5]), 0.8);
  expect(ctl.calibrated).toBe(true);
}

describe("axis decoupling", () => {
  /**
   * The operator's complaint: a hand raised straight up also pushed the
   * arm in or out. A tilted palm projects differently depending on
   * where in the frame it is, so the apparent-size depth cue moved when
   * only the height had changed.
   */
  it("holds depth while the hand moves straight up", () => {
    const aspect = 16 / 9;
    const depth = 0.55;
    const ctl = new TeleopController(stubScene());
    ctl.frameAspect = aspect;
    // A palm tilted toward the camera: the worst case for the cue.
    const hand = syntheticOpenHand().map((p) => matVec3(rotX(0.5), p));
    const at = (metresUp: number): HandObservation => ({
      imageLandmarks: projectPinhole(hand, depth, aspect, [0, -metresUp]),
      worldLandmarks: hand,
    });

    run(ctl, at(0), 1.0);
    expect(ctl.calibrated).toBe(true);
    const start = ctl.targetPos;

    run(ctl, at(0.06), 2.0); // 6 cm straight up, nothing else
    expect(ctl.targetPos[2] - start[2]).toBeGreaterThan(0.1); // the arm rose
    expect(Math.abs(ctl.targetPos[1] - start[1])).toBeLessThan(0.005);
    expect(Math.abs(ctl.targetPos[0] - start[0])).toBeLessThan(0.01);
  });

  it("still follows a deliberate push toward the camera", () => {
    const aspect = 16 / 9;
    const ctl = new TeleopController(stubScene());
    ctl.frameAspect = aspect;
    const hand = syntheticOpenHand();
    const at = (depth: number): HandObservation => ({
      imageLandmarks: projectPinhole(hand, depth, aspect),
      worldLandmarks: hand,
    });

    run(ctl, at(0.55), 1.0);
    const start = ctl.targetPos;
    // 10% closer: a fifth of the depth window, well clear of the slop.
    run(ctl, at(0.5), 3.0);
    expect(ctl.targetPos[0] - start[0]).toBeGreaterThan(0.05);
    // Not quite zero, and inherently so: the lateral cue is the palm's
    // position in the image, and the palm sits a few centimetres from
    // the hand centre the camera pivots it around, so approaching the
    // lens slides it slightly across the frame. Roughly a centimetre of
    // tool travel per 10% of depth — far below the arm's own motion.
    expect(Math.abs(ctl.targetPos[2] - start[2])).toBeLessThan(0.02);
  });
});

describe("gripper", () => {
  it("closes on pinch and reopens with hysteresis", () => {
    const ctl = make();
    ctl.tick(obsAt([0.5, 0.5], 2, syntheticPinch()), DT);
    expect(ctl.gripping).toBe(true);
    expect(ctl.gripOpening).toBe(0);

    // Open hand: pinch ratio rises well past pinchOpen once the low-pass
    // filter catches up, releasing the grip and saturating the aperture.
    run(ctl, obsAt([0.5, 0.5]), 2.0);
    expect(ctl.gripping).toBe(false);
    expect(ctl.gripOpening).toBe(1);
  });

  it("never closes on a thumb-tucked fist", () => {
    const ctl = make();
    settle(ctl);
    run(ctl, obsAt([0.5, 0.5], 2, syntheticTuckedFist()), 0.5);
    expect(ctl.clutched).toBe(true);
    expect(ctl.gripping).toBe(false);
    expect(ctl.gripOpening).toBe(1);
  });
});

describe("clutch", () => {
  it("engages on a held fist, freezes the target, and latches a held grasp", () => {
    const ctl = make();
    settle(ctl);
    // Pinch first so a grasp is active when the clutch releases.
    run(ctl, obsAt([0.5, 0.5], 2, syntheticPinch()), 0.3);
    expect(ctl.gripping).toBe(true);

    run(ctl, obsAt([0.5, 0.5], 2, syntheticFist()), 0.2);
    expect(ctl.clutched).toBe(true);
    const frozenTarget = [...ctl.targetPos];
    run(ctl, obsAt([0.7, 0.3], 2, syntheticFist()), 0.3); // hand moves while clutched
    expect(ctl.targetPos).toEqual(frozenTarget); // frozen: no target update

    // Open hand releases the clutch; the grasp latches so the payload
    // is not dropped, and a fresh pinch re-arms the latch.
    run(ctl, obsAt([0.7, 0.3]), 0.2);
    expect(ctl.clutched).toBe(false);
    expect(ctl.gripLatched).toBe(true);
    expect(ctl.gripOpening).toBe(0);
    run(ctl, obsAt([0.7, 0.3], 2, syntheticPinch()), 0.5);
    expect(ctl.gripLatched).toBe(false);
  });

  it("ignores single-frame fist and open-hand glitches", () => {
    const ctl = make();
    settle(ctl);
    ctl.tick(obsAt([0.5, 0.5], 2, syntheticFist()), DT);
    expect(ctl.clutched).toBe(false);
    run(ctl, obsAt([0.5, 0.5], 2, syntheticFist()), 0.3);
    expect(ctl.clutched).toBe(true);
    ctl.tick(obsAt([0.5, 0.5]), DT);
    expect(ctl.clutched).toBe(true);
    ctl.tick(obsAt([0.5, 0.5], 2, syntheticFist()), DT);
    run(ctl, obsAt([0.5, 0.5]), 0.3);
    expect(ctl.clutched).toBe(false);
  });

  it("resumes from the frozen pose, not from the hand's new absolute spot", () => {
    const ctl = make();
    settle(ctl);
    run(ctl, obsAt([0.35, 0.5]), 1.0);

    // Clutch, then carry the hand across the frame and closer to the camera.
    run(ctl, obsAt([0.35, 0.5], 2, syntheticFist()), 0.3);
    expect(ctl.clutched).toBe(true);
    const frozen: Vec3 = [...ctl.targetPos] as Vec3;
    run(ctl, obsAt([0.75, 0.65], 2.4, syntheticFist()), 0.3);
    expect(ctl.targetPos).toEqual(frozen);

    // Release: with an absolute map the arm would swing to the hand's new
    // spot; the offset rebase keeps it where it froze.
    const away = ctl.mapper.mapPosition([0.75, 0.65], 2.4);
    expect(vecNorm(vecSub(away, frozen))).toBeGreaterThan(0.15);
    run(ctl, obsAt([0.75, 0.65], 2.4), 0.5);
    expect(ctl.clutched).toBe(false);
    expect(vecNorm(vecSub(ctl.targetPos, frozen))).toBeLessThan(1e-6);

    // Motion relative to the new anchor still steers the arm.
    run(ctl, obsAt([0.8, 0.65], 2.4), 1.0);
    expect(ctl.targetPos[1]).toBeGreaterThan(frozen[1] + 0.03);
  });

  it("ignores where the hand went while tracking was lost", () => {
    const ctl = make();
    settle(ctl);
    // Establish a non-trivial pose: off-centre, with the wrist tilted.
    const tilted = syntheticOpenHand().map((p) => matVec3(rotX(0.5), p));
    run(ctl, obsAt([0.4, 0.45], 2.0, tilted), 1.0);
    const frozen: Vec3 = [...ctl.targetPos] as Vec3;
    const qFrozen = ctl.targetQuat;
    expect(quatAngleBetween(qFrozen, TOOL_DOWN_QUAT)).toBeGreaterThan(0.2);

    // The hand leaves the frame for well past the hold threshold.
    run(ctl, null, 1.0);
    expect(ctl.targetPos).toEqual(frozen);

    // It returns somewhere else entirely, closer in and tilted the
    // other way. None of that travel happened while the arm was
    // following, so none of it may move the arm.
    const back = syntheticOpenHand().map((p) => matVec3(rotX(-0.4), p));
    const away = ctl.mapper.mapPosition([0.75, 0.7], 2.5);
    expect(vecNorm(vecSub(away, frozen))).toBeGreaterThan(0.15);
    run(ctl, obsAt([0.75, 0.7], 2.5, back), 0.5);
    expect(vecNorm(vecSub(ctl.targetPos, frozen))).toBeLessThan(1e-6);
    expect(quatAngleBetween(ctl.targetQuat, qFrozen)).toBeLessThan(1e-6);

    // Motion relative to the new anchor still steers the arm.
    run(ctl, obsAt([0.8, 0.7], 2.5, back), 1.0);
    expect(ctl.targetPos[1]).toBeGreaterThan(frozen[1] + 0.03);
  });

  it("counts a freeze toggled with no hand in view", () => {
    const ctl = make();
    settle(ctl);
    run(ctl, obsAt([0.4, 0.45]), 1.0);
    const frozen: Vec3 = [...ctl.targetPos] as Vec3;
    ctl.tick(null, DT);
    ctl.manualFreeze = true; // pressed and released between frames
    ctl.manualFreeze = false;
    run(ctl, obsAt([0.75, 0.65], 2.4), 0.5);
    expect(vecNorm(vecSub(ctl.targetPos, frozen))).toBeLessThan(1e-6);
  });

  it("rebases after a manual freeze too", () => {
    const ctl = make();
    settle(ctl);
    ctl.manualFreeze = true;
    const frozen: Vec3 = [...ctl.targetPos] as Vec3;
    run(ctl, obsAt([0.8, 0.2]), 0.5);
    expect(ctl.frozen).toBe(true);
    expect(ctl.targetPos).toEqual(frozen);
    ctl.manualFreeze = false;
    run(ctl, obsAt([0.8, 0.2]), 0.5);
    expect(vecNorm(vecSub(ctl.targetPos, frozen))).toBeLessThan(1e-6);
  });
});

describe("calibration gate", () => {
  it("waits for a still, fully visible hand before taking it as neutral", () => {
    const ctl = make();
    // Sweeping in from the edge: never still, partly out of frame.
    for (let i = 0; i < 20; i++) ctl.tick(obsAt([-0.1 + i * 0.03, 0.5]), DT);
    expect(ctl.calibrated).toBe(false);
    expect(ctl.calibProgress).toBe(0);
    // Holding still: progress climbs, then locks on.
    run(ctl, obsAt([0.5, 0.5]), CONTROL.calibSettleSec * 0.6);
    expect(ctl.calibrated).toBe(false);
    expect(ctl.calibProgress).toBeGreaterThan(0.4);
    run(ctl, obsAt([0.5, 0.5]), CONTROL.calibSettleSec * 0.6);
    expect(ctl.calibrated).toBe(true);
  });

  it("calibrates immediately on request when a hand is in view", () => {
    const ctl = make();
    ctl.tick(obsAt([0.6, 0.4]), DT); // one frame, not yet calibrated
    expect(ctl.calibrated).toBe(false);
    ctl.requestCalibration();
    expect(ctl.calibrated).toBe(true);
    expect(ctl.mapper.scaleRef).not.toBeNull();
    // Without a hand, the request re-arms the gate instead.
    ctl.tick(null, DT);
    ctl.requestCalibration();
    expect(ctl.calibrated).toBe(false);
  });
});

describe("tracking", () => {
  it("holds the arm after the hand is lost for enough frames", () => {
    const ctl = make();
    settle(ctl);
    ctl.tick(obsAt([0.75, 0.3]), DT); // target jumps away; arm still moving
    const qAfterTracking = [...ctl.qCmd];

    // Within the grace window the solver keeps pulling toward the target.
    for (let i = 0; i < 4; i++) ctl.tick(null, DT);
    expect(ctl.qCmd).not.toEqual(qAfterTracking);

    // Past holdAfterLostSec of lost tracking the command freezes.
    ctl.tick(null, DT);
    const qHeld = [...ctl.qCmd];
    for (let i = 0; i < 10; i++) ctl.tick(null, DT);
    expect(ctl.qCmd).toEqual(qHeld);
  });

  it("follows a fast sweep with little lag", () => {
    const ctl = make();
    settle(ctl);
    run(ctl, obsAt([0.35, 0.5]), 1.0);
    // Sweep 0.3 of the image width in 0.4 s (~0.7 m/s across the workspace).
    const seconds = 0.4;
    const n = Math.round(seconds / DT);
    let x = 0.35;
    for (let i = 1; i <= n; i++) {
      x = 0.35 + (0.3 * i) / n;
      ctl.tick(obsAt([x, 0.5]), DT);
    }
    const ideal = ctl.mapper.mapPosition([x, 0.5], 2.0);
    // Image units/s -> m/s through the lateral gain (aspect is 1 here).
    const speed = (0.3 / seconds) * MAPPING.posGain;
    const lagSec = Math.abs(ideal[1] - ctl.targetPos[1]) / speed;
    expect(lagSec).toBeLessThan(0.09);

    // And the commanded arm is on the target shortly after.
    run(ctl, obsAt([x, 0.5]), 0.3);
    const tcp = fk(ctl.qCmd).pos;
    expect(vecNorm(vecSub(tcp, ctl.targetPos))).toBeLessThan(0.01);
  });

  it("keeps reach depth fixed when the palm tilts", () => {
    const ctl = make();
    settle(ctl);
    run(ctl, obsAt([0.5, 0.5]), 0.5);
    const xFlat = ctl.targetPos[0];
    const pitched = syntheticOpenHand().map((p) => matVec3(rotX(0.7), p));
    run(ctl, obsAt([0.5, 0.5], 2.0, pitched), 1.0);
    expect(Math.abs(ctl.targetPos[0] - xFlat)).toBeLessThan(0.01);
    // The wrist did follow the tilt.
    expect(ctl.orientationOn).toBe(true);
    expect(quatAngleBetween(ctl.targetQuat, fk(ctl.qCmd).quat)).toBeLessThan(0.2);
  });
});

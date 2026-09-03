import { describe, expect, it } from "vitest";

import { type ArmPlant, TeleopController } from "../src/app";
import { CONTROL } from "../src/config";
import { type HandObservation } from "../src/handTracker";
import { fk } from "../src/kinematics";
import { type Landmarks } from "../src/poseFeatures";
import { matVec3, quatAngleBetween, rotX, type Vec3, vecNorm, vecSub } from "../src/transforms";
import {
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
    // 0.3 of the image in 0.4 s: ~0.7 m/s across the workspace.
    const seconds = 0.4;
    const n = Math.round(seconds / DT);
    let x = 0.35;
    for (let i = 1; i <= n; i++) {
      x = 0.35 + (0.3 * i) / n;
      ctl.tick(obsAt([x, 0.5]), DT);
    }
    const ideal = ctl.mapper.mapPosition([x, 0.5], 2.0);
    const speed = (0.3 / seconds) * (0.68 / 0.7); // m/s in robot y
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

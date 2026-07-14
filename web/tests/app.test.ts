import { describe, expect, it } from "vitest";

import { TeleopController } from "../src/app";
import { type ArmScene } from "../src/armScene";
import { type HandObservation } from "../src/handTracker";
import { type Landmarks } from "../src/poseFeatures";

const DT = 1 / 30;

/** Same synthetic hands as poseFeatures.test.ts / the Python tests
 * (duplicated here: importing a .test file would re-register its tests). */
function syntheticOpenHand(): Landmarks {
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

function syntheticFist(): Landmarks {
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

function syntheticPinch(): Landmarks {
  const lm = syntheticOpenHand();
  lm[4] = [0.031, 0.12, -0.01];
  lm[8] = [0.03, 0.12, -0.01];
  lm[6] = [0.03, 0.115, 0];
  return lm;
}

function stubScene(): ArmScene {
  return {
    setJointTargets: () => {},
    setGripper: () => {},
    setTargetMarker: () => {},
    step: () => {},
  } as unknown as ArmScene;
}

function obs(lm: Landmarks): HandObservation {
  return { imageLandmarks: lm, worldLandmarks: lm };
}

describe("TeleopController", () => {
  it("closes the gripper on pinch and reopens with hysteresis", () => {
    const ctl = new TeleopController(stubScene());
    ctl.tick(obs(syntheticPinch()), DT);
    expect(ctl.gripping).toBe(true);
    expect(ctl.gripOpening).toBe(0);

    // Open hand: pinch ratio rises well past pinchOpen once the low-pass
    // filter catches up, releasing the grip and saturating the aperture.
    for (let i = 0; i < 60; i++) ctl.tick(obs(syntheticOpenHand()), DT);
    expect(ctl.gripping).toBe(false);
    expect(ctl.gripOpening).toBe(1);
  });

  it("clutches on a fist, freezes the target, and latches a held grasp", () => {
    const ctl = new TeleopController(stubScene());
    // Pinch first so a grasp is active when the clutch releases.
    ctl.tick(obs(syntheticPinch()), DT);
    expect(ctl.gripping).toBe(true);

    ctl.tick(obs(syntheticFist()), DT);
    expect(ctl.clutched).toBe(true);
    const frozenTarget = [...ctl.targetPos];
    ctl.tick(obs(syntheticFist()), DT);
    expect(ctl.targetPos).toEqual(frozenTarget); // frozen: no target update

    // Open hand releases the clutch; the grasp latches so the payload
    // is not dropped, and a fresh pinch re-arms the latch.
    ctl.tick(obs(syntheticOpenHand()), DT);
    expect(ctl.clutched).toBe(false);
    expect(ctl.gripLatched).toBe(true);
    for (let i = 0; i < 60; i++) ctl.tick(obs(syntheticPinch()), DT);
    expect(ctl.gripLatched).toBe(false);
  });

  it("holds the arm after the hand is lost for enough frames", () => {
    const ctl = new TeleopController(stubScene());
    ctl.tick(obs(syntheticOpenHand()), DT);
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
});

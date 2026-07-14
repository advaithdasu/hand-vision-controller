/**
 * Teleoperation controller — TypeScript port of handarm/app.py.
 *
 * Owns the control state (targets, filters, gestures, clutch/grasp latch)
 * and turns hand observations into joint commands for the scene.
 */

import { ArmScene } from "./armScene";
import { CONTROL, HOME_Q } from "./config";
import { OneEuroFilter, QuaternionLowPass, ScalarLowPass } from "./filters";
import { type HandObservation } from "./handTracker";
import { type IKResult, solveIK } from "./ik";
import { fk } from "./kinematics";
import { HandToRobotMapper, TOOL_DOWN_QUAT } from "./mapping";
import * as pf from "./poseFeatures";
import { type Landmarks } from "./poseFeatures";
import { type Quat, type Vec3 } from "./transforms";

export class TeleopController {
  readonly mapper = new HandToRobotMapper();
  private posFilter = new OneEuroFilter(
    CONTROL.posMinCutoff, CONTROL.posBeta, CONTROL.posDCutoff);
  private oriFilter = new QuaternionLowPass(CONTROL.oriCutoff);
  private pinchFilter = new ScalarLowPass(CONTROL.pinchCutoff);

  qCmd = [...HOME_Q];
  targetPos: Vec3;
  targetQuat: Quat;
  gripOpening = 1;
  gripping = false;
  gripLatched = false;
  clutched = false;
  manualFreeze = false;
  orientationOn = true;
  calibrated = false;
  lostTime = 999; // seconds since the hand was last seen
  lastIK: IKResult | null = null;
  frameAspect = 16 / 9;

  constructor(private scene: ArmScene) {
    const f = fk(HOME_Q);
    this.targetPos = f.pos;
    this.targetQuat = f.quat;
  }

  get frozen(): boolean {
    return this.clutched || this.manualFreeze;
  }

  resetControl(): void {
    this.qCmd = [...HOME_Q];
    const f = fk(HOME_Q);
    this.targetPos = f.pos;
    this.targetQuat = f.quat;
    this.gripOpening = 1;
    this.gripping = false;
    this.gripLatched = false;
    this.clutched = false;
    this.manualFreeze = false;
    this.posFilter.reset();
    this.oriFilter.reset();
    this.pinchFilter.reset();
  }

  /**
   * Apparent hand size in aspect-corrected normalized units (MediaPipe
   * normalizes x by width and y by height; without correction, in-plane
   * rotation masquerades as depth motion on a 16:9 camera).
   */
  handScaleCorrected(imageLandmarks: Landmarks): number {
    const lm = imageLandmarks.map(
      (p): [number, number, number] => [p[0] * this.frameAspect, p[1], p[2]],
    );
    return pf.handScale(lm);
  }

  calibrate(obs: HandObservation): void {
    this.mapper.calibrate(
      this.handScaleCorrected(obs.imageLandmarks),
      pf.palmFrame(obs.worldLandmarks),
    );
    this.calibrated = true;
    this.posFilter.reset();
    this.oriFilter.reset();
  }

  processHand(obs: HandObservation, dt: number): void {
    const { imageLandmarks: img, worldLandmarks: world } = obs;
    const ext = pf.fingerExtensions(world);
    const pinchRaw = pf.pinchRatio(world);

    // Clutch: a fist freezes tracking; an open hand releases it. If an
    // object is held at release, latch the grasp until the next pinch so
    // opening the hand doesn't drop the payload.
    if (pf.isFistFrom(ext, pinchRaw)) {
      this.clutched = true;
    } else if (this.clutched && pf.countExtendedFrom(ext) >= 3) {
      this.clutched = false;
      this.pinchFilter.reset();
      if (this.gripping) this.gripLatched = true;
    }

    if (this.frozen) return;

    const pinch = this.pinchFilter.apply(pinchRaw, dt);
    if (this.gripLatched) {
      if (pinch < CONTROL.pinchClose) this.gripLatched = false; // re-armed
    } else {
      if (this.gripping && pinch > CONTROL.pinchOpen) this.gripping = false;
      else if (!this.gripping && pinch < CONTROL.pinchClose) this.gripping = true;
      this.gripOpening = Math.min(Math.max(
        (pinch - CONTROL.pinchClose) /
          (CONTROL.pinchOpen + CONTROL.apertureMargin - CONTROL.pinchClose),
        0), 1);
    }

    if (!this.calibrated) this.calibrate(obs);

    const scale = this.handScaleCorrected(img);
    const center = pf.palmCenter(img);
    const rawPos = this.mapper.mapPosition([center[0], center[1]], scale);
    this.targetPos = this.posFilter.apply(rawPos, dt) as Vec3;
    const rawQuat = this.orientationOn
      ? this.mapper.mapOrientation(pf.palmFrame(world))
      : TOOL_DOWN_QUAT;
    this.targetQuat = this.oriFilter.apply(rawQuat, dt);
  }

  solveAndCommand(dt: number): void {
    const res = solveIK(this.qCmd, this.targetPos, this.targetQuat);
    this.lastIK = res;
    const maxDq = CONTROL.maxJointVel * Math.max(dt, 1e-3);
    this.qCmd = this.qCmd.map((v, i) =>
      v + Math.min(Math.max(res.q[i] - v, -maxDq), maxDq),
    );
    this.scene.setJointTargets(this.qCmd);
    this.scene.setGripper(this.gripOpening);
    this.scene.setTargetMarker(this.targetPos);
  }

  /** One control tick, as the render loop calls it. */
  tick(obs: HandObservation | null, dt: number): void {
    if (obs) {
      this.lostTime = 0;
      this.processHand(obs, dt);
    } else {
      // Wall-clock, not ticks: the loop runs at display refresh rate, so a
      // frame count would make the hold threshold monitor-dependent.
      this.lostTime += dt;
    }
    // hold implies a lost hand: lostTime resets to 0 whenever obs exists.
    const hold = this.lostTime >= CONTROL.holdAfterLostSec;
    if (!hold) this.solveAndCommand(dt);
    this.scene.step(dt);
  }
}

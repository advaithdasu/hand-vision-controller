/**
 * Teleoperation controller — TypeScript port of handarm/app.py.
 *
 * Owns the control state (targets, filters, gestures, clutch/grasp latch,
 * calibration gate) and turns hand observations into joint commands for
 * the scene.
 */

import type { ArmScene } from "./armScene";
import { CONTROL, HOME_Q } from "./config";
import { OneEuroFilter, QuaternionLowPass, ScalarLowPass } from "./filters";
import { type HandObservation } from "./handTracker";
import { type IKResult, solveIK } from "./ik";
import { fk } from "./kinematics";
import { HandToRobotMapper, TOOL_DOWN_QUAT } from "./mapping";
import * as pf from "./poseFeatures";
import { type Quat, type Vec3 } from "./transforms";

/** The subset of ArmScene the controller drives (stubbed in tests). */
export type ArmPlant = Pick<
  ArmScene,
  "setJointTargets" | "setGripper" | "setTargetMarker" | "step"
>;

export class TeleopController {
  readonly mapper = new HandToRobotMapper();
  private posFilter = new OneEuroFilter(
    CONTROL.posMinCutoff, CONTROL.posBeta, CONTROL.posDCutoff);
  private oriFilter = new QuaternionLowPass(
    CONTROL.oriMinCutoff, CONTROL.oriBeta, CONTROL.oriDCutoff);
  private pinchFilter = new ScalarLowPass(CONTROL.pinchCutoff);

  qCmd = [...HOME_Q];
  targetPos: Vec3;
  targetQuat: Quat;
  gripOpening = 1;
  gripping = false;
  gripLatched = false;
  clutched = false;
  calibrated = false;
  /** 0..1 progress of the hold-still auto-calibration gate. */
  calibProgress = 0;
  lostTime = 999; // seconds since the hand was last seen
  lastIK: IKResult | null = null;
  lastObs: HandObservation | null = null;
  frameAspect = 16 / 9;

  private _manualFreeze = false;
  private _orientationOn = true;
  // Gesture debounce timers (seconds the current shape has persisted).
  private fistTime = 0;
  private openTime = 0;
  // Set when tracking resumes after a freeze: the next processed frame
  // re-anchors the maps so the arm continues from its frozen pose.
  private rebasePosPending = false;
  private rebaseOriPending = false;
  // Calibration gate state.
  private calibHold = 0;
  private calibPrevCenter: [number, number] | null = null;

  constructor(private scene: ArmPlant) {
    const f = fk(HOME_Q);
    this.targetPos = f.pos;
    this.targetQuat = f.quat;
  }

  get frozen(): boolean {
    return this.clutched || this._manualFreeze;
  }

  get manualFreeze(): boolean {
    return this._manualFreeze;
  }

  set manualFreeze(on: boolean) {
    if (this._manualFreeze && !on) this.markRebase();
    this._manualFreeze = on;
  }

  get orientationOn(): boolean {
    return this._orientationOn;
  }

  set orientationOn(on: boolean) {
    // Turning orientation back on would otherwise snap the wrist from
    // tool-down to the hand's absolute tilt.
    if (!this._orientationOn && on) this.rebaseOriPending = true;
    this._orientationOn = on;
  }

  private markRebase(): void {
    this.rebasePosPending = true;
    this.rebaseOriPending = true;
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
    this._manualFreeze = false;
    this.fistTime = 0;
    this.openTime = 0;
    this.rebasePosPending = false;
    this.rebaseOriPending = false;
    this.mapper.posOffset = [0, 0, 0];
    this.posFilter.reset();
    this.oriFilter.reset();
    this.pinchFilter.reset();
  }

  /**
   * Take the current hand pose as neutral. With a hand in view this is
   * immediate (what the Calibrate button should do, frozen or not);
   * otherwise the hold-still gate re-arms for the next detection.
   */
  requestCalibration(): void {
    if (this.lastObs) {
      this.calibrate(this.lastObs);
    } else {
      this.calibrated = false;
      this.calibHold = 0;
      this.calibProgress = 0;
    }
  }

  calibrate(obs: HandObservation): void {
    this.mapper.calibrate(
      pf.apparentScale(obs.imageLandmarks, obs.worldLandmarks, this.frameAspect),
      pf.palmFrame(obs.worldLandmarks),
    );
    this.calibrated = true;
    this.calibProgress = 1;
    this.calibHold = 0;
    this.rebasePosPending = false;
    this.rebaseOriPending = false;
    this.posFilter.reset();
    this.oriFilter.reset();
  }

  /**
   * Auto-calibration gate: the first detection is usually a hand entering
   * at the frame edge, so wait until it is fully in view and nearly still.
   */
  private updateCalibrationGate(obs: HandObservation, dt: number): void {
    const c = pf.palmCenter(obs.imageLandmarks);
    const xy: [number, number] = [c[0] * this.frameAspect, c[1]];
    let speed = 0;
    if (this.calibPrevCenter && dt > 0) {
      speed = Math.hypot(xy[0] - this.calibPrevCenter[0], xy[1] - this.calibPrevCenter[1]) / dt;
    }
    this.calibPrevCenter = xy;
    const steady =
      pf.fullyInFrame(obs.imageLandmarks, CONTROL.calibEdgeMargin) &&
      speed < CONTROL.calibMaxSpeed;
    this.calibHold = steady ? this.calibHold + dt : 0;
    this.calibProgress = Math.min(this.calibHold / CONTROL.calibSettleSec, 1);
    if (this.calibHold >= CONTROL.calibSettleSec) this.calibrate(obs);
  }

  private updateGripper(pinchRaw: number, dt: number): void {
    const pinch = this.pinchFilter.apply(pinchRaw, dt);
    if (this.gripLatched) {
      if (pinch < CONTROL.pinchClose) this.gripLatched = false; // re-armed
      return;
    }
    if (this.gripping && pinch > CONTROL.pinchOpen) this.gripping = false;
    else if (!this.gripping && pinch < CONTROL.pinchClose) this.gripping = true;
    this.gripOpening = Math.min(Math.max(
      (pinch - CONTROL.pinchClose) /
        (CONTROL.pinchOpen + CONTROL.apertureMargin - CONTROL.pinchClose),
      0), 1);
  }

  processHand(obs: HandObservation, dt: number): void {
    const { imageLandmarks: img, worldLandmarks: world } = obs;
    const ext = pf.fingerExtensions(world);
    const pinchRaw = pf.pinchRatio(world);

    // Clutch: a held fist freezes tracking; a held open hand releases it.
    // If an object is grasped at release, latch the grasp until the next
    // pinch so opening the hand doesn't drop the payload.
    const fist = pf.isFistFrom(ext);
    const open = pf.countExtendedFrom(ext) >= 3;
    this.fistTime = fist ? this.fistTime + dt : 0;
    this.openTime = open ? this.openTime + dt : 0;
    if (!this.clutched) {
      if (this.fistTime >= CONTROL.clutchEngageSec) this.clutched = true;
    } else if (this.openTime >= CONTROL.clutchReleaseSec) {
      this.clutched = false;
      this.pinchFilter.reset();
      if (this.gripping) this.gripLatched = true;
      this.markRebase();
    }

    if (this.frozen) return;

    // A fist-shaped frame never drives the gripper, even before the
    // clutch debounce elapses: a tucked thumb can read as a tight pinch.
    if (!fist) this.updateGripper(pinchRaw, dt);

    if (!this.calibrated) {
      this.updateCalibrationGate(obs, dt);
      if (!this.calibrated) return;
    }

    const scale = pf.apparentScale(img, world, this.frameAspect);
    const c = pf.palmCenter(img);
    const xy: [number, number] = [c[0], c[1]];
    if (this.rebasePosPending) {
      this.mapper.rebasePosition(xy, scale, this.targetPos);
      this.posFilter.reset();
      this.rebasePosPending = false;
    }
    this.targetPos = this.posFilter.apply(this.mapper.mapPosition(xy, scale), dt) as Vec3;

    let rawQuat: Quat = TOOL_DOWN_QUAT;
    if (this._orientationOn) {
      const palmRot = pf.palmFrame(world);
      if (this.rebaseOriPending) {
        this.mapper.rebaseOrientation(palmRot, this.targetQuat);
        this.oriFilter.reset();
      }
      rawQuat = this.mapper.mapOrientation(palmRot);
    }
    this.rebaseOriPending = false;
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

  /**
   * One control tick with the targets already set by a script (the
   * autopilot): same IK -> rate-limited command -> physics path as a hand
   * frame, minus gesture processing. A manual freeze holds the arm.
   */
  tickScripted(dt: number): void {
    this.lastObs = null;
    if (!this.frozen) this.solveAndCommand(dt);
    this.scene.step(dt);
  }

  /** One control tick, as the render loop calls it. */
  tick(obs: HandObservation | null, dt: number): void {
    this.lastObs = obs;
    if (obs) {
      this.lostTime = 0;
      this.processHand(obs, dt);
    } else {
      // Wall-clock, not ticks: the loop runs at display refresh rate, so a
      // frame count would make the hold threshold monitor-dependent.
      this.lostTime += dt;
      this.fistTime = 0;
      this.openTime = 0;
      this.calibPrevCenter = null;
    }
    // hold implies a lost hand: lostTime resets to 0 whenever obs exists.
    const hold = this.lostTime >= CONTROL.holdAfterLostSec;
    if (!hold) this.solveAndCommand(dt);
    this.scene.step(dt);
  }
}

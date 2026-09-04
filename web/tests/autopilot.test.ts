import { describe, expect, it } from "vitest";

import { type ArmPlant, TeleopController } from "../src/app";
import {
  Autopilot,
  type AutopilotPlant,
  DROP_SLOTS,
  planPickAndPlace,
  TRAY_CENTER,
} from "../src/autopilot";
import { HOME_Q } from "../src/config";
import { solveIK } from "../src/ik";
import { fk } from "../src/kinematics";
import { TOOL_DOWN_QUAT } from "../src/mapping";
import { type Vec3, vecNorm, vecSub } from "../src/transforms";

const DT = 1 / 60;
const CUBE_HALF = 0.02;
const CUBE_STARTS: Vec3[] = [
  [0.42, -0.12, CUBE_HALF],
  [0.5, 0.06, CUBE_HALF],
];
const GRASP_RADIUS = 0.055; // as armScene.ts
const TRAY_INNER = 0.078; // half extent inside the walls

/**
 * Stand-in for ArmScene's grasp model: a cube within reach of the tool
 * when the gripper closes rides along with it and drops straight down
 * when the gripper opens. No physics, so what this exercises is the
 * script's sequencing and whether the joint-rate-limited arm actually
 * reaches each waypoint within the time the script allows.
 */
class FakeScene implements ArmPlant, AutopilotPlant {
  cubes: Vec3[] = CUBE_STARTS.map((c) => [...c] as Vec3);
  private q = [...HOME_Q];
  private grip = 1;
  private heldIdx = -1;
  private holdOffset: Vec3 = [0, 0, 0];
  closes = 0;
  constructor(private graspWorks = true) {}

  setJointTargets(q: number[]): void {
    this.q = [...q];
  }
  setGripper(opening: number): void {
    if (opening < 0.25 && this.grip >= 0.25) this.closes++;
    this.grip = opening;
  }
  setTargetMarker(): void {}
  step(): void {
    const tcp = fk(this.q).pos;
    if (this.grip < 0.25) {
      if (this.heldIdx < 0 && this.graspWorks) {
        this.cubes.forEach((c, i) => {
          if (this.heldIdx < 0 && vecNorm(vecSub(c, tcp)) < GRASP_RADIUS) {
            this.heldIdx = i;
            this.holdOffset = vecSub(c, tcp);
          }
        });
      }
    } else if (this.heldIdx >= 0) {
      const c = this.cubes[this.heldIdx];
      this.cubes[this.heldIdx] = [c[0], c[1], CUBE_HALF]; // falls straight down
      this.heldIdx = -1;
    }
    if (this.heldIdx >= 0) {
      const o = this.holdOffset;
      this.cubes[this.heldIdx] = [tcp[0] + o[0], tcp[1] + o[1], tcp[2] + o[2]];
    }
  }
  cubePositions(): Vec3[] {
    return this.cubes.map((c) => [...c] as Vec3);
  }
  get holding(): boolean {
    return this.heldIdx >= 0;
  }
}

function rig(graspWorks = true): { scene: FakeScene; ap: Autopilot; ctl: TeleopController } {
  const scene = new FakeScene(graspWorks);
  const ctl = new TeleopController(scene);
  const ap = new Autopilot(ctl, scene);
  return { scene, ap, ctl };
}

function runUntil(ap: Autopilot, done: () => boolean, maxSeconds: number): number {
  let t = 0;
  while (!done() && t < maxSeconds) {
    ap.tick(DT);
    t += DT;
  }
  return t;
}

describe("planPickAndPlace", () => {
  it("produces waypoints the arm can reach with tool-down orientation", () => {
    for (const [i, cube] of CUBE_STARTS.entries()) {
      let q = [...HOME_Q];
      for (const seg of planPickAndPlace(cube, DROP_SLOTS[i])) {
        // Warm-start through the path as the live loop would.
        let res = solveIK(q, seg.pos, TOOL_DOWN_QUAT);
        for (let k = 0; k < 8 && !res.converged; k++) res = solveIK(res.q, seg.pos, TOOL_DOWN_QUAT);
        expect(res.posErr, `${seg.label} for cube ${i}`).toBeLessThan(3e-3);
        expect(res.oriErr, `${seg.label} for cube ${i}`).toBeLessThan(0.03);
        q = res.q;
      }
    }
  });

  it("drops every slot inside the tray walls", () => {
    for (const slot of DROP_SLOTS) {
      expect(Math.abs(slot[0] - TRAY_CENTER[0])).toBeLessThan(TRAY_INNER - CUBE_HALF);
      expect(Math.abs(slot[1] - TRAY_CENTER[1])).toBeLessThan(TRAY_INNER - CUBE_HALF);
    }
  });
});

describe("Autopilot", () => {
  it("picks both cubes and places them in the tray", () => {
    const { scene, ap } = rig();
    const t = runUntil(ap, () => ap.roundComplete, 40);
    expect(ap.roundComplete).toBe(true);
    expect(t).toBeLessThan(30);
    expect(scene.holding).toBe(false);
    scene.cubes.forEach((c, i) => {
      expect(Math.abs(c[0] - TRAY_CENTER[0]), `cube ${i} x`).toBeLessThan(TRAY_INNER - CUBE_HALF);
      expect(Math.abs(c[1] - TRAY_CENTER[1]), `cube ${i} y`).toBeLessThan(TRAY_INNER - CUBE_HALF);
      expect(vecNorm(vecSub([c[0], c[1], 0], DROP_SLOTS[i])), `cube ${i} slot`).toBeLessThan(0.02);
    });
    // Exactly one grasp per cube: no re-grabs, no fumbles.
    expect(scene.closes).toBe(2);
  });

  it("drives through the real IK path with converged solves", () => {
    const { ap, ctl } = rig();
    let worst = 0;
    runUntil(ap, () => {
      if (ctl.lastIK) worst = Math.max(worst, ctl.lastIK.posErr);
      return ap.roundComplete;
    }, 40);
    // The solver itself must keep up with the eased targets (the joint
    // rate limit, not the solve, is what lags during moves).
    expect(worst).toBeLessThan(0.02);
    expect(ctl.lastIK?.converged).toBe(true);
  });

  it("retries a failed grasp a bounded number of times, then moves on", () => {
    const { scene, ap } = rig(false);
    runUntil(ap, () => ap.roundComplete, 90);
    expect(ap.roundComplete).toBe(true);
    // Three attempts per cube, each of which closes the gripper once.
    expect(scene.closes).toBe(6);
    // Nothing was placed.
    scene.cubes.forEach((c, i) => expect(vecNorm(vecSub(c, CUBE_STARTS[i]))).toBeLessThan(1e-9));
  });

  it("pauses while frozen and resumes on unfreeze", () => {
    const { ap, ctl } = rig();
    runUntil(ap, () => ap.status === "reaching for the cube", 5);
    ap.tick(DT);
    const before = [...ctl.targetPos];
    ctl.manualFreeze = true;
    for (let i = 0; i < 30; i++) ap.tick(DT);
    expect(ctl.targetPos).toEqual(before);
    ctl.manualFreeze = false;
    ap.tick(DT);
    expect(ctl.targetPos).not.toEqual(before);
  });

  it("restart() begins a fresh round after a scene reset", () => {
    const { scene, ap, ctl } = rig();
    runUntil(ap, () => ap.roundComplete, 40);
    scene.cubes = CUBE_STARTS.map((c) => [...c] as Vec3);
    ctl.resetControl();
    ap.restart();
    expect(ap.roundComplete).toBe(false);
    runUntil(ap, () => ap.roundComplete, 40);
    expect(ap.roundComplete).toBe(true);
    expect(scene.closes).toBe(4);
  });
});

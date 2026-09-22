/**
 * Autopilot: a scripted pick-and-place that drives the arm with no camera.
 *
 * It exists so a visitor who won't (or can't) grant camera access still
 * sees the arm work. The script feeds Cartesian targets into the *same*
 * path the hand does — TeleopController.solveAndCommand (warm-started
 * DLS IK, joint-rate limiting) and the scene's physics — so what plays is
 * the real control stack, not a canned animation. Mirrors
 * handarm/autopilot.py.
 */

import type { TeleopController } from "./app";
import { SCENE } from "./config";
import { TOOL_DOWN_QUAT } from "./mapping";
import { type Quat, quatSlerp, type Vec3 } from "./transforms";

/** The subset of the scene the script needs (stubbed in tests). */
export interface AutopilotPlant {
  /** Current cube centers, world frame, in spawn order. */
  cubePositions(): Vec3[];
  /** True while a cube is grasped. */
  readonly holding: boolean;
}

export interface Segment {
  /** Where the tool center point should be at the end of the segment. */
  pos: Vec3;
  /** Gripper opening to hold through the segment (0 closed .. 1 open). */
  grip: number;
  /** Seconds the move takes. */
  duration: number;
  label: string;
  /** After this segment, the cube must be held or the pick is retried. */
  verifyGrasp?: boolean;
}

/** Drop slots inside the tray, one per cube, spread along its x axis. */
export const DROP_SLOTS: Vec3[] = [
  [SCENE.trayCenter[0] - 0.035, SCENE.trayCenter[1], 0],
  [SCENE.trayCenter[0] + 0.035, SCENE.trayCenter[1], 0],
];

const HOVER_Z = 0.15; // above the cube center before descending
const GRASP_DZ = 0.015; // tcp just above the cube center, as the Python test
const TRAY_HOVER_Z = 0.2;
const TRAY_DROP_Z = 0.1; // low enough that the cube can't bounce out

/**
 * Waypoints for picking `cube` and dropping it at `slot`. Positions are
 * tool-center-point targets; the gripper closes in place before the lift.
 */
export function planPickAndPlace(cube: Vec3, slot: Vec3): Segment[] {
  const hover: Vec3 = [cube[0], cube[1], cube[2] + HOVER_Z];
  const grasp: Vec3 = [cube[0], cube[1], cube[2] + GRASP_DZ];
  const above: Vec3 = [slot[0], slot[1], TRAY_HOVER_Z];
  const drop: Vec3 = [slot[0], slot[1], TRAY_DROP_Z];
  return [
    { pos: hover, grip: 1, duration: 1.4, label: "reaching for the cube" },
    { pos: grasp, grip: 1, duration: 1.0, label: "descending" },
    { pos: grasp, grip: 0, duration: 0.6, label: "closing the gripper" },
    { pos: hover, grip: 0, duration: 1.0, label: "lifting", verifyGrasp: true },
    { pos: above, grip: 0, duration: 1.6, label: "carrying to the tray" },
    { pos: drop, grip: 0, duration: 0.8, label: "lowering" },
    { pos: drop, grip: 1, duration: 0.5, label: "releasing" },
    { pos: above, grip: 1, duration: 0.8, label: "retreating" },
  ];
}

/** Cubic ease-in/out on [0, 1]: no velocity discontinuity at waypoints. */
function smoothstep(t: number): number {
  const u = Math.min(Math.max(t, 0), 1);
  return u * u * (3 - 2 * u);
}

/** Cubes at least this far from their drop slot still need placing. */
const PLACED_RADIUS = 0.06;
const MAX_ATTEMPTS = 3; // per cube before moving on
const START_PAUSE = 1.0; // seconds to rest before the first pick

type Phase =
  | {
      kind: "pick";
      cube: number;
      attempt: number;
      segs: Segment[];
      i: number;
      t: number;
      from: Vec3;
      /** Orientation at the segment start; eased to tool-down like position. */
      fromQuat: Quat;
    }
  | { kind: "pause"; t: number };

export class Autopilot {
  private phase: Phase = { kind: "pause", t: START_PAUSE };
  /** Cubes given up on this round (grasp failed MAX_ATTEMPTS times). */
  private skipped = new Set<number>();
  /**
   * True once every cube is in its slot. The script then parks; the
   * caller resets the scene and calls restart() to loop.
   */
  roundComplete = false;
  status = "starting";

  constructor(
    private ctl: TeleopController,
    private scene: AutopilotPlant,
  ) {}

  /** Start over from a fresh scene (after a reset). */
  restart(): void {
    this.phase = { kind: "pause", t: 0.3 };
    this.skipped.clear();
    this.roundComplete = false;
    this.status = "starting";
  }

  /** Index of the next cube not yet in its slot, or -1 when done. */
  private nextCube(): number {
    const cubes = this.scene.cubePositions();
    for (let i = 0; i < cubes.length && i < DROP_SLOTS.length; i++) {
      if (this.skipped.has(i)) continue;
      const slot = DROP_SLOTS[i];
      const dxy = Math.hypot(cubes[i][0] - slot[0], cubes[i][1] - slot[1]);
      if (dxy > PLACED_RADIUS) return i;
    }
    return -1;
  }

  private beginPick(cube: number, attempt: number): void {
    const p = this.scene.cubePositions()[cube];
    this.phase = {
      kind: "pick",
      cube,
      attempt,
      segs: planPickAndPlace(p, DROP_SLOTS[cube]),
      i: 0,
      t: 0,
      from: [...this.ctl.targetPos] as Vec3,
      fromQuat: [...this.ctl.targetQuat] as Quat,
    };
  }

  /**
   * One tick: advance the script, write the controller's targets, and run
   * the usual IK -> command -> physics step. Freezing the controller
   * pauses the script (the scene still steps so held cubes settle).
   */
  tick(dt: number): void {
    if (!this.ctl.frozen) this.advance(dt);
    this.ctl.tickScripted(dt);
  }

  private advance(dt: number): void {
    const ph = this.phase;
    if (ph.kind === "pause") {
      ph.t -= dt;
      this.status = "resting";
      if (ph.t > 0) return;
      const next = this.nextCube();
      if (next < 0) {
        this.roundComplete = true;
        this.status = "done";
        ph.t = Infinity; // park until restart()
        return;
      }
      this.beginPick(next, 1);
      return;
    }

    const seg = ph.segs[ph.i];
    ph.t += dt;
    const a = smoothstep(ph.t / seg.duration);
    this.ctl.targetPos = [
      ph.from[0] + (seg.pos[0] - ph.from[0]) * a,
      ph.from[1] + (seg.pos[1] - ph.from[1]) * a,
      ph.from[2] + (seg.pos[2] - ph.from[2]) * a,
    ];
    // The home pose is not tool-down; snapping the wrist there at the
    // edge of reach would ask IK for an unreachable pose on the first tick.
    this.ctl.targetQuat = quatSlerp(ph.fromQuat, TOOL_DOWN_QUAT, a);
    // Aperture eases over the segment too, so a close doesn't snap.
    const gripFrom = ph.i > 0 ? ph.segs[ph.i - 1].grip : 1;
    this.ctl.gripOpening = gripFrom + (seg.grip - gripFrom) * a;
    this.status = seg.label;

    if (ph.t < seg.duration) return;

    // Segment done. After the lift, verify the grasp took; otherwise retry
    // from the cube's current spot (it may have been nudged).
    ph.from = [...seg.pos] as Vec3;
    ph.fromQuat = [...TOOL_DOWN_QUAT];
    ph.t = 0;
    ph.i++;
    if (seg.verifyGrasp && !this.scene.holding) {
      if (ph.attempt >= MAX_ATTEMPTS) {
        this.skipped.add(ph.cube);
        this.phase = { kind: "pause", t: 0.3 };
      } else {
        this.beginPick(ph.cube, ph.attempt + 1);
      }
      return;
    }
    if (ph.i >= ph.segs.length) this.phase = { kind: "pause", t: 0.4 };
  }
}

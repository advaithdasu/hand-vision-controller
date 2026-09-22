/**
 * Camera-space hand pose -> robot-space end-effector target — TypeScript
 * port of handarm/mapping.py.
 *
 * Position: the palm's (x, y) in the mirrored image maps to the robot's
 * (y, z); apparent hand size maps to depth along robot x, with a hand
 * reaching toward the camera extending the arm. Both are
 * *relative* maps, anchored on the pose captured at calibration: that
 * pose sits at the centre of the workspace and the hand moves the tool
 * away from it by a fixed gain, so the operator can recentre their
 * working volume at any time by recalibrating. A position offset lets
 * the operator clutch: after a fist-freeze the arm resumes from where
 * it stopped, not from wherever the hand now happens to be.
 *
 * Orientation: the palm's rotation *relative to a reference pose* is
 * remapped into robot axes on top of the default tool-down pose. The
 * reference is captured at calibration and re-anchored on clutch release
 * so the wrist, too, resumes from its frozen pose.
 */

import { MAPPING, WORKSPACE } from "./config";
import {
  type Mat3,
  matMul3,
  matrixFromQuat,
  matTranspose3,
  type Quat,
  quatAngleBetween,
  quatFromMatrix,
  quatSlerp,
  type Vec3,
} from "./transforms";

/**
 * Default end-effector orientation: tool axis pointing straight down
 * (rotation of pi about world y), reachable with zero wrist roll.
 */
export const TOOL_DOWN_QUAT: Quat = [0, 1, 0, 0];
const TOOL_DOWN_MAT = matrixFromQuat(TOOL_DOWN_QUAT);
const TOOL_DOWN_MAT_T = matTranspose3(TOOL_DOWN_MAT);

/**
 * Camera axes -> robot axes, chosen so the arm matches what the operator
 * sees: camera x (right on the mirrored image) -> robot +y (screen-right
 * for the scene camera), camera y (down) -> robot -z, camera z (into the
 * scene) -> robot +x.
 *
 * The depth column is the mirror's sense, not the position map's (which
 * extends the arm as the hand comes forward): the two are independent
 * here, because flipping that column also flips the angle sign, and the
 * two cancel on the only axis the depth direction names — roll. It is
 * pitch and yaw that the choice decides, and those are set by the
 * mirrored preview, which is what the operator watches their wrist in.
 *
 * This map is deliberately improper (orthogonal, det = -1): the preview is
 * a mirror, so making rotations *look* right requires mirroring their
 * sense too. Conjugating with it still yields proper rotations (det of
 * M R Mᵀ is +1), flips roll/yaw the way a mirror does, and leaves pitch
 * (toward/away tilt) unchanged. This intentionally diverges from
 * handarm/mapping.py, whose MuJoCo viewer sits on the other side of the
 * scene.
 */
export const CAM_TO_ROBOT: Mat3 = [
  [0, 0, 1],
  [1, 0, 0],
  [0, -1, 0],
];
const CAM_TO_ROBOT_T = matTranspose3(CAM_TO_ROBOT);

function clampToWorkspace(p: Vec3): Vec3 {
  return [
    Math.min(Math.max(p[0], WORKSPACE.x[0]), WORKSPACE.x[1]),
    Math.min(Math.max(p[1], WORKSPACE.y[0]), WORKSPACE.y[1]),
    Math.min(Math.max(p[2], WORKSPACE.z[0]), WORKSPACE.z[1]),
  ];
}

const mid = (r: [number, number]): number => (r[0] + r[1]) / 2;
const span = (r: [number, number]): number => r[1] - r[0];

export class HandToRobotMapper {
  scaleRef: number | null = null;
  rotRef: Mat3 | null = null;
  /**
   * Palm position (normalized image coords) that maps to the centre of
   * the workspace. Stored un-scaled so a mid-session resolution change
   * can't silently shift the anchor.
   */
  centerRef: [number, number] | null = null;
  /** Frame width / height, kept in sync by the controller. */
  frameAspect = 16 / 9;
  /** Robot-space offset added to the absolute map (clutch rebasing). */
  posOffset: Vec3 = [0, 0, 0];

  /** Capture the current hand pose as the neutral reference. */
  calibrate(palmXY: [number, number], handScale: number, palmRot: Mat3): void {
    this.centerRef = [palmXY[0], palmXY[1]];
    // Floor the reference so a degenerate detection (all landmarks
    // coincident -> scale 0) can't collapse the depth window into a
    // divide-by-zero that feeds NaN into the IK target.
    this.scaleRef = Math.max(handScale, 1e-4);
    this.rotRef = palmRot.map((r) => [...r]);
    this.posOffset = [0, 0, 0];
  }

  /**
   * Normalized image coords -> image *height* units. Without this the
   * same hand travel covers a smaller fraction of the (wider) frame
   * horizontally than vertically, and sideways motion feels sluggish by
   * exactly the aspect ratio.
   */
  private isotropic(xy: [number, number]): [number, number] {
    return [xy[0] * this.frameAspect, xy[1]];
  }

  /** Absolute map, before the clutch offset and workspace clamp. */
  private mapPositionAbsolute(palmXY: [number, number], handScale: number): Vec3 {
    const [hx, hy] = this.isotropic(palmXY);
    const [cx, cy] = this.isotropic(this.centerRef ?? [0.5, 0.5]);
    // Image is mirrored, so image-right = operator-right. Position tracks
    // the robot's own body frame, not the screen: with the arm's neutral
    // reach along +x (forward) and +z up, the robot's own right is -y
    // (forward x up), so hand-right -> -y and hand-left -> +y. That's
    // control from the robot's point of view, like driving from behind
    // it, rather than mirroring what the operator sees on screen. Image
    // y grows downward, so it negates into robot +z (up).
    const y = mid(WORKSPACE.y) - (hx - cx) * MAPPING.posGain;
    const z = mid(WORKSPACE.z) - (hy - cy) * MAPPING.posGain;
    // Apparent size is proportional to 1 / distance; invert it so the
    // depth axis is linear in how far the hand actually travelled,
    // rather than bunching the far half of the reach into a sliver of
    // hand motion. 1.0 is the calibrated neutral distance.
    //
    // Reaching *toward* the camera extends the arm, hence the minus:
    // the operator's forward is the robot's forward. The opposite sense
    // treats the screen as a mirror — push forward, arm retracts — and
    // that is the one thing operators reliably trip over, because both
    // views contradict it. The hand looms larger in the (mirrored)
    // preview as it comes forward, and the scene camera sits out in
    // front of the arm, so an extending tool looms larger too; only
    // this sign has the two grow together.
    const ref = this.scaleRef ?? MAPPING.defaultHandScale;
    const dist = ref / Math.max(handScale, 1e-4);
    const x = mid(WORKSPACE.x) -
      ((dist - 1) / (MAPPING.depthFarDist - MAPPING.depthNearDist)) * span(WORKSPACE.x);
    return [x, y, z];
  }

  /** (normalized image x, y) + apparent hand size -> robot xyz target. */
  mapPosition(palmXY: [number, number], handScale: number): Vec3 {
    const p = this.mapPositionAbsolute(palmXY, handScale);
    return clampToWorkspace([
      p[0] + this.posOffset[0],
      p[1] + this.posOffset[1],
      p[2] + this.posOffset[2],
    ]);
  }

  /**
   * Re-anchor the position map so the current hand pose maps exactly to
   * `target` — called when a clutch releases, so the arm continues from
   * where it froze instead of swinging to the hand's new absolute spot.
   */
  rebasePosition(palmXY: [number, number], handScale: number, target: Vec3): void {
    const p = this.mapPositionAbsolute(palmXY, handScale);
    this.posOffset = [target[0] - p[0], target[1] - p[1], target[2] - p[2]];
  }

  /** Palm rotation (camera frame) -> end-effector target quaternion. */
  mapOrientation(palmRot: Mat3): Quat {
    if (this.rotRef === null) return [...TOOL_DOWN_QUAT];
    // Rotation of the palm since calibration, conjugated into robot axes.
    const rRelCam = matMul3(palmRot, matTranspose3(this.rotRef));
    const rRelRobot = matMul3(matMul3(CAM_TO_ROBOT, rRelCam), CAM_TO_ROBOT_T);
    const qTarget = quatFromMatrix(matMul3(rRelRobot, TOOL_DOWN_MAT));
    return this.clampTilt(qTarget);
  }

  /**
   * Re-anchor the orientation reference so the current palm rotation
   * maps exactly to `target` (the orientation counterpart of
   * rebasePosition). Inverts mapOrientation: with R_rel,robot = T·D⁻¹ the
   * camera-frame relative rotation is Mᵀ·R_rel,robot·M, and the reference
   * that yields it from the current palm rotation P is R_rel,camᵀ·P.
   */
  rebaseOrientation(palmRot: Mat3, target: Quat): void {
    const rRelRobot = matMul3(matrixFromQuat(target), TOOL_DOWN_MAT_T);
    const rRelCam = matMul3(matMul3(CAM_TO_ROBOT_T, rRelRobot), CAM_TO_ROBOT);
    this.rotRef = matMul3(matTranspose3(rRelCam), palmRot);
  }

  /** Limit deviation from the default pose to maxTiltRad. */
  private clampTilt(q: Quat): Quat {
    const ang = quatAngleBetween(TOOL_DOWN_QUAT, q);
    if (ang <= MAPPING.maxTiltRad) return q;
    return quatSlerp(TOOL_DOWN_QUAT, q, MAPPING.maxTiltRad / ang);
  }
}

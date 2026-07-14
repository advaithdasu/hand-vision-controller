/**
 * Camera-space hand pose -> robot-space end-effector target — TypeScript
 * port of handarm/mapping.py.
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

/**
 * Camera axes -> robot axes, chosen so the arm matches what the operator
 * sees: camera x (right on the mirrored image) -> robot +y (screen-right
 * for the scene camera), camera y (down) -> robot -z, camera z (into the
 * scene) -> robot +x.
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

function remap(v: number, src: [number, number], dst: [number, number]): number {
  const t = Math.min(Math.max((v - src[0]) / (src[1] - src[0]), 0), 1);
  return dst[0] + t * (dst[1] - dst[0]);
}

function clampToWorkspace(p: Vec3): Vec3 {
  return [
    Math.min(Math.max(p[0], WORKSPACE.x[0]), WORKSPACE.x[1]),
    Math.min(Math.max(p[1], WORKSPACE.y[0]), WORKSPACE.y[1]),
    Math.min(Math.max(p[2], WORKSPACE.z[0]), WORKSPACE.z[1]),
  ];
}

export class HandToRobotMapper {
  scaleRef: number | null = null;
  rotRef: Mat3 | null = null;

  /** Capture the current hand pose as the neutral reference. */
  calibrate(handScale: number, palmRot: Mat3): void {
    // Floor the reference so a degenerate detection (all landmarks
    // coincident -> scale 0) can't collapse the depth window into a
    // divide-by-zero that feeds NaN into the IK target.
    this.scaleRef = Math.max(handScale, 1e-4);
    this.rotRef = palmRot.map((r) => [...r]);
  }

  /** (normalized image x, y) + apparent hand size -> robot xyz target. */
  mapPosition(palmXY: [number, number], handScale: number): Vec3 {
    // Image is mirrored, so image-right = operator-right. The scene camera
    // puts robot +y on screen-right, so hand-right -> +y keeps the arm
    // moving the same direction the operator sees their hand move.
    const y = remap(palmXY[0], MAPPING.imgXRange, [WORKSPACE.y[0], WORKSPACE.y[1]]);
    const z = remap(palmXY[1], MAPPING.imgYRange, [WORKSPACE.z[1], WORKSPACE.z[0]]);
    // Depth window is relative to the calibrated neutral hand size.
    const ref = this.scaleRef ?? MAPPING.defaultHandScale;
    const near = ref * MAPPING.depthNearRatio;
    const far = ref * MAPPING.depthFarRatio;
    const x = remap(handScale, [near, far], WORKSPACE.x);
    return clampToWorkspace([x, y, z]);
  }

  /** Palm rotation (camera frame) -> end-effector target quaternion. */
  mapOrientation(palmRot: Mat3): Quat {
    if (this.rotRef === null) return [...TOOL_DOWN_QUAT];
    // Rotation of the palm since calibration, conjugated into robot axes.
    const rRelCam = matMul3(palmRot, matTranspose3(this.rotRef));
    const rRelRobot = matMul3(
      matMul3(CAM_TO_ROBOT, rRelCam),
      matTranspose3(CAM_TO_ROBOT),
    );
    const qTarget = quatFromMatrix(matMul3(rRelRobot, TOOL_DOWN_MAT));
    return this.clampTilt(qTarget);
  }

  /** Limit deviation from the default pose to maxTiltRad. */
  private clampTilt(q: Quat): Quat {
    const ang = quatAngleBetween(TOOL_DOWN_QUAT, q);
    if (ang <= MAPPING.maxTiltRad) return q;
    return quatSlerp(TOOL_DOWN_QUAT, q, MAPPING.maxTiltRad / ang);
  }
}

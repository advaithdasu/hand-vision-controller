/**
 * Forward kinematics and the geometric Jacobian for the 6-DOF arm.
 *
 * The chain mirrors assets/scene.xml exactly (yaw / pitch / pitch / roll /
 * pitch / roll with the same link offsets), so joint solutions are
 * interchangeable with the Python + MuJoCo implementation.
 */

import {
  identity3,
  matMul3,
  matVec3,
  type Mat3,
  type Quat,
  quatFromMatrix,
  rotY,
  rotZ,
  type Vec3,
  vecAdd,
  vecCross,
  vecSub,
} from "./transforms";

export interface ChainLink {
  offset: Vec3; // translation from parent joint frame, pre-rotation
  axis: "y" | "z";
}

/** Same layout as the MJCF: body offsets then hinge about the local axis. */
export const CHAIN: ChainLink[] = [
  { offset: [0, 0, 0.1], axis: "z" },
  { offset: [0, 0, 0.08], axis: "y" },
  { offset: [0, 0, 0.3], axis: "y" },
  { offset: [0, 0, 0.25], axis: "z" },
  { offset: [0, 0, 0.06], axis: "y" },
  { offset: [0, 0, 0.06], axis: "z" },
];

export const TCP_OFFSET: Vec3 = [0, 0, 0.1];

export const JOINT_LOWER = [-2.96, -1.92, -2.44, -6.2832, -2.09, -6.2832];
export const JOINT_UPPER = [2.96, 1.92, 2.44, 6.2832, 2.09, 6.2832];

const LOCAL_AXIS: Record<"y" | "z", Vec3> = { y: [0, 1, 0], z: [0, 0, 1] };

export interface FKResult {
  pos: Vec3;
  quat: Quat;
  rot: Mat3;
  /** World-frame origin and rotation axis of each joint, for the Jacobian. */
  jointOrigins: Vec3[];
  jointAxes: Vec3[];
  /** World rotation of each link frame (post joint rotation). */
  linkRots: Mat3[];
}

export function clampJoints(q: number[]): number[] {
  return q.map((v, i) =>
    Math.min(Math.max(v, JOINT_LOWER[i]), JOINT_UPPER[i]),
  );
}

export function fk(q: number[]): FKResult {
  let p: Vec3 = [0, 0, 0];
  let R = identity3();
  const jointOrigins: Vec3[] = [];
  const jointAxes: Vec3[] = [];
  const linkRots: Mat3[] = [];

  for (let i = 0; i < CHAIN.length; i++) {
    const { offset, axis } = CHAIN[i];
    p = vecAdd(p, matVec3(R, offset));
    jointOrigins.push(p);
    // The joint axis is invariant under its own rotation.
    jointAxes.push(matVec3(R, LOCAL_AXIS[axis]));
    R = matMul3(R, axis === "z" ? rotZ(q[i]) : rotY(q[i]));
    linkRots.push(R);
  }

  const pos = vecAdd(p, matVec3(R, TCP_OFFSET));
  return { pos, quat: quatFromMatrix(R), rot: R, jointOrigins, jointAxes, linkRots };
}

/**
 * 6x6 geometric Jacobian of the tcp (world frame, rows = [linear; angular]).
 * For revolute joint i: Jp_i = a_i x (p_tcp - p_i), Jr_i = a_i.
 */
export function jacobian(q: number[], fkResult?: FKResult): number[][] {
  const f = fkResult ?? fk(q);
  const J: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (let i = 0; i < 6; i++) {
    const lever = vecSub(f.pos, f.jointOrigins[i]);
    const jp = vecCross(f.jointAxes[i], lever);
    for (let r = 0; r < 3; r++) {
      J[r][i] = jp[r];
      J[r + 3][i] = f.jointAxes[i][r];
    }
  }
  return J;
}

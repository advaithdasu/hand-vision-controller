/**
 * Minimal quaternion / rotation utilities — TypeScript port of
 * handarm/transforms.py.
 *
 * Quaternions are [x, y, z, w]. Rotation matrices are row-major 3x3
 * (number[3][3]) whose columns are the frame axes in the parent frame.
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Mat3 = number[][];

export function vecAdd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vecSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vecScale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function vecNorm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vecCross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function vecNormalize(a: Vec3): Vec3 {
  const n = vecNorm(a) + 1e-9;
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function matMul3(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return out;
}

export function matVec3(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

export function matTranspose3(m: Mat3): Mat3 {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}

export function identity3(): Mat3 {
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
}

export function rotX(a: number): Mat3 {
  const c = Math.cos(a), s = Math.sin(a);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

export function rotY(a: number): Mat3 {
  const c = Math.cos(a), s = Math.sin(a);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

export function rotZ(a: number): Mat3 {
  const c = Math.cos(a), s = Math.sin(a);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n < 1e-12) return [0, 0, 0, 1];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Hamilton product a*b: rotate by b first, then by a. */
export function quatMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatConjugate(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
}

/** Rotation vector (axis * angle) for a unit quaternion. */
export function quatToAxisAngle(q: Quat): Vec3 {
  let [x, y, z, w] = quatNormalize(q);
  if (w < 0) {
    x = -x; y = -y; z = -z; w = -w; // shortest arc
  }
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return [0, 0, 0];
  const angle = 2 * Math.atan2(s, w);
  return [(x / s) * angle, (y / s) * angle, (z / s) * angle];
}

/** Shepperd's method: rotation matrix -> quaternion [x, y, z, w]. */
export function quatFromMatrix(m: Mat3): Quat {
  const t = m[0][0] + m[1][1] + m[2][2];
  let q: Quat;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [
      (m[2][1] - m[1][2]) / s,
      (m[0][2] - m[2][0]) / s,
      (m[1][0] - m[0][1]) / s,
      0.25 * s,
    ];
  } else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [
      0.25 * s,
      (m[0][1] + m[1][0]) / s,
      (m[0][2] + m[2][0]) / s,
      (m[2][1] - m[1][2]) / s,
    ];
  } else if (m[1][1] >= m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [
      (m[0][1] + m[1][0]) / s,
      0.25 * s,
      (m[1][2] + m[2][1]) / s,
      (m[0][2] - m[2][0]) / s,
    ];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [
      (m[0][2] + m[2][0]) / s,
      (m[1][2] + m[2][1]) / s,
      0.25 * s,
      (m[1][0] - m[0][1]) / s,
    ];
  }
  return quatNormalize(q);
}

export function matrixFromQuat(q: Quat): Mat3 {
  const [x, y, z, w] = quatNormalize(q);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  a = quatNormalize(a);
  b = quatNormalize(b);
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (dot < 0) {
    b = [-b[0], -b[1], -b[2], -b[3]];
    dot = -dot;
  }
  if (dot > 0.9995) {
    return quatNormalize([
      a[0] + t * (b[0] - a[0]),
      a[1] + t * (b[1] - a[1]),
      a[2] + t * (b[2] - a[2]),
      a[3] + t * (b[3] - a[3]),
    ]);
  }
  const theta = Math.acos(Math.min(Math.max(dot, -1), 1));
  const s = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / s;
  const wb = Math.sin(t * theta) / s;
  return quatNormalize([
    wa * a[0] + wb * b[0],
    wa * a[1] + wb * b[1],
    wa * a[2] + wb * b[2],
    wa * a[3] + wb * b[3],
  ]);
}

/** World-frame rotation vector that takes qCurrent to qTarget. */
export function orientationError(qTarget: Quat, qCurrent: Quat): Vec3 {
  return quatToAxisAngle(quatMultiply(qTarget, quatConjugate(qCurrent)));
}

/** Absolute rotation angle (radians) between two orientations. */
export function quatAngleBetween(a: Quat, b: Quat): number {
  return vecNorm(orientationError(b, a));
}

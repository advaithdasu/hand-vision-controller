import { describe, expect, it } from "vitest";

import { WORKSPACE } from "../src/config";
import { CAM_TO_ROBOT, HandToRobotMapper, TOOL_DOWN_QUAT } from "../src/mapping";
import {
  identity3,
  matMul3,
  matrixFromQuat,
  matTranspose3,
  quatAngleBetween,
  quatFromMatrix,
  rotX,
  rotY,
  rotZ,
  type Vec3,
} from "../src/transforms";

describe("CAM_TO_ROBOT", () => {
  it("is orthogonal with det -1 (view-consistent mirror map)", () => {
    const shouldBeI = matMul3(CAM_TO_ROBOT, matTranspose3(CAM_TO_ROBOT));
    const I = identity3();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(shouldBeI[i][j]).toBeCloseTo(I[i][j], 9);
    const m = CAM_TO_ROBOT;
    const det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    expect(det).toBeCloseTo(-1, 9);
  });
});

function expectInWorkspace(p: Vec3): void {
  expect(p[0]).toBeGreaterThanOrEqual(WORKSPACE.x[0]);
  expect(p[0]).toBeLessThanOrEqual(WORKSPACE.x[1]);
  expect(p[1]).toBeGreaterThanOrEqual(WORKSPACE.y[0]);
  expect(p[1]).toBeLessThanOrEqual(WORKSPACE.y[1]);
  expect(p[2]).toBeGreaterThanOrEqual(WORKSPACE.z[0]);
  expect(p[2]).toBeLessThanOrEqual(WORKSPACE.z[1]);
}

describe("position mapping", () => {
  it("stays inside the workspace for any input", () => {
    const m = new HandToRobotMapper();
    for (let i = 0; i < 200; i++) {
      const xy: [number, number] = [Math.sin(i) * 0.8 + 0.5, Math.cos(i * 1.3) * 0.8 + 0.5];
      expectInWorkspace(m.mapPosition(xy, 0.02 + ((i * 37) % 100) / 200));
    }
  });

  it("maps directions to match the on-screen view", () => {
    const m = new HandToRobotMapper();
    const p0 = m.mapPosition([0.5, 0.5], 1.9);
    expect(m.mapPosition([0.7, 0.5], 1.9)[1]).toBeGreaterThan(p0[1]); // right -> +y (screen-right)
    expect(m.mapPosition([0.5, 0.3], 1.9)[2]).toBeGreaterThan(p0[2]); // up -> +z
    expect(m.mapPosition([0.5, 0.5], 2.5)[0]).toBeLessThan(p0[0]); // closer -> retract
  });

  it("stays finite when calibrated with a degenerate hand scale", () => {
    const m = new HandToRobotMapper();
    m.calibrate(0, identity3());
    const p = m.mapPosition([0.5, 0.5], 0);
    for (const v of p) expect(Number.isFinite(v)).toBe(true);
    expectInWorkspace(p);
  });

  it("recenters the depth window on calibration", () => {
    const m = new HandToRobotMapper();
    m.calibrate(1.0, identity3());
    const xSmall = m.mapPosition([0.5, 0.5], 1.0)[0];
    m.calibrate(3.0, identity3());
    const xBig = m.mapPosition([0.5, 0.5], 3.0)[0];
    expect(xSmall).toBeCloseTo(xBig, 9);
    expect(xSmall).toBeGreaterThan(WORKSPACE.x[0]);
    expect(xSmall).toBeLessThan(WORKSPACE.x[1]);
  });
});

describe("clutch rebasing", () => {
  it("makes the current hand pose map onto the frozen target", () => {
    const m = new HandToRobotMapper();
    m.calibrate(2.0, identity3());
    const frozen = m.mapPosition([0.5, 0.5], 2.0);
    // Hand moved far away while clutched, then released.
    m.rebasePosition([0.8, 0.3], 2.6, frozen);
    const resumed = m.mapPosition([0.8, 0.3], 2.6);
    for (let i = 0; i < 3; i++) expect(resumed[i]).toBeCloseTo(frozen[i], 9);
    // Relative motion still works from the new anchor, in the same sense.
    expect(m.mapPosition([0.85, 0.3], 2.6)[1]).toBeGreaterThan(frozen[1]);
    expectInWorkspace(m.mapPosition([0.2, 0.9], 1.0));
  });

  it("clears the offset on calibration", () => {
    const m = new HandToRobotMapper();
    m.calibrate(2.0, identity3());
    m.rebasePosition([0.9, 0.9], 2.0, [0.4, 0, 0.3]);
    expect(m.posOffset.some((v) => Math.abs(v) > 1e-9)).toBe(true);
    m.calibrate(2.0, identity3());
    expect(m.posOffset).toEqual([0, 0, 0]);
  });

  it("re-anchors orientation so the current palm maps to the frozen wrist pose", () => {
    const m = new HandToRobotMapper();
    m.calibrate(2.0, identity3());
    const frozen = m.mapOrientation(rotY(0.4));
    const palmNow = matMul3(rotZ(0.9), rotX(-0.5));
    m.rebaseOrientation(palmNow, frozen);
    expect(quatAngleBetween(m.mapOrientation(palmNow), frozen)).toBeLessThan(1e-9);
    // Further palm rotation still moves the wrist away from the anchor.
    const later = m.mapOrientation(matMul3(rotX(0.3), palmNow));
    expect(quatAngleBetween(later, frozen)).toBeGreaterThan(0.2);
  });
});

describe("orientation mapping", () => {
  it("returns tool-down before calibration and at the neutral pose", () => {
    const m = new HandToRobotMapper();
    expect(quatAngleBetween(m.mapOrientation(identity3()), TOOL_DOWN_QUAT)).toBeLessThan(1e-9);
    const R = rotY(0.7);
    m.calibrate(0.2, R);
    expect(quatAngleBetween(m.mapOrientation(R), TOOL_DOWN_QUAT)).toBeLessThan(1e-9);
  });

  it("mirrors roll sense and preserves pitch, matching the mirrored view", () => {
    const m = new HandToRobotMapper();
    m.calibrate(0.2, identity3());
    const toolDown = matrixFromQuat(TOOL_DOWN_QUAT);
    // Roll about camera z -> rotation about robot +x with mirrored sense.
    const roll = m.mapOrientation(rotZ(0.3));
    const rollExpected = quatFromMatrix(matMul3(rotX(-0.3), toolDown));
    expect(quatAngleBetween(roll, rollExpected)).toBeLessThan(1e-9);
    // Pitch (toward/away tilt about camera x) is unchanged by the mirror.
    const pitch = m.mapOrientation(rotX(0.4));
    const pitchExpected = quatFromMatrix(matMul3(rotY(-0.4), toolDown));
    expect(quatAngleBetween(pitch, pitchExpected)).toBeLessThan(1e-9);
  });

  it("preserves rotation magnitude and clamps extreme tilt", () => {
    const m = new HandToRobotMapper();
    m.calibrate(0.2, identity3());
    const q = m.mapOrientation(rotY(0.5));
    expect(quatAngleBetween(q, TOOL_DOWN_QUAT)).toBeCloseTo(0.5, 6);
    // 180-degree flip must clamp to maxTiltRad.
    const qFlip = m.mapOrientation(rotY(Math.PI));
    expect(quatAngleBetween(qFlip, TOOL_DOWN_QUAT)).toBeLessThanOrEqual(1.2 + 1e-6);
  });
});

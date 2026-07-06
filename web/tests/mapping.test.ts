import { describe, expect, it } from "vitest";

import { WORKSPACE } from "../src/config";
import { CAM_TO_ROBOT, HandToRobotMapper, TOOL_DOWN_QUAT } from "../src/mapping";
import {
  identity3,
  matMul3,
  matTranspose3,
  quatAngleBetween,
  rotY,
} from "../src/transforms";

describe("CAM_TO_ROBOT", () => {
  it("is a proper rotation", () => {
    const shouldBeI = matMul3(CAM_TO_ROBOT, matTranspose3(CAM_TO_ROBOT));
    const I = identity3();
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) expect(shouldBeI[i][j]).toBeCloseTo(I[i][j], 9);
  });
});

describe("position mapping", () => {
  it("stays inside the workspace for any input", () => {
    const m = new HandToRobotMapper();
    for (let i = 0; i < 200; i++) {
      const xy: [number, number] = [Math.sin(i) * 0.8 + 0.5, Math.cos(i * 1.3) * 0.8 + 0.5];
      const p = m.mapPosition(xy, 0.02 + ((i * 37) % 100) / 200);
      expect(p[0]).toBeGreaterThanOrEqual(WORKSPACE.x[0]);
      expect(p[0]).toBeLessThanOrEqual(WORKSPACE.x[1]);
      expect(p[1]).toBeGreaterThanOrEqual(WORKSPACE.y[0]);
      expect(p[1]).toBeLessThanOrEqual(WORKSPACE.y[1]);
      expect(p[2]).toBeGreaterThanOrEqual(WORKSPACE.z[0]);
      expect(p[2]).toBeLessThanOrEqual(WORKSPACE.z[1]);
    }
  });

  it("maps directions like the Python implementation", () => {
    const m = new HandToRobotMapper();
    const p0 = m.mapPosition([0.5, 0.5], 0.19);
    expect(m.mapPosition([0.7, 0.5], 0.19)[1]).toBeLessThan(p0[1]); // right -> -y
    expect(m.mapPosition([0.5, 0.3], 0.19)[2]).toBeGreaterThan(p0[2]); // up -> +z
    expect(m.mapPosition([0.5, 0.5], 0.25)[0]).toBeLessThan(p0[0]); // closer -> retract
  });

  it("recenters the depth window on calibration", () => {
    const m = new HandToRobotMapper();
    m.calibrate(0.1, identity3());
    const xSmall = m.mapPosition([0.5, 0.5], 0.1)[0];
    m.calibrate(0.3, identity3());
    const xBig = m.mapPosition([0.5, 0.5], 0.3)[0];
    expect(xSmall).toBeCloseTo(xBig, 9);
    expect(xSmall).toBeGreaterThan(WORKSPACE.x[0]);
    expect(xSmall).toBeLessThan(WORKSPACE.x[1]);
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

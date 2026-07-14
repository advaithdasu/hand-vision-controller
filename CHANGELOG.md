# Changelog

All notable changes to this project are documented in this file.

## [0.1.0.0] - 2026-07-14

### Added
- Browser demo (`web/`): drive the 6-DOF robot arm with your hand entirely
  in the browser — webcam hand tracking (MediaPipe), damped-least-squares IK,
  and Rapier physics with pick-and-place cubes, no install required.
- TypeScript port of the math core (forward kinematics, DLS IK, camera→robot
  mapping, One Euro filtering, pose features) with a vitest suite mirroring
  the Python tests, plus new controller and quaternion-math coverage
  (36 tests).
- Landing screen with camera-permission flow, live HUD (fps, detect/tick
  timing, IK convergence), hand-skeleton overlay, and Calibrate /
  Orientation / Freeze / Reset controls with keyboard shortcuts.

### Changed
- Hand movement is no longer mirrored: moving your hand right moves the arm
  right on screen, and wrist roll/yaw follow the same on-screen sense, so
  the arm tracks what you see in the camera preview.
- The hand-skeleton overlay now lines up with your hand across the whole
  frame (object-fit crop accounted for) and renders sharply on high-DPI
  displays.

### Fixed
- The camera is released and the app returns to the start screen if startup
  fails, the camera is unplugged, or the session crashes — previously the
  webcam stayed on with no way to recover short of a reload.
- A stalled camera no longer keeps steering the arm with the last seen hand
  pose; the arm now holds position when frames stop arriving.
- Reset now reopens the gripper and homes the arm immediately, even while
  the hand is out of frame.
- Cmd/Ctrl shortcuts (copy, find) no longer trigger the demo hotkeys.
- Degenerate hand detections can no longer feed NaN targets into the IK
  solver, and dropped cubes no longer inherit spin from the wrist.

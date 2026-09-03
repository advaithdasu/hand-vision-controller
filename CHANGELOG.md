# Changelog

All notable changes to this project are documented in this file.

## [0.2.0.0] - 2026-09-03

Control-quality release: the arm tracks the hand with less lag and fewer
false gestures, and the demo is ready to host as a static site.

### Added
- Clutch that actually clutches: releasing a fist (or the Freeze button)
  re-anchors the position and orientation maps, so the arm resumes from
  where it froze instead of swinging to the hand's new absolute spot.
  Toggling orientation back on re-anchors the wrist the same way.
- Calibration lock-on: auto-calibration waits until the hand is fully in
  frame and still for half a second (progress bar in the HUD) instead of
  firing on the first detection at the frame edge. The Calibrate button
  and `c` key act immediately on the last observation, frozen or not.
- Gesture debouncing: a fist must persist ~120 ms to clutch and an open
  hand ~80 ms to release, so one misdetected frame cannot freeze the arm
  or drop it back into tracking.
- Speed-adaptive (One Euro) filtering on orientation as well as position.
- Browser demo: cubes glow when the open gripper is within grasp range, a
  floor ring and drop line under the tool give a depth cue, and the HUD
  hints what to do next (lock on, pinch, release).
- Browser demo: `requestVideoFrameCallback` frame pacing, WebGL
  context-loss recovery, a Content-Security-Policy that pins every
  resource to the page's own origin, and a "no camera found" message.
- GitHub Actions CI (vitest + build, pytest); `BASE_PATH` build variable
  for hosting the demo under a subpath.
- Tests for all of the above on both sides (68 Python, 53 vitest).

### Changed
- Reach depth now comes from an orientation-invariant size estimate (the
  ratio of image-plane to world-space palm extent) instead of the raw
  wrist-to-knuckle image length, so tilting the palm no longer reads as
  reaching. Both implementations share the estimator.
- Fist detection ignores the thumb: a thumb tucked over the fingers used
  to read as a pinch and close the gripper instead of clutching.
- Position filter retuned: the One Euro `beta` was so small the filter was
  a fixed 1.2 Hz low-pass (~160 ms lag at any speed); it now opens to
  ~3.5 Hz during a 1 m/s sweep. Joint velocity limit raised to 3.5 rad/s.
- Browser demo: held cubes stay dynamic and are servoed to the tool frame
  every physics substep, so they respect contacts (no more tunneling
  through tray walls) and no longer lag the gripper by a frame. Dropped
  cubes keep a capped velocity and no spin.
- Browser demo: the engine (Three.js, Rapier, MediaPipe) is loaded on
  demand; the landing page is an 8 kB entry chunk instead of 1 MB.
- Browser demo: the HUD is built from text nodes instead of `innerHTML`.
- Project brief moved to `docs/brief.md`.

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

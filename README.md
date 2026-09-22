# Hand-Controlled Robot Arm

Real-time teleoperation of a simulated 6-DOF robot arm using nothing but a laptop webcam. Your hand is the controller: move it in the air and the arm follows, pinch to grip, make a fist to clutch. No hardware required.

<p align="center">
  <img src="docs/sim_scene.png" width="420" alt="Simulated 6-DOF arm hovering over a cube, gripper open"/>
</p>

## What this is

Controlling a robot arm intuitively is a hard interface problem — joysticks, teach pendants, and gantry controls all put a translation layer between the operator's intent and the robot's motion. This project puts the human hand directly in the loop: a webcam sees the hand, a vision pipeline turns it into a 6-DOF pose, and a numerical IK solver drives a physically simulated arm to match it, at interactive framerates, on commodity hardware. It's the software core of what powers teleoperated surgical robots, prosthetic control, and remote manipulation — perception, mapping, and control, decoupled from any particular plant.

The system exists twice: a Python/MuJoCo reference implementation, and a from-scratch TypeScript/Three.js/Rapier port that runs entirely client-side in the browser, sharing the same math and the same design decisions.

## How it works

```
webcam ──▶ hand pose estimation ──▶ pose-to-command mapping ──▶ IK solver ──▶ physics sim
 30 fps     MediaPipe HandLandmarker    camera → robot space      damped        MuJoCo
            21 keypoints + world 3D     One Euro filtering        least squares  500 Hz
                    │                                                              │
                    └────────────── unified split-screen display ◀────────────────┘
                                    skeleton overlay + latency HUD
```

1. **Hand pose estimation** — MediaPipe's HandLandmarker extracts 21 keypoints per frame in both normalized image coordinates and metric hand-centered 3D. Pure-numpy geometry (`pose_features.py`, ported to `poseFeatures.ts`) turns those into a palm orientation frame (forward/lateral/normal from three knuckle vectors), a pinch ratio, and per-finger curl ratios — kept free of any MediaPipe dependency, so the geometry is unit-testable against synthetic hands.

2. **Pose-to-command mapping** — palm position maps to the arm's lateral/vertical target; palm rotation *relative to a calibrated reference* maps to end-effector orientation, clamped to a safe tilt cone; apparent hand size maps to reach depth. Apparent size is the tricky part: the naive wrist-to-knuckle image length shrinks when the palm just *tilts* toward the camera, which reads as a reach that never happened. The fix is a ratio of the image-plane palm extent to the metric world-space extent MediaPipe also returns, projected onto the same plane — that ratio cancels foreshortening and leaves something proportional to true distance. Everything is a *relative* map anchored at calibration, so the operator can recenter the working volume at any time, and a clutch (fist) freezes the arm while the maps re-anchor on release so it resumes from where it froze instead of snapping to the hand's new position.

3. **Filtering** — One Euro filters (Casiez et al., 2012) smooth the position and orientation targets: heavy smoothing at rest kills jitter, and the cutoff opens with signal speed during fast moves to keep lag low without a fixed trade-off. Gestures are time-debounced (~120 ms to clutch, ~80 ms to release) so a single misdetected frame can't freeze or drop the arm, and calibration itself waits for the hand to be fully in frame and still for half a second rather than locking onto a frame-edge detection.

4. **Inverse kinematics** — a damped-least-squares (Levenberg–Marquardt) solver computes the six joint angles for each target pose from the 6×6 geometric Jacobian, every frame. See [below](#the-interesting-part-the-ik-solver).

5. **Physics** — a hand-authored 6R arm (yaw/pitch/pitch/roll/pitch/roll wrist cluster) with a parallel-jaw gripper runs in MuJoCo at 500 Hz with position-servo actuators and gravity compensation; the browser port swaps in Rapier, running the same actuation model in WASM. The scene includes two cubes and a tray for pick-and-place.

6. **Display** — one window: webcam feed with a skeleton overlay on the left, the simulation on the right, with a HUD showing FPS, per-stage latency, IK convergence, and gripper state.

## Engineering notes

**Position is mapped from the robot's own point of view, not the screen's.** With the arm's neutral reach along +x and up along +z, its own right is −y (forward × up); hand-right maps to −y, so moving your hand left moves the arm to *its* own left — like driving from behind it, not watching a mirror. Wrist orientation deliberately diverges from that: roll/yaw follow the *mirrored* camera preview the operator is actually watching, via an intentionally improper (det = −1) camera→robot axis map, because that's the one thing operators reliably fight when it doesn't match. The Python app keeps the opposite orientation convention throughout, since the MuJoCo viewer sits on the other side of the scene. Both apps agree that reaching toward the camera extends the arm — the browser's scene camera sits out front, so an extending tool looms larger on screen just as the hand does in the mirrored preview.

**No camera, no problem.** An autopilot feeds scripted Cartesian targets through the exact same controller, IK solver, and physics the hand normally drives — not a canned animation — so a visitor who won't grant camera access still watches a real pick-and-place with a real friction grasp. "Take over" switches to live control mid-run without reloading anything.

**The browser port had to earn its keep as a zero-install demo.** The landing page is an 8 kB shell; Three.js, Rapier, and MediaPipe's ~20 MB of wasm and model weights load only after the visitor presses start (with byte-level download progress), and hovering a start button prefetches the engine so the click feels instant. The page ships a same-origin Content-Security-Policy that pins every resource to its own origin, so "video never leaves your machine" is enforced by the browser rather than just promised. `requestVideoFrameCallback` paces detection to actual new frames instead of the render loop, and a `webglcontextlost` handler rebuilds the engine instead of leaving the demo dead after a GPU driver hiccup. Cubes held by the gripper are re-servoed to the tool frame every physics substep rather than parented to it, so they keep respecting contacts (no tunneling through tray walls) instead of lagging a frame behind.

## The interesting part: the IK solver

6-DOF IK has no closed-form solution path that handles limits and singularities for free; this project solves it numerically every frame:

- **Damped least squares**: `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`, with the 6×6 geometric Jacobian computed from a scratch kinematic model so live sim state is never touched by solver iterations.
- **Singularity robustness**: manipulability `w = √det(JJᵀ)` is monitored on the raw Jacobian; as the arm approaches a singular configuration (e.g., full extension), extra damping ramps in smoothly, trading a little tracking error for bounded joint velocities instead of the wild spins a plain pseudo-inverse produces.
- **Joint limits**: every iterate is clamped to the model's joint ranges, so the solver only explores feasible space.
- **Continuity**: per-frame solves warm-start from the current configuration with clamped task-space error and step size, so the arm never snaps between IK branches. Cold-start solves add an aimed-base-yaw seed plus random restarts for global coverage.

## Latency

Measured on an M-series MacBook (a budget test enforces this in CI):

| Stage | Time |
|---|---|
| Hand detection (MediaPipe, VIDEO mode) | ~10–15 ms |
| Mapping + filtering + IK solve | ~0.6 ms |
| Physics (1/30 s of 500 Hz MuJoCo) | ~0.2 ms |
| Sim render (640 px offscreen) | ~2 ms |

End-to-end motion-to-motion latency is dominated by camera exposure and detection; the control side is effectively free. The HUD displays live per-stage numbers.

## From simulation to hardware

The architecture keeps the perception → mapping → IK stack independent of the plant. `ArmSim` exposes exactly two commands — `set_joint_targets(q)` and `set_gripper(opening)` — so swapping in a real arm means implementing that same interface over a serial/CAN/ROS bridge and tuning the joint limits in one config file. This is the software stack of a real teleoperation system, running in simulation.

## Project layout

```
handarm/
  app.py            real-time loop, HUD, record/replay, autopilot loop
  autopilot.py      scripted pick-and-place (no camera)
  hand_tracker.py   MediaPipe HandLandmarker wrapper
  pose_features.py  palm frame, pinch, finger curl (pure numpy)
  mapping.py        camera-space → robot-space, calibration
  filters.py        One Euro, quaternion low-pass
  ik.py             damped least squares solver
  kinematics.py     FK / Jacobians (MuJoCo scratch model)
  sim.py            MuJoCo world: arm, gripper, cubes, tray
  overlay.py        skeleton overlay, HUD, split-screen compositor
  latency.py        rolling per-stage profiler
  recorder.py       JSONL trajectory record / replay
assets/scene.xml    6-DOF arm + gripper + scene (MJCF)
web/
  src/              browser port: TS math core, Three.js scene, Rapier
                     physics, MediaPipe hand tracking
  public/           vendored MediaPipe wasm + hand-landmarker model
```

Release notes live in [CHANGELOG.md](CHANGELOG.md); known gaps and planned work in [TODOS.md](TODOS.md); the original project brief in [docs/brief.md](docs/brief.md).

## Running it

```bash
# Python + MuJoCo (3.10-3.12)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m handarm                 # --autopilot for no-camera mode, --replay <file> to play back a recording

# Browser
cd web && pnpm install && pnpm dev
```

Both suites (71 Python, 70 vitest) run in CI on every push.

## License

[MIT](LICENSE)

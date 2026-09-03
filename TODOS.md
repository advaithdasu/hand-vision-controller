# TODOS

## Web Demo

### Record the demo video

**What:** Screen-capture the browser demo (split view: hand + arm) picking up a cube and dropping it in the tray, and embed it at the top of the README in place of the placeholder.

**Why:** The README and the brief both call the video the centerpiece; a portfolio visitor who won't grant camera access needs to see the demo work.

**Effort:** S
**Priority:** P1

### Gripper finger colliders

**What:** Give the two fingers and the palm kinematic Rapier colliders so the arm can nudge cubes it isn't holding, with collision groups excluding the currently held cube.

**Why:** Today the arm passes through unheld cubes; bumping them around would make the scene feel physical.

**Context:** `web/src/armScene.ts`. Deliberately deferred: kinematic fingers pressing a resting cube into the floor produce ugly sink/pop artifacts unless the contact is tuned, and that needs a live WebGL session to iterate on.

**Effort:** M
**Priority:** P3

### Reuse scratch buffers in the IK hot loop

**What:** Preallocate matrices/vectors reused across `solveIK` iterations and add a det-only elimination that skips back-substitution.

**Why:** Each 60 Hz frame allocates thousands of short-lived arrays (fk results, 6×6 products, augmented matrices), inviting periodic GC jank in a real-time loop.

**Context:** `web/src/ik.ts` (`descend`, `matMulT`, `solveLinear6`), `web/src/kinematics.ts` (`fk`).

**Effort:** M
**Priority:** P3

### Single source of truth for the accent color

**What:** Derive the `#f27317` accent used in `style.css`, `armScene.ts` (`COLORS.orange`), and `main.ts` (`ACCENT`) from one definition.

**Why:** A rebrand currently requires three synchronized edits in two formats.

**Effort:** S
**Priority:** P4

## Python App

### Replay recordings in the browser

**What:** Accept a `recordings/*.jsonl` file (drag-and-drop) in the browser demo and play the joint trajectory back on the Three.js arm.

**Why:** Lets the portfolio page show a canned manipulation for visitors without a webcam.

**Effort:** M
**Priority:** P3

## Completed

- **Rebase the clutch instead of using absolute mapping** (v0.2.0.0): position and orientation maps are re-anchored on fist release, manual unfreeze, and orientation re-enable.
- **Distinguish a thumb-tucked fist from a pinch** (v0.2.0.0): fist detection ignores the thumb; the clutch and gripper are debounced.
- **Calibration quality gate** (v0.2.0.0): hold-still auto-calibration with HUD progress; the Calibrate button acts immediately.
- **Lazy-load the engine bundle** (v0.2.0.0): 8 kB entry chunk, engine chunk loaded on start.
- **Held-cube kinematic target ordering and tunneling** (v0.2.0.0): held cubes are dynamic and velocity-servoed per substep with interpolated targets.
- **Harden frame delivery and rendering paths** (v0.2.0.0): `requestVideoFrameCallback`, `webglcontextlost` handling with engine rebuild.
- **Add a Content-Security-Policy** (v0.2.0.0): same-origin CSP in `index.html`; HUD built from text nodes.

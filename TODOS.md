# TODOS

## Web Demo

### Rebase the clutch instead of using absolute mapping

**What:** Make the fist-clutch actually reposition: capture a position offset on clutch release so the arm resumes from where it froze instead of snapping to the hand's new absolute position.

**Why:** The landing copy advertises "Fist — clutch: freeze the arm to reposition your hand", but `mapPosition` is absolute, so on release the arm swings back to wherever the hand now maps — repositioning achieves nothing and the transit is a fast un-commanded move.

**Context:** Found by the pre-ship adversarial review (v0.1.0.0). Either add an offset rebase in `web/src/mapping.ts` (and mirror it in `handarm/mapping.py`, which has the same semantics), or change the copy/UX intent. The position filter is also not reset on clutch release (`web/src/app.ts`), which softens but does not remove the lurch.

**Effort:** M
**Priority:** P1

### Distinguish a thumb-tucked fist from a pinch

**What:** Make `isFistFrom` recognize a fist with the thumb wrapped over the fingers, which currently reads as a pinch (small thumb–index distance) and closes the gripper instead of clutching.

**Why:** The user's "freeze" gesture can silently command a grip close — worst case it grabs or drops a cube.

**Context:** `web/src/poseFeatures.ts` (`isFistFrom` requires `pinch > 0.5`); inherited from `handarm/pose_features.py`. Needs gesture-design judgment: e.g. gate the fist check on finger extensions only, or use thumb-tip-to-palm distance rather than thumb–index.

**Effort:** M
**Priority:** P2

### Lazy-load the engine bundle

**What:** Dynamic-import Three.js/Rapier/MediaPipe in `start()` so the landing shell is a small entry chunk.

**Why:** The single 2.9 MB (1 MB gzip) bundle downloads, parses, and compiles before the landing page is interactive, even though none of it runs until the user clicks start.

**Context:** `web/src/main.ts` statically imports `armScene`/`handTracker`; `vite.config.ts` raises `chunkSizeWarningLimit` to accommodate. `enginePromise` already isolates engine creation, so the import can move inside it.

**Effort:** S
**Priority:** P2

### Calibration quality gate

**What:** Calibrate synchronously from the last observation when the button is pressed, and gate auto/first-frame calibration on a few stable tracked frames.

**Why:** Auto-calibration fires on the very first detection (often while the hand is entering the frame edge), yielding a bad neutral pose; pressing Calibrate while clutched/frozen silently defers to an arbitrary later frame.

**Context:** `web/src/main.ts` (`recalibrate`), `web/src/app.ts` (`processHand` calls `calibrate` only when not frozen). Python `handarm/app.py` calibrates immediately from the current observation.

**Effort:** M
**Priority:** P2

### Reuse scratch buffers in the IK hot loop

**What:** Preallocate matrices/vectors reused across `solveIK` iterations and add a det-only elimination that skips back-substitution.

**Why:** Each 60 Hz frame allocates thousands of short-lived arrays (fk results, 6×6 products, augmented matrices), inviting periodic GC jank in a real-time loop.

**Context:** `web/src/ik.ts` (`descend`, `matMulT`, `solveLinear6`), `web/src/kinematics.ts` (`fk`). Also trim the 21-landmark map in `app.ts handScaleCorrected` (only 2 landmarks are read).

**Effort:** M
**Priority:** P3

### Held-cube kinematic target ordering and tunneling

**What:** Set the held cube's kinematic target from the current FK before stepping the physics world (interpolated per fixed substep), and consider clamping the held pose above the floor.

**Why:** The body lags the rendered pose by one frame, catch-up frames absorb the whole jump in one substep, and a kinematically held cube can be driven through tray walls, ejecting the other cube.

**Context:** `web/src/armScene.ts` `step()`/`updateGrasp()`. The file header documents kinematic grasping as a deliberate browser-demo tradeoff; this is about softening its visible artifacts.

**Effort:** M
**Priority:** P3

### Harden frame delivery and rendering paths

**What:** Use `requestVideoFrameCallback` (fallback to the current `currentTime` compare) for new-frame detection, and handle `webglcontextlost` on both canvases.

**Why:** `currentTime` is quantized on some browsers (missed/duplicate detections), and a lost WebGL context currently surfaces only via the generic loop crash handler.

**Context:** `web/src/main.ts` loop; `web/src/armScene.ts` renderer.

**Effort:** S
**Priority:** P3

### Add a Content-Security-Policy

**What:** Serve/meta a CSP (`script-src 'self'; connect-src 'self'` plus `wasm-unsafe-eval` as needed) and build the HUD from `textContent` spans instead of `innerHTML`.

**Why:** Enforces the "video never leaves your machine" privacy claim at the platform level and removes the fragile innerHTML pattern before any externally influenced string ever reaches the HUD.

**Context:** `web/index.html`, `web/src/main.ts updateHud`. All assets are already same-origin (vendored MediaPipe wasm/models under `web/public/`).

**Effort:** S
**Priority:** P3

### Single source of truth for the accent color

**What:** Derive the `#f27317` accent used in `style.css`, `armScene.ts` (`COLORS.orange`), and `main.ts` (skeleton dots) from one definition.

**Why:** A rebrand currently requires three synchronized edits in two formats.

**Effort:** S
**Priority:** P4

## Completed

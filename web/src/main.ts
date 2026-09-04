/**
 * Bootstrap and render loop.
 *
 * Two ways to drive the arm share one loop: the camera (hand -> tracker ->
 * controller) and the autopilot (scripted targets -> controller), both
 * ending in the same IK -> command -> physics -> render path.
 */

import { TeleopController } from "./app";
import type { ArmScene } from "./armScene";
import type { Autopilot } from "./autopilot";
import type { HandObservation, HandTracker, ProgressFn } from "./handTracker";
import { HAND_CONNECTIONS, type Landmarks } from "./poseFeatures";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const simCanvas = $<HTMLCanvasElement>("sim");
const hud = $<HTMLDivElement>("hud");
const landing = $<HTMLDivElement>("landing");
const landingStatus = $<HTMLParagraphElement>("landing-status");
const landingProgress = $<HTMLProgressElement>("landing-progress");
const startBtn = $<HTMLButtonElement>("start");
const autopilotBtn = $<HTMLButtonElement>("btn-autopilot");
const camPlaceholder = $<HTMLDivElement>("cam-placeholder");
const takeoverBtn = $<HTMLButtonElement>("btn-takeover");
const takeoverStatus = $<HTMLParagraphElement>("takeover-status");
const btnCal = $<HTMLButtonElement>("btn-calibrate");
const btnOri = $<HTMLButtonElement>("btn-orientation");
const btnFreeze = $<HTMLButtonElement>("btn-freeze");
const btnReset = $<HTMLButtonElement>("btn-reset");

class RollingMean {
  private buf: number[] = [];
  constructor(private n = 60) {}
  push(v: number): void {
    this.buf.push(v);
    if (this.buf.length > this.n) this.buf.shift();
  }
  mean(): number {
    return this.buf.length
      ? this.buf.reduce((s, v) => s + v, 0) / this.buf.length
      : 0;
  }
}

const detectMs = new RollingMean();
const tickMs = new RollingMean();
const frameMs = new RollingMean();

// ---------------- engine loading ----------------

/**
 * Heavy modules are loaded on demand (the landing shell stays a small
 * entry chunk) and created once, then reused across sessions: a second
 * ArmScene would attach a second WebGLRenderer to the same canvas and
 * leak the Rapier world; a second HandLandmarker leaks its predecessor.
 * The scene and the tracker load separately so the autopilot never pays
 * for the ~20 MB of hand-tracking wasm + model it doesn't use.
 */
let scenePromise: Promise<ArmScene> | null = null;
let trackerPromise: Promise<HandTracker> | null = null;
/** Whoever is currently showing a loading UI receives model progress. */
let progressListener: ProgressFn | null = null;

function loadScene(): Promise<ArmScene> {
  scenePromise ??= import("./armScene")
    .then(({ ArmScene }) => ArmScene.create(simCanvas))
    .catch((e) => {
      scenePromise = null; // allow a clean retry
      throw e;
    });
  return scenePromise;
}

function loadTracker(): Promise<HandTracker> {
  trackerPromise ??= import("./handTracker")
    .then(({ HandTracker }) => HandTracker.create((l, t) => progressListener?.(l, t)))
    .catch((e) => {
      trackerPromise = null;
      throw e;
    });
  return trackerPromise;
}

/** Tear the engine down so the next start builds a fresh one. */
function discardEngine(): void {
  const s = scenePromise;
  const t = trackerPromise;
  scenePromise = null;
  trackerPromise = null;
  s?.then((scene) => scene.dispose()).catch(() => {});
  t?.then((tracker) => tracker.close()).catch(() => {});
}

// ---------------- sessions ----------------

type Session =
  | { mode: "hand"; ctl: TeleopController; scene: ArmScene; stream: MediaStream }
  | { mode: "autopilot"; ctl: TeleopController; scene: ArmScene; autopilot: Autopilot };

/** The running session, if any — event handlers wired once refer to this. */
let active: Session | null = null;

function endSession(): void {
  if (active?.mode === "hand") active.stream.getTracks().forEach((t) => t.stop());
  active = null;
  video.srcObject = null;
  camPlaceholder.hidden = true;
  takeoverStatus.textContent = "";
  overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
}

/** Stop everything and drop back to the landing card with a message. */
function stopToLanding(message: string): void {
  endSession();
  landing.classList.remove("hidden");
  setLoading(message, null);
  startBtn.disabled = false;
  autopilotBtn.disabled = false;
  for (const b of [btnCal, btnOri, btnFreeze, btnReset]) b.disabled = true;
  btnFreeze.classList.remove("active");
}

function setLoading(text: string, frac: number | null): void {
  landingStatus.textContent = text;
  landingProgress.hidden = frac === null;
  if (frac !== null) landingProgress.value = frac;
}

function enterSession(s: Session): void {
  endSession();
  active = s;
  landing.classList.add("hidden");
  setLoading("", null);
  startBtn.disabled = false;
  autopilotBtn.disabled = false;
  camPlaceholder.hidden = s.mode !== "autopilot";
  // Calibration and orientation only mean something with a hand.
  btnCal.disabled = btnOri.disabled = s.mode !== "hand";
  btnFreeze.disabled = btnReset.disabled = false;
  btnFreeze.classList.remove("active");
  btnOri.textContent = `Orientation: ${s.ctl.orientationOn ? "on" : "off"}`;
}

// ---------------- drawing ----------------

/** Cap the backing store like the Three.js renderer does. */
const dpr = (): number => Math.min(window.devicePixelRatio || 1, 2);

/**
 * Size the overlay to the panel (in device pixels) only when it changed:
 * assigning canvas width/height clears the bitmap and resets context state
 * even when the value is identical, so an unconditional per-frame write
 * costs a backing-store clear plus a forced layout.
 */
function sizeOverlay(panelW: number, panelH: number): void {
  const w = Math.round(panelW * dpr());
  const h = Math.round(panelH * dpr());
  if (overlay.width !== w) overlay.width = w;
  if (overlay.height !== h) overlay.height = h;
}

/** Skeleton keypoints use the page accent (style.css --accent). */
const ACCENT =
  getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#f27317";
const SKELETON = "rgba(80, 200, 255, 0.9)";
const SKELETON_FROZEN = "rgba(255, 92, 92, 0.9)";

function drawSkeleton(lm: Landmarks, frozen: boolean): void {
  const ctx = overlay.getContext("2d")!;
  const { width: w, height: h } = overlay;
  ctx.clearRect(0, 0, w, h);
  // The video fills the panel with object-fit: cover, so the rendered
  // frame is scaled by max(w/videoW, h/videoH) and center-cropped; project
  // the normalized landmarks through that same transform or the skeleton
  // drifts off the hand toward the panel edges.
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const scale = Math.max(w / vw, h / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const offX = (dispW - w) / 2;
  const offY = (dispH - h) / 2;
  const px = (p: readonly number[]): [number, number] => [
    p[0] * dispW - offX,
    p[1] * dispH - offY,
  ];
  ctx.lineWidth = 2.5 * dpr();
  ctx.strokeStyle = frozen ? SKELETON_FROZEN : SKELETON;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(...px(lm[a]));
    ctx.lineTo(...px(lm[b]));
    ctx.stroke();
  }
  ctx.fillStyle = ACCENT;
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(...px(p), 4 * dpr(), 0, Math.PI * 2);
    ctx.fill();
  }
}

/** One HUD line is a list of text runs, optionally styled. */
type Run = { t: string; c?: "warn" | "good" | "muted" };

let lastHudKey = "";

/**
 * Render the HUD from text runs (no innerHTML: nothing here is markup,
 * and this keeps the page CSP-clean). Skipped when nothing changed.
 */
function renderHud(lines: Run[][]): void {
  const key = JSON.stringify(lines);
  if (key === lastHudKey) return;
  lastHudKey = key;
  const frag = document.createDocumentFragment();
  lines.forEach((runs, i) => {
    if (i > 0) frag.append("\n");
    for (const r of runs) {
      if (!r.c) {
        frag.append(r.t);
        continue;
      }
      const span = document.createElement("span");
      span.className = r.c;
      span.textContent = r.t;
      frag.append(span);
    }
  });
  hud.replaceChildren(frag);
}

function progressBar(frac: number, width = 10): string {
  const filled = Math.round(Math.min(Math.max(frac, 0), 1) * width);
  return "▮".repeat(filled) + "▯".repeat(width - filled);
}

function updateHud(s: Session, handVisible: boolean): void {
  const { ctl, scene } = s;
  const ik = ctl.lastIK;
  const fps = frameMs.mean() > 0 ? 1000 / frameMs.mean() : 0;
  const mode: Run = ctl.manualFreeze
    ? { t: "FROZEN (button)", c: "warn" }
    : s.mode === "autopilot"
      ? { t: `AUTOPILOT — ${s.autopilot.status}`, c: "good" }
      : ctl.clutched
        ? { t: "CLUTCHED (fist) — open your hand to resume", c: "warn" }
        : { t: ctl.orientationOn ? "pos + orientation" : "position only" };
  const grip: Run = ctl.gripLatched
    ? { t: "LATCHED", c: "warn" }
    : ctl.gripping || (s.mode === "autopilot" && ctl.gripOpening < 0.5)
      ? { t: "CLOSED", c: "warn" }
      : { t: "open" };
  const timing = s.mode === "hand"
    ? `fps ${fps.toFixed(0).padStart(3)}   detect ${detectMs.mean().toFixed(1)} ms   tick ${tickMs.mean().toFixed(2)} ms`
    : `fps ${fps.toFixed(0).padStart(3)}   tick ${tickMs.mean().toFixed(2)} ms`;
  const lines: Run[][] = [[{ t: timing }], [{ t: "mode: " }, mode]];
  if (s.mode === "hand") {
    lines.push([
      { t: "hand: " },
      handVisible ? { t: "tracking", c: "good" } : { t: "NOT FOUND", c: "warn" },
    ]);
  }
  lines.push([
    { t: "grip: " }, grip, { t: `  aperture ${ctl.gripOpening.toFixed(2)}` },
    ...(scene.holding ? [{ t: "  ● holding cube", c: "good" as const }] : []),
  ]);
  if (ik) {
    lines.push([
      { t: `ik: ${ik.iters} it  ${(ik.posErr * 1000).toFixed(1)} mm / ${((ik.oriErr * 180) / Math.PI).toFixed(1)} deg  w ${ik.manipulability.toFixed(3)}` },
      ...(ik.converged ? [] : [{ t: " (!)", c: "warn" as const }]),
    ]);
  }
  if (s.mode === "autopilot") {
    lines.push([{ t: "same IK + physics as hand control — enable the camera to take over", c: "muted" }]);
  } else if (!ctl.calibrated) {
    lines.push([
      handVisible
        ? { t: `hold still to lock on  ${progressBar(ctl.calibProgress)}`, c: "warn" }
        : { t: "show your open hand to the camera", c: "warn" },
    ]);
  } else if (scene.holding) {
    lines.push([{ t: "open your hand to release", c: "muted" }]);
  } else if (scene.canGrasp) {
    lines.push([{ t: "cube in reach — pinch to grab", c: "good" }]);
  }
  renderHud(lines);
}

// ---------------- the loop ----------------

/**
 * Drive one session at display rate. `frame` produces this tick's hand
 * observation (null when none) after advancing the controller; the loop
 * owns timing, drawing, the HUD, and crash handling.
 */
function runLoop(
  s: Session,
  frame: (dt: number, now: number) => HandObservation | null | "stop",
): void {
  let tPrev = performance.now();
  const loop = (): void => {
    // Any uncaught throw here (GPU context loss, wasm fault) would
    // otherwise kill the rAF chain silently with the camera still live.
    try {
      if (active !== s) return; // superseded by another session
      const now = performance.now();
      frameMs.push(now - tPrev); // frame interval -> real display fps
      const dt = Math.min((now - tPrev) / 1000, 0.1);
      tPrev = now;

      const obs = frame(dt, now);
      if (obs === "stop") return;

      const camPanel = video.parentElement!;
      sizeOverlay(camPanel.clientWidth, camPanel.clientHeight);
      if (obs) drawSkeleton(obs.imageLandmarks, s.ctl.frozen);
      else overlay.getContext("2d")!.clearRect(0, 0, overlay.width, overlay.height);

      const simPanel = simCanvas.parentElement!;
      s.scene.render(simPanel.clientWidth, simPanel.clientHeight);

      updateHud(s, obs !== null);
      requestAnimationFrame(loop);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // A lost GPU context never comes back for this renderer; rebuild
      // the engine on the next start instead of reusing a dead one.
      if (/context lost/i.test(msg)) discardEngine();
      stopToLanding(`The session crashed (${msg}). Start again to retry.`);
    }
  };
  requestAnimationFrame(loop);
}

/** Treat the hand as lost if the camera stops delivering frames this long. */
const STALLED_FRAME_MS = 1500;
/** Rest with the cubes in the tray before the autopilot loops. */
const ROUND_DONE_PAUSE_MS = 2500;

function startupError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "Camera access was denied — enable it in your browser's site settings and try again.";
  }
  if (err instanceof DOMException && err.name === "NotFoundError") {
    return "No camera found. Plug one in (or allow camera access) and try again.";
  }
  if (/WebGL/i.test(msg)) {
    return "This browser has no WebGL support (required for the 3D simulation). Try Chrome, Edge, or Safari with hardware acceleration enabled.";
  }
  return `Failed to start: ${msg}`;
}

const mb = (bytes: number): string => (bytes / 1e6).toFixed(1);

/** Camera-driven session. Can take over from a running autopilot. */
async function start(): Promise<void> {
  startBtn.disabled = true;
  takeoverBtn.disabled = true;
  const fromAutopilot = active?.mode === "autopilot";
  // While the autopilot keeps running, report progress in its panel.
  const status = (text: string, frac: number | null): void => {
    if (fromAutopilot) takeoverStatus.textContent = text;
    else setLoading(text, frac);
  };
  let stream: MediaStream | null = null;
  try {
    status("Loading 3D engine…", null);
    const scene = await loadScene();
    status("Loading hand tracker…", null);
    progressListener = (loaded, total) =>
      status(
        total
          ? `Downloading hand model  ${mb(loaded)} / ${mb(total)} MB`
          : `Downloading hand model  ${mb(loaded)} MB`,
        total ? loaded / total : null,
      );
    const tracker = await loadTracker();
    progressListener = null;

    status("Requesting camera…", null);
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise<void>((res, rej) => {
      video.onloadedmetadata = () => res();
      // A granted stream that never delivers metadata (stalled virtual
      // camera) would otherwise hang the UI at "Requesting camera…" forever.
      setTimeout(() => rej(new Error("Camera stream never became ready.")), 10_000);
    });
    await video.play();

    // Device unplugged / permission revoked mid-session: stop pretending
    // the last observation is live.
    let cameraLost = false;
    stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      cameraLost = true;
    });

    if (fromAutopilot) scene.reset(); // hand the operator a fresh scene
    const ctl = new TeleopController(scene);
    ctl.frameAspect = video.videoWidth / video.videoHeight;
    const s: Session = { mode: "hand", ctl, scene, stream };
    enterSession(s);

    let lastObs: HandObservation | null = null;
    let lastFrameAt = performance.now();

    // New-frame detection: requestVideoFrameCallback fires exactly once
    // per delivered camera frame where supported; elsewhere fall back to
    // watching currentTime, which some browsers quantize coarsely.
    let newFrame = false;
    let lastVideoTime = -1;
    const hasRvfc = typeof video.requestVideoFrameCallback === "function";
    const onVideoFrame = (): void => {
      newFrame = true;
      if (active === s) video.requestVideoFrameCallback(onVideoFrame);
    };
    if (hasRvfc) video.requestVideoFrameCallback(onVideoFrame);

    runLoop(s, (dt, now) => {
      if (cameraLost) {
        stopToLanding("Camera disconnected — start again to reconnect.");
        return "stop";
      }
      if (!hasRvfc && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        newFrame = true;
      }
      // Detect only on new video frames; reuse the last observation between.
      if (newFrame && video.readyState >= 2) {
        newFrame = false;
        lastFrameAt = now;
        const t0 = performance.now();
        lastObs = tracker.detect(video, now);
        detectMs.push(performance.now() - t0);
      } else if (now - lastFrameAt > STALLED_FRAME_MS) {
        // No fresh frames: without this, a stalled camera keeps feeding the
        // arm the last observation forever and the hold failsafe never fires.
        lastObs = null;
      }
      const t1 = performance.now();
      ctl.tick(lastObs, dt);
      tickMs.push(performance.now() - t1);
      return lastObs;
    });
  } catch (err) {
    progressListener = null;
    // Release the camera on any failure, or the indicator light stays on
    // (and a retry would acquire a second stream) while the app looks dead.
    stream?.getTracks().forEach((t) => t.stop());
    if (active?.mode === "autopilot") {
      // Keep the show running; report the problem where the button is.
      takeoverStatus.textContent = startupError(err);
      startBtn.disabled = false;
    } else {
      stopToLanding(startupError(err));
    }
  } finally {
    takeoverBtn.disabled = false;
  }
}

/** Scripted pick-and-place, no camera. */
async function startAutopilot(): Promise<void> {
  autopilotBtn.disabled = true;
  try {
    setLoading("Loading 3D engine…", null);
    const [scene, { Autopilot }] = await Promise.all([loadScene(), import("./autopilot")]);
    scene.reset();
    const ctl = new TeleopController(scene);
    const autopilot = new Autopilot(ctl, scene);
    const s: Session = { mode: "autopilot", ctl, scene, autopilot };
    enterSession(s);

    let doneAt: number | null = null;
    runLoop(s, (dt, now) => {
      const t1 = performance.now();
      autopilot.tick(dt);
      tickMs.push(performance.now() - t1);
      if (autopilot.roundComplete) {
        doneAt ??= now;
        if (now - doneAt > ROUND_DONE_PAUSE_MS) {
          doneAt = null;
          resetScene();
        }
      }
      return null;
    });
  } catch (err) {
    stopToLanding(startupError(err));
  }
}

// ---------------- controls ----------------

function resetScene(): void {
  if (!active) return;
  const { ctl, scene } = active;
  scene.reset();
  ctl.resetControl();
  // Push the reset state to the scene immediately: while the lost-hand
  // hold is active the controller stops commanding, and without this the
  // cubes teleport home but the arm keeps its stale pose and grip.
  scene.setJointTargets(ctl.qCmd);
  scene.setGripper(ctl.gripOpening);
  scene.setTargetMarker(ctl.targetPos);
  if (active.mode === "autopilot") active.autopilot.restart();
  btnFreeze.classList.remove("active");
}

/**
 * Wired once at module load; handlers act on the current session via
 * `active`, so a restart doesn't stack duplicate listeners.
 */
function wireControls(): void {
  const toggleOrientation = (): void => {
    if (active?.mode !== "hand") return;
    active.ctl.orientationOn = !active.ctl.orientationOn;
    btnOri.textContent = `Orientation: ${active.ctl.orientationOn ? "on" : "off"}`;
  };
  const toggleFreeze = (): void => {
    if (!active) return;
    active.ctl.manualFreeze = !active.ctl.manualFreeze;
    btnFreeze.classList.toggle("active", active.ctl.manualFreeze);
  };
  const recalibrate = (): void => {
    if (active?.mode === "hand") active.ctl.requestCalibration();
  };

  btnCal.addEventListener("click", recalibrate);
  btnOri.addEventListener("click", toggleOrientation);
  btnFreeze.addEventListener("click", toggleFreeze);
  btnReset.addEventListener("click", resetScene);

  window.addEventListener("keydown", (e) => {
    // Bare keys only: Cmd+C must copy, Ctrl+F must find, held keys no-op.
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    if (e.key === "c") recalibrate();
    else if (e.key === "o") toggleOrientation();
    else if (e.key === "f") toggleFreeze();
    else if (e.key === "x") resetScene();
  });

  // Track resolution changes (device rotation, browser downgrading the
  // stream) or the aspect correction silently corrupts the depth axis.
  video.addEventListener("resize", () => {
    if (active?.mode === "hand" && video.videoHeight > 0) {
      active.ctl.frameAspect = video.videoWidth / video.videoHeight;
    }
  });

  startBtn.addEventListener("click", () => void start());
  takeoverBtn.addEventListener("click", () => void start());
  autopilotBtn.addEventListener("click", () => void startAutopilot());

  // Intent-based prefetch: hovering or focusing a start button is a good
  // sign the ~30 MB engine download is about to be wanted, and starting
  // it now makes the click feel instant without loading it for visitors
  // who only came to read.
  const prefetch = (btn: HTMLElement, load: () => Promise<unknown>): void => {
    const once = (): void => {
      load().catch(() => {}); // the click path reports errors
    };
    btn.addEventListener("pointerenter", once, { once: true });
    btn.addEventListener("focus", once, { once: true });
  };
  prefetch(startBtn, () => Promise.all([loadScene(), loadTracker()]));
  prefetch(autopilotBtn, loadScene);
}

wireControls();

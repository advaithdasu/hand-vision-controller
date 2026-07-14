/** Bootstrap and render loop: camera -> tracker -> controller -> scene. */

import { TeleopController } from "./app";
import { ArmScene } from "./armScene";
import { HandTracker, type HandObservation } from "./handTracker";
import { HAND_CONNECTIONS, type Landmarks } from "./poseFeatures";

const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const simCanvas = document.getElementById("sim") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;
const landing = document.getElementById("landing") as HTMLDivElement;
const landingStatus = document.getElementById("landing-status") as HTMLParagraphElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const controlButtons = ["btn-calibrate", "btn-orientation", "btn-freeze", "btn-reset"]
  .map((id) => document.getElementById(id) as HTMLButtonElement);

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

/**
 * Heavy engine modules are created once and reused across start() retries:
 * a second ArmScene would attach a second WebGLRenderer to the same canvas
 * and leak the Rapier world; a second HandLandmarker leaks its predecessor.
 */
let enginePromise: Promise<[ArmScene, HandTracker]> | null = null;

/** The running session, if any — event handlers wired once refer to this. */
let active: { ctl: TeleopController; scene: ArmScene } | null = null;

/** Stop the camera and drop back to the landing card with a message. */
function stopToLanding(stream: MediaStream | null, message: string): void {
  active = null;
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  landing.classList.remove("hidden");
  landingStatus.textContent = message;
  startBtn.disabled = false;
  for (const b of controlButtons) b.disabled = true;
}

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

function drawSkeleton(lm: Landmarks): void {
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
  ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(...px(lm[a]));
    ctx.lineTo(...px(lm[b]));
    ctx.stroke();
  }
  ctx.fillStyle = "#f27317";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(...px(p), 4 * dpr(), 0, Math.PI * 2);
    ctx.fill();
  }
}

let lastHudHtml = "";

function updateHud(ctl: TeleopController, scene: ArmScene, handVisible: boolean): void {
  const ik = ctl.lastIK;
  const fps = frameMs.mean() > 0 ? 1000 / frameMs.mean() : 0;
  const mode = ctl.manualFreeze
    ? '<span class="warn">FROZEN (button)</span>'
    : ctl.clutched
      ? '<span class="warn">CLUTCHED (fist)</span>'
      : ctl.orientationOn
        ? "pos + orientation"
        : "position only";
  const grip = ctl.gripLatched
    ? '<span class="warn">LATCHED</span>'
    : ctl.gripping
      ? '<span class="warn">CLOSED</span>'
      : "open";
  const lines = [
    `fps ${fps.toFixed(0).padStart(3)}   detect ${detectMs.mean().toFixed(1)} ms   tick ${tickMs.mean().toFixed(2)} ms`,
    `mode: ${mode}`,
    `hand: ${handVisible ? '<span class="good">tracking</span>' : '<span class="warn">NOT FOUND</span>'}`,
    `grip: ${grip}  aperture ${ctl.gripOpening.toFixed(2)}${scene.holding ? '  <span class="good">● holding cube</span>' : ""}`,
  ];
  if (ik) {
    const conv = ik.converged ? "" : ' <span class="warn">(!)</span>';
    lines.push(
      `ik: ${ik.iters} it  ${(ik.posErr * 1000).toFixed(1)} mm / ${((ik.oriErr * 180) / Math.PI).toFixed(1)} deg  w ${ik.manipulability.toFixed(3)}${conv}`,
    );
  }
  if (!ctl.calibrated) {
    lines.push('<span class="warn">show your hand to calibrate</span>');
  }
  const html = lines.join("\n");
  // Skip the write (and the DOM teardown/re-parse it causes) when unchanged.
  if (html !== lastHudHtml) {
    lastHudHtml = html;
    hud.innerHTML = html;
  }
}

/** Treat the hand as lost if the camera stops delivering frames this long. */
const STALLED_FRAME_MS = 1500;

async function start(): Promise<void> {
  startBtn.disabled = true;
  let stream: MediaStream | null = null;
  try {
    landingStatus.textContent = "Loading physics + hand tracking models…";
    enginePromise ??= Promise.all([ArmScene.create(simCanvas), HandTracker.create()]);
    const [scene, tracker] = await enginePromise.catch((e) => {
      enginePromise = null; // engine init failed — allow a clean retry
      throw e;
    });

    landingStatus.textContent = "Requesting camera…";
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

    const ctl = new TeleopController(scene);
    ctl.frameAspect = video.videoWidth / video.videoHeight;
    active = { ctl, scene };
    landing.classList.add("hidden");
    landingStatus.textContent = "";
    for (const b of controlButtons) b.disabled = false; // session is live

    let lastObs: HandObservation | null = null;
    let lastVideoTime = -1;
    let lastFrameAt = performance.now();
    let tPrev = performance.now();

    const loop = (): void => {
      // Any uncaught throw here (GPU context loss, wasm fault) would
      // otherwise kill the rAF chain silently with the camera still live.
      try {
        if (active?.ctl !== ctl) return; // superseded by a restart
        const now = performance.now();
        frameMs.push(now - tPrev); // frame interval -> real display fps
        const dt = Math.min((now - tPrev) / 1000, 0.1);
        tPrev = now;

        if (cameraLost) {
          stopToLanding(stream, "Camera disconnected — start again to reconnect.");
          return;
        }

        // Detect only on new video frames; reuse the last observation between.
        if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
          lastVideoTime = video.currentTime;
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

        const camPanel = video.parentElement!;
        sizeOverlay(camPanel.clientWidth, camPanel.clientHeight);
        if (lastObs) drawSkeleton(lastObs.imageLandmarks);
        else overlay.getContext("2d")!.clearRect(0, 0, overlay.width, overlay.height);

        const simPanel = simCanvas.parentElement!;
        scene.render(simPanel.clientWidth, simPanel.clientHeight);

        updateHud(ctl, scene, lastObs !== null);
        requestAnimationFrame(loop);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        stopToLanding(stream, `The session crashed (${msg}). Start again to retry.`);
      }
    };
    requestAnimationFrame(loop);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    // Release the camera on any failure, or the indicator light stays on
    // (and a retry would acquire a second stream) while the app looks dead.
    stopToLanding(
      stream,
      err instanceof DOMException && err.name === "NotAllowedError"
        ? "Camera access was denied — enable it in your browser's site settings and try again."
        : /WebGL/i.test(msg)
          ? "This browser has no WebGL support (required for the 3D simulation). Try Chrome, Edge, or Safari with hardware acceleration enabled."
          : `Failed to start: ${msg}`,
    );
  }
}

/**
 * Wired once at module load; handlers act on the current session via
 * `active`, so a camera-lost restart doesn't stack duplicate listeners.
 */
function wireControls(): void {
  const btnCal = document.getElementById("btn-calibrate") as HTMLButtonElement;
  const btnOri = document.getElementById("btn-orientation") as HTMLButtonElement;
  const btnFreeze = document.getElementById("btn-freeze") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-reset") as HTMLButtonElement;
  const toggleOrientation = (): void => {
    if (!active) return;
    active.ctl.orientationOn = !active.ctl.orientationOn;
    btnOri.textContent = `Orientation: ${active.ctl.orientationOn ? "on" : "off"}`;
  };
  const toggleFreeze = (): void => {
    if (!active) return;
    active.ctl.manualFreeze = !active.ctl.manualFreeze;
    btnFreeze.classList.toggle("active", active.ctl.manualFreeze);
  };
  const resetScene = (): void => {
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
    btnFreeze.classList.remove("active");
  };
  const recalibrate = (): void => {
    if (active) active.ctl.calibrated = false; // re-zero on the next tracked frame
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
    if (active && video.videoHeight > 0) {
      active.ctl.frameAspect = video.videoWidth / video.videoHeight;
    }
  });
}

wireControls();
startBtn.addEventListener("click", () => void start());

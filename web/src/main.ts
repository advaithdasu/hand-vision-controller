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
const ikMs = new RollingMean();
const frameMs = new RollingMean();

function drawSkeleton(lm: Landmarks): void {
  const ctx = overlay.getContext("2d")!;
  const { width: w, height: h } = overlay;
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lm[a][0] * w, lm[a][1] * h);
    ctx.lineTo(lm[b][0] * w, lm[b][1] * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#f27317";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p[0] * w, p[1] * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

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
    `fps ${fps.toFixed(0).padStart(3)}   detect ${detectMs.mean().toFixed(1)} ms   ik ${ikMs.mean().toFixed(2)} ms`,
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
  hud.innerHTML = lines.join("\n");
}

async function start(): Promise<void> {
  startBtn.disabled = true;
  try {
    landingStatus.textContent = "Loading physics + hand tracking models…";
    const [scene, tracker] = await Promise.all([
      ArmScene.create(simCanvas),
      HandTracker.create(),
    ]);

    landingStatus.textContent = "Requesting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise<void>((res) => {
      video.onloadedmetadata = () => res();
    });
    await video.play();

    const ctl = new TeleopController(scene);
    ctl.frameAspect = video.videoWidth / video.videoHeight;
    wireControls(ctl, scene);
    landing.classList.add("hidden");

    let lastObs: HandObservation | null = null;
    let lastVideoTime = -1;
    let tPrev = performance.now();

    const loop = (): void => {
      const now = performance.now();
      const dt = Math.min((now - tPrev) / 1000, 0.1);
      tPrev = now;

      // Detect only on new video frames; reuse the last observation between.
      if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
        lastVideoTime = video.currentTime;
        const t0 = performance.now();
        lastObs = tracker.detect(video, now);
        detectMs.push(performance.now() - t0);
      }

      const t1 = performance.now();
      ctl.tick(lastObs, dt);
      ikMs.push(performance.now() - t1);

      const camPanel = video.parentElement!;
      overlay.width = camPanel.clientWidth;
      overlay.height = camPanel.clientHeight;
      if (lastObs) drawSkeleton(lastObs.imageLandmarks);
      else overlay.getContext("2d")!.clearRect(0, 0, overlay.width, overlay.height);

      const simPanel = simCanvas.parentElement!;
      scene.render(simPanel.clientWidth, simPanel.clientHeight);

      updateHud(ctl, scene, lastObs !== null);
      frameMs.push(performance.now() - now);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (err) {
    startBtn.disabled = false;
    const msg = err instanceof Error ? err.message : String(err);
    landingStatus.textContent =
      err instanceof DOMException && err.name === "NotAllowedError"
        ? "Camera access was denied — enable it in your browser's site settings and try again."
        : /WebGL/i.test(msg)
          ? "This browser has no WebGL support (required for the 3D simulation). Try Chrome, Edge, or Safari with hardware acceleration enabled."
          : `Failed to start: ${msg}`;
  }
}

function wireControls(ctl: TeleopController, scene: ArmScene): void {
  const btnCal = document.getElementById("btn-calibrate") as HTMLButtonElement;
  const btnOri = document.getElementById("btn-orientation") as HTMLButtonElement;
  const btnFreeze = document.getElementById("btn-freeze") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-reset") as HTMLButtonElement;

  const toggleOrientation = (): void => {
    ctl.orientationOn = !ctl.orientationOn;
    btnOri.textContent = `Orientation: ${ctl.orientationOn ? "on" : "off"}`;
  };
  const toggleFreeze = (): void => {
    ctl.manualFreeze = !ctl.manualFreeze;
    btnFreeze.classList.toggle("active", ctl.manualFreeze);
  };
  const resetScene = (): void => {
    scene.reset();
    ctl.resetControl();
    btnFreeze.classList.remove("active");
  };

  btnCal.addEventListener("click", () => {
    ctl.calibrated = false; // re-zero on the next tracked frame
  });
  btnOri.addEventListener("click", toggleOrientation);
  btnFreeze.addEventListener("click", toggleFreeze);
  btnReset.addEventListener("click", resetScene);

  window.addEventListener("keydown", (e) => {
    if (e.key === "c") ctl.calibrated = false;
    else if (e.key === "o") toggleOrientation();
    else if (e.key === "f") toggleFreeze();
    else if (e.key === "x") resetScene();
  });
}

startBtn.addEventListener("click", () => void start());

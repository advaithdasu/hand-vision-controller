/**
 * MediaPipe HandLandmarker wrapper (Tasks API, VIDEO mode).
 *
 * Landmarks are mirrored (x -> 1-x for image coords, x -> -x for metric
 * world coords) so the geometry matches the Python app, which flips the
 * frame before detection. The video element itself is mirrored with CSS.
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

import { type Landmarks } from "./poseFeatures";

export interface HandObservation {
  imageLandmarks: Landmarks; // normalized, mirrored
  worldLandmarks: Landmarks; // meters, hand-centered, mirrored
}

/** Download progress callback: bytes so far and the total, if known. */
export type ProgressFn = (loaded: number, total: number | null) => void;

/**
 * Fetch the model with byte-level progress so the landing card can show
 * a real bar during the ~8 MB first-visit download instead of a static
 * "loading…" line. Falls back to a plain download when the body stream
 * or Content-Length is unavailable.
 */
async function fetchWithProgress(url: string, onProgress?: ProgressFn): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  const len = Number(res.headers.get("Content-Length"));
  const total = Number.isFinite(len) && len > 0 ? len : null;
  if (!res.body || !onProgress) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

export class HandTracker {
  private constructor(private landmarker: HandLandmarker) {}

  static async create(onProgress?: ProgressFn): Promise<HandTracker> {
    // The wasm runtime and the model download in parallel; MediaPipe
    // loads the wasm itself, so only the model reports byte progress.
    const [fileset, modelAssetBuffer] = await Promise.all([
      FilesetResolver.forVisionTasks(import.meta.env.BASE_URL + "mediapipe/wasm"),
      fetchWithProgress(import.meta.env.BASE_URL + "models/hand_landmarker.task", onProgress),
    ]);
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: {
        // A fresh copy per attempt: MediaPipe may detach the buffer it is
        // handed, and the CPU fallback needs an intact one.
        modelAssetBuffer: modelAssetBuffer.slice(),
        delegate,
      },
      runningMode: "VIDEO" as const,
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    try {
      return new HandTracker(
        await HandLandmarker.createFromOptions(fileset, options("GPU")),
      );
    } catch (err) {
      // Usually no usable WebGL2 context — fall back to CPU inference
      // (slower but works everywhere). Surface the original error so a
      // non-GPU failure (bad model path, network) isn't masked by the retry.
      console.warn("GPU hand-landmarker init failed, retrying on CPU:", err);
      return new HandTracker(
        await HandLandmarker.createFromOptions(fileset, options("CPU")),
      );
    }
  }

  private lastTs = -1;

  detect(video: HTMLVideoElement, timestampMs: number): HandObservation | null {
    // MediaPipe VIDEO mode requires strictly increasing timestamps.
    if (timestampMs <= this.lastTs) timestampMs = this.lastTs + 1;
    this.lastTs = timestampMs;

    const result = this.landmarker.detectForVideo(video, timestampMs);
    if (!result.landmarks.length) return null;

    const imageLandmarks = result.landmarks[0].map(
      (p): [number, number, number] => [1 - p.x, p.y, p.z],
    );
    const worldLandmarks = result.worldLandmarks[0].map(
      (p): [number, number, number] => [-p.x, p.y, p.z],
    );
    return { imageLandmarks, worldLandmarks };
  }

  close(): void {
    this.landmarker.close();
  }
}

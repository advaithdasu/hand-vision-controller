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

export class HandTracker {
  private constructor(private landmarker: HandLandmarker) {}

  static async create(): Promise<HandTracker> {
    const fileset = await FilesetResolver.forVisionTasks(
      import.meta.env.BASE_URL + "mediapipe/wasm",
    );
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: {
        modelAssetPath: import.meta.env.BASE_URL + "models/hand_landmarker.task",
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

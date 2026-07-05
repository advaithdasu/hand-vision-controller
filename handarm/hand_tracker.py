"""MediaPipe HandLandmarker wrapper (Tasks API, VIDEO mode).

Produces a HandObservation per frame: normalized-image landmarks for
position mapping and drawing, metric world landmarks for orientation and
gesture geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from .config import HAND_MODEL


@dataclass
class HandObservation:
    image_landmarks: np.ndarray  # (21, 3) normalized image coords
    world_landmarks: np.ndarray  # (21, 3) meters, hand-centered
    handedness: str              # "Left" / "Right" (as seen in the frame)


class HandTracker:
    def __init__(self, model_path=HAND_MODEL, num_hands: int = 1):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        """Run detection on an RGB frame. Returns None if no hand is found."""
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        img_lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        world_lm = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
        handedness = result.handedness[0][0].category_name if result.handedness else "Right"
        return HandObservation(img_lm, world_lm, handedness)

    def close(self) -> None:
        self._landmarker.close()

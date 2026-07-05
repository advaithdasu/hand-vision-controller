"""Drawing: hand skeleton overlay, HUD, and the unified split-screen view."""

from __future__ import annotations

import cv2
import numpy as np

from .pose_features import HAND_CONNECTIONS

_BONE = (80, 200, 255)
_JOINT = (0, 120, 255)
_HUD_BG = (18, 18, 22)
_HUD_FG = (235, 235, 235)
_ACCENT = (60, 220, 120)
_WARN = (60, 60, 240)


def draw_hand_skeleton(frame: np.ndarray, image_landmarks: np.ndarray) -> None:
    """Draw the 21-point skeleton on a BGR frame (in place)."""
    h, w = frame.shape[:2]
    pts = np.column_stack(
        [image_landmarks[:, 0] * w, image_landmarks[:, 1] * h]
    ).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), _BONE, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, _JOINT, -1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[tuple[str, bool]]) -> None:
    """Semi-transparent status panel, top-left. lines = [(text, is_warning)]."""
    pad, lh = 10, 22
    text_widths = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        for text, _ in lines
    ]
    width = min(max(text_widths, default=0) + 2 * pad, frame.shape[1])
    height = pad * 2 + lh * len(lines)
    panel = frame[0:height, 0:width]
    overlay = np.full_like(panel, _HUD_BG)
    cv2.addWeighted(overlay, 0.65, panel, 0.35, 0, dst=panel)
    for i, (text, warn) in enumerate(lines):
        color = _WARN if warn else _HUD_FG
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def draw_gripper_bar(frame: np.ndarray, opening: float, gripping: bool) -> None:
    """Vertical aperture bar on the right edge of the webcam panel."""
    h, w = frame.shape[:2]
    x0, x1 = w - 28, w - 12
    y0, y1 = 60, h - 60
    cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)
    fill_top = int(y1 - (y1 - y0) * np.clip(opening, 0, 1))
    color = _WARN if gripping else _ACCENT
    cv2.rectangle(frame, (x0 + 2, fill_top), (x1 - 2, y1 - 2), color, -1)
    cv2.putText(frame, "GRIP", (x0 - 8, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _HUD_FG, 1, cv2.LINE_AA)


def compose_split(webcam_bgr: np.ndarray, sim_rgb: np.ndarray, height: int) -> np.ndarray:
    """Side-by-side view: webcam feed left, simulation right."""

    def fit(img: np.ndarray) -> np.ndarray:
        scale = height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), height))

    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
    return np.hstack([fit(webcam_bgr), fit(sim_bgr)])

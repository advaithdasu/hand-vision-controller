"""Record and replay joint-space trajectories (learning-from-demo preview).

Recordings are JSON Lines: one {"t", "q", "grip"} object per frame, with
t in seconds relative to recording start.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self._fh = None
        self._t0 = 0.0
        self.path: Path | None = None

    @property
    def active(self) -> bool:
        return self._fh is not None

    def start(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.out_dir / f"demo-{stamp}.jsonl"
        self._fh = open(self.path, "w")  # noqa: SIM115 — closed in stop()
        self._t0 = time.perf_counter()
        return self.path

    def record(self, q: np.ndarray, grip: float) -> None:
        if self._fh is None:
            return
        row = {
            "t": round(time.perf_counter() - self._t0, 4),
            "q": [round(float(v), 5) for v in q],
            "grip": round(float(grip), 4),
        }
        self._fh.write(json.dumps(row) + "\n")

    def stop(self) -> Path | None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            return self.path
        return None


def load_recording(path: Path) -> list[dict]:
    frames = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames

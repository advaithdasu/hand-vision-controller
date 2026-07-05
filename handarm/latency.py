"""Rolling latency profiler for the perception -> IK -> physics pipeline."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class LatencyProfiler:
    """Accumulates per-stage durations and reports rolling means in ms."""

    def __init__(self, window: int = 60):
        self._window = window
        self._stages: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._t0: float | None = None
        self._last: float | None = None

    def frame_start(self) -> None:
        self._t0 = self._last = time.perf_counter()

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        if self._last is not None:
            self._stages[stage].append(now - self._last)
        self._last = now

    def frame_end(self) -> None:
        if self._t0 is not None:
            self._stages["total"].append(time.perf_counter() - self._t0)

    def mean_ms(self, stage: str) -> float:
        buf = self._stages.get(stage)
        if not buf:
            return 0.0
        return 1000.0 * sum(buf) / len(buf)

    def fps(self) -> float:
        total = self._stages.get("total")
        if not total or sum(total) == 0:
            return 0.0
        return len(total) / sum(total)

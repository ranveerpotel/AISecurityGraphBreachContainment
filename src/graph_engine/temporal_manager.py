"""
Manages the three sliding temporal windows described in the paper:
  current (5 min), short-term (24 hr), long-term (7 days).

Also provides per-node and per-edge statistics for the temporal feature
vector fed to the GNN.
"""
from __future__ import annotations
import math
from collections import deque, defaultdict
from datetime import datetime, timedelta
from threading import RLock
from typing import Deque, Dict, List, Optional, Tuple


class WindowStats:
    """Running stats (count, mean bytes, std bytes) over a fixed-time window."""

    def __init__(self, window_seconds: float) -> None:
        self._window_sec = window_seconds
        self._entries: Deque[Tuple[datetime, float]] = deque()
        self._sum = 0.0
        self._sum_sq = 0.0

    def add(self, timestamp: datetime, value: float) -> None:
        self._entries.append((timestamp, value))
        self._sum += value
        self._sum_sq += value * value
        self._evict(timestamp)

    def _evict(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._window_sec)
        while self._entries and self._entries[0][0] < cutoff:
            _, v = self._entries.popleft()
            self._sum -= v
            self._sum_sq -= v * v

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def mean(self) -> float:
        n = len(self._entries)
        return self._sum / n if n > 0 else 0.0

    @property
    def std(self) -> float:
        n = len(self._entries)
        if n < 2:
            return 0.0
        variance = (self._sum_sq - self._sum ** 2 / n) / (n - 1)
        return math.sqrt(max(0.0, variance))

    @property
    def rate_per_minute(self) -> float:
        return self.count / (self._window_sec / 60.0)


class TemporalManager:
    """
    Tracks per-(src, dst) edge activity across all three temporal windows.
    Used by the anomaly detector to build temporal feature vectors and
    compute temporal deviation scores.
    """

    _WINDOW_5MIN = 5 * 60
    _WINDOW_24HR = 24 * 3600
    _WINDOW_7DAY = 7 * 24 * 3600

    def __init__(self) -> None:
        self._lock = RLock()
        # (src, dst) -> window name -> WindowStats
        self._stats: Dict[Tuple[str, str], Dict[str, WindowStats]] = defaultdict(
            lambda: {
                "current": WindowStats(self._WINDOW_5MIN),
                "short": WindowStats(self._WINDOW_24HR),
                "long": WindowStats(self._WINDOW_7DAY),
            }
        )
        # Per-node activity for structural features
        self._node_activity: Dict[str, WindowStats] = defaultdict(
            lambda: WindowStats(self._WINDOW_24HR)
        )

    def record(self, src: str, dst: str, timestamp: datetime, bytes_val: float = 0.0) -> None:
        with self._lock:
            for ws in self._stats[(src, dst)].values():
                ws.add(timestamp, bytes_val)
            self._node_activity[src].add(timestamp, bytes_val)
            self._node_activity[dst].add(timestamp, bytes_val)

    def edge_features(self, src: str, dst: str, now: Optional[datetime] = None) -> List[float]:
        """Return 6-dimensional temporal feature for edge (src, dst)."""
        now = now or datetime.utcnow()
        with self._lock:
            windows = self._stats.get((src, dst))
            if windows is None:
                return [0.0] * 6
            c, s, l_ = windows["current"], windows["short"], windows["long"]
            return [
                c.rate_per_minute,
                c.mean,
                s.rate_per_minute,
                s.mean,
                l_.rate_per_minute,
                l_.mean,
            ]

    def node_temporal_features(self, node_id: str) -> List[float]:
        """Return 4-dimensional temporal feature for a node."""
        with self._lock:
            ws = self._node_activity.get(node_id)
            if ws is None:
                return [0.0] * 4
            return [ws.count, ws.rate_per_minute, ws.mean, ws.std]

    def temporal_deviation(self, src: str, dst: str) -> float:
        """
        Ratio of current-window rate to short-window baseline rate.
        Returns value in [0, 1].
        """
        with self._lock:
            windows = self._stats.get((src, dst))
            if windows is None:
                return 0.0
            cur_rate = windows["current"].rate_per_minute
            base_rate = windows["short"].rate_per_minute
            if base_rate == 0.0:
                return min(1.0, cur_rate)
            deviation = abs(cur_rate - base_rate) / base_rate
            return min(1.0, deviation)

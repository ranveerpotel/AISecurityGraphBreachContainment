"""
Concept drift detection and online learning (§9 of the paper).

Mechanism:
  - Population Stability Index (PSI) computed weekly over rolling 30-day windows.
  - PSI > 0.25 triggers model retraining on last 60 days of labelled data.
  - New model runs in shadow mode for 7 days before deployment.
  - Tracks 3 feature groups: behavioral, temporal, structural.
"""
from __future__ import annotations
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..common.models import DriftEvent
from ..common.config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# PSI bucket thresholds
_N_BUCKETS = 10
_PSI_WARN = 0.10
_PSI_CRITICAL = 0.25


def _psi(expected: np.ndarray, actual: np.ndarray, n_buckets: int = _N_BUCKETS) -> float:
    """
    Population Stability Index between two 1-D distributions.
    PSI < 0.10: no drift.  PSI 0.10-0.25: moderate.  PSI > 0.25: significant.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    # Define bucket edges on the expected distribution
    percentiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(expected, percentiles)
    edges[0] -= 1e-9   # include the minimum
    edges[-1] += 1e-9  # include the maximum

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e_pct = e_counts / len(expected)
    a_pct = a_counts / max(len(actual), 1)

    # Replace zeros to avoid log(0)
    e_pct = np.where(e_pct == 0, 1e-6, e_pct)
    a_pct = np.where(a_pct == 0, 1e-6, a_pct)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


class FeatureBuffer:
    """Rolling buffer storing the last `window_days` of feature observations."""

    def __init__(self, window_days: int = 30) -> None:
        self._window = window_days
        self._entries: Deque[Tuple[datetime, np.ndarray]] = deque()

    def add(self, timestamp: datetime, features: np.ndarray) -> None:
        self._entries.append((timestamp, features.copy()))
        self._evict(timestamp)

    def _evict(self, now: datetime) -> None:
        cutoff = now - timedelta(days=self._window)
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def as_matrix(self) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        return np.vstack([e for _, e in self._entries])

    def split(self, cutoff: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Return (before_cutoff, after_cutoff) arrays."""
        before = [e for t, e in self._entries if t < cutoff]
        after = [e for t, e in self._entries if t >= cutoff]
        b = np.vstack(before) if before else np.empty((0,))
        a = np.vstack(after) if after else np.empty((0,))
        return b, a


class ConceptDriftDetector:
    """
    Monitors feature distribution drift across 3 feature groups.
    Triggers GNN retraining when PSI exceeds 0.25 in any group.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config.drift
        self._buffers: Dict[str, FeatureBuffer] = {
            "behavioral": FeatureBuffer(self._cfg.rolling_window_days),
            "temporal": FeatureBuffer(self._cfg.rolling_window_days),
            "structural": FeatureBuffer(self._cfg.rolling_window_days),
        }
        self._last_check: Optional[datetime] = None
        self._drift_events: List[DriftEvent] = []
        self._psi_history: Dict[str, List[Tuple[datetime, float]]] = {
            g: [] for g in self._buffers
        }
        # Shadow model tracking
        self._shadow_start: Optional[datetime] = None
        self._in_shadow_mode: bool = False

    def record_features(
        self,
        timestamp: datetime,
        behavioral: np.ndarray,
        temporal: np.ndarray,
        structural: np.ndarray,
    ) -> None:
        self._buffers["behavioral"].add(timestamp, behavioral)
        self._buffers["temporal"].add(timestamp, temporal)
        self._buffers["structural"].add(timestamp, structural)

    def check_drift(self, now: Optional[datetime] = None) -> List[DriftEvent]:
        """
        Run PSI check. Returns any new DriftEvent objects detected.
        Should be called weekly (check_interval_days).
        """
        now = now or datetime.utcnow()
        if self._last_check and (now - self._last_check).days < self._cfg.check_interval_days:
            return []

        self._last_check = now
        new_events: List[DriftEvent] = []

        for group, buf in self._buffers.items():
            mat = buf.as_matrix()
            if mat is None or mat.shape[0] < 20:
                continue

            cutoff = now - timedelta(days=self._cfg.check_interval_days)
            expected, actual = buf.split(cutoff)
            if expected.shape[0] == 0 or actual.shape[0] == 0:
                continue

            # PSI per feature, then take the mean
            psi_vals = []
            n_features = min(expected.shape[1] if expected.ndim > 1 else 1,
                             actual.shape[1] if actual.ndim > 1 else 1)
            for fi in range(n_features):
                e_col = expected[:, fi] if expected.ndim > 1 else expected
                a_col = actual[:, fi] if actual.ndim > 1 else actual
                psi_vals.append(_psi(e_col, a_col))
            psi_score = float(np.mean(psi_vals))

            self._psi_history[group].append((now, psi_score))
            logger.info("PSI [%s] = %.4f at %s", group, psi_score, now.isoformat())

            if psi_score > _PSI_CRITICAL:
                event = DriftEvent(
                    detected_at=now,
                    feature_group=group,
                    psi_score=psi_score,
                    triggered_retrain=True,
                )
                self._drift_events.append(event)
                new_events.append(event)
                logger.warning(
                    "Significant drift detected in [%s] PSI=%.4f — triggering retraining",
                    group, psi_score,
                )
            elif psi_score > _PSI_WARN:
                logger.info("Moderate drift in [%s] PSI=%.4f — monitoring", group, psi_score)

        return new_events

    def start_shadow_mode(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.utcnow()
        self._shadow_start = now
        self._in_shadow_mode = True
        logger.info("Shadow mode started at %s (runs for %d days)", now, self._cfg.shadow_mode_days)

    def should_promote_shadow_model(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        if not self._in_shadow_mode or self._shadow_start is None:
            return False
        return (now - self._shadow_start).days >= self._cfg.shadow_mode_days

    def promote_shadow_model(self) -> None:
        self._in_shadow_mode = False
        self._shadow_start = None
        logger.info("Shadow model promoted to production")

    @property
    def recent_drift_events(self) -> List[DriftEvent]:
        return self._drift_events[-20:]

    def psi_summary(self) -> Dict[str, float]:
        result = {}
        for group, history in self._psi_history.items():
            result[group] = history[-1][1] if history else 0.0
        return result

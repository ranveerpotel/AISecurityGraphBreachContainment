"""
Risk scoring per §4.4 of the paper.

Risk(p) = P(path_used) × Impact(target)

where P(path_used) = Π_edges(weight) × (1 - p_detect)

Thresholds:
  risk > 0.8  → automated isolation
  0.5 < risk ≤ 0.8 → SOC alert with prepared containment policies
  risk ≤ 0.5  → log and monitor
"""
from __future__ import annotations
from typing import List

from ..common.models import AttackPath
from ..common.config import AppConfig, DEFAULT_CONFIG


class RiskScorer:

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config.containment

    def classify(self, path: AttackPath) -> str:
        """Return 'AUTO_ISOLATE', 'SOC_ALERT', or 'MONITOR'."""
        if path.risk_score > self._cfg.auto_isolate_threshold:
            return "AUTO_ISOLATE"
        if path.risk_score > self._cfg.soc_alert_threshold:
            return "SOC_ALERT"
        return "MONITOR"

    def aggregate_risk(self, paths: List[AttackPath]) -> float:
        """Max risk across all paths (conservative approach)."""
        if not paths:
            return 0.0
        return max(p.risk_score for p in paths)

    def severity_label(self, risk_score: float) -> str:
        if risk_score > 0.8:
            return "CRITICAL"
        if risk_score > 0.5:
            return "HIGH"
        if risk_score > 0.2:
            return "MEDIUM"
        return "LOW"

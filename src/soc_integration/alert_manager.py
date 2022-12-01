"""
Alert manager: formats SecurityAlert objects and dispatches them to
SIEM/SOAR systems (§4.5). Includes MITRE ATT&CK tactic mapping.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from ..common.models import (
    AnomalyScore,
    AttackPath,
    ContainmentAction,
    SecurityAlert,
    TelemetryEvent,
)
from ..common.config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Simplified MITRE ATT&CK tactic mapping based on process name / event type
_MITRE_MAP: Dict[str, List[str]] = {
    "psexec": ["TA0008 Lateral Movement", "T1021.002 SMB/Windows Admin Shares"],
    "ssh": ["TA0008 Lateral Movement", "T1021.004 SSH"],
    "rdp": ["TA0008 Lateral Movement", "T1021.001 Remote Desktop Protocol"],
    "wmic": ["TA0008 Lateral Movement", "T1047 Windows Management Instrumentation"],
    "powershell": ["TA0002 Execution", "T1059.001 PowerShell"],
    "authentication": ["TA0006 Credential Access", "T1078 Valid Accounts"],
    "file_access": ["TA0009 Collection", "T1005 Data from Local System"],
    "dns": ["TA0011 Command and Control", "T1071.004 DNS"],
}


def _severity_from_risk(risk: float) -> str:
    if risk > 0.8:
        return "CRITICAL"
    if risk > 0.5:
        return "HIGH"
    if risk > 0.2:
        return "MEDIUM"
    return "LOW"


def _map_mitre(anomaly_scores: List[AnomalyScore], events: Optional[List[TelemetryEvent]] = None) -> List[str]:
    tactics: List[str] = []
    if events:
        for ev in events:
            for keyword, tactic_list in _MITRE_MAP.items():
                if keyword in (ev.process or "").lower() or keyword == ev.event_type:
                    for t in tactic_list:
                        if t not in tactics:
                            tactics.append(t)
    if not tactics:
        # Default for lateral movement pattern
        tactics = ["TA0008 Lateral Movement", "T1021 Remote Services"]
    return tactics


class AlertManager:
    """Builds, stores, and dispatches security alerts."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        self._alerts: List[SecurityAlert] = []
        self._kafka_producer = None
        if config.use_kafka:
            self._init_kafka()

    def _init_kafka(self) -> None:
        try:
            from kafka import KafkaProducer
            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self._cfg.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            )
        except ImportError:
            logger.warning("kafka-python not installed; Kafka alerting disabled")

    def create_alert(
        self,
        anomaly_scores: List[AnomalyScore],
        attack_paths: List[AttackPath],
        containment: Optional[ContainmentAction] = None,
        triggering_events: Optional[List[TelemetryEvent]] = None,
    ) -> SecurityAlert:
        flagged = [s for s in anomaly_scores if s.is_flagged]
        compromised = list({s.source for s in flagged} | {s.destination for s in flagged})
        risk = max((p.risk_score for p in attack_paths), default=0.0)
        severity = _severity_from_risk(risk)
        mitre = _map_mitre(flagged, triggering_events)

        recommendations = self._build_recommendations(attack_paths, containment)

        alert = SecurityAlert(
            severity=severity,
            title=f"[{severity}] Potential Lateral Movement Detected — {len(compromised)} node(s)",
            description=(
                f"GNN anomaly detector flagged {len(flagged)} suspicious edges. "
                f"Highest risk path: {attack_paths[0].source} → {attack_paths[0].target} "
                f"(risk={risk:.2f}, hops={attack_paths[0].hop_count})"
                if attack_paths else
                f"GNN anomaly detector flagged {len(flagged)} suspicious edges."
            ),
            compromised_nodes=compromised,
            attack_paths=attack_paths[:5],  # top 5 for brevity
            anomaly_scores=flagged[:20],
            recommended_actions=recommendations,
            mitre_tactics=mitre,
            containment_action=containment,
            created_at=datetime.utcnow(),
            metadata={
                "total_flagged_edges": len(flagged),
                "total_attack_paths": len(attack_paths),
                "max_risk_score": risk,
            },
        )
        self._alerts.append(alert)
        self._dispatch(alert)
        return alert

    def _build_recommendations(
        self,
        paths: List[AttackPath],
        containment: Optional[ContainmentAction],
    ) -> List[str]:
        recs = []
        if containment and containment.auto_applied:
            recs.append(f"Automated isolation applied ({len(containment.firewall_rules)} rules active).")
        elif containment and containment.action_type == "SOC_ALERT":
            recs.append("Review and approve prepared containment policies in the dashboard.")
        else:
            recs.append("Initiate manual incident response process.")
        if paths:
            top = paths[0]
            recs.append(
                f"Investigate path: {' → '.join(top.nodes)} (risk={top.risk_score:.2f})."
            )
        recs.append("Conduct forensic analysis of all compromised nodes.")
        recs.append("Reset credentials for all users associated with compromised workloads.")
        return recs

    def _dispatch(self, alert: SecurityAlert) -> None:
        payload = self._to_json(alert)
        logger.warning("ALERT [%s] %s", alert.severity, alert.title)
        if self._kafka_producer:
            try:
                self._kafka_producer.send(self._cfg.kafka.alert_topic, value=payload)
            except Exception as exc:
                logger.error("Failed to publish alert to Kafka: %s", exc)

    @staticmethod
    def _to_json(alert: SecurityAlert) -> dict:
        return {
            "alert_id": alert.alert_id,
            "severity": alert.severity,
            "title": alert.title,
            "description": alert.description,
            "compromised_nodes": alert.compromised_nodes,
            "mitre_tactics": alert.mitre_tactics,
            "recommended_actions": alert.recommended_actions,
            "attack_paths": [
                {
                    "path_id": p.path_id,
                    "nodes": p.nodes,
                    "risk_score": p.risk_score,
                    "hop_count": p.hop_count,
                }
                for p in alert.attack_paths
            ],
            "containment": {
                "action_type": alert.containment_action.action_type,
                "auto_applied": alert.containment_action.auto_applied,
                "rules_count": len(alert.containment_action.firewall_rules),
                "business_impact": alert.containment_action.business_impact_estimate,
            } if alert.containment_action else None,
            "created_at": alert.created_at.isoformat(),
        }

    def get_alerts(self, severity: Optional[str] = None, limit: int = 100) -> List[SecurityAlert]:
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def acknowledge(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

"""
FastAPI REST endpoints for the SOC dashboard (§4.5).

Endpoints:
  GET  /health
  GET  /graph/stats
  GET  /graph/nodes/{node_id}
  GET  /alerts
  POST /alerts/{alert_id}/acknowledge
  GET  /containment/active-rules
  POST /containment/{action_id}/rollback
  GET  /drift/status
  GET  /detection/scores

All responses are JSON. Authentication/TLS handled by reverse proxy in production.
"""
from __future__ import annotations
import logging
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from ..common.models import SecurityAlert, ContainmentAction, FirewallRule
from ..common.config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def create_app(
    alert_manager,
    containment_engine,
    security_graph,
    drift_detector,
    config: AppConfig = DEFAULT_CONFIG,
) -> "FastAPI":
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="AI Security Graph — Breach Containment API",
        description="Real-time lateral movement detection and containment (Potel 2022)",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    @app.get("/graph/stats")
    def graph_stats():
        return {
            "node_count": security_graph.node_count,
            "edge_count": security_graph.edge_count,
            "sensitive_nodes": len(security_graph.sensitive_nodes()),
        }

    @app.get("/graph/nodes/{node_id}")
    def get_node(node_id: str):
        attrs = security_graph.get_node_attrs(node_id)
        if not attrs:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        return {
            "node_id": attrs.node_id,
            "node_type": attrs.node_type,
            "sensitivity": attrs.sensitivity,
            "is_critical": attrs.is_critical,
            "ip_address": attrs.ip_address,
            "hostname": attrs.hostname,
        }

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    @app.get("/alerts")
    def list_alerts(
        severity: Optional[str] = Query(None, regex="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
        limit: int = Query(50, ge=1, le=500),
    ):
        alerts = alert_manager.get_alerts(severity=severity, limit=limit)
        return {
            "total": len(alerts),
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "title": a.title,
                    "compromised_nodes": a.compromised_nodes,
                    "mitre_tactics": a.mitre_tactics,
                    "created_at": a.created_at.isoformat(),
                    "acknowledged": a.acknowledged,
                }
                for a in alerts
            ],
        }

    @app.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(alert_id: str):
        ok = alert_manager.acknowledge(alert_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"status": "acknowledged", "alert_id": alert_id}

    # ------------------------------------------------------------------
    # Containment
    # ------------------------------------------------------------------
    @app.get("/containment/active-rules")
    def active_rules():
        rules: List[FirewallRule] = containment_engine.active_rules
        return {
            "count": len(rules),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "action": r.action,
                    "source": r.source,
                    "destination": r.destination,
                    "reason": r.reason,
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                }
                for r in rules
            ],
        }

    @app.post("/containment/{action_id}/rollback")
    def rollback_containment(action_id: str):
        ok = containment_engine.rollback(action_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Containment action not found")
        return {"status": "rolled_back", "action_id": action_id}

    @app.get("/containment/history")
    def containment_history(limit: int = Query(20, ge=1, le=200)):
        history: List[ContainmentAction] = containment_engine.history[-limit:]
        return {
            "count": len(history),
            "actions": [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "risk_score": a.risk_score,
                    "auto_applied": a.auto_applied,
                    "business_impact": a.business_impact_estimate,
                    "created_at": a.created_at.isoformat(),
                    "applied_at": a.applied_at.isoformat() if a.applied_at else None,
                    "rules_count": len(a.firewall_rules),
                }
                for a in history
            ],
        }

    # ------------------------------------------------------------------
    # Drift
    # ------------------------------------------------------------------
    @app.get("/drift/status")
    def drift_status():
        return {
            "psi_scores": drift_detector.psi_summary(),
            "in_shadow_mode": drift_detector._in_shadow_mode,
            "recent_drift_events": [
                {
                    "drift_id": e.drift_id,
                    "feature_group": e.feature_group,
                    "psi_score": e.psi_score,
                    "triggered_retrain": e.triggered_retrain,
                    "detected_at": e.detected_at.isoformat(),
                }
                for e in drift_detector.recent_drift_events[-5:]
            ],
        }

    return app

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import uuid

from .config import NodeType, EdgeType, SensitivityLevel


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class NodeAttributes:
    node_id: str
    node_type: NodeType
    sensitivity: SensitivityLevel = SensitivityLevel.LOW
    owner: str = ""
    ip_address: str = ""
    hostname: str = ""
    os_type: str = ""          # linux, windows, container, cloud
    cloud_provider: str = ""   # aws, azure, onprem
    is_critical: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EdgeAttributes:
    source: str
    destination: str
    edge_type: EdgeType = EdgeType.COMMUNICATION
    weight: float = 0.0            # combined w(e) = alpha*f + beta*s + gamma*d
    frequency_score: float = 0.0   # f_norm — normalized event frequency
    sensitivity_score: float = 0.0 # s(e) — target sensitivity contribution
    deviation_score: float = 0.0   # d(e) — temporal anomaly component
    last_seen: datetime = field(default_factory=datetime.utcnow)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    event_count: int = 0
    bytes_total: int = 0
    protocols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryEvent:
    event_id: str = field(default_factory=_uid)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_node: str = ""
    destination_node: str = ""
    source_ip: str = ""
    destination_ip: str = ""
    event_type: str = ""       # TelemetrySource value
    protocol: str = ""
    port: int = 0
    bytes_transferred: int = 0
    user: str = ""
    process: str = ""          # PsExec, ssh, rdp, wmi, etc.
    success: bool = True
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeFeatureVector:
    """64-dimensional input feature vector for the GNN."""
    node_id: str
    # 24D: behavioral (outgoing/incoming traffic rates, unique peers, etc.)
    behavioral: List[float] = field(default_factory=lambda: [0.0] * 24)
    # 16D: temporal (hour-of-day, day-of-week, window activity rates)
    temporal: List[float] = field(default_factory=lambda: [0.0] * 16)
    # 16D: structural (degree, clustering coeff, centrality metrics, path lengths)
    structural: List[float] = field(default_factory=lambda: [0.0] * 16)
    # 8D: node type one-hot + sensitivity encoding
    node_type_enc: List[float] = field(default_factory=lambda: [0.0] * 8)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_vector(self) -> List[float]:
        return self.behavioral + self.temporal + self.structural + self.node_type_enc


@dataclass
class AnomalyScore:
    source: str
    destination: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    behavioral_deviation: float = 0.0  # D_b
    temporal_deviation: float = 0.0    # D_t
    structural_deviation: float = 0.0  # D_g
    total_score: float = 0.0           # A(i,j) = alpha_b*D_b + alpha_t*D_t + alpha_g*D_g
    is_flagged: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackPath:
    path_id: str = field(default_factory=_uid)
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    source: str = ""
    target: str = ""
    probability: float = 0.0    # Π edge_weights × (1 - p_detect)
    risk_score: float = 0.0     # probability × impact(target)
    impact_score: float = 0.0
    hop_count: int = 0
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FirewallRule:
    rule_id: str = field(default_factory=_uid)
    action: str = "DENY"           # DENY, RATE_LIMIT, MONITOR
    source: str = ""
    destination: str = ""
    protocol: str = "ANY"
    port: int = 0
    priority: int = 100
    expires_at: Optional[datetime] = None
    reason: str = ""


@dataclass
class ContainmentAction:
    action_id: str = field(default_factory=_uid)
    action_type: str = ""          # AUTO_ISOLATE, SOC_ALERT, MANUAL_REVIEW
    target_nodes: List[str] = field(default_factory=list)
    target_edges: List[Tuple[str, str]] = field(default_factory=list)
    risk_score: float = 0.0
    attack_paths: List[AttackPath] = field(default_factory=list)
    firewall_rules: List[FirewallRule] = field(default_factory=list)
    business_impact_estimate: float = 0.0
    auto_applied: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None
    rollback_applied: bool = False


@dataclass
class SecurityAlert:
    alert_id: str = field(default_factory=_uid)
    severity: str = "MEDIUM"       # LOW, MEDIUM, HIGH, CRITICAL
    title: str = ""
    description: str = ""
    compromised_nodes: List[str] = field(default_factory=list)
    attack_paths: List[AttackPath] = field(default_factory=list)
    anomaly_scores: List[AnomalyScore] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)  # ATT&CK mapping
    containment_action: Optional[ContainmentAction] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftEvent:
    drift_id: str = field(default_factory=_uid)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    feature_group: str = ""        # behavioral, temporal, structural
    psi_score: float = 0.0
    triggered_retrain: bool = False
    retraining_completed_at: Optional[datetime] = None

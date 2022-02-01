from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    WORKLOAD = "workload"
    USER = "user"
    ASSET = "asset"


class EdgeType(str, Enum):
    COMMUNICATION = "communication"
    PRIVILEGE = "privilege"


class SensitivityLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TelemetrySource(str, Enum):
    NETWORK_FLOW = "network_flow"
    PROCESS_EXEC = "process_exec"
    CLOUD_AUDIT = "cloud_audit"
    CONTAINER = "container"
    AUTH = "authentication"
    APPLICATION = "application"
    DNS = "dns"
    VPN = "vpn"
    FILE_ACCESS = "file_access"


@dataclass
class GraphConfig:
    # Edge weight formula: w(e) = alpha*f_norm + beta*s + gamma*d
    alpha: float = 0.4       # frequency component weight
    beta: float = 0.3        # sensitivity component weight
    gamma: float = 0.3       # deviation component weight
    decay_lambda: float = 0.1  # exponential decay rate per hour
    prune_after_days: int = 7
    window_current_min: int = 5
    window_short_hr: int = 24
    window_long_days: int = 7
    max_nodes: int = 100_000
    max_edges: int = 500_000


@dataclass
class GNNConfig:
    # 3-layer GraphSAGE: 64D input -> 128D -> 64D -> 32D embeddings
    input_dim: int = 64       # 24D behavioral + 16D temporal + 16D structural + 8D node type
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    embedding_dim: int = 32
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 50
    # Feature sub-dimensions
    behavioral_dim: int = 24
    temporal_dim: int = 16
    structural_dim: int = 16
    node_type_dim: int = 8


@dataclass
class DetectionConfig:
    alpha_b: float = 0.5     # behavioral deviation weight
    alpha_t: float = 0.3     # temporal deviation weight
    alpha_g: float = 0.2     # structural deviation weight
    anomaly_threshold: float = 0.7
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class ContainmentConfig:
    auto_isolate_threshold: float = 0.8
    soc_alert_threshold: float = 0.5
    max_hops: int = 5
    max_business_impact_auto: float = 0.05   # auto-apply if impact < 5%
    max_business_impact_soc: float = 0.20    # SOC alert if impact 5-20%
    containment_latency_target_sec: int = 35


@dataclass
class DriftConfig:
    psi_threshold: float = 0.25
    check_interval_days: int = 7
    rolling_window_days: int = 30
    retrain_window_days: int = 60
    shadow_mode_days: int = 7


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "kafka:9092"
    telemetry_topic: str = "telemetry-events"
    alert_topic: str = "security-alerts"
    containment_topic: str = "containment-actions"
    group_id: str = "aisgbc-consumer"
    auto_offset_reset: str = "latest"


@dataclass
class AppConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    containment: ContainmentConfig = field(default_factory=ContainmentConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    use_gpu: bool = False
    log_level: str = "INFO"
    use_kafka: bool = False   # set False for standalone / testing
    use_neo4j: bool = False   # set False to use in-memory NetworkX


DEFAULT_CONFIG = AppConfig()

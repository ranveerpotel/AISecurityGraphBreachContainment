"""
Smoke tests verifying each component initialises and processes events correctly.
Run with: pytest tests/
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timedelta
import numpy as np
import pytest

from src.common.config import DEFAULT_CONFIG, NodeType, SensitivityLevel
from src.common.models import TelemetryEvent, NodeAttributes
from src.telemetry.collector import SimulatedTelemetryGenerator
from src.telemetry.normalizer import TelemetryNormalizer
from src.graph_engine.security_graph import SecurityGraph
from src.graph_engine.temporal_manager import TemporalManager
from src.detection.graphsage_model import GNNInferenceEngine, NumpyGraphSAGEApproximation
from src.detection.anomaly_detector import AnomalyDetector, FeatureExtractor, RuleBasedDetector
from src.detection.concept_drift import ConceptDriftDetector, _psi
from src.containment.path_prioritizer import AttackPathPrioritizer
from src.containment.risk_scorer import RiskScorer
from src.containment.containment_engine import ContainmentEngine
from src.soc_integration.alert_manager import AlertManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return DEFAULT_CONFIG


@pytest.fixture
def graph(config):
    return SecurityGraph(config)


@pytest.fixture
def temporal_mgr():
    return TemporalManager()


@pytest.fixture
def gnn(config):
    cfg = config.detection.gnn
    return GNNInferenceEngine(
        input_dim=cfg.input_dim,
        hidden_dim_1=cfg.hidden_dim_1,
        hidden_dim_2=cfg.hidden_dim_2,
        embedding_dim=cfg.embedding_dim,
    )


@pytest.fixture
def detector(gnn, temporal_mgr, config):
    return AnomalyDetector(gnn, temporal_mgr, config)


@pytest.fixture
def populated_graph(graph):
    """Graph with 5 nodes and known edges for deterministic tests."""
    for nid, ntype, sens in [
        ("wl-001", NodeType.WORKLOAD, SensitivityLevel.LOW),
        ("wl-002", NodeType.WORKLOAD, SensitivityLevel.LOW),
        ("wl-003", NodeType.WORKLOAD, SensitivityLevel.MEDIUM),
        ("usr-001", NodeType.USER, SensitivityLevel.LOW),
        ("db-001", NodeType.ASSET, SensitivityLevel.CRITICAL),
    ]:
        graph.add_or_update_node(NodeAttributes(
            node_id=nid, node_type=ntype, sensitivity=sens,
            is_critical=(ntype == NodeType.ASSET),
        ))
    events = [
        ("wl-001", "wl-002"), ("wl-001", "wl-003"),
        ("wl-002", "db-001"), ("usr-001", "wl-001"),
        ("wl-003", "db-001"),
    ]
    for src, dst in events:
        event = TelemetryEvent(
            source_node=src, destination_node=dst,
            event_type="network_flow", port=443, bytes_transferred=1024,
        )
        graph.ingest_event(event)
    return graph


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestSimulatedGenerator:
    def test_generates_events(self):
        gen = SimulatedTelemetryGenerator(num_workloads=100, num_users=50, num_databases=5)
        events = list(gen.generate(duration_sec=0.01, start_time=datetime.utcnow()))
        assert len(events) > 0

    def test_event_has_required_fields(self):
        gen = SimulatedTelemetryGenerator(num_workloads=50, num_users=20)
        events = list(gen.generate(duration_sec=0.001))
        ev = events[0]
        assert ev.source_node
        assert ev.destination_node
        assert ev.event_type
        assert isinstance(ev.timestamp, datetime)


class TestNormalizer:
    def test_normalize_raw_dict(self):
        normalizer = TelemetryNormalizer()
        raw = {
            "event_id": "test-123",
            "timestamp": "2024-01-01T12:00:00Z",
            "source_node": "wl-001",
            "destination_node": "db-001",
            "event_type": "network_flow",
            "protocol": "tcp",
            "port": 5432,
            "bytes_transferred": 8192,
            "user": "usr-001",
            "process": "psql",
            "success": True,
        }
        event = normalizer.normalize(raw)
        assert event is not None
        assert event.source_node == "wl-001"
        assert event.destination_node == "db-001"
        assert event.protocol == "TCP"

    def test_suspicious_process_detection(self):
        normalizer = TelemetryNormalizer()
        ev = TelemetryEvent(source_node="wl-001", destination_node="wl-002", process="psexec")
        assert normalizer.is_suspicious_process(ev)
        ev2 = TelemetryEvent(source_node="wl-001", destination_node="wl-002", process="chrome")
        assert not normalizer.is_suspicious_process(ev2)


# ---------------------------------------------------------------------------
# Graph Engine
# ---------------------------------------------------------------------------

class TestSecurityGraph:
    def test_ingest_creates_nodes_and_edges(self, graph):
        ev = TelemetryEvent(
            source_node="wl-001", destination_node="db-001",
            event_type="network_flow", port=5432, bytes_transferred=2048,
        )
        graph.ingest_event(ev)
        assert graph.node_count >= 2
        assert graph.edge_count >= 1

    def test_edge_weight_in_range(self, populated_graph):
        for _, _, attrs in populated_graph.get_all_edges():
            assert 0.0 <= attrs.weight <= 1.0, f"Weight out of range: {attrs.weight}"

    def test_sensitive_nodes_identified(self, populated_graph):
        sensitive = populated_graph.sensitive_nodes()
        assert "db-001" in sensitive

    def test_prune_does_not_remove_fresh_edges(self, graph):
        ev = TelemetryEvent(source_node="a", destination_node="b", event_type="network_flow")
        graph.ingest_event(ev)
        removed = graph.prune_stale_edges()
        assert removed == 0
        assert graph.edge_count == 1

    def test_temporal_decay_reduces_weight(self, graph):
        ev = TelemetryEvent(source_node="x", destination_node="y", event_type="network_flow")
        graph.ingest_event(ev)
        edge_before = graph.get_edge_attrs("x", "y")
        w_before = edge_before.weight if edge_before else 0.0
        # Decay with a timestamp far in the past
        from datetime import timedelta
        graph.apply_temporal_decay(now=datetime.utcnow() + timedelta(hours=24))
        edge_after = graph.get_edge_attrs("x", "y")
        w_after = edge_after.weight if edge_after else 0.0
        assert w_after <= w_before


class TestTemporalManager:
    def test_record_and_retrieve(self, temporal_mgr):
        now = datetime.utcnow()
        temporal_mgr.record("a", "b", now, 1000.0)
        features = temporal_mgr.edge_features("a", "b", now)
        assert len(features) == 6
        assert any(f > 0 for f in features)

    def test_deviation_zero_for_new_edge(self, temporal_mgr):
        dev = temporal_mgr.temporal_deviation("unknown-src", "unknown-dst")
        assert dev == 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestNumpyGraphSAGE:
    def test_output_shape(self):
        model = NumpyGraphSAGEApproximation(input_dim=64, embedding_dim=32)
        features = np.random.randn(10, 64).astype(np.float32)
        adj = [[1, 2], [0], [0, 3], [2], [], [], [], [], [], []]
        out = model.forward(features, adj)
        assert out.shape == (10, 32), f"Expected (10, 32), got {out.shape}"

    def test_output_deterministic(self):
        model = NumpyGraphSAGEApproximation(seed=42)
        features = np.ones((5, 64), dtype=np.float32)
        adj = [[], [], [], [], []]
        out1 = model.forward(features, adj)
        out2 = model.forward(features, adj)
        np.testing.assert_array_equal(out1, out2)


class TestGNNInferenceEngine:
    def test_embed_returns_correct_shape(self, gnn):
        features = np.random.randn(20, 64).astype(np.float32)
        adj = [[1, 2], [0], [0]] + [[] for _ in range(17)]
        embeddings = gnn.embed(features, adj)
        assert embeddings.shape[0] == 20
        assert embeddings.shape[1] == 32

    def test_embed_no_edges(self, gnn):
        features = np.random.randn(5, 64).astype(np.float32)
        adj = [[] for _ in range(5)]
        embeddings = gnn.embed(features, adj)
        assert embeddings.shape == (5, 32)


class TestAnomalyDetector:
    def test_detect_returns_scores(self, detector, populated_graph):
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        assert len(scores) > 0
        for s in scores:
            assert 0.0 <= s.total_score

    def test_score_components_non_negative(self, detector, populated_graph):
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        for s in scores:
            assert s.behavioral_deviation >= 0.0
            assert s.temporal_deviation >= 0.0
            assert s.structural_deviation >= 0.0

    def test_flagged_edges_above_threshold(self, detector, populated_graph):
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        flagged = detector.flagged_edges(scores)
        threshold = DEFAULT_CONFIG.detection.anomaly_threshold
        for s in flagged:
            assert s.total_score > threshold

    def test_new_edge_to_critical_asset_gets_novelty_score(self, detector, graph, config):
        """Edge to a critical asset created < 1hr ago must score above threshold (R1)."""
        graph.add_or_update_node(NodeAttributes(
            node_id="wl-src", node_type=NodeType.WORKLOAD, sensitivity=SensitivityLevel.LOW
        ))
        graph.add_or_update_node(NodeAttributes(
            node_id="db-crit", node_type=NodeType.ASSET,
            sensitivity=SensitivityLevel.CRITICAL, is_critical=True
        ))
        event = TelemetryEvent(
            source_node="wl-src", destination_node="db-crit",
            event_type="network_flow", port=5432, bytes_transferred=4096,
        )
        graph.ingest_event(event)
        snapshot = graph.snapshot()
        scores = detector.detect(snapshot)
        edge_scores = [s for s in scores if s.source == "wl-src" and s.destination == "db-crit"]
        assert edge_scores, "Edge wl-src→db-crit not scored"
        assert edge_scores[0].total_score > config.detection.anomaly_threshold, (
            f"Expected score > {config.detection.anomaly_threshold}, got {edge_scores[0].total_score:.4f}"
        )
        assert edge_scores[0].is_flagged

    def test_score_details_contains_rule_info(self, detector, populated_graph):
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        for s in scores:
            assert "gnn_score" in s.details
            assert "rule_score" in s.details
            assert "novelty" in s.details


class TestRuleBasedDetector:
    def test_new_edge_to_sensitive_node_flagged(self, populated_graph):
        rd = RuleBasedDetector()
        snapshot = populated_graph.snapshot()
        # wl-002 → db-001 is a fresh edge (first_seen is now)
        now = datetime.utcnow()
        score, rule = rd.score_edge("wl-002", "db-001", snapshot, now)
        assert score >= 0.5, f"Expected novelty score ≥ 0.5, got {score}"
        assert "R1" in rule

    def test_established_edge_no_novelty_rule(self, populated_graph):
        rd = RuleBasedDetector()
        snapshot = populated_graph.snapshot()
        # Manually age the edge by 2 hours
        edge_attrs = snapshot["wl-002"]["db-001"]["attrs"]
        edge_attrs.first_seen = datetime.utcnow() - timedelta(hours=2)
        edge_attrs.last_seen = datetime.utcnow()
        now = datetime.utcnow()
        score, rule = rd.score_edge("wl-002", "db-001", snapshot, now)
        # With R1 no longer applicable and no other rules triggered, score should be low
        assert score < 0.5 or "R1" not in rule


# ---------------------------------------------------------------------------
# Concept Drift
# ---------------------------------------------------------------------------

class TestPSI:
    def test_identical_distributions_zero_psi(self):
        data = np.random.randn(500)
        assert _psi(data, data) < 0.05

    def test_shifted_distribution_high_psi(self):
        expected = np.random.randn(500)
        actual = np.random.randn(500) + 5.0  # large shift
        assert _psi(expected, actual) > 0.1


class TestConceptDriftDetector:
    def test_no_drift_on_first_check(self, config):
        detector = ConceptDriftDetector(config)
        events = detector.check_drift(now=datetime.utcnow())
        assert events == []

    def test_records_features_without_error(self, config):
        detector = ConceptDriftDetector(config)
        behavioral = np.random.randn(10, 24).astype(np.float32)
        temporal = np.random.randn(10, 16).astype(np.float32)
        structural = np.random.randn(10, 16).astype(np.float32)
        detector.record_features(datetime.utcnow(), behavioral, temporal, structural)


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------

class TestAttackPathPrioritizer:
    def test_enumerates_paths_to_sensitive_assets(self, populated_graph, config):
        prioritizer = AttackPathPrioritizer(config)
        snapshot = populated_graph.snapshot()
        compromised = {"wl-001"}
        sensitive = {"db-001"}
        paths = prioritizer.enumerate_paths(snapshot, compromised, sensitive)
        # wl-001 → wl-002 → db-001 and wl-001 → wl-003 → db-001 should be found
        assert len(paths) > 0
        targets = {p.target for p in paths}
        assert "db-001" in targets

    def test_path_probability_decreasing_with_hops(self, populated_graph, config):
        prioritizer = AttackPathPrioritizer(config)
        snapshot = populated_graph.snapshot()
        paths = prioritizer.enumerate_paths(snapshot, {"wl-001"}, {"db-001"})
        if len(paths) >= 2:
            assert paths[0].risk_score >= paths[-1].risk_score


class TestRiskScorer:
    def test_high_risk_classified_as_auto_isolate(self, config):
        scorer = RiskScorer(config)
        from src.common.models import AttackPath
        path = AttackPath(risk_score=0.9)
        assert scorer.classify(path) == "AUTO_ISOLATE"

    def test_medium_risk_classified_as_soc_alert(self, config):
        scorer = RiskScorer(config)
        from src.common.models import AttackPath
        path = AttackPath(risk_score=0.65)
        assert scorer.classify(path) == "SOC_ALERT"

    def test_low_risk_classified_as_monitor(self, config):
        scorer = RiskScorer(config)
        from src.common.models import AttackPath
        path = AttackPath(risk_score=0.1)
        assert scorer.classify(path) == "MONITOR"


class TestContainmentEngine:
    def test_generates_rules_for_high_risk_paths(self, populated_graph, config):
        engine = ContainmentEngine(config)
        prioritizer = AttackPathPrioritizer(config, min_probability=0.0)
        snapshot = populated_graph.snapshot()
        paths = prioritizer.enumerate_paths(snapshot, {"wl-001"}, {"db-001"})
        # Force high risk for test
        for p in paths:
            p.risk_score = 0.95
        actions = engine.evaluate_and_contain(paths, snapshot, {"wl-001"})
        assert len(actions) > 0

    def test_rollback_removes_rules(self, populated_graph, config):
        engine = ContainmentEngine(config)
        prioritizer = AttackPathPrioritizer(config, min_probability=0.0)
        snapshot = populated_graph.snapshot()
        paths = prioritizer.enumerate_paths(snapshot, {"wl-001"}, {"db-001"})
        for p in paths:
            p.risk_score = 0.95
        actions = engine.evaluate_and_contain(paths, snapshot, {"wl-001"})
        if actions and actions[0].auto_applied:
            ok = engine.rollback(actions[0].action_id)
            assert ok


# ---------------------------------------------------------------------------
# SOC Integration
# ---------------------------------------------------------------------------

class TestAlertManager:
    def test_creates_alert(self, config, populated_graph, detector):
        mgr = AlertManager(config)
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        alert = mgr.create_alert(anomaly_scores=scores, attack_paths=[])
        assert alert.alert_id
        assert alert.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_acknowledge_alert(self, config, detector, populated_graph):
        mgr = AlertManager(config)
        snapshot = populated_graph.snapshot()
        scores = detector.detect(snapshot)
        alert = mgr.create_alert(anomaly_scores=scores, attack_paths=[])
        ok = mgr.acknowledge(alert.alert_id)
        assert ok
        retrieved = mgr.get_alerts()
        ack_alert = next(a for a in retrieved if a.alert_id == alert.alert_id)
        assert ack_alert.acknowledged


# ---------------------------------------------------------------------------
# Integration: end-to-end pipeline smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_pipeline_runs_without_error(self):
        from src.main import AISecurityGraphPipeline
        pipeline = AISecurityGraphPipeline(DEFAULT_CONFIG)
        gen = SimulatedTelemetryGenerator(num_workloads=200, num_users=50, num_databases=5)
        events = list(gen.generate(duration_sec=0.1))
        for ev in events[:100]:
            pipeline.ingest(ev)
        assert pipeline.graph.node_count > 0
        assert pipeline.graph.edge_count > 0
        assert pipeline._event_counter == min(100, len(events))

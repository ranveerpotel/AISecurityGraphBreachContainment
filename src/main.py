"""
Main pipeline orchestrator.

Wires all five components:
  A. Telemetry Collection
  B. Security Graph Engine
  C. AI Detection Module
  D. Risk Scoring & Containment
  E. SOC Integration

Runs in a continuous processing loop. Can be started with:
  python -m src.main
  python -m src.main --simulate        (use built-in event generator)
  python -m src.main --api-only        (start REST API only)
"""
from __future__ import annotations
import argparse
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from .common.config import AppConfig, DEFAULT_CONFIG
from .common.models import TelemetryEvent
from .telemetry.collector import SimulatedTelemetryGenerator, KafkaTelemetryConsumer
from .telemetry.normalizer import TelemetryNormalizer
from .graph_engine.security_graph import SecurityGraph
from .graph_engine.temporal_manager import TemporalManager
from .detection.graphsage_model import GNNInferenceEngine
from .detection.anomaly_detector import AnomalyDetector
from .detection.concept_drift import ConceptDriftDetector
from .containment.path_prioritizer import AttackPathPrioritizer
from .containment.containment_engine import ContainmentEngine
from .soc_integration.alert_manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Detection runs every N ingested events (balances latency vs CPU cost)
_DETECTION_INTERVAL = 500
# Decay applied every N events
_DECAY_INTERVAL = 5_000
# Drift check every N events
_DRIFT_CHECK_INTERVAL = 100_000


class AISecurityGraphPipeline:
    """
    Coordinates all five system components in a single pipeline.
    Thread-safe: graph ingestion runs on the main thread; API runs on a daemon thread.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        logger.info("Initialising AI Security Graph pipeline...")

        # Component B: Security Graph Engine
        self.graph = SecurityGraph(config)
        self.temporal_mgr = TemporalManager()

        # Component C: AI Detection
        gnn_cfg = config.detection.gnn
        self.gnn = GNNInferenceEngine(
            input_dim=gnn_cfg.input_dim,
            hidden_dim_1=gnn_cfg.hidden_dim_1,
            hidden_dim_2=gnn_cfg.hidden_dim_2,
            embedding_dim=gnn_cfg.embedding_dim,
            dropout=gnn_cfg.dropout,
            use_gpu=config.use_gpu,
        )
        self.detector = AnomalyDetector(self.gnn, self.temporal_mgr, config)
        self.drift_detector = ConceptDriftDetector(config)

        # Component D: Containment
        self.prioritizer = AttackPathPrioritizer(config)
        self.containment_engine = ContainmentEngine(config)

        # Component E: SOC Integration
        self.alert_manager = AlertManager(config)
        self.normalizer = TelemetryNormalizer()

        self._event_counter = 0
        self._running = False

    def ingest(self, event: TelemetryEvent) -> None:
        """Process a single telemetry event through the pipeline."""
        # Component A → B: update graph
        src_attrs, dst_attrs = self.normalizer.extract_node_attributes(event)
        self.graph.add_or_update_node(src_attrs)
        self.graph.add_or_update_node(dst_attrs)
        self.graph.ingest_event(event)
        self.temporal_mgr.record(
            event.source_node,
            event.destination_node,
            event.timestamp,
            float(event.bytes_transferred),
        )

        self._event_counter += 1

        # Periodic decay
        if self._event_counter % _DECAY_INTERVAL == 0:
            self.graph.apply_temporal_decay()

        # Component C: run detection every N events
        if self._event_counter % _DETECTION_INTERVAL == 0:
            self._run_detection()

    def _run_detection(self) -> None:
        snapshot = self.graph.snapshot()
        if snapshot.number_of_nodes() < 2:
            return

        # Anomaly detection
        scores = self.detector.detect(snapshot)
        flagged = self.detector.flagged_edges(scores)

        if not flagged:
            return

        # Identify compromised nodes: only the highest-confidence flagged sources.
        # Using top-5 by score prevents the false-positive flood from the cold-start
        # novelty scorer, which flags many new edges simultaneously on first detection pass.
        sensitive = self.graph.sensitive_nodes()
        top_flagged = sorted(flagged, key=lambda s: s.total_score, reverse=True)[:5]
        compromised = {s.source for s in top_flagged if s.total_score > 0.85}
        if not compromised:
            # Fall back to single highest-scoring edge source only
            compromised = {top_flagged[0].source} if top_flagged else set()

        # Component D: attack path prioritisation and containment
        paths = self.prioritizer.enumerate_paths(snapshot, compromised, sensitive)
        actions = self.containment_engine.evaluate_and_contain(paths, snapshot, compromised)

        # Component E: create and dispatch alert
        if flagged:
            alert = self.alert_manager.create_alert(
                anomaly_scores=scores,
                attack_paths=paths,
                containment=actions[0] if actions else None,
            )
            logger.info(
                "Alert %s created | severity=%s | compromised=%d | paths=%d | actions=%d",
                alert.alert_id[:8], alert.severity, len(compromised), len(paths), len(actions),
            )

        # Drift monitoring
        if self._event_counter % _DRIFT_CHECK_INTERVAL == 0:
            self._check_drift(scores)

    def _check_drift(self, scores) -> None:
        if not scores:
            return
        import numpy as np
        behavioral = np.array([[s.behavioral_deviation] for s in scores], dtype=np.float32)
        temporal = np.array([[s.temporal_deviation] for s in scores], dtype=np.float32)
        structural = np.array([[s.structural_deviation] for s in scores], dtype=np.float32)
        self.drift_detector.record_features(datetime.utcnow(), behavioral, temporal, structural)
        drift_events = self.drift_detector.check_drift()
        for de in drift_events:
            logger.warning("Drift event: feature_group=%s PSI=%.4f", de.feature_group, de.psi_score)
            self.drift_detector.start_shadow_mode()

    def run_simulation(
        self,
        duration_sec: float = 300.0,
        event_rate: float = 1_000.0,
    ) -> None:
        """Run the full pipeline on simulated telemetry."""
        logger.info("Starting simulation: duration=%.0fs rate=%.0f events/s", duration_sec, event_rate)
        generator = SimulatedTelemetryGenerator(event_rate_per_sec=event_rate)
        self._running = True
        t0 = time.time()
        for event in generator.generate(duration_sec=duration_sec):
            if not self._running:
                break
            self.ingest(event)
        elapsed = time.time() - t0
        logger.info(
            "Simulation complete: %d events in %.1fs (%.0f events/s) | graph: %d nodes, %d edges",
            self._event_counter, elapsed, self._event_counter / max(elapsed, 1),
            self.graph.node_count, self.graph.edge_count,
        )

    def start_api(self, block: bool = True) -> None:
        """Start the FastAPI REST server."""
        try:
            import uvicorn
            from .soc_integration.dashboard_api import create_app
        except ImportError:
            logger.error("Install fastapi and uvicorn: pip install fastapi uvicorn")
            return

        app = create_app(
            alert_manager=self.alert_manager,
            containment_engine=self.containment_engine,
            security_graph=self.graph,
            drift_detector=self.drift_detector,
            config=self._cfg,
        )
        host, port = self._cfg.api_host, self._cfg.api_port
        logger.info("Starting API server at http://%s:%d", host, port)
        if block:
            uvicorn.run(app, host=host, port=port, log_level="info")
        else:
            t = threading.Thread(
                target=uvicorn.run,
                kwargs={"app": app, "host": host, "port": port, "log_level": "error"},
                daemon=True,
            )
            t.start()

    def stop(self) -> None:
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Security Graph — Breach Containment")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated telemetry")
    parser.add_argument("--api-only", action="store_true", help="Start REST API only")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (sec)")
    parser.add_argument("--rate", type=float, default=1000.0, help="Simulated events/sec")
    args = parser.parse_args()

    pipeline = AISecurityGraphPipeline(DEFAULT_CONFIG)

    if args.api_only:
        pipeline.start_api(block=True)
        return

    # Start API in background
    pipeline.start_api(block=False)

    if args.simulate:
        pipeline.run_simulation(duration_sec=args.duration, event_rate=args.rate)
    else:
        # Kafka consumer mode
        consumer = KafkaTelemetryConsumer(DEFAULT_CONFIG)
        consumer.start()
        logger.info("Consuming from Kafka topic '%s'...", DEFAULT_CONFIG.kafka.telemetry_topic)
        try:
            consumer.consume(pipeline.ingest)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            consumer.stop()


if __name__ == "__main__":
    main()

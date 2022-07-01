"""
Anomaly detection module implementing §4.3 and Algorithm 2 from the paper.

Anomaly score per edge:
  A(i,j) = alpha_b * D_b + alpha_t * D_t + alpha_g * D_g

Where:
  D_b = behavioral deviation (vs historical baseline embeddings)
        + novelty component (cold-start: new edges to sensitive targets score high)
  D_t = temporal deviation   (current rate vs 24-hr average)
  D_g = structural deviation (graph-structural embedding distance)

Edges with A(i,j) > tau are flagged as suspicious.

RuleBasedDetector (§8.1 ensemble):
  Detects lateral movement tools, critical-asset access by new nodes, and
  anomalous admin-port usage independent of GNN training state.
  Scores are merged with GNN scores via max() before thresholding.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

from .graphsage_model import GNNInferenceEngine
from ..common.models import AnomalyScore, NodeFeatureVector
from ..common.config import AppConfig, DEFAULT_CONFIG, NodeType, SensitivityLevel
from ..graph_engine.temporal_manager import TemporalManager

logger = logging.getLogger(__name__)

# Lateral-movement tool keywords matched against the 'process' metadata field
_LM_TOOLS: Set[str] = {"psexec", "wmic", "winrm", "dcom", "mimikatz", "cobalt"}
# Admin protocols that should rarely originate from user/workload nodes
_ADMIN_PORTS: Set[int] = {22, 3389, 445, 135, 5985, 5986}
# Minimum age (hours) for an edge to be considered "established" (not novel)
_NOVELTY_AGE_HOURS = 1.0
# Novelty boost for brand-new edges (< 1 hr old).
# Sensitive/critical targets: 0.75 → always exceeds the flag threshold (0.70).
# A new workload→critical-DB connection is inherently suspicious and should be flagged.
_NOVELTY_SENSITIVE_BOOST = 0.75
_NOVELTY_NORMAL_BOOST = 0.20


class FeatureExtractor:
    """Builds the 64-dimensional node feature vector required by the GNN."""

    def __init__(self, temporal_manager: TemporalManager) -> None:
        self._tm = temporal_manager

    def extract(self, node_id: str, graph: nx.DiGraph, now: Optional[datetime] = None) -> NodeFeatureVector:
        now = now or datetime.utcnow()
        node_data = graph.nodes.get(node_id, {})
        attrs = node_data.get("attrs")

        # ---- 24D behavioral ------------------------------------------------
        out_degree = graph.out_degree(node_id)
        in_degree = graph.in_degree(node_id)
        peers_out = len(list(graph.successors(node_id)))
        peers_in = len(list(graph.predecessors(node_id)))
        out_weights = [graph[node_id][n]["attrs"].weight for n in graph.successors(node_id)]
        in_weights = [graph[n][node_id]["attrs"].weight for n in graph.predecessors(node_id)]
        total_out_bytes = sum(
            graph[node_id][n]["attrs"].bytes_total for n in graph.successors(node_id)
        )
        total_in_bytes = sum(
            graph[n][node_id]["attrs"].bytes_total for n in graph.predecessors(node_id)
        )
        max_out_w = max(out_weights, default=0.0)
        mean_out_w = float(np.mean(out_weights)) if out_weights else 0.0
        max_in_w = max(in_weights, default=0.0)
        mean_in_w = float(np.mean(in_weights)) if in_weights else 0.0
        proto_div = len(
            set(p for n in graph.successors(node_id) for p in graph[node_id][n]["attrs"].protocols)
        )
        behavioral = [
            float(out_degree), float(in_degree),
            float(peers_out), float(peers_in),
            max_out_w, mean_out_w, max_in_w, mean_in_w,
            float(total_out_bytes) / 1e6, float(total_in_bytes) / 1e6,
            float(proto_div),
        ] + [0.0] * 13  # pad to 24D

        # ---- 16D temporal --------------------------------------------------
        node_tf = self._tm.node_temporal_features(node_id)   # 4D
        hour_sin = float(np.sin(2 * np.pi * now.hour / 24))
        hour_cos = float(np.cos(2 * np.pi * now.hour / 24))
        dow_sin = float(np.sin(2 * np.pi * now.weekday() / 7))
        dow_cos = float(np.cos(2 * np.pi * now.weekday() / 7))
        temporal = node_tf + [hour_sin, hour_cos, dow_sin, dow_cos] + [0.0] * 8

        # ---- 16D structural ------------------------------------------------
        try:
            pr = nx.pagerank(graph, max_iter=50).get(node_id, 0.0)
        except Exception:
            pr = 0.0
        try:
            cc = nx.clustering(nx.Graph(graph), node_id)
        except Exception:
            cc = 0.0
        structural = [
            pr, cc,
            float(out_degree) / max(1, graph.number_of_nodes()),
            float(in_degree) / max(1, graph.number_of_nodes()),
        ] + [0.0] * 12

        # ---- 8D node type --------------------------------------------------
        if attrs:
            nt_vec = [
                1.0 if attrs.node_type == NodeType.WORKLOAD else 0.0,
                1.0 if attrs.node_type == NodeType.USER else 0.0,
                1.0 if attrs.node_type == NodeType.ASSET else 0.0,
                float(attrs.sensitivity.value) / 4.0,
                1.0 if attrs.is_critical else 0.0,
            ] + [0.0] * 3
        else:
            nt_vec = [0.0] * 8

        return NodeFeatureVector(
            node_id=node_id,
            behavioral=behavioral[:24],
            temporal=temporal[:16],
            structural=structural[:16],
            node_type_enc=nt_vec[:8],
            timestamp=now,
        )


# ---------------------------------------------------------------------------
# Rule-based detection layer (§8.1 ensemble component)
# ---------------------------------------------------------------------------

class RuleBasedDetector:
    """
    Lightweight heuristic detector that complements the GNN.
    Works without a trained model and catches obvious lateral-movement patterns
    the paper describes defending against via ensemble (§8.1).

    Rules:
      R1 – New edge (< _NOVELTY_AGE_HOURS old) to a HIGH/CRITICAL sensitivity node → high score
      R2 – Edge metadata contains a lateral-movement tool name (psexec, wmic, etc.)
      R3 – Admin port (22, 3389, 445, …) used by a WORKLOAD/USER node for the first time
      R4 – Sudden spike: out-degree of a node doubles within the current window
    """

    def __init__(self) -> None:
        # Track per-node baseline out-degree to detect R4 spikes
        self._baseline_out_degree: Dict[str, float] = {}
        self._ema_alpha = 0.1

    def score_edge(
        self,
        src: str,
        dst: str,
        graph: nx.DiGraph,
        now: datetime,
    ) -> Tuple[float, str]:
        """
        Returns (rule_score ∈ [0,1], rule_name_that_triggered).
        rule_score = 0 means no rule matched.
        """
        edge_data = graph.edges.get((src, dst), {})
        edge_attrs = edge_data.get("attrs")
        dst_data = graph.nodes.get(dst, {})
        dst_attrs = dst_data.get("attrs")
        src_data = graph.nodes.get(src, {})
        src_attrs = src_data.get("attrs")

        if edge_attrs is None:
            return 0.0, ""

        # R1 — new edge to sensitive/critical target
        age_hours = (now - edge_attrs.first_seen).total_seconds() / 3600.0
        if age_hours < _NOVELTY_AGE_HOURS:
            freshness = 1.0 - (age_hours / _NOVELTY_AGE_HOURS)
            if dst_attrs is not None and (
                dst_attrs.sensitivity >= SensitivityLevel.HIGH or dst_attrs.is_critical
            ):
                return _NOVELTY_SENSITIVE_BOOST * freshness, "R1:new_edge_to_sensitive"
            return _NOVELTY_NORMAL_BOOST * freshness, "R1:new_edge"

        # R2 — lateral movement tool in edge metadata
        lm_tool = edge_attrs.metadata.get("process", "").lower()
        if any(tool in lm_tool for tool in _LM_TOOLS):
            return 0.80, "R2:lateral_movement_tool"

        # R3 — admin port used by a workload/user node it hasn't used before
        if edge_attrs.event_count <= 2:  # practically first time
            for port in edge_attrs.metadata.get("ports", []):
                if port in _ADMIN_PORTS:
                    src_type = src_attrs.node_type if src_attrs else NodeType.WORKLOAD
                    if src_type != NodeType.ASSET:
                        return 0.65, f"R3:admin_port_{port}"

        # R4 — out-degree spike: current out-degree > 2× baseline
        current_out = graph.out_degree(src)
        baseline = self._baseline_out_degree.get(src, float(current_out))
        if baseline > 0 and current_out > 2.5 * baseline:
            return 0.60, "R4:degree_spike"

        # Update baseline out-degree EMA
        self._baseline_out_degree[src] = (
            (1 - self._ema_alpha) * self._baseline_out_degree.get(src, float(current_out))
            + self._ema_alpha * float(current_out)
        )
        return 0.0, ""

    def score_all(
        self, graph: nx.DiGraph, now: Optional[datetime] = None
    ) -> Dict[Tuple[str, str], Tuple[float, str]]:
        now = now or datetime.utcnow()
        return {
            (u, v): self.score_edge(u, v, graph, now)
            for u, v in graph.edges()
        }


# ---------------------------------------------------------------------------
# Main detector (GNN + rule-based ensemble)
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Algorithm 2 from the paper — GNN + rule-based ensemble.

    Steps:
      1. Extract 64D feature vectors for all nodes.
      2. Run GraphSAGE to get 32D node embeddings.
      3. Maintain rolling EMA baseline embeddings per node.
      4. For each edge: compute D_b (+ novelty), D_t, D_g → GNN score.
      5. Run rule-based detector in parallel.
      6. Final score = max(GNN_score, rule_score) — conservative ensemble.
      7. Flag edges where final score exceeds threshold.
    """

    def __init__(
        self,
        gnn: GNNInferenceEngine,
        temporal_manager: TemporalManager,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        self._gnn = gnn
        self._tm = temporal_manager
        self._cfg_d = config.detection
        self._extractor = FeatureExtractor(temporal_manager)
        self._rule_detector = RuleBasedDetector()
        # EMA baseline embeddings: slow drift so sustained anomalies keep scoring high
        self._baseline_embeddings: Dict[str, np.ndarray] = {}
        self._ema_alpha = 0.05
        # Track how many detection passes each node has seen (for cold-start novelty)
        self._node_passes: Dict[str, int] = {}

    def detect(self, graph: nx.DiGraph, now: Optional[datetime] = None) -> List[AnomalyScore]:
        """
        Full detection pass. Returns scores sorted by total_score descending.
        """
        now = now or datetime.utcnow()
        nodes = list(graph.nodes())
        if not nodes:
            return []

        node_index = {n: i for i, n in enumerate(nodes)}

        # ---- Build feature matrix ------------------------------------------
        feature_vecs = []
        for n in nodes:
            fv = self._extractor.extract(n, graph, now)
            feature_vecs.append(fv.to_vector())
        feature_matrix = np.array(feature_vecs, dtype=np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=0.0)

        # ---- GNN inference -------------------------------------------------
        adj = [
            [node_index[nbr] for nbr in graph.successors(n) if nbr in node_index]
            for n in nodes
        ]
        embeddings = self._gnn.embed(feature_matrix, adj)   # (N, 32)

        # ---- Update EMA baselines (do NOT update until 2nd pass to get D_b) -
        first_pass_nodes: Set[str] = set()
        for i, n in enumerate(nodes):
            emb = embeddings[i]
            if n in self._baseline_embeddings:
                self._baseline_embeddings[n] = (
                    (1 - self._ema_alpha) * self._baseline_embeddings[n]
                    + self._ema_alpha * emb
                )
            else:
                # First time: store as baseline but mark as unseen
                self._baseline_embeddings[n] = emb.copy()
                first_pass_nodes.add(n)
            self._node_passes[n] = self._node_passes.get(n, 0) + 1

        # ---- Rule-based scores ---------------------------------------------
        rule_scores = self._rule_detector.score_all(graph, now)

        # ---- Score each edge -----------------------------------------------
        scores: List[AnomalyScore] = []
        for u, v in graph.edges():
            score = self._score_edge(
                u, v, graph, embeddings, node_index, now,
                first_pass_nodes, rule_scores
            )
            scores.append(score)

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def _score_edge(
        self,
        src: str,
        dst: str,
        graph: nx.DiGraph,
        embeddings: np.ndarray,
        node_index: Dict[str, int],
        now: datetime,
        first_pass_nodes: Set[str],
        rule_scores: Dict[Tuple[str, str], Tuple[float, str]],
    ) -> AnomalyScore:
        cfg = self._cfg_d
        i, j = node_index.get(src, -1), node_index.get(dst, -1)
        if i < 0 or j < 0:
            return AnomalyScore(source=src, destination=dst)

        emb_i = embeddings[i]
        emb_j = embeddings[j]

        # D_b: embedding distance from baseline.
        # On first pass D_b = 0 (no history). We supplement with edge novelty
        # so the cold-start period still catches new connections to sensitive assets.
        d_b_embed = self._embedding_distance(
            emb_i, self._baseline_embeddings.get(src, emb_i)
        ) if src not in first_pass_nodes else 0.0

        # Novelty boost: new edges that target sensitive nodes get elevated D_b
        edge_attrs = graph.edges.get((src, dst), {}).get("attrs")
        dst_attrs = graph.nodes.get(dst, {}).get("attrs")
        novelty = self._novelty_score(edge_attrs, dst_attrs, now)
        d_b = max(d_b_embed, novelty)

        # D_t: temporal deviation
        d_t = self._tm.temporal_deviation(src, dst)

        # D_g: structural deviation between peer embeddings
        d_g = self._embedding_distance(emb_i, emb_j)

        gnn_score = cfg.alpha_b * d_b + cfg.alpha_t * d_t + cfg.alpha_g * d_g

        # Ensemble: take max of GNN score and rule-based score
        rule_score, rule_name = rule_scores.get((src, dst), (0.0, ""))
        total = max(gnn_score, rule_score)

        details = {
            "gnn_score": round(gnn_score, 4),
            "rule_score": round(rule_score, 4),
            "rule_triggered": rule_name,
            "d_b_embed": round(d_b_embed, 4),
            "novelty": round(novelty, 4),
            "threshold": cfg.anomaly_threshold,
        }

        return AnomalyScore(
            source=src,
            destination=dst,
            timestamp=now,
            behavioral_deviation=float(d_b),
            temporal_deviation=float(d_t),
            structural_deviation=float(d_g),
            total_score=float(total),
            is_flagged=total > cfg.anomaly_threshold,
            details=details,
        )

    @staticmethod
    def _novelty_score(edge_attrs, dst_attrs, now: datetime) -> float:
        """
        Returns a novelty contribution for D_b.
        New edges (< 1 hr) to sensitive/critical targets get a high score,
        reflecting that in a trained deployment these connections would deviate
        from the established behavioral baseline.
        """
        if edge_attrs is None:
            return 0.0
        age_hours = (now - edge_attrs.first_seen).total_seconds() / 3600.0
        if age_hours >= _NOVELTY_AGE_HOURS:
            return 0.0
        # Freshness factor: score decays from 1.0 → 0 as edge ages to threshold
        freshness = 1.0 - (age_hours / _NOVELTY_AGE_HOURS)
        if dst_attrs is not None and (
            dst_attrs.sensitivity >= SensitivityLevel.HIGH or dst_attrs.is_critical
        ):
            return _NOVELTY_SENSITIVE_BOOST * freshness
        return _NOVELTY_NORMAL_BOOST * freshness

    @staticmethod
    def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised cosine distance in [0, 1]."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
        return (1.0 - cosine_sim) / 2.0

    def flagged_edges(self, scores: List[AnomalyScore]) -> List[AnomalyScore]:
        return [s for s in scores if s.is_flagged]

    def reset_baseline(self) -> None:
        self._baseline_embeddings.clear()
        self._node_passes.clear()

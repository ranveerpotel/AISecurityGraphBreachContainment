"""
Dynamic security graph G(t) implementing:
  - Weighted edges: w(e) = alpha*f_norm + beta*s + gamma*d
  - Exponential temporal decay: weight *= exp(-lambda * delta_t_hours)
  - Sliding windows: 5-min current, 24-hr short, 7-day long
  - Edge pruning after 7 days of inactivity
  - Node sensitivity classification
"""
from __future__ import annotations
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..common.models import (
    NodeAttributes,
    EdgeAttributes,
    TelemetryEvent,
)
from ..common.config import (
    AppConfig,
    DEFAULT_CONFIG,
    EdgeType,
    NodeType,
    SensitivityLevel,
)

logger = logging.getLogger(__name__)


class SecurityGraph:
    """
    Thread-safe dynamic directed graph for the hybrid cloud security domain.

    Nodes carry NodeAttributes; edges carry EdgeAttributes with a composite
    weight updated on every ingested TelemetryEvent.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config.graph
        self._G: nx.DiGraph = nx.DiGraph()
        self._lock = RLock()
        # Per-edge frequency history for baseline deviation computation
        self._edge_history: Dict[Tuple[str, str], List[Tuple[datetime, int]]] = defaultdict(list)
        # Per-node event counts for structural features
        self._node_event_counts: Dict[str, int] = defaultdict(int)
        self._last_pruned: datetime = datetime.utcnow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_or_update_node(self, attrs: NodeAttributes) -> None:
        with self._lock:
            if self._G.has_node(attrs.node_id):
                existing = self._G.nodes[attrs.node_id]["attrs"]
                # Retain higher sensitivity classification
                if attrs.sensitivity > existing.sensitivity:
                    existing.sensitivity = attrs.sensitivity
                    existing.is_critical = attrs.is_critical
            else:
                self._G.add_node(attrs.node_id, attrs=attrs)

    def ingest_event(self, event: TelemetryEvent) -> None:
        """Process one telemetry event: update graph structure and edge weights."""
        src, dst = event.source_node, event.destination_node
        if src == dst:
            return

        with self._lock:
            self._ensure_nodes(src, dst, event)
            self._update_edge(src, dst, event)
            self._node_event_counts[src] += 1
            self._node_event_counts[dst] += 1

        # Periodic pruning (once per 6 hours of wall-clock time)
        now = datetime.utcnow()
        if (now - self._last_pruned).total_seconds() > 21_600:
            self.prune_stale_edges()
            self._last_pruned = now

    def prune_stale_edges(self) -> int:
        """Remove edges not seen in `prune_after_days` days. Returns count removed."""
        cutoff = datetime.utcnow() - timedelta(days=self._cfg.prune_after_days)
        to_remove = []
        with self._lock:
            for u, v, data in self._G.edges(data=True):
                edge_attrs: EdgeAttributes = data["attrs"]
                if edge_attrs.last_seen < cutoff:
                    to_remove.append((u, v))
            for u, v in to_remove:
                self._G.remove_edge(u, v)
            if to_remove:
                logger.info("Pruned %d stale edges", len(to_remove))
        return len(to_remove)

    def apply_temporal_decay(self, now: Optional[datetime] = None) -> None:
        """Apply exponential decay to all edge weights (call periodically)."""
        now = now or datetime.utcnow()
        with self._lock:
            for _, _, data in self._G.edges(data=True):
                attrs: EdgeAttributes = data["attrs"]
                delta_hours = (now - attrs.last_seen).total_seconds() / 3600.0
                decay = math.exp(-self._cfg.decay_lambda * delta_hours)
                attrs.weight = max(0.0, attrs.weight * decay)

    def get_node_attrs(self, node_id: str) -> Optional[NodeAttributes]:
        with self._lock:
            node = self._G.nodes.get(node_id)
            return node["attrs"] if node else None

    def get_edge_attrs(self, src: str, dst: str) -> Optional[EdgeAttributes]:
        with self._lock:
            edge = self._G.edges.get((src, dst))
            return edge["attrs"] if edge else None

    def neighbors_of(self, node_id: str) -> List[str]:
        with self._lock:
            return list(self._G.successors(node_id))

    def predecessors_of(self, node_id: str) -> List[str]:
        with self._lock:
            return list(self._G.predecessors(node_id))

    def sensitive_nodes(self) -> Set[str]:
        with self._lock:
            return {
                n
                for n, d in self._G.nodes(data=True)
                if d["attrs"].sensitivity >= SensitivityLevel.HIGH or d["attrs"].is_critical
            }

    @property
    def node_count(self) -> int:
        return self._G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._G.number_of_edges()

    def snapshot(self) -> nx.DiGraph:
        """Return a shallow copy of the current graph (for GNN inference)."""
        with self._lock:
            return self._G.copy()

    def get_all_edges(self) -> List[Tuple[str, str, EdgeAttributes]]:
        with self._lock:
            return [(u, v, d["attrs"]) for u, v, d in self._G.edges(data=True)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_nodes(self, src: str, dst: str, event: TelemetryEvent) -> None:
        for nid in (src, dst):
            if not self._G.has_node(nid):
                node_type = NodeType.WORKLOAD
                sensitivity = SensitivityLevel.LOW
                is_critical = False
                if nid.startswith("db-"):
                    node_type = NodeType.ASSET
                    sensitivity = SensitivityLevel.CRITICAL
                    is_critical = True
                elif nid.startswith("usr-"):
                    node_type = NodeType.USER
                attrs = NodeAttributes(
                    node_id=nid,
                    node_type=node_type,
                    sensitivity=sensitivity,
                    is_critical=is_critical,
                )
                self._G.add_node(nid, attrs=attrs)

    def _update_edge(self, src: str, dst: str, event: TelemetryEvent) -> None:
        now = event.timestamp
        dst_attrs: NodeAttributes = self._G.nodes[dst]["attrs"]

        if self._G.has_edge(src, dst):
            attrs: EdgeAttributes = self._G[src][dst]["attrs"]
            attrs.event_count += 1
            attrs.bytes_total += event.bytes_transferred
            attrs.last_seen = now
            if event.protocol and event.protocol not in attrs.protocols:
                attrs.protocols.append(event.protocol)
        else:
            attrs = EdgeAttributes(
                source=src,
                destination=dst,
                edge_type=EdgeType.COMMUNICATION,
                event_count=1,
                bytes_total=event.bytes_transferred,
                last_seen=now,
                first_seen=now,
                protocols=[event.protocol] if event.protocol else [],
            )
            self._G.add_edge(src, dst, attrs=attrs)

        # Record history entry for deviation computation
        self._edge_history[(src, dst)].append((now, attrs.event_count))
        # Keep only last 7 days of history entries
        cutoff = now - timedelta(days=7)
        self._edge_history[(src, dst)] = [
            (t, c) for t, c in self._edge_history[(src, dst)] if t >= cutoff
        ]

        attrs.frequency_score = self._compute_frequency_score(src, dst)
        attrs.sensitivity_score = dst_attrs.sensitivity.value / SensitivityLevel.CRITICAL.value
        attrs.deviation_score = self._compute_deviation_score(src, dst, now)
        attrs.weight = self._composite_weight(attrs)

    def _compute_frequency_score(self, src: str, dst: str) -> float:
        """Normalize event count relative to the busiest edge from this source."""
        count = self._G[src][dst]["attrs"].event_count
        max_count = max(
            (self._G[src][n]["attrs"].event_count for n in self._G.successors(src)),
            default=1,
        )
        return count / max_count if max_count > 0 else 0.0

    def _compute_deviation_score(self, src: str, dst: str, now: datetime) -> float:
        """
        d(e): how much current activity deviates from the 24-hour baseline.
        Simple ratio of recent (5-min window) rate vs 24-hour average rate.
        """
        history = self._edge_history[(src, dst)]
        window_5min = now - timedelta(minutes=5)
        window_24hr = now - timedelta(hours=24)
        recent = sum(1 for t, _ in history if t >= window_5min)
        baseline_period = [t for t, _ in history if t >= window_24hr]
        if not baseline_period:
            return 0.0
        duration_5min = 5 / (24 * 60)  # fraction of day
        baseline_rate = len(baseline_period) / (24 * 60)  # events per minute baseline
        recent_rate = recent / 5.0 if recent > 0 else 0.0
        if baseline_rate == 0:
            return min(1.0, recent_rate)
        deviation = abs(recent_rate - baseline_rate) / baseline_rate
        return min(1.0, deviation)

    def _composite_weight(self, attrs: EdgeAttributes) -> float:
        cfg = self._cfg
        return (
            cfg.alpha * attrs.frequency_score
            + cfg.beta * attrs.sensitivity_score
            + cfg.gamma * attrs.deviation_score
        )

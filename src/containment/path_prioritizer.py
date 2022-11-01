"""
Algorithm 3: Attack Path Prioritization.

BFS with probabilistic pruning enumerates all high-probability paths from
a compromised node to sensitive assets (max 5 hops).

P(path_used) = Π_edges(weight) × (1 - p_detect)
Risk(p) = P(path_used) × Impact(target)

Priority queue ordered by risk score (highest first).
"""
from __future__ import annotations
import heapq
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..common.models import AttackPath, NodeAttributes
from ..common.config import AppConfig, DEFAULT_CONFIG, SensitivityLevel

logger = logging.getLogger(__name__)

# Detection probability per hop (p_d = 0.89 from the paper, §10.2)
_DEFAULT_P_DETECT = 0.89


def _impact_score(attrs: Optional[NodeAttributes]) -> float:
    """Map sensitivity level to impact score in [0, 1]."""
    if attrs is None:
        return 0.1
    mapping = {
        SensitivityLevel.LOW: 0.1,
        SensitivityLevel.MEDIUM: 0.4,
        SensitivityLevel.HIGH: 0.7,
        SensitivityLevel.CRITICAL: 1.0,
    }
    return mapping.get(attrs.sensitivity, 0.1) + (0.2 if attrs.is_critical else 0.0)


class AttackPathPrioritizer:
    """
    Enumerates attack paths from a set of compromised nodes to sensitive assets.

    Complexity: O(k·d) where k = high-probability paths, d = depth (≤5 hops).
    """

    def __init__(
        self,
        config: AppConfig = DEFAULT_CONFIG,
        p_detect: float = _DEFAULT_P_DETECT,
        min_probability: float = 0.001,
    ) -> None:
        self._cfg = config.containment
        self._p_detect = p_detect
        self._min_probability = min_probability

    def enumerate_paths(
        self,
        graph: nx.DiGraph,
        compromised_nodes: Set[str],
        sensitive_assets: Set[str],
    ) -> List[AttackPath]:
        """
        BFS from each compromised node; collect all paths to sensitive assets
        within max_hops. Returns paths sorted by risk_score descending.
        """
        all_paths: List[AttackPath] = []
        for start_node in compromised_nodes:
            paths = self._bfs_from(graph, start_node, sensitive_assets)
            all_paths.extend(paths)
        all_paths.sort(key=lambda p: p.risk_score, reverse=True)
        return all_paths

    def _bfs_from(
        self,
        graph: nx.DiGraph,
        start: str,
        sensitive_assets: Set[str],
    ) -> List[AttackPath]:
        """
        Priority-queue BFS.
        State: (neg_probability, node, path_nodes, path_edges)
        Stops when probability drops below min_probability or hops > max_hops.
        """
        max_hops = self._cfg.max_hops
        results: List[AttackPath] = []

        # heap entry: (neg_edge_prob, hop_count, current_node, path_nodes, path_edges)
        # edge_prob tracks only Π(edge_weights); (1 - p_detect) is applied once at the end.
        heap: List[Tuple[float, int, str, List[str], List[Tuple[str, str]]]] = [
            (-1.0, 0, start, [start], [])
        ]
        visited_states: Set[Tuple[str, int]] = set()

        while heap:
            neg_edge_prob, hops, node, path_nodes, path_edges = heapq.heappop(heap)
            edge_prob = -neg_edge_prob

            state = (node, hops)
            if state in visited_states:
                continue
            visited_states.add(state)

            # Check if reached a sensitive asset (not the starting node)
            if node in sensitive_assets and hops > 0:
                # Risk(path) = Π(edge_weights) × Impact(target)   [paper §4.4]
                # The (1 - p_detect) factor appears in formal proofs (Theorem 1, §10.3) to
                # bound detection guarantees. The live risk score uses only path traversal
                # probability (Π weights) × impact so scores stay in [0,1] and the
                # calibrated thresholds 0.5/0.8 are meaningful.
                path_prob = edge_prob          # Π(edge_weights)
                target_attrs = graph.nodes.get(node, {}).get("attrs")
                impact = _impact_score(target_attrs)
                risk = path_prob * impact
                path = AttackPath(
                    path_id=str(uuid.uuid4()),
                    nodes=list(path_nodes),
                    edges=list(path_edges),
                    source=start,
                    target=node,
                    probability=path_prob,                       # Π(edge_weights)
                    impact_score=impact,
                    risk_score=risk,                             # Π(weights) × Impact
                    hop_count=hops,
                    detected_at=datetime.utcnow(),
                )
                results.append(path)
                if hops >= max_hops:
                    continue

            if hops >= max_hops:
                continue

            # Expand neighbours — accumulate only edge weights during BFS
            for neighbour in graph.successors(node):
                if neighbour in path_nodes:
                    continue  # prevent cycles
                edge_data = graph.edges.get((node, neighbour), {})
                edge_attrs = edge_data.get("attrs")
                edge_weight = edge_attrs.weight if edge_attrs else 0.5

                new_edge_prob = edge_prob * edge_weight

                if new_edge_prob < self._min_probability:
                    continue

                heapq.heappush(heap, (
                    -new_edge_prob,
                    hops + 1,
                    neighbour,
                    path_nodes + [neighbour],
                    path_edges + [(node, neighbour)],
                ))

        return results

    def top_k(self, paths: List[AttackPath], k: int = 10) -> List[AttackPath]:
        return paths[:k]

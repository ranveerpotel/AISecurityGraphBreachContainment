"""
Algorithm 4: Automated Containment.

For high-risk paths:
  1. Extract edges on the attack path.
  2. Generate firewall DENY rules for those edges.
  3. Estimate business impact (fraction of legitimate traffic disrupted).
  4. Apply automatically if impact < 5%.
  5. Alert SOC for 5-20% impact paths.
  6. Recommend manual review for > 20% impact.

Achieves 35-second mean time-to-containment per paper §7.2.
"""
from __future__ import annotations
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Set, Tuple

import networkx as nx

from .risk_scorer import RiskScorer
from ..common.models import AttackPath, ContainmentAction, FirewallRule
from ..common.config import AppConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _business_impact_estimate(
    edges: List[Tuple[str, str]],
    graph: nx.DiGraph,
    total_edges: int,
) -> float:
    """
    Estimate fraction of legitimate traffic that would be disrupted.
    Heuristic: ratio of blocked edge weights to total graph weight.
    """
    if total_edges == 0 or not edges:
        return 0.0
    blocked_weight = sum(
        graph.edges[e]["attrs"].weight
        for e in edges
        if graph.has_edge(*e) and graph.edges[e].get("attrs")
    )
    total_weight = sum(
        d["attrs"].weight
        for _, _, d in graph.edges(data=True)
        if d.get("attrs")
    )
    return blocked_weight / max(total_weight, 1e-9)


class ContainmentEngine:
    """
    Generates and (simulates) applying containment actions.

    In production, firewall_rules are pushed to:
      - AWS Security Groups via boto3
      - Azure NSG via azure-sdk
      - on-prem firewall via REST/CLI
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config.containment
        self._scorer = RiskScorer(config)
        self._applied_rules: List[FirewallRule] = []
        self._containment_history: List[ContainmentAction] = []

    def evaluate_and_contain(
        self,
        attack_paths: List[AttackPath],
        graph: nx.DiGraph,
        compromised_nodes: Set[str],
    ) -> List[ContainmentAction]:
        """
        Main entry point. Returns list of containment actions taken or recommended.
        """
        if not attack_paths:
            return []

        actions: List[ContainmentAction] = []
        # Group paths by classification
        auto_paths = [p for p in attack_paths if self._scorer.classify(p) == "AUTO_ISOLATE"]
        soc_paths = [p for p in attack_paths if self._scorer.classify(p) == "SOC_ALERT"]

        if auto_paths:
            action = self._build_action(auto_paths, graph, compromised_nodes, auto=True)
            if action:
                self._apply_action(action)
                actions.append(action)

        if soc_paths:
            action = self._build_action(soc_paths, graph, compromised_nodes, auto=False)
            if action:
                actions.append(action)

        return actions

    def _build_action(
        self,
        paths: List[AttackPath],
        graph: nx.DiGraph,
        compromised_nodes: Set[str],
        auto: bool,
    ) -> Optional[ContainmentAction]:
        # Collect unique edges across all paths
        path_edges = list({e for p in paths for e in p.edges})
        target_nodes = list({p.target for p in paths} | compromised_nodes)
        max_risk = max(p.risk_score for p in paths)

        firewall_rules = self._generate_firewall_rules(path_edges, compromised_nodes, max_risk)
        impact = _business_impact_estimate(path_edges, graph, graph.number_of_edges())

        # Determine action type based on impact threshold
        if auto:
            if impact < self._cfg.max_business_impact_auto:
                action_type = "AUTO_ISOLATE"
            elif impact < self._cfg.max_business_impact_soc:
                action_type = "SOC_ALERT"
                auto = False
            else:
                action_type = "MANUAL_REVIEW"
                auto = False
        else:
            action_type = "SOC_ALERT"

        action = ContainmentAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            target_nodes=target_nodes,
            target_edges=path_edges,
            risk_score=max_risk,
            attack_paths=paths,
            firewall_rules=firewall_rules,
            business_impact_estimate=impact,
            auto_applied=auto and action_type == "AUTO_ISOLATE",
            created_at=datetime.utcnow(),
        )
        return action

    def _generate_firewall_rules(
        self,
        edges: List[Tuple[str, str]],
        compromised_nodes: Set[str],
        risk_score: float,
    ) -> List[FirewallRule]:
        rules: List[FirewallRule] = []
        expires = datetime.utcnow() + timedelta(hours=24)

        # Block all outbound from compromised nodes
        for node in compromised_nodes:
            rules.append(FirewallRule(
                rule_id=str(uuid.uuid4()),
                action="DENY",
                source=node,
                destination="ANY",
                protocol="ANY",
                port=0,
                priority=10,  # high priority
                expires_at=expires,
                reason=f"Compromised node isolation (risk={risk_score:.2f})",
            ))

        # Block specific attack path edges
        for src, dst in edges:
            rules.append(FirewallRule(
                rule_id=str(uuid.uuid4()),
                action="DENY",
                source=src,
                destination=dst,
                protocol="ANY",
                port=0,
                priority=20,
                expires_at=expires,
                reason=f"Attack path edge block (risk={risk_score:.2f})",
            ))

        return rules

    def _apply_action(self, action: ContainmentAction) -> None:
        """Simulate rule application. Replace with real firewall/SDN calls in production."""
        action.applied_at = datetime.utcnow()
        latency_ms = (action.applied_at - action.created_at).total_seconds() * 1000
        self._applied_rules.extend(action.firewall_rules)
        self._containment_history.append(action)
        logger.info(
            "AUTO-CONTAINED: %d rules applied in %.1fms | risk=%.2f | impact=%.1f%%",
            len(action.firewall_rules), latency_ms,
            action.risk_score, action.business_impact_estimate * 100,
        )

    def rollback(self, action_id: str) -> bool:
        """Revoke containment action by removing its firewall rules."""
        action = next((a for a in self._containment_history if a.action_id == action_id), None)
        if not action:
            return False
        rule_ids = {r.rule_id for r in action.firewall_rules}
        self._applied_rules = [r for r in self._applied_rules if r.rule_id not in rule_ids]
        action.rollback_applied = True
        logger.info("Rolled back containment action %s", action_id)
        return True

    @property
    def active_rules(self) -> List[FirewallRule]:
        now = datetime.utcnow()
        return [r for r in self._applied_rules if r.expires_at is None or r.expires_at > now]

    @property
    def history(self) -> List[ContainmentAction]:
        return list(self._containment_history)

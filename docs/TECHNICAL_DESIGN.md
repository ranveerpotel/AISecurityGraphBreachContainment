# Technical Design: AI-Driven Security Graphs for Real-Time Breach Containment

**Paper:** Ranveer Potel, "AI-Driven Security Graphs for Real-Time Breach Containment in Hybrid Cloud Environments,"  
IJAIBDCMS Vol. 3, Issue 4, pp. 123–131, 2022. DOI: 10.63282/3050-9416.IJAIBDCMS-V3I4P113

---

## 1. Stack Decision

### Candidates Evaluated

| Criterion | **Python + PyG + NetworkX** | Java/Scala + Flink + DL4J | Go + Custom GNN |
|---|---|---|---|
| GNN ecosystem | **Best** (PyTorch Geometric) | Weak (DL4J immature) | Absent |
| Real-time streaming | Good (Kafka-Python) | **Best** (Flink native) | Good |
| Graph DB integration | Good (py2neo) | Good | Average |
| ML research → production | **Seamless** | Friction | Not applicable |
| Consistency with existing projects | **Yes** (preemptiveCyberDefence, quantumSolutionForCounterfeiting) | No | No |
| Community / support | **Largest** | Medium | Small |
| GPU acceleration | **Native CUDA** | Limited | None |

### Decision: Python + PyTorch Geometric + NetworkX + FastAPI + Kafka

**Rationale:**
- PyTorch Geometric is the gold standard for production GNN systems (used at Pinterest, Twitter, Uber).
- NetworkX handles in-memory graph operations up to 100K nodes (within paper's target scale); Neo4j can be swapped in as a backend for 250K+ node deployments via the graph abstraction layer.
- FastAPI delivers sub-millisecond REST response times — adequate for the SOC dashboard latency budget.
- Consistent with the author's existing Python research stack.

**NumPy fallback:** The `GNNInferenceEngine` class transparently falls back to a pure-NumPy GraphSAGE approximation when PyTorch Geometric is not installed, so the full pipeline runs in any Python environment.

---

## 2. System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Hybrid Cloud Enterprise                          │
│  Workloads (6,500 servers + 2,500 containers + 1,000 laptops)            │
│  Users (6,000)   ·   Applications (350)   ·   Databases (40 critical)    │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │ 9 Telemetry Sources (520K events/sec peak)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  A. Telemetry Collection                                                 │
│  eBPF (Linux) / ETW (Windows) → Apache Kafka → TelemetryNormalizer      │
│  Sources: NetFlow/sFlow, Sysmon/osquery, CloudTrail/AzureMonitor,        │
│           K8s audit, Auth, App logs, DNS, VPN, File access               │
│  Latency: p50=1.8s  p95=3.2s  p99=5.1s  (event → graph update)         │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  B. Security Graph Engine                                                │
│  Dynamic G(t) = (V, E)   V = workloads ∪ users ∪ assets                 │
│  Edge weight: w(e) = 0.4·f_norm + 0.3·s(e) + 0.3·d(e)                  │
│  Temporal decay: weight ×= exp(−0.1·Δt_hours)                           │
│  Windows: 5-min (current) · 24-hr (short) · 7-day (long)                │
│  Pruning: edges unseen for 7 days removed                                │
│  Storage: NetworkX in-memory (≤100K nodes) / Neo4j (>100K nodes)        │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ Graph snapshot every 500 events
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  C. AI Detection Module                                                  │
│  GraphSAGE (3 layers, mean aggregation)                                  │
│    Input:  64D = [24D behavioral | 16D temporal | 16D structural | 8D type] │
│    L1: 64D → 128D   L2: 128D → 64D   L3: 64D → 32D embeddings           │
│  Anomaly score: A(i,j) = 0.5·D_b + 0.3·D_t + 0.2·D_g                   │
│  Flagged if A(i,j) > τ = 0.70                                            │
│  AUC = 0.95  ·  FPR = 7%  ·  Detection latency p50 = 2.9 min            │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ Flagged anomalous edges
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  D. Risk Scoring & Containment                                           │
│  Attack path enum: BFS (max 5 hops, probabilistic pruning)               │
│  Risk(p) = Π_edges(w) × (1 − p_detect) × Impact(target)                 │
│  p_detect = 0.89 per hop → P(5 hops undetected) = 1.6×10⁻⁵             │
│                                                                          │
│  risk > 0.8  → AUTO_ISOLATE  (firewall DENY rules, 35-sec mean TTR)     │
│  0.5–0.8     → SOC_ALERT     (prepared policies, analyst approval)      │
│  < 0.5       → MONITOR                                                   │
│                                                                          │
│  Business impact check: auto-apply only if impact < 5%                  │
│  Results: 73% blast radius reduction · 91% lateral movement prevention  │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  E. SOC Integration                                                      │
│  FastAPI REST  ·  MITRE ATT&CK tactic mapping  ·  Kafka alert topic     │
│  Dashboard: /alerts · /graph/stats · /containment/active-rules          │
│  Analyst feedback → online learning loop                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Source File Map

```
AISecurityGraphBreachContainment/
├── src/
│   ├── common/
│   │   ├── config.py          # All thresholds, weights, window params
│   │   └── models.py          # TelemetryEvent, AnomalyScore, AttackPath, etc.
│   │
│   ├── telemetry/
│   │   ├── collector.py       # Simulated generator + Kafka consumer
│   │   └── normalizer.py      # Raw event → TelemetryEvent + NodeAttributes
│   │
│   ├── graph_engine/
│   │   ├── security_graph.py  # Dynamic G(t): ingest, weight, decay, prune
│   │   └── temporal_manager.py# 3-window stats for D_t computation
│   │
│   ├── detection/
│   │   ├── graphsage_model.py # PyG + NumPy fallback GNN
│   │   ├── anomaly_detector.py# Feature extraction + A(i,j) scoring
│   │   └── concept_drift.py   # PSI drift detection + online retraining
│   │
│   ├── containment/
│   │   ├── path_prioritizer.py# BFS attack path enumeration
│   │   ├── risk_scorer.py     # Risk thresholds + severity classification
│   │   └── containment_engine.py # Firewall rule generation + apply/rollback
│   │
│   ├── soc_integration/
│   │   ├── alert_manager.py   # Alert creation, MITRE mapping, Kafka dispatch
│   │   └── dashboard_api.py   # FastAPI REST endpoints
│   │
│   └── main.py                # Pipeline orchestrator + CLI entry point
│
├── tests/
│   └── test_smoke.py          # Component + integration tests
├── docs/
│   └── TECHNICAL_DESIGN.md    # This document
├── docker-compose.yml         # Kafka, Neo4j, app, Prometheus, Grafana
└── requirements.txt
```

---

## 4. Key Algorithms

### 4.1 Edge Weight Formula (§4.2)

```
w(e) = α · f_norm(e) + β · s(e) + γ · d(e)

α = 0.4 (frequency)     f_norm: normalised event count vs busiest peer
β = 0.3 (sensitivity)   s(e): target node sensitivity / CRITICAL
γ = 0.3 (deviation)     d(e): current 5-min rate / 24-hr baseline rate

Temporal decay: w(e) ← w(e) · exp(−0.1 · Δt_hours)
```

### 4.2 GNN Node Embeddings (§4.3 / Algorithm 2)

```
Z = GraphSAGE_θ(G, X)          # 3-layer mean aggregation
X ∈ ℝ^{N×64}  :  [behavioral₂₄ | temporal₁₆ | structural₁₆ | type₈]

A(i,j) = 0.5 · D_b + 0.3 · D_t + 0.2 · D_g
  D_b = cosine_dist(Z_i, baseline_Z_i)   # embedding drift from EMA baseline
  D_t = |current_rate − 24hr_rate| / 24hr_rate
  D_g = cosine_dist(Z_i, Z_j)            # structural dissimilarity of peers

Flag edge if A(i,j) > 0.70
```

### 4.3 Attack Path Prioritisation (§4.4 / Algorithm 3)

```
BFS with probabilistic pruning (max depth 5):
  P(path) = ∏_{e ∈ path} w(e) × (1 − p_detect)^hops
  Risk(p) = P(path) × Impact(target)
  Prune branch if P < 0.001

Complexity: O(k·d) where k ≈ 100 high-probability paths, d ≤ 5
```

### 4.4 Automated Containment (§4.4 / Algorithm 4)

```
For each path with Risk > 0.8:
  1. Generate firewall DENY rules for path edges
  2. Generate DENY-ALL-OUTBOUND rules for compromised nodes
  3. Estimate business_impact = Σ(blocked_weights) / Σ(all_weights)
  4. If impact < 5%  → apply immediately (AUTO_ISOLATE)
     If 5–20%        → SOC_ALERT with prepared rules
     If > 20%        → MANUAL_REVIEW recommendation
```

### 4.5 Concept Drift (§9 / PSI)

```
PSI(feature_group) = Σ_buckets (actual% − expected%) · ln(actual% / expected%)

PSI < 0.10  → stable
PSI 0.10–0.25 → moderate drift (monitor)
PSI > 0.25  → retrain GNN on 60-day window, shadow mode 7 days, then promote
```

---

## 5. Formal Security Guarantees (§10)

| Theorem | Statement | With p_d = 0.89 |
|---|---|---|
| T1: Geometric Detection Bound | P(d hops undetected) ≤ (1 − p_d)^d | d=5: P ≤ 1.6×10⁻⁵ |
| T2: Expected Blast Radius | E[\|A_τ\|] ≤ 1 + 1/p_d | ≤ 2.12 nodes (theory) |
| T3: Containment Race | P(contain before next hop) ≥ 1 − C/L_a | C=35s, L_a=1hr: P ≥ 0.99 |
| Corollary | P(sensitive asset compromised) ≤ (1 − p_d)^d_min | d_min=3: P ≤ 0.0013 |

---

## 6. Performance Targets (from paper §7)

| Metric | Target | Achieved |
|---|---|---|
| Detection latency (mean) | < 5 min | 2.9 min |
| Blast radius reduction | > 70% | 73% |
| Lateral movement prevention | > 85% | 91% |
| AUC | > 0.90 | 0.95 |
| FPR (steady-state) | < 10% | 7% |
| Time to containment | < 60 sec | 35 sec |
| Scalability | 100K nodes | Linear to 100K+ |
| Adversarial TPR (combined) | > 75% | 79% |

---

## 7. Configuration Parameters (src/common/config.py)

| Parameter | Value | Description |
|---|---|---|
| `graph.alpha` | 0.4 | Frequency weight in w(e) |
| `graph.beta` | 0.3 | Sensitivity weight in w(e) |
| `graph.gamma` | 0.3 | Deviation weight in w(e) |
| `graph.decay_lambda` | 0.1/hr | Exponential decay rate |
| `graph.prune_after_days` | 7 | Edge inactivity TTL |
| `detection.alpha_b` | 0.5 | Behavioral weight in A(i,j) |
| `detection.alpha_t` | 0.3 | Temporal weight in A(i,j) |
| `detection.alpha_g` | 0.2 | Structural weight in A(i,j) |
| `detection.anomaly_threshold` | 0.70 | Flag threshold τ |
| `containment.auto_isolate_threshold` | 0.8 | AUTO_ISOLATE trigger |
| `containment.soc_alert_threshold` | 0.5 | SOC_ALERT trigger |
| `containment.max_hops` | 5 | BFS depth limit |
| `drift.psi_threshold` | 0.25 | Retraining trigger |
| `drift.shadow_mode_days` | 7 | New model shadow period |

---

## 8. Deployment

### Quick start (simulation mode, no external services)
```bash
pip install -r requirements.txt
python -m src.main --simulate --duration 120 --rate 1000
# API available at http://localhost:8000
```

### Run tests
```bash
pytest tests/ -v
```

### Production (Docker Compose)
```bash
docker compose up -d
# Services: Kafka, Neo4j, aisgbc app, Prometheus, Grafana
```

### Phased Rollout (§12.4)
1. **Weeks 1–6:** `--simulate` mode; tune thresholds; establish baselines.
2. **Week 7:** Deploy with `use_kafka=True`; alerting only (no automated containment).
3. **Week 8+:** Enable automated containment for non-critical assets; escalate gradually.

---

## 9. Adversarial Robustness (§8)

| Attack Type | Defense | TPR | FPR |
|---|---|---|---|
| Mimicry | Ensemble: GNN + rule-based UEBA | 87% | 11% |
| Low-and-slow | Cumulative risk scoring, multi-window | 89% | 9% |
| Graph poisoning | Trust-weighted learning from high-confidence periods | 84% | 14% |
| Combined adversarial | All defenses combined | 79% | 16% |

---

## 10. Ethical Considerations (§12.3)

- **Privacy:** Behavioural profiles retained max 90 days; anonymised for non-security analytics.
- **Bias:** Quarterly audits across user groups; fairness-aware training data.
- **Disruption:** Automated containment only if business impact < 5%; human approval for critical assets.
- **Dual-use:** Export controls; ethical deployment guidelines for authoritarian-context prevention.

---

## 11. Future Work (§13)

1. Federated learning for cross-organisation threat intelligence.
2. OT/IoT integration for operational technology environments.
3. Certified adversarial robustness (randomised smoothing).
4. Explainable AI (XAI) for analyst interpretability.
5. Automated response orchestration: credential revocation, process termination, data quarantine.

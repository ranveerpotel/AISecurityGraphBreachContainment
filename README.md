# AI-Driven Security Graphs for Real-Time Breach Containment in Hybrid Cloud Environments

> **Implementation of the peer-reviewed paper:**  
> Ranveer Potel — *International Journal of AI, BigData, Computational and Management Studies (IJAIBDCMS)*  
> Volume 3, Issue 4, pp. 123–131, 2022 · ISSN 3050-9416  
> 🔗 **[https://doi.org/10.63282/3050-9416.IJAIBDCMS-V3I4P113](https://doi.org/10.63282/3050-9416.IJAIBDCMS-V3I4P113)**

---

## Overview

Modern cyberattacks increasingly leverage **lateral movement** to compromise critical assets. This project implements an AI-driven security graph framework for real-time detection and automated containment of threats in hybrid cloud environments.

The system:
- Constructs a **dynamic security graph** G(t) representing workloads, users, and assets
- Performs **behavior-based anomaly detection** using Graph Neural Networks (GraphSAGE)
- Prioritizes **high-risk attack paths** using probabilistic BFS
- Enforces **automated micro-segmentation** with firewall rule generation

### Key Results (from paper §7)

| Metric | Baseline | This System | Improvement |
|--------|----------|-------------|-------------|
| Detection latency (mean) | 18.4 min | **2.9 min** | 84% faster |
| Blast radius | 23.1 nodes | **6.2 nodes** | 73% reduction |
| Lateral movement prevention | 13% | **91%** | — |
| AUC-ROC | — | **0.95** | — |
| False positive rate | 21% | **7%** | 67% reduction |
| Time to containment | Manual (18.4 min) | **35 seconds** | — |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  A. Telemetry Collection  (9 sources, 520K events/sec)   │
│     NetFlow · Sysmon · CloudTrail · K8s · Auth · DNS … │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  B. Security Graph Engine                                │
│     w(e) = 0.4·f_norm + 0.3·sensitivity + 0.3·deviation │
│     Exponential decay · 3 sliding windows · 7-day prune  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  C. AI Detection Module                                  │
│     GraphSAGE (3-layer, 64D→128D→64D→32D embeddings)    │
│     A(i,j) = 0.5·D_b + 0.3·D_t + 0.2·D_g               │
│     + RuleBasedDetector ensemble (§8.1)                  │
│     + PSI concept drift detection + online retraining    │
└──────────────────────────┬──────────────────────────────┘
                           │ Flagged edges (score > 0.70)
┌──────────────────────────▼──────────────────────────────┐
│  D. Risk Scoring & Containment                           │
│     BFS attack paths (max 5 hops)                        │
│     Risk = Π(edge weights) × Impact(target)              │
│     > 0.8 → AUTO_ISOLATE  ·  0.5–0.8 → SOC_ALERT        │
│     Firewall DENY rules · business impact check · rollback│
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  E. SOC Integration                                      │
│     FastAPI REST · MITRE ATT&CK mapping · Kafka alerts   │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
src/
├── common/
│   ├── config.py          # All paper constants (weights, thresholds, windows)
│   └── models.py          # TelemetryEvent, AnomalyScore, AttackPath, …
├── telemetry/
│   ├── collector.py       # Simulated generator + Kafka consumer
│   └── normalizer.py      # Raw event → canonical TelemetryEvent
├── graph_engine/
│   ├── security_graph.py  # Dynamic G(t): ingest, weight, decay, prune
│   └── temporal_manager.py# 3-window statistics for D_t computation
├── detection/
│   ├── graphsage_model.py # PyG + NumPy-fallback GNN
│   ├── anomaly_detector.py# Feature extraction, A(i,j) scoring, RuleBasedDetector
│   └── concept_drift.py   # PSI drift detection + online retraining
├── containment/
│   ├── path_prioritizer.py# BFS attack path enumeration
│   ├── risk_scorer.py     # Risk thresholds + severity classification
│   └── containment_engine.py # Firewall rule generation + apply/rollback
├── soc_integration/
│   ├── alert_manager.py   # Alert creation, MITRE mapping, Kafka dispatch
│   └── dashboard_api.py   # FastAPI REST endpoints
└── main.py                # Pipeline orchestrator + CLI entry point
```

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `networkx`, `numpy`, `scipy`, `fastapi`, `uvicorn`  
Optional (full GNN): `torch`, `torch-geometric`

### Run simulation mode (no external services needed)

```bash
python -m src.main --simulate --duration 120 --rate 1000
# REST API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Run tests

```bash
pytest tests/ -v
# 36 tests across all 5 components
```

### Production (Docker Compose)

```bash
docker compose up -d
# Starts: Kafka, Neo4j, app, Prometheus, Grafana
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/graph/stats` | Node/edge counts, sensitive node count |
| GET | `/graph/nodes/{id}` | Node attributes and sensitivity |
| GET | `/alerts` | List alerts (filter by severity) |
| POST | `/alerts/{id}/acknowledge` | Acknowledge an alert |
| GET | `/containment/active-rules` | Active firewall rules |
| POST | `/containment/{id}/rollback` | Rollback a containment action |
| GET | `/containment/history` | Containment action history |
| GET | `/drift/status` | PSI scores and drift event log |

---

## Formal Security Guarantees (§10)

| Theorem | Statement | p_d = 0.89 |
|---------|-----------|------------|
| T1: Geometric Detection Bound | P(d hops undetected) ≤ (1 − p_d)^d | d=5: P ≤ 1.6×10⁻⁵ |
| T2: Expected Blast Radius | E[\|A_τ\|] ≤ 1 + 1/p_d | ≤ 2.12 nodes |
| T3: Containment Race | P(contain before next hop) ≥ 1 − C/L_a | C=35s, L_a=1hr: P ≥ 0.99 |

---

## Adversarial Robustness (§8)

| Attack Type | Defense | TPR | FPR |
|-------------|---------|-----|-----|
| Mimicry | GNN + rule-based UEBA ensemble | 87% | 11% |
| Low-and-slow | Cumulative risk, multi-window analysis | 89% | 9% |
| Graph poisoning | Trust-weighted learning | 84% | 14% |
| Combined adversarial | All defenses | 79% | 16% |

---

## Citation

```bibtex
@article{potel2022aisecurity,
  author    = {Ranveer Potel},
  title     = {AI-Driven Security Graphs for Real-Time Breach Containment
               in Hybrid Cloud Environments},
  journal   = {International Journal of AI, BigData, Computational
               and Management Studies},
  volume    = {3},
  number    = {4},
  pages     = {123--131},
  year      = {2022},
  issn      = {3050-9416},
  doi       = {10.63282/3050-9416.IJAIBDCMS-V3I4P113},
  url       = {https://doi.org/10.63282/3050-9416.IJAIBDCMS-V3I4P113}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

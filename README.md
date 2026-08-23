# ACOS — Agentic Commerce Operating System

**© 2026 Nagendra Singh Yadav. All Rights Reserved.** See [LICENSE](LICENSE)
for terms — this code is not open source; no reuse, modification, or
redistribution is permitted without prior written consent.

## Overview

ACOS is a deterministic multi-agent architecture for coordinated
commerce decision-making. It integrates specialized agents for demand
forecasting, inventory management, pricing, procurement, fraud-risk
assessment, governance, and analytics through dependency-aware
orchestration, with cross-agent reconciliation and a policy engine
that reviews every decision.

Every agent's `reason()` step is a closed-form or well-defined
algorithm (exponential smoothing, EOQ / safety-stock formulas,
constrained margin optimization, TOPSIS multi-criteria ranking, robust
z-score anomaly detection). Determinism makes every decision
unit-testable, replayable, and auditable through the PolicyEngine.

## Architecture

- **Orchestrator + Planner + ConsensusResolver** — the planner maps
  supported goals to predefined task templates and constructs
  dependency-aware directed acyclic graphs for deterministic
  execution (`core/planner.py`); the orchestrator schedules and runs
  them; the consensus resolver reconciles the one specified cross-agent
  dependency (price → inventory).
- **Agent layer** — seven specialized agents (below) communicating
  through a shared blackboard and event bus.
- **Persistence** — SQLite long-term memory, a semantic retrieval
  layer, and a knowledge graph.
- **Governance** — a rule-based PolicyEngine that reviews every
  decision and produces an approve/reject/escalate ruling with a
  rationale.

The retrieval layer (`core/vector_memory.py`) represents documents
using TF-IDF features followed by truncated SVD, indexed with FAISS
for similarity-based retrieval. The knowledge graph
(`core/knowledge_graph.py`) is an in-process `networkx` graph.

## Implemented Agents

| Agent | Method |
|---|---|
| DemandForecastAgent | Holt's linear exponential smoothing |
| InventoryAgent | EOQ + safety-stock reorder point |
| PricingAgent | Bounded grid search over a margin-optimization objective |
| ProcurementAgent | TOPSIS multi-criteria supplier ranking |
| FraudRiskAgent | Robust (median/MAD) z-score ensemble |
| GovernanceAgent | Rule-based policy evaluation |
| AnalyticsAgent | Deterministic KPI aggregation |

## Installation

```bash
pip install -r requirements.txt --break-system-packages
```

Python 3.10+. Dependencies: numpy, scipy, scikit-learn, networkx,
pandas, matplotlib, FastAPI, uvicorn, pydantic, pytest, FAISS,
python-docx.

## Running ACOS

```bash
python3 main.py                # full demo workflow end to end
uvicorn api.main:app --reload  # REST API locally; see /docs for Swagger UI
```

## Running Tests

```bash
python3 -m pytest tests/ -v    # 45 tests (unit, integration, API)
```

## Reproducing Experiments

Every script under `experiments/` is independently runnable and
regenerates its own results file and figure(s) — see the docstring at
the top of each script for what it measures.

```bash
python3 experiments/run_experiments.py   # regenerate experiments/results.json (E1-E7)
python3 experiments/generate_figures.py  # regenerate core figures/*.png
```

## Experimental Comparisons

LangGraph, AutoGen-core, and CrewAI (`experiments/comparison/`) are
used exclusively in the comparative experiments reported in the
accompanying study. They are not components or runtime dependencies
of the ACOS architecture — installing and running ACOS itself only
requires `requirements.txt` above.

Isolation Forest (`experiments/fraud_baselines_and_ablation.py`,
`experiments/fraud_baselines_multiseed.py`) is included solely as an
external baseline for the fraud-risk experiments and is not part of
the ACOS FraudRiskAgent.

## Repository Structure

```
core/            event_bus.py, memory.py, vector_memory.py, knowledge_graph.py,
                 planner.py, policy.py, consensus.py, orchestrator.py
agents/          base.py + 7 specialized agents
api/             main.py — FastAPI REST service (bearer-token auth)
data/            synthetic_data.py — seeded, reproducible synthetic dataset generator
                 external/ — real datasets fetched here at runtime (not shipped)
experiments/     run_experiments.py, generate_figures.py, generate_uml_diagrams.py, stats_utils.py

                 real_data_validation.py, real_data_validation_rossmann.py,
                 rolling_origin_backtest.py, rolling_origin_clustered_analysis.py,
                 forecast_baselines.py, real_data_baseline_comparison.py,
                 holt_parameter_sensitivity.py
                   — forecasting: real-data validation, rolling-origin backtest with
                     cluster-aware statistics, baseline comparisons, parameter sensitivity

                 fraud_threshold_sweep.py, fraud_baselines_and_ablation.py,
                 fraud_baselines_multiseed.py
                   — fraud: threshold sweep, baseline/ablation comparison, and a
                     31-seed robustness check on that comparison with Holm/BH correction

                 ablation_study_v2.py, governance_stress_multiseed.py
                   — architecture: equal-work-baseline ablation (multi-seed, independent
                     downstream outcome measure) and a 10-seed governance stress test;
                     ablation_study.py is an earlier, single-baseline version kept for
                     reference but superseded by v2 everywhere it's cited

                 multi_seed_robustness.py — 31-seed re-check of the most-scrutinized
                     single-seed quantities elsewhere in the codebase

                 comparison/ — LangGraph/AutoGen-core/CrewAI equivalent-DAG scripts +
                     larger_dag_benchmark.py; see "Experimental Comparisons" above
figures/         21 generated PNG figures (300 DPI)
tests/           45 tests (unit, integration, API) — all passing
main.py          end-to-end demo entry point
```

## Data and Result Artifacts

- All data used in E1-E7 is synthetic and seeded (`data/synthetic_data.py`,
  distributions documented in its module docstring).
- Real-data validation (`experiments/real_data_validation.py`,
  `experiments/real_data_validation_rossmann.py`,
  `experiments/rolling_origin_backtest.py`,
  `experiments/rolling_origin_clustered_analysis.py`) covers only
  DemandForecastAgent, on two independent real datasets (UCI Online
  Retail, Rossmann Store Sales). PricingAgent, ProcurementAgent, and
  FraudRiskAgent are not validated against real data anywhere in this
  codebase, since neither public dataset contains cost, supplier, or
  fraud-label fields.
- A seed-propagation defect in `main.py::build_system()` was
  identified, regression-tested
  (`tests/test_integration.py::test_build_system_respects_seed_parameter`),
  and corrected before the results in the accompanying paper were
  generated.
- **Code availability is currently incomplete**: this repository is not
  yet archived at a permanent, DOI-backed location (e.g. Zenodo/Software
  Heritage).

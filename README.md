# ACOS — Agentic Commerce Operating System

**© 2026 Nagendra Singh Yadav. All Rights Reserved.** See [LICENSE](LICENSE)
for terms — this code is not open source; no reuse, modification, or
redistribution is permitted without prior written consent.

A fully implemented, tested multi-agent decision-intelligence platform
for enterprise commerce operations: demand forecasting, inventory
management, dynamic pricing, supplier selection, fraud screening,
analytics, and rule-based governance.

## Important, upfront: what this system is (and isn't)

This is a **deterministic, classical-algorithm multi-agent system**,
not an LLM-agent framework. Every agent's `reason()` step uses a real,
closed-form or well-defined algorithm (exponential smoothing, EOQ /
safety-stock formulas, constrained margin optimization, TOPSIS
multi-criteria ranking, robust z-score anomaly detection) rather than
a call to a language model. This was a deliberate response to a real
constraint (no credentialed LLM API access in the build environment),
not a disguised limitation. `agents/base.py`'s `BaseAgent` interface is
written so an LLM-backed reasoning agent could be substituted in behind
it without changing the surrounding orchestration, governance, or
memory layers.

Similarly: `core/vector_memory.py` uses TF-IDF + SVD instead of a
dense neural encoder (no model-hub access), and `core/knowledge_graph.py`
uses an in-process `networkx` graph instead of a standalone Neo4j
server (no daemon services available in the build environment). Both
are written behind interfaces designed to make swapping in the "real"
component a drop-in change.

## Quickstart

```bash
pip install -r requirements.txt --break-system-packages
python3 main.py                          # run the full demo workflow end to end
python3 -m pytest tests/ -v              # 43 tests, should all pass
python3 experiments/run_experiments.py   # regenerate experiments/results.json (E1-E7)
python3 experiments/generate_figures.py  # regenerate core figures/*.png
uvicorn api.main:app --reload            # run the REST API locally, see /docs for Swagger UI
```

Every script under `experiments/` is independently runnable and
regenerates its own results file and figure(s) — see the docstring at
the top of each script for what it measures and how to run it.

## Project layout

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
                     larger_dag_benchmark.py, run by installing and executing each
                     framework directly rather than reading documentation
figures/         21 generated PNG figures (300 DPI): experiment plots, UML diagrams,
                 real-data backtests, fraud ROC/PR + baselines, forecast baselines,
                 rolling-origin backtest, architectural ablation
tests/           43 tests (unit, integration, API) — all passing
main.py          end-to-end demo entry point
```

## Honesty notes

- All experimental numbers were produced by actually running the
  scripts in `experiments/`; none were invented.
- All data used in E1-E7 is synthetic and seeded (`data/synthetic_data.py`,
  distributions documented in its module docstring); this is stated
  explicitly wherever results are reported.
- `experiments/comparison/`'s framework comparison (AutoGen, LangGraph,
  CrewAI) was produced by installing and executing each framework's
  actual code, including a real, unplanned finding — LangGraph raised
  `InvalidUpdateError` on concurrent state writes until an explicit
  reducer was added — reported as-is rather than edited out.
- `agents/pricing_agent.py`'s optimality math was found to be
  internally inconsistent during review (a unimodality claim didn't
  hold for the full elasticity range actually used) and was re-derived
  from scratch and corrected, with the correct two-case derivation
  presented directly — see `tests/test_core_algorithms.py` for the
  regression tests backing it.
- `experiments/multi_seed_robustness.py` re-runs the most-scrutinized
  synthetic measurements across 31 seeds. An earlier version of this
  experiment reported perfectly invariant reorder/reconciliation counts
  across seeds (SD=0.0); investigating why found a hardcoded seed in a
  shared helper function (`main.py::build_system()`), not a genuine
  architectural property. Fixed, with a regression test added
  (`tests/test_integration.py::test_build_system_respects_seed_parameter`).
- `experiments/ablation_study_v2.py` is the primary test of whether
  ACOS's architecture (governance, cross-agent reconciliation,
  orchestration) produces measurable effects over a computation-
  equivalent sequential baseline using the same algorithms — the
  single most important experiment in this codebase. It includes a
  multi-seed run (n=20), an equal-work baseline (isolating architecture
  overhead from agent-count differences), and an independent downstream
  economic outcome measure showing reconciliation's effect is
  conditional on whether its elasticity assumption matches realized
  demand, not unconditionally beneficial.
- `experiments/fraud_baselines_and_ablation.py` compares the built-in
  fraud ensemble against Isolation Forest and a leave-one-signal-out
  ablation on one seed; `experiments/fraud_baselines_multiseed.py`
  re-runs that same comparison across 31 seeds with Holm/BH-corrected
  paired tests. Both competitors beat the full ensemble on 29-30 of
  31 seeds. `experiments/real_data_baseline_comparison.py` similarly
  compares the forecasting agent against seasonal-naive and Croston's
  method on real data; a seasonal-naive baseline shows lower
  single-holdout error on one of the two datasets tested.
- Real-data validation (`experiments/real_data_validation.py`,
  `experiments/real_data_validation_rossmann.py`,
  `experiments/rolling_origin_backtest.py`,
  `experiments/rolling_origin_clustered_analysis.py`) covers only the
  DemandForecastAgent, on two independent real datasets (UCI Online
  Retail, Rossmann Store Sales), with formal paired statistical
  testing (`experiments/stats_utils.py`), a single-holdout backtest,
  and a true rolling-origin, multi-horizon backtest reanalyzed at the
  series level to avoid overstating independence from pooled folds.
  This is not whole-system validation — PricingAgent, ProcurementAgent,
  and FraudRiskAgent are not validated against real data anywhere in
  this codebase, since neither public dataset contains cost, supplier,
  or fraud-label fields.
- **Code availability is currently incomplete**: this repository is not
  yet archived at a permanent, DOI-backed location (e.g. Zenodo/Software
  Heritage).

"""
experiments/multi_seed_robustness.py

Addresses a real gap: E1-E7 (experiments/run_experiments.py)
and the fraud threshold sweep were originally run on a single fixed seed
(42). A single seed establishes computational reproducibility (same
seed -> same result) but says nothing about whether the reported
numbers are stable properties of the ACOS algorithms or artifacts of
that one particular synthetic draw.

This script re-runs the most-scrutinized measurements -- fraud
detection AUC, the consensus-reconciliation count, and end-to-end
workflow latency -- across 20 independent seeds and reports mean +/-
SD (and, for count-like statistics, the full distribution), so the
report can state whether the seed-42 numbers are representative or
outliers.
"""
from __future__ import annotations

import json
import statistics as _stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.fraud_risk_agent import FraudRiskAgent
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from data.synthetic_data import build_full_dataset
from experiments.fraud_threshold_sweep import compute_roc_pr
from main import build_system

N_SEEDS = 30
OUT_PATH = Path(__file__).resolve().parent / "multi_seed_results.json"


def run_one_seed(seed: int) -> dict:
    dataset = build_full_dataset(seed=seed)

    # --- fraud detection AUC + the seed-42 operating point's precision/recall
    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path=f"/tmp/acos_ms_fraud_{seed}.db"))
    decision = agent.reason({"transactions": dataset["transactions"]})
    scored = {s["transaction_id"]: s for s in decision.output["scored"]}
    ground_truth = set(dataset["fraud_ground_truth"])
    txn_ids = list(scored.keys())
    y_true = np.array([1 if tid in ground_truth else 0 for tid in txn_ids])
    scores = np.array([scored[tid]["risk_score"] for tid in txn_ids])
    curves = compute_roc_pr(y_true, scores)

    pred_07 = scores >= 0.70
    tp = int(np.sum(pred_07 & (y_true == 1)))
    fp = int(np.sum(pred_07 & (y_true == 0)))
    fn = int(np.sum(~pred_07 & (y_true == 1)))
    precision_07 = tp / (tp + fp) if (tp + fp) else None
    recall_07 = tp / (tp + fn) if (tp + fn) else None
    agent.long_term_memory.close()

    # --- full workflow: consensus reconciliation count + latency
    orchestrator, dataset2, _, ltm, _ = build_system(db_path=f"/tmp/acos_ms_wf_{seed}.db", seed=seed)
    inputs = {
        "sales_history": dataset2["sales_history"], "catalog": dataset2["catalog"],
        "current_inventory": dataset2["current_inventory"], "suppliers": dataset2["suppliers"],
        "transactions": dataset2["transactions"],
    }
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    reorder_plan = result.task_outputs.get("assess_inventory", {}).get("reorder_plan", {})
    n_needing_reorder = sum(1 for r in reorder_plan.values() if r.get("needs_reorder"))
    ltm.close()

    return {
        "seed": seed,
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
        "precision_at_0.70": round(precision_07, 4) if precision_07 is not None else None,
        "recall_at_0.70": round(recall_07, 4) if recall_07 is not None else None,
        "n_conflicts_resolved": len(result.conflicts),
        "n_skus_needing_reorder": n_needing_reorder,
        "wall_clock_seconds": round(result.wall_clock_seconds, 5),
        "overall_governance_ruling": result.governance.get("overall_ruling"),
    }


def main():
    seeds = list(range(1, N_SEEDS + 1))
    if 42 not in seeds:
        seeds.append(42)  # ensure the seed used everywhere else in this codebase is included for direct comparison
    per_seed = [run_one_seed(seed) for seed in seeds]

    def bootstrap_ci_mean(vals, n_boot=10000, alpha=0.05, seed=0):
        import random as _random
        rng = _random.Random(seed)
        n = len(vals)
        means = []
        for _ in range(n_boot):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()
        lo = means[int((alpha / 2) * n_boot)]
        hi = means[int((1 - alpha / 2) * n_boot) - 1]
        return round(lo, 4), round(hi, 4)

    def agg(key):
        vals = [r[key] for r in per_seed if r[key] is not None]
        ci = bootstrap_ci_mean(vals, seed=hash(key) % 10000) if len(vals) > 1 else (None, None)
        return {
            "mean": round(_stats.mean(vals), 4), "sd": round(_stats.pstdev(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4),
            "bootstrap_ci_95_of_mean": list(ci),
            "seed42_value": next((r[key] for r in per_seed if r["seed"] == 42), None),
        } if vals else None

    summary = {
        "n_seeds": len(seeds), "seeds_used": seeds,
        "seed_count_justification": (
            "30 was chosen as a round number in the range typically recommended for a robustness "
            "check like this (roughly 10-30 seeds), balancing statistical utility against "
            "the computational cost of running the full workflow "
            "once per seed (~1-2s each including deterministic computation and I/O); it is not "
            "derived from a formal power calculation. The bootstrap 95% CIs reported below give a direct, " 
            "empirical sense of the remaining estimation uncertainty at this sample size, which a reader " 
            "can judge for their own purposes rather than accept 30 as inherently sufficient."
        ),
        "note": "seed=42 is the seed used for every other experiment (E1-E7, fraud sweep) in this codebase; "
                "its value is called out in each aggregate below so the single-seed numbers reported "
                "elsewhere can be checked against the multi-seed distribution.",
        "roc_auc": agg("roc_auc"),
        "pr_auc": agg("pr_auc"),
        "precision_at_0.70": agg("precision_at_0.70"),
        "recall_at_0.70": agg("recall_at_0.70"),
        "n_conflicts_resolved": agg("n_conflicts_resolved"),
        "n_skus_needing_reorder": agg("n_skus_needing_reorder"),
        "wall_clock_seconds": agg("wall_clock_seconds"),
        "governance_ruling_distribution": {
            r: sum(1 for x in per_seed if x["overall_governance_ruling"] == r)
            for r in set(x["overall_governance_ruling"] for x in per_seed)
        },
        "per_seed": per_seed,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()

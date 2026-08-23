"""
experiments/fraud_baselines_multiseed.py

Section 8.3's headline fraud finding (Isolation Forest and the
no-amount two-signal ablation both outperform FraudRiskAgent's full
ensemble) was originally measured on seed=42 alone. Section 8.5
multi-seed-checks the ensemble's own ROC/PR/precision across 31 seeds,
but not the two baselines it's being compared against -- so the
robustness of the RANKING itself was never actually tested. This
script closes that gap: runs the full ensemble, Isolation Forest, and
the no-amount ablation over the same 31-seed set used elsewhere
(multi_seed_robustness.py: seeds 1-30 plus 42), and reports paired
differences with uncertainty, not just three more single-point
estimates.

Seed=42 is not a clean confirmatory observation for the no-amount
ablation specifically: fraud_baselines_and_ablation.py selected
no-amount as the strongest of three leave-one-signal-out ablations by
inspecting its performance ON seed=42. Re-including seed=42 in the
no-amount ablation's inferential test would let the same observation
that selected the comparator also help validate it. Isolation Forest
has no such issue -- it is an a-priori external baseline never
inspected against seed=42 before being chosen, so its test correctly
uses the full 31-seed set. For the no-amount ablation, the paired
test therefore uses only seeds 1-30 (excluding the exploratory
seed=42) as the confirmatory sample; descriptive summary statistics
are still reported across all 31 seeds for context.

Usage:
    python3 experiments/fraud_baselines_multiseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import IsolationForest

from agents.fraud_risk_agent import FraudRiskAgent
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from data.synthetic_data import build_full_dataset
from experiments.fraud_baselines_and_ablation import build_features
from experiments.fraud_threshold_sweep import compute_roc_pr
from experiments.stats_utils import paired_comparison, holm_bonferroni, benjamini_hochberg

N_SEEDS = 30
EXPLORATORY_SEED = 42  # selected the no-amount ablation in fraud_baselines_and_ablation.py
OUT_PATH = Path(__file__).resolve().parent / "fraud_baselines_multiseed_results.json"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_one_seed(seed: int) -> dict:
    dataset = build_full_dataset(seed=seed)
    transactions = dataset["transactions"]
    ground_truth = set(dataset["fraud_ground_truth"])
    txn_ids = [t["transaction_id"] for t in transactions]
    y_true = np.array([1 if tid in ground_truth else 0 for tid in txn_ids])

    z_scores, velocity_flags, geo_flags = build_features(transactions)

    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path=f"/tmp/acos_fbm_{seed}.db"))
    decision = agent.reason({"transactions": transactions})
    scored = {s["transaction_id"]: s["risk_score"] for s in decision.output["scored"]}
    ensemble_scores = np.array([scored[tid] for tid in txn_ids])
    agent.long_term_memory.close()

    # no-amount ablation: the strongest of the three single-signal-removed
    # ablations in fraud_baselines_and_ablation.py (removing the amount
    # z-score improved both ROC-AUC and PR-AUC on seed=42)
    W_VELOCITY, W_GEO = 0.9, 0.7
    no_amount_scores = sigmoid(W_VELOCITY * velocity_flags + W_GEO * geo_flags - 2.0)

    X = np.column_stack([np.abs(z_scores), velocity_flags, geo_flags])
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)

    out = {}
    for name, scores in [("full_ensemble", ensemble_scores),
                          ("isolation_forest", iso_scores),
                          ("no_amount_ablation", no_amount_scores)]:
        curves = compute_roc_pr(y_true, scores)
        out[f"{name}_roc_auc"] = curves["roc_auc"]
        out[f"{name}_pr_auc"] = curves["pr_auc"]
    out["seed"] = seed
    return out


def main():
    seeds = list(range(1, N_SEEDS + 1))
    if EXPLORATORY_SEED not in seeds:
        seeds.append(EXPLORATORY_SEED)
    per_seed = [run_one_seed(seed) for seed in seeds]
    print(f"Completed {len(per_seed)} seeds")

    def summarize(key, rows):
        vals = [r[key] for r in rows]
        return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4),
                "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4)}

    # Full 31-seed arrays: used for descriptive stats and for Isolation
    # Forest's inferential test (a-priori baseline, no selection issue).
    full_roc = [r["full_ensemble_roc_auc"] for r in per_seed]
    iso_roc = [r["isolation_forest_roc_auc"] for r in per_seed]
    abl_roc = [r["no_amount_ablation_roc_auc"] for r in per_seed]
    full_pr = [r["full_ensemble_pr_auc"] for r in per_seed]
    iso_pr = [r["isolation_forest_pr_auc"] for r in per_seed]
    abl_pr = [r["no_amount_ablation_pr_auc"] for r in per_seed]

    # Seeds 1-30 only (excludes the exploratory seed=42 that selected the
    # no-amount ablation): the clean confirmatory sample for that ablation.
    confirmatory_rows = [r for r in per_seed if r["seed"] != EXPLORATORY_SEED]
    full_roc_conf = [r["full_ensemble_roc_auc"] for r in confirmatory_rows]
    abl_roc_conf = [r["no_amount_ablation_roc_auc"] for r in confirmatory_rows]
    full_pr_conf = [r["full_ensemble_pr_auc"] for r in confirmatory_rows]
    abl_pr_conf = [r["no_amount_ablation_pr_auc"] for r in confirmatory_rows]

    summary = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "exploratory_seed_excluded_from_no_amount_confirmatory_test": EXPLORATORY_SEED,
        "n_confirmatory_seeds_no_amount": len(confirmatory_rows),
        "per_seed": per_seed,
        "summary": {
            "full_ensemble_roc_auc": summarize("full_ensemble_roc_auc", per_seed),
            "isolation_forest_roc_auc": summarize("isolation_forest_roc_auc", per_seed),
            "no_amount_ablation_roc_auc": summarize("no_amount_ablation_roc_auc", per_seed),
            "full_ensemble_pr_auc": summarize("full_ensemble_pr_auc", per_seed),
            "isolation_forest_pr_auc": summarize("isolation_forest_pr_auc", per_seed),
            "no_amount_ablation_pr_auc": summarize("no_amount_ablation_pr_auc", per_seed),
        },
        "paired_tests": {
            "isolation_forest_vs_full_ensemble_roc_auc": paired_comparison(iso_roc, full_roc, seed=101),
            "isolation_forest_vs_full_ensemble_pr_auc": paired_comparison(iso_pr, full_pr, seed=102),
            "no_amount_ablation_vs_full_ensemble_roc_auc": paired_comparison(abl_roc_conf, full_roc_conf, seed=103),
            "no_amount_ablation_vs_full_ensemble_pr_auc": paired_comparison(abl_pr_conf, full_pr_conf, seed=104),
        },
    }

    # Fraud robustness family: these 4 tests (2 competitors x 2 metrics) are a
    # single confirmatory family -- all planned together as the multi-seed
    # check on Section 8.3's headline ranking. paired_comparison() rounds its
    # stored p-value to 4dp, which collapses these (all far below 0.0001) to
    # 0.0, so we recompute the exact Wilcoxon p-values here for the Holm/BH
    # correction rather than correcting the already-rounded values.
    from scipy import stats as _stats
    test_order = ["isolation_forest_vs_full_ensemble_roc_auc", "isolation_forest_vs_full_ensemble_pr_auc",
                  "no_amount_ablation_vs_full_ensemble_roc_auc", "no_amount_ablation_vs_full_ensemble_pr_auc"]
    exact_pairs = {
        "isolation_forest_vs_full_ensemble_roc_auc": (iso_roc, full_roc),
        "isolation_forest_vs_full_ensemble_pr_auc": (iso_pr, full_pr),
        "no_amount_ablation_vs_full_ensemble_roc_auc": (abl_roc_conf, full_roc_conf),
        "no_amount_ablation_vs_full_ensemble_pr_auc": (abl_pr_conf, full_pr_conf),
    }
    exact_raw_ps = []
    for name in test_order:
        a, b = exact_pairs[name]
        _, p_exact = _stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        exact_raw_ps.append(float(p_exact))
    holm_adj = holm_bonferroni(exact_raw_ps)
    bh_adj = benjamini_hochberg(exact_raw_ps)
    for name, p_exact, p_holm, p_bh in zip(test_order, exact_raw_ps, holm_adj, bh_adj):
        summary["paired_tests"][name]["wilcoxon_p_value_exact"] = p_exact
        summary["paired_tests"][name]["holm_adjusted_p"] = p_holm
        summary["paired_tests"][name]["bh_adjusted_p"] = p_bh

    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["summary"], indent=2))
    print(json.dumps(summary["paired_tests"], indent=2, default=str))
    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()

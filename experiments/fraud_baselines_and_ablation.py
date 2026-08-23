"""
experiments/fraud_baselines_and_ablation.py

FraudRiskAgent's ensemble detector needs a competing baseline, not
just a threshold sweep of itself, plus a signal ablation showing
whether each of its three components (amount z-score, velocity, geo
mismatch) individually earns its place in the ensemble.

Isolation Forest here is an external experimental baseline used only
for this comparison -- it is not part of the ACOS architecture or the
FraudRiskAgent implementation, which remains a deterministic,
rule-based ensemble.

Baselines compared, all computed from the same transactions/scores so
comparisons are apples-to-apples:
  - amount-only: robust z-score on amount alone
  - velocity-only: binary velocity flag alone (as a degenerate "score")
  - geo-only: binary geo-mismatch flag alone
  - unsupervised: an Isolation-Forest baseline using scikit-learn
    (already a project dependency) over amount, velocity, and geo
    features -- a different, model-based approach, not just a rule
    ensemble with fewer rules.
  - FraudRiskAgent (full ensemble): the system as implemented.

All are scored with ROC-AUC and PR-AUC on the same synthetic ground
truth (seed=42), reusing the trapezoidal-AUC helper already used by
fraud_threshold_sweep.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import IsolationForest

from agents.fraud_risk_agent import FraudRiskAgent, robust_zscore
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from data.synthetic_data import build_full_dataset
from experiments.fraud_threshold_sweep import compute_roc_pr

OUT_PATH = Path(__file__).resolve().parent / "fraud_baselines_results.json"


def build_features(transactions):
    from collections import defaultdict

    amounts = [t["amount"] for t in transactions]
    z_scores = robust_zscore(amounts) if len(set(amounts)) > 1 else [0.0] * len(amounts)

    by_customer = defaultdict(list)
    for t in transactions:
        by_customer[t["customer_id"]].append(t)

    velocity_flags, geo_flags = [], []
    for t in transactions:
        cust_txns = sorted(by_customer[t["customer_id"]], key=lambda x: x["timestamp"])
        window = [x for x in cust_txns if abs(x["timestamp"] - t["timestamp"]) <= 3600]
        velocity_flags.append(1.0 if len(window) >= 4 else 0.0)
        geo_flags.append(1.0 if t.get("shipping_country") != t.get("billing_country") else 0.0)

    return np.array(z_scores), np.array(velocity_flags), np.array(geo_flags)


def main():
    dataset = build_full_dataset(seed=42)
    transactions = dataset["transactions"]
    ground_truth = set(dataset["fraud_ground_truth"])
    txn_ids = [t["transaction_id"] for t in transactions]
    y_true = np.array([1 if tid in ground_truth else 0 for tid in txn_ids])

    z_scores, velocity_flags, geo_flags = build_features(transactions)

    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path="/tmp/acos_fraud_baseline.db"))
    decision = agent.reason({"transactions": transactions})
    scored = {s["transaction_id"]: s["risk_score"] for s in decision.output["scored"]}
    ensemble_scores = np.array([scored[tid] for tid in txn_ids])
    agent.long_term_memory.close()

    amount_only_scores = np.abs(z_scores)
    velocity_only_scores = velocity_flags
    geo_only_scores = geo_flags

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    W_AMOUNT, W_VELOCITY, W_GEO = 0.6, 0.9, 0.7
    full_reimpl = sigmoid(W_AMOUNT * np.abs(z_scores) + W_VELOCITY * velocity_flags + W_GEO * geo_flags - 2.0)
    no_amount = sigmoid(W_VELOCITY * velocity_flags + W_GEO * geo_flags - 2.0)
    no_velocity = sigmoid(W_AMOUNT * np.abs(z_scores) + W_GEO * geo_flags - 2.0)
    no_geo = sigmoid(W_AMOUNT * np.abs(z_scores) + W_VELOCITY * velocity_flags - 2.0)

    X = np.column_stack([np.abs(z_scores), velocity_flags, geo_flags])
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)

    methods = {
        "FraudRiskAgent_full_ensemble": ensemble_scores,
        "amount_only_zscore": amount_only_scores,
        "velocity_only": velocity_only_scores,
        "geo_only": geo_only_scores,
        "ensemble_reimplementation_check": full_reimpl,
        "ablation_no_amount": no_amount,
        "ablation_no_velocity": no_velocity,
        "ablation_no_geo": no_geo,
        "isolation_forest_unsupervised": iso_scores,
    }

    results = {}
    for name, scores in methods.items():
        curves = compute_roc_pr(y_true, scores)
        results[name] = {"roc_auc": curves["roc_auc"], "pr_auc": curves["pr_auc"]}

    summary = {
        "n_transactions": len(transactions),
        "n_fraud": len(ground_truth),
        "results": results,
        "reimplementation_sanity_check": (
            "ensemble_reimplementation_check should closely match FraudRiskAgent_full_ensemble; "
            "a large gap would indicate the re-implementation here doesn't match the real agent's formula. "
            f"AUC difference: {abs(results['FraudRiskAgent_full_ensemble']['roc_auc'] - results['ensemble_reimplementation_check']['roc_auc']):.4f}"
        ),
        "interpretation": (
            "Comparing FraudRiskAgent_full_ensemble against amount_only/velocity_only/geo_only shows "
            "whether combining signals earns its place over any single signal alone. The ablation_no_* "
            "rows show each signal's marginal contribution within the ensemble formula itself (holding "
            "the other two signals' weights fixed) -- this isolates contribution without "
            "re-deriving new weights per ablation, which would answer a different question (what is the "
            "best 2-signal ensemble) than the one asked (does the 3rd signal help the ACTUAL implemented "
            "ensemble). isolation_forest_unsupervised is a different, model-based method (not a "
            "hand-tuned rule combination) over the same three numeric features, included so the comparison "
            "is not limited to variants of the same rule-based approach."
        ),
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {OUT_PATH}")

    generate_figure(results)


def generate_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                          "axes.spines.top": False, "axes.spines.right": False})

    order = ["FraudRiskAgent_full_ensemble", "isolation_forest_unsupervised", "amount_only_zscore",
             "velocity_only", "geo_only", "ablation_no_amount", "ablation_no_velocity", "ablation_no_geo"]
    labels = ["Full ensemble\n(FraudRiskAgent)", "Isolation Forest\n(unsupervised)", "Amount-only\n(z-score)",
              "Velocity-only", "Geo-only", "Ablation:\nno amount", "Ablation:\nno velocity", "Ablation:\nno geo"]
    roc_aucs = [results[k]["roc_auc"] for k in order]
    pr_aucs = [results[k]["pr_auc"] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(order))
    colors = ["#2563eb"] + ["#7c3aed"] + ["#94a3b8"] * 3 + ["#f59e0b"] * 3

    axes[0].bar(x, roc_aucs, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("ROC-AUC: Baselines and Signal Ablation")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)

    axes[1].bar(x, pr_aucs, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("PR-AUC: Baselines and Signal Ablation")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fraud_baselines_ablation.png")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_fraud_baselines_ablation.png'}")


if __name__ == "__main__":
    main()

"""
experiments/fraud_threshold_sweep.py

Extends E3 (single-threshold fraud evaluation, precision=1.00,
recall=0.44 at risk>=0.70) with a full threshold sweep: precision,
recall, F1 at every distinct risk score, plus ROC-AUC and PR-AUC. This
directly addresses the concern that a single reported operating point
can look more impressive than the underlying detector's discriminative
power actually is.

Also worth noting: the injected fraud patterns (extreme amount, geo
mismatch, velocity burst) are the same three signal families the
detector explicitly scores on -- so this experiment measures the
detector's ability to separate a specific, documented anomaly-
generating process from normal transactions, not general real-world
fraud detection capability. That distinction is stated here explicitly
rather than left implicit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.fraud_risk_agent import FraudRiskAgent
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from data.synthetic_data import build_full_dataset

OUT_PATH = Path(__file__).resolve().parent / "fraud_threshold_sweep_results.json"


def compute_roc_pr(y_true: np.ndarray, scores: np.ndarray):
    """Manual ROC/PR curve computation (no sklearn dependency): sweep
    every distinct score as a threshold, compute confusion-matrix-derived
    rates at each, and integrate via the trapezoidal rule for AUC."""
    thresholds = np.sort(np.unique(scores))[::-1]
    P = y_true.sum()
    N = len(y_true) - P

    tprs, fprs, precisions, recalls = [], [], [], []
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        tprs.append(tpr); fprs.append(fpr)
        precisions.append(precision); recalls.append(recall)

    # prepend/append boundary points for clean AUC integration
    fprs = [0.0] + fprs + [1.0]
    tprs = [0.0] + tprs + [1.0]
    recalls_sorted_idx = np.argsort(recalls)
    recalls_for_auc = [0.0] + list(np.array(recalls)[recalls_sorted_idx]) + [1.0]
    precisions_for_auc = [1.0] + list(np.array(precisions)[recalls_sorted_idx]) + [0.0]

    roc_auc = float(np.trapezoid(tprs, fprs))
    pr_auc = float(np.trapezoid(precisions_for_auc, recalls_for_auc))

    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "tpr": [round(float(x), 4) for x in tprs[1:-1]],
        "fpr": [round(float(x), 4) for x in fprs[1:-1]],
        "precision": [round(float(x), 4) for x in precisions],
        "recall": [round(float(x), 4) for x in recalls],
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def main():
    dataset = build_full_dataset(seed=42)
    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path="/tmp/acos_fraud_sweep.db"))
    decision = agent.reason({"transactions": dataset["transactions"]})
    scored = {s["transaction_id"]: s for s in decision.output["scored"]}
    ground_truth = set(dataset["fraud_ground_truth"])

    txn_ids = list(scored.keys())
    y_true = np.array([1 if tid in ground_truth else 0 for tid in txn_ids])
    scores = np.array([scored[tid]["risk_score"] for tid in txn_ids])

    curves = compute_roc_pr(y_true, scores)

    # operating-point table at a representative set of thresholds
    operating_points = []
    for t in [0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        pred = scores >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(len(y_true) - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None
        operating_points.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        })

    summary = {
        "n_transactions": len(y_true),
        "n_fraud": int(y_true.sum()),
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
        "operating_points": operating_points,
        "note_on_injection_process": (
            "The injected fraud transactions use exactly the three signal families "
            "(extreme amount, geo mismatch, velocity burst) that FraudRiskAgent's "
            "reason() scores on. ROC-AUC and PR-AUC reported here therefore measure "
            "the detector's ability to separate this specific, documented anomaly-"
            "generating process from normal transactions -- a measure of "
            "implementation correctness and internal consistency -- not a claim "
            "about detection capability against real, adversarial, previously-"
            "unseen fraud patterns that may not match these three signal families."
        ),
        "curves": curves,
    }

    agent.long_term_memory.close()
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "curves"}, indent=2))
    print(f"\nFull results (incl. full curves) written to {OUT_PATH}")

    generate_figure(curves)


def generate_figure(curves: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    fpr = [0.0] + curves["fpr"] + [1.0]
    tpr = [0.0] + curves["tpr"] + [1.0]
    axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC (AUC={curves['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=8)

    order = np.argsort(curves["recall"])
    recall_sorted = np.array(curves["recall"])[order]
    precision_sorted = np.array(curves["precision"])[order]
    axes[1].plot(recall_sorted, precision_sorted, color="#dc2626", linewidth=2,
                 label=f"PR (AUC={curves['pr_auc']:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(fontsize=8)

    fig.suptitle("Fraud-Screening Threshold Sweep (synthetic dataset, seed=42, n=315 transactions)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fraud_roc_pr.png")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_fraud_roc_pr.png'}")


if __name__ == "__main__":
    main()

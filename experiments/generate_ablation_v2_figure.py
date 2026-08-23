"""
experiments/generate_ablation_v2_figure.py

Regenerates figures/fig_ablation_v2.png (Figure 7 in the manuscript)
from experiments/ablation_v2_results.json. This figure previously had
no generator script in the repository -- it was produced ad hoc and
never saved as a reusable file, so it could not be regenerated after
ablation_study_v2.py's results changed (e.g. after the randomized-
order/warm-up timing protocol fix).

Three panels: (1) latency by configuration with SD error bars, (2)
downstream outcome cost, reconciled vs. unreconciled, under both
demand models, (3) governance catch rate across stress levels.

Usage:
    python3 experiments/generate_ablation_v2_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).resolve().parent / "ablation_v2_results.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "figures" / "fig_ablation_v2.png"


def main():
    d = json.loads(RESULTS_PATH.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Ablation v2: Equal-Work Baseline, Multi-Seed (n=20), Independent Outcome Measure", fontsize=10)

    # Panel 1: latency by configuration
    lat = d["latency_by_configuration_seconds"]
    labels = ["EQUAL_WORK\nSEQUENTIAL", "ACOS_NO\nGOVERNANCE", "ACOS_NO\nRECONCILIATION", "ACOS_FULL"]
    keys = ["equal_work_sequential", "acos_no_governance", "acos_no_reconciliation", "acos_full"]
    means_ms = [lat[k]["mean"] * 1000 for k in keys]
    sds_ms = [lat[k]["sd"] * 1000 for k in keys]
    colors = ["#a0a0a0", "#4a90d9", "#4a90d9", "#4a90d9"]
    axes[0].bar(labels, means_ms, yerr=sds_ms, capsize=4, color=colors)
    axes[0].set_ylabel("Latency (ms), mean +/- SD, 20 seeds")
    axes[0].set_title("Latency by Configuration\n(equal computational work)", fontsize=9)

    # Panel 2: downstream outcome cost, reconciled vs unreconciled
    om = d["outcome_measure"]
    regimes = ["Price-independent\ndemand", "Price-elastic\ndemand\n(model assumption holds)"]
    reconciled = [om["price_independent"]["reconciled_mean_total_cost"], om["price_elastic"]["reconciled_mean_total_cost"]]
    unreconciled = [om["price_independent"]["unreconciled_mean_total_cost"], om["price_elastic"]["unreconciled_mean_total_cost"]]
    x = range(len(regimes))
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], reconciled, width, label="Reconciled", color="#2255aa")
    axes[1].bar([i + width / 2 for i in x], unreconciled, width, label="Unreconciled", color="#a8c4e8")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(regimes, fontsize=8)
    axes[1].set_ylabel("Mean total cost (holding + lost-margin stockout), $")
    axes[1].set_title("Downstream Outcome Cost:\nReconciled vs Unreconciled", fontsize=9)
    axes[1].legend(fontsize=8)

    # Panel 3: governance catch rate across stress levels
    gov = d["governance_stress_sweep"]
    levels = sorted(gov.keys(), key=lambda k: gov[k]["max_price_change_pct"])
    pcts = [gov[k]["max_price_change_pct"] * 100 for k in levels]
    would_ship = [gov[k]["naive_extreme_would_ship"] for k in levels]
    rejected = [gov[k]["acos_full_n_rejected"] for k in levels]
    axes[2].plot(pcts, would_ship, marker="o", color="#cc3333", label="Would ship (naive, no governance)")
    axes[2].plot(pcts, rejected, marker="s", color="#2a8a4a", label="Rejected by governance (ACOS_FULL)")
    axes[2].set_xlabel("PricingAgent max allowed price change (%)")
    axes[2].set_ylabel("Number of extreme-price actions")
    axes[2].set_title("Governance Catch Rate\nAcross Stress Levels", fontsize=9)
    axes[2].legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

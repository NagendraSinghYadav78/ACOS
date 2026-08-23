"""
experiments/governance_stress_multiseed.py

Runs the governance stress sweep (Section 8.4.2) across multiple seeds
instead of just seed=42, to check whether the catch rate reported in
the single-seed version holds up on independent synthetic draws.

Reuses build_dataset_with_future() and governance_stress_sweep() from
ablation_study_v2.py rather than duplicating dataset/sweep logic.

Usage:
    python3 experiments/governance_stress_multiseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ablation_study_v2 import build_dataset_with_future, governance_stress_sweep

N_SEEDS = 10
LEVELS = (0.25, 0.35, 0.45, 0.60, 0.80)
OUT_PATH = Path(__file__).parent / "governance_stress_multiseed_results.json"


def main():
    all_results = {}
    for seed in range(1, N_SEEDS + 1):
        dataset = build_dataset_with_future(seed=seed, future_weeks=4)
        inputs = {
            "sales_history": dataset["sales_history"],
            "catalog": dataset["catalog"],
            "current_inventory": dataset["current_inventory"],
            "suppliers": dataset["suppliers"],
            "transactions": dataset["transactions"],
        }
        all_results[seed] = governance_stress_sweep(dataset, inputs, seed=seed, levels=LEVELS)
        print(f"seed={seed} done")

    # Summarize: does every violation get caught at every above-threshold level,
    # across every seed?
    summary = {}
    for level_key in ["45pct", "60pct", "80pct"]:
        total_would_ship = sum(r[level_key]["naive_extreme_would_ship"] for r in all_results.values())
        total_rejected = sum((r[level_key]["acos_full_n_rejected"] or 0) for r in all_results.values())
        per_seed_counts = [r[level_key]["naive_extreme_would_ship"] for r in all_results.values()]
        summary[level_key] = {
            "total_would_ship": total_would_ship,
            "total_rejected": total_rejected,
            "all_caught": total_would_ship == total_rejected,
            "per_seed_violation_counts": per_seed_counts,
            "min": min(per_seed_counts),
            "max": max(per_seed_counts),
            "mean": round(sum(per_seed_counts) / len(per_seed_counts), 2),
        }

    output = {"n_seeds": N_SEEDS, "levels": LEVELS, "per_seed": all_results, "summary": summary}
    OUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()

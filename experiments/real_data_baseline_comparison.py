"""
experiments/real_data_baseline_comparison.py

Extends the UCI and Rossmann out-of-sample backtests with two
additional baselines beyond naive persistence -- seasonal-naive
and Croston's method (the standard classical method for intermittent
demand), since "Holt vs. one trivial comparator" is too weak a
baseline set on its own -- and directly tests (rather than just
asserting) whether the UCI dataset's null result is attributable to
demand intermittency: if Croston's method meaningfully outperforms Holt
specifically on UCI (where per-SKU series are sparser) but not on
Rossmann (where they are not), that is real evidence for the
intermittency explanation; if it does not, this script's output says
so and the causal claim should be softened accordingly.
"""
from __future__ import annotations

import json
import statistics as _stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.demand_forecast_agent import holt_linear_forecast
from experiments.forecast_baselines import croston_forecast, evaluate_forecast, seasonal_naive_forecast
from experiments.real_data_validation import build_series as build_series_uci
from experiments.real_data_validation import load_and_aggregate as load_uci
from experiments.real_data_validation_rossmann import build_series as build_series_rossmann
from experiments.real_data_validation_rossmann import load_and_aggregate as load_rossmann
from experiments.stats_utils import paired_comparison

HOLDOUT = 4


def backtest_all_methods(series: dict, holdout: int = HOLDOUT) -> dict:
    results = {}
    for key, values in series.items():
        if len(values) < holdout + 8:
            continue
        train, test = values[:-holdout], values[-holdout:]

        holt_fc, _, _, _ = holt_linear_forecast(train, horizon=holdout)
        naive_fc = [train[-1]] * holdout
        seasonal_fc = seasonal_naive_forecast(train, holdout, season_length=4)
        croston_fc, croston_meta = croston_forecast(train, holdout)

        results[key] = {
            "holt": evaluate_forecast(test, holt_fc),
            "naive": evaluate_forecast(test, naive_fc),
            "seasonal_naive": evaluate_forecast(test, seasonal_fc),
            "croston": evaluate_forecast(test, croston_fc),
            "croston_meta": croston_meta,
        }
    return results


def summarize(results: dict, dataset_name: str) -> dict:
    methods = ["holt", "naive", "seasonal_naive", "croston"]
    summary = {"dataset": dataset_name, "n_series": len(results)}

    for m in methods:
        mapes = [r[m]["mape"] for r in results.values() if r[m]["mape"] is not None]
        summary[f"{m}_median_mape"] = round(_stats.median(mapes), 4) if mapes else None
        summary[f"{m}_mean_mape"] = round(_stats.mean(mapes), 4) if mapes else None

    avg_frac_zero = _stats.mean(r["croston_meta"]["fraction_zero_periods"] for r in results.values())
    summary["avg_fraction_zero_periods_in_training"] = round(avg_frac_zero, 4)

    # paired test: Holt vs Croston specifically, to test the intermittency claim
    holt_mapes, croston_mapes = [], []
    for r in results.values():
        if r["holt"]["mape"] is not None and r["croston"]["mape"] is not None:
            holt_mapes.append(r["holt"]["mape"])
            croston_mapes.append(r["croston"]["mape"])
    if len(holt_mapes) > 1:
        summary["holt_vs_croston_paired_test"] = paired_comparison(holt_mapes, croston_mapes, seed=42)

    # paired test: Holt vs seasonal-naive
    holt_mapes2, seas_mapes = [], []
    for r in results.values():
        if r["holt"]["mape"] is not None and r["seasonal_naive"]["mape"] is not None:
            holt_mapes2.append(r["holt"]["mape"])
            seas_mapes.append(r["seasonal_naive"]["mape"])
    if len(holt_mapes2) > 1:
        summary["holt_vs_seasonal_naive_paired_test"] = paired_comparison(holt_mapes2, seas_mapes, seed=43)

    return summary


def main():
    out = {}

    print("=== UCI Online Retail ===")
    try:
        df_uci, _, _, top_skus_uci = load_uci()
        series_uci = build_series_uci(df_uci, top_skus_uci)
        results_uci = backtest_all_methods(series_uci)
        summary_uci = summarize(results_uci, "UCI Online Retail")
        out["uci"] = summary_uci
        print(json.dumps(summary_uci, indent=2, default=str))
    except FileNotFoundError:
        print("UCI raw data not found, skipping (see experiments/real_data_validation.py for fetch instructions).")

    print("\n=== Rossmann Store Sales ===")
    try:
        df_ross, _, top_stores = load_rossmann()
        series_ross = build_series_rossmann(df_ross, top_stores)
        results_ross = backtest_all_methods(series_ross)
        summary_ross = summarize(results_ross, "Rossmann Store Sales")
        out["rossmann"] = summary_ross
        print(json.dumps(summary_ross, indent=2, default=str))
    except FileNotFoundError:
        print("Rossmann raw data not found, skipping.")

    out_path = Path(__file__).resolve().parent / "real_data_baseline_comparison_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull results written to {out_path}")

    generate_figure(out)


def generate_figure(out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})

    datasets = [k for k in ["uci", "rossmann"] if k in out]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    methods = ["holt", "naive", "seasonal_naive", "croston"]
    labels = ["Holt", "Naive", "Seasonal-naive", "Croston's\n(intermittent)"]
    colors = ["#2563eb", "#94a3b8", "#f59e0b", "#059669"]

    for ax, key in zip(axes, datasets):
        s = out[key]
        vals = [s[f"{m}_median_mape"] for m in methods]
        ax.bar(labels, vals, color=colors)
        ax.set_ylabel("Median out-of-sample MAPE")
        ax.set_title(f"{s['dataset']}\n(n={s['n_series']} series)")

    fig.suptitle("Forecasting Method Comparison: Holt vs. Three Baselines", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_forecast_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_forecast_baseline_comparison.png'}")


if __name__ == "__main__":
    main()

"""
experiments/rolling_origin_backtest.py

Extends the single-holdout backtest used in real_data_validation.py /
real_data_validation_rossmann.py (one train/test split, 4-week horizon
only) with a genuine rolling-origin evaluation: the training origin is
advanced one week at a time, and at each origin we forecast multiple
horizons (1, 4, 8 weeks), scoring against the actual subsequently-
observed values. A single train/test split is a holdout, not a
rolling-origin evaluation -- this script is what that term actually
means, and the original single-split experiments remain as a separate,
explicitly-labeled single-holdout result rather than being conflated
with this one.
"""
from __future__ import annotations

import json
import statistics as _stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.demand_forecast_agent import holt_linear_forecast
from experiments.real_data_validation import build_series as build_series_uci
from experiments.real_data_validation import load_and_aggregate as load_uci
from experiments.real_data_validation_rossmann import build_series as build_series_rossmann
from experiments.real_data_validation_rossmann import load_and_aggregate as load_rossmann
from experiments.stats_utils import paired_comparison

HORIZONS = [1, 4, 8]
MIN_TRAIN_LENGTH = 16  # minimum weeks of history before the first origin
STEP = 2  # advance the origin by 2 weeks between folds (keeps runtime reasonable)


def rolling_origin_evaluate(series: dict, horizons=HORIZONS, min_train=MIN_TRAIN_LENGTH, step=STEP) -> dict:
    """For each series and each horizon, generate forecasts from every
    valid origin (advancing by `step` weeks), score against the actual
    values, and aggregate. Returns per-horizon MAPE distributions for
    Holt and a naive baseline, i.e. a real multi-fold backtest rather
    than one train/test split."""
    per_horizon_holt = {h: [] for h in horizons}
    per_horizon_naive = {h: [] for h in horizons}
    n_folds_total = 0

    for key, values in series.items():
        n = len(values)
        max_horizon = max(horizons)
        # valid origins: enough history to train, enough future to test the largest horizon
        origins = list(range(min_train, n - max_horizon, step))
        for origin in origins:
            train = values[:origin]
            try:
                holt_fc, _, _, _ = holt_linear_forecast(train, horizon=max_horizon)
            except ValueError:
                continue
            naive_fc = [train[-1]] * max_horizon

            for h in horizons:
                actual = values[origin:origin + h]
                if len(actual) < h:
                    continue
                holt_pred = holt_fc[:h]
                naive_pred = naive_fc[:h]

                holt_errors = [abs((actual[i] - holt_pred[i]) / actual[i]) for i in range(h) if actual[i] != 0]
                naive_errors = [abs((actual[i] - naive_pred[i]) / actual[i]) for i in range(h) if actual[i] != 0]
                if holt_errors:
                    per_horizon_holt[h].append(sum(holt_errors) / len(holt_errors))
                if naive_errors:
                    per_horizon_naive[h].append(sum(naive_errors) / len(naive_errors))
            n_folds_total += 1

    results = {}
    for h in horizons:
        holt_vals = per_horizon_holt[h]
        naive_vals = per_horizon_naive[h]
        n_paired = min(len(holt_vals), len(naive_vals))
        entry = {
            "n_fold_observations_holt": len(holt_vals),
            "n_fold_observations_naive": len(naive_vals),
            "holt_median_mape": round(_stats.median(holt_vals), 4) if holt_vals else None,
            "holt_mean_mape": round(_stats.mean(holt_vals), 4) if holt_vals else None,
            "naive_median_mape": round(_stats.median(naive_vals), 4) if naive_vals else None,
            "naive_mean_mape": round(_stats.mean(naive_vals), 4) if naive_vals else None,
        }
        if len(holt_vals) == len(naive_vals) and len(holt_vals) > 1:
            entry["paired_test"] = paired_comparison(holt_vals, naive_vals, seed=h)
        results[f"horizon_{h}w"] = entry

    return {"n_series": len(series), "n_series_origin_folds_total": n_folds_total, "by_horizon": results}


def main():
    out = {}

    print("=== UCI Online Retail: rolling-origin, multi-horizon ===")
    try:
        df_uci, _, _, top_skus_uci = load_uci()
        series_uci = build_series_uci(df_uci, top_skus_uci)
        out["uci"] = rolling_origin_evaluate(series_uci)
        print(json.dumps(out["uci"], indent=2, default=str))
    except FileNotFoundError:
        print("UCI raw data not found, skipping.")

    print("\n=== Rossmann Store Sales: rolling-origin, multi-horizon ===")
    try:
        df_ross, _, top_stores = load_rossmann()
        series_ross = build_series_rossmann(df_ross, top_stores)
        out["rossmann"] = rolling_origin_evaluate(series_ross)
        print(json.dumps(out["rossmann"], indent=2, default=str))
    except FileNotFoundError:
        print("Rossmann raw data not found, skipping.")

    out_path = Path(__file__).resolve().parent / "rolling_origin_results.json"
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
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 18,
                          "axes.spines.top": False, "axes.spines.right": False,
                          "axes.titlesize": 18, "axes.labelsize": 18,
                          "xtick.labelsize": 16, "ytick.labelsize": 16,
                          "legend.fontsize": 16})

    datasets = [k for k in ["uci", "rossmann"] if k in out]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.5 * len(datasets), 5.2))
    if len(datasets) == 1:
        axes = [axes]

    for ax, key in zip(axes, datasets):
        d = out[key]
        horizons = sorted(int(k.split("_")[1][:-1]) for k in d["by_horizon"].keys())
        holt_vals = [d["by_horizon"][f"horizon_{h}w"]["holt_median_mape"] for h in horizons]
        naive_vals = [d["by_horizon"][f"horizon_{h}w"]["naive_median_mape"] for h in horizons]
        x = np.arange(len(horizons))
        w = 0.35
        ax.bar(x - w / 2, holt_vals, w, label="Holt", color="#2563eb")
        ax.bar(x + w / 2, naive_vals, w, label="Naive", color="#94a3b8")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}-week" for h in horizons])
        ax.set_ylabel("Median MAPE across all folds")
        title = "UCI Online Retail" if key == "uci" else "Rossmann Store Sales"
        ax.set_title(f"{title}\n({d['n_series_origin_folds_total']} origin-folds)")
        ax.legend()

    fig.suptitle("Rolling-Origin Backtest: Median MAPE by Forecast Horizon", fontsize=20, y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rolling_origin_backtest.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_rolling_origin_backtest.png'}")


if __name__ == "__main__":
    main()

"""
experiments/real_data_validation_rossmann.py

A second, independent real-data validation of DemandForecastAgent,
using the Rossmann Store Sales dataset (Kaggle competition data;
1,017,209 real daily sales records across 1,115 German drug stores,
Jan 2013-Jul 2015), retrieved via a public GitHub mirror since this
sandboxed environment has no direct network path to kaggle.com.

This complements experiments/real_data_validation.py (UCI Online
Retail): two independent real datasets, same rigorous methodology
(genuine out-of-sample holdout, naive-baseline comparison), now with
formal paired statistical testing (experiments/stats_utils.py) rather
than only point-estimate summary statistics.

Honesty notes:
  - Only the top 20 stores by total sales volume are used (for
    comparability with the 20-SKU/20-store scale used elsewhere in
    this codebase), not a random or cherry-picked sample.
  - Store-closure days (Open=0) are included in the weekly aggregate
    as zero-sales days, since that is the real, correct weekly total.
  - This dataset, like the UCI Online Retail dataset, provides no
    cost, supplier, or fraud-label fields, so only DemandForecastAgent
    is validated here.
"""
from __future__ import annotations

import json
import statistics as _stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from agents.demand_forecast_agent import holt_linear_forecast
from experiments.stats_utils import paired_comparison
from experiments.real_data_validation import compute_additional_metrics

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "external" / "rossmann_train.csv"
OUT_PATH = Path(__file__).resolve().parent / "real_data_results_rossmann.json"

N_STORES = 20
HOLDOUT_WEEKS = 4


def load_and_aggregate():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    n_raw = len(df)
    df["Date"] = pd.to_datetime(df["Date"])

    top_stores = df.groupby("Store")["Sales"].sum().sort_values(ascending=False).head(N_STORES).index.tolist()
    df = df[df["Store"].isin(top_stores)]
    return df, n_raw, top_stores


def build_series(df: pd.DataFrame, top_stores: list) -> dict:
    series = {}
    for store in top_stores:
        s = df[df["Store"] == store].set_index("Date")["Sales"]
        weekly = s.resample("W").sum().astype(float)
        series[str(store)] = weekly.tolist()
    return series


def backtest_forecast(series: dict, holdout: int = HOLDOUT_WEEKS) -> dict:
    results = {}
    for store, values in series.items():
        if len(values) < holdout + 8:
            continue
        train, test = values[:-holdout], values[-holdout:]
        try:
            forecasts, in_sample_mape, level, trend = holt_linear_forecast(train, horizon=holdout)
        except ValueError:
            continue

        errors = [abs((test[i] - forecasts[i]) / test[i]) for i in range(holdout) if test[i] != 0]
        oos_mape = sum(errors) / len(errors) if errors else None

        naive_forecast = [train[-1]] * holdout
        naive_errors = [abs((test[i] - naive_forecast[i]) / test[i])
                         for i in range(holdout) if test[i] != 0]
        naive_mape = sum(naive_errors) / len(naive_errors) if naive_errors else None

        holt_extra = compute_additional_metrics(train, test, forecasts)
        naive_extra = compute_additional_metrics(train, test, naive_forecast)

        results[store] = {
            "n_weeks_total": len(values),
            "in_sample_mape_train_only": round(in_sample_mape, 4),
            "out_of_sample_mape_holt": round(oos_mape, 4) if oos_mape is not None else None,
            "out_of_sample_mape_naive_baseline": round(naive_mape, 4) if naive_mape is not None else None,
            "out_of_sample_wape_holt": holt_extra["wape"],
            "out_of_sample_wape_naive_baseline": naive_extra["wape"],
            "out_of_sample_mase_holt": holt_extra["mase"],
            "out_of_sample_mase_naive_baseline": naive_extra["mase"],
            "holt_beats_naive": (oos_mape < naive_mape) if (oos_mape is not None and naive_mape is not None) else None,
            "holt_beats_naive_by_wape": (holt_extra["wape"] < naive_extra["wape"]) if (holt_extra["wape"] is not None and naive_extra["wape"] is not None) else None,
        }
    return results


def main():
    if not RAW_PATH.exists():
        print(f"Rossmann dataset not found at {RAW_PATH}. Skipping.")
        return

    df, n_raw, top_stores = load_and_aggregate()
    series = build_series(df, top_stores)
    backtest = backtest_forecast(series)

    holt_mapes = [r["out_of_sample_mape_holt"] for r in backtest.values() if r["out_of_sample_mape_holt"] is not None]
    naive_mapes = [r["out_of_sample_mape_naive_baseline"] for r in backtest.values() if r["out_of_sample_mape_naive_baseline"] is not None]
    n_beats_naive = sum(1 for r in backtest.values() if r["holt_beats_naive"])
    holt_wapes = [r["out_of_sample_wape_holt"] for r in backtest.values() if r["out_of_sample_wape_holt"] is not None]
    naive_wapes = [r["out_of_sample_wape_naive_baseline"] for r in backtest.values() if r["out_of_sample_wape_naive_baseline"] is not None]
    holt_mases = [r["out_of_sample_mase_holt"] for r in backtest.values() if r["out_of_sample_mase_holt"] is not None]
    naive_mases = [r["out_of_sample_mase_naive_baseline"] for r in backtest.values() if r["out_of_sample_mase_naive_baseline"] is not None]
    n_beats_naive_wape = sum(1 for r in backtest.values() if r["holt_beats_naive_by_wape"])
    wape_stat_test = paired_comparison(holt_wapes, naive_wapes, seed=42) if holt_wapes else None

    stat_test = paired_comparison(holt_mapes, naive_mapes, seed=42)

    summary = {
        "dataset": "Rossmann Store Sales (Kaggle competition data), retrieved via "
                   "RPI-DATA/tutorials-intro GitHub mirror",
        "n_raw_rows": n_raw,
        "n_stores_used": len(series),
        "stores": top_stores,
        "holdout_weeks": HOLDOUT_WEEKS,
        "mean_out_of_sample_mape_holt": round(_stats.mean(holt_mapes), 4) if holt_mapes else None,
        "median_out_of_sample_mape_holt": round(_stats.median(holt_mapes), 4) if holt_mapes else None,
        "mean_out_of_sample_mape_naive_baseline": round(_stats.mean(naive_mapes), 4) if naive_mapes else None,
        "median_out_of_sample_mape_naive_baseline": round(_stats.median(naive_mapes), 4) if naive_mapes else None,
        "n_stores_where_holt_beats_naive": n_beats_naive,
        "mean_wape_holt": round(_stats.mean(holt_wapes), 4) if holt_wapes else None,
        "median_wape_holt": round(_stats.median(holt_wapes), 4) if holt_wapes else None,
        "mean_wape_naive_baseline": round(_stats.mean(naive_wapes), 4) if naive_wapes else None,
        "median_wape_naive_baseline": round(_stats.median(naive_wapes), 4) if naive_wapes else None,
        "n_stores_where_holt_beats_naive_by_wape": n_beats_naive_wape,
        "wape_paired_statistical_test": wape_stat_test,
        "mean_mase_holt": round(_stats.mean(holt_mases), 4) if holt_mases else None,
        "median_mase_holt": round(_stats.median(holt_mases), 4) if holt_mases else None,
        "mean_mase_naive_baseline": round(_stats.mean(naive_mases), 4) if naive_mases else None,
        "median_mase_naive_baseline": round(_stats.median(naive_mases), 4) if naive_mases else None,
        "n_stores_evaluated": len(backtest),
        "paired_statistical_test_holt_vs_naive": stat_test,
        "per_store_backtest": backtest,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_store_backtest"}, indent=2, default=str))
    print(f"\nFull results written to {OUT_PATH}")

    generate_figure(backtest)


def generate_figure(backtest: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)

    stores = list(backtest.keys())
    holt = [backtest[s]["out_of_sample_mape_holt"] for s in stores]
    naive = [backtest[s]["out_of_sample_mape_naive_baseline"] for s in stores]

    order = np.argsort(naive)
    stores = [stores[i] for i in order]
    holt = [holt[i] for i in order]
    naive = [naive[i] for i in order]

    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(stores))
    w = 0.38
    ax.bar(x - w / 2, holt, w, label="Holt's linear smoothing", color="#2563eb")
    ax.bar(x + w / 2, naive, w, label="Naive (last-value) baseline", color="#94a3b8")
    ax.set_xticks(x)
    ax.set_xticklabels(stores, rotation=90, fontsize=7)
    ax.set_ylabel("Out-of-sample MAPE")
    ax.set_title("Holt vs. Naive Baseline: 4-Week-Ahead Out-of-Sample Forecast\n"
                 "(Rossmann Store Sales dataset, real transactions, n=20 stores)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_rossmann_backtest.png")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_rossmann_backtest.png'}")


if __name__ == "__main__":
    main()

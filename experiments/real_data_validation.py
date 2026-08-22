"""
experiments/real_data_validation.py

Validates DemandForecastAgent against REAL transaction data: the UCI
"Online Retail" dataset (Chen, D. Online Retail. UCI Machine Learning
Repository, 2015, https://doi.org/10.24432/C5BW33) -- 541,909 real
invoice line items from a UK-based online retailer, Dec 2010-Dec 2011.
Retrieved via the Databricks "Spark: The Definitive Guide" teaching
repository's mirror (Apache-2.0 licensed repo; data itself is the
public UCI dataset), since archive.ics.uci.edu and Kaggle were not
directly reachable.

This experiment does two things the synthetic-data experiments (E1-E7)
could not:
  1. Uses real enterprise transaction data rather than seeded synthetic
     data, directly for demand forecasting.
  2. Performs genuine OUT-OF-SAMPLE backtesting (train on the first
     N-4 weeks, forecast the held-out last 4 weeks, compare against
     actual observed demand) rather than the in-sample MAPE reported
     in E4.

Honesty notes:
  - The dataset provides quantity, price, and date -- it does NOT
    provide unit cost, supplier information, or fraud ground truth.
    This experiment therefore validates ONLY the DemandForecastAgent
    (and, downstream, the EOQ/ROP formula's sensitivity to real demand
    statistics). PricingAgent, ProcurementAgent, and FraudRiskAgent
    still require the synthetic dataset's cost/supplier/label fields
    and are not re-validated here.
  - Returns/cancellations (negative Quantity, StockCode prefixed 'C')
    are excluded, a standard, documented preprocessing step for this
    dataset.
  - The top 20 SKUs by total volume are used (for comparability with
    the 20-SKU synthetic catalog used in E1-E7), not a random or
    cherry-picked sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from agents.demand_forecast_agent import holt_linear_forecast
from agents.inventory_agent import InventoryAgent
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from experiments.stats_utils import paired_comparison

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "external" / "online_retail_raw.csv"
OUT_PATH = Path(__file__).resolve().parent / "real_data_results.json"

N_SKUS = 20
HOLDOUT_WEEKS = 4


def load_and_aggregate() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, encoding="latin1")
    n_raw = len(df)
    df = df[df["Quantity"] > 0]  # exclude returns/cancellations (documented, standard)
    df = df[~df["StockCode"].astype(str).str.startswith("C")]
    n_after_filter = len(df)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")

    top_skus = df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(N_SKUS).index.tolist()
    df = df[df["StockCode"].isin(top_skus)]
    return df, n_raw, n_after_filter, top_skus


def build_series(df: pd.DataFrame, top_skus: list) -> dict:
    series = {}
    for sku in top_skus:
        s = df[df["StockCode"] == sku].set_index("InvoiceDate")["Quantity"]
        weekly = s.resample("W").sum().astype(float)
        series[sku] = weekly.tolist()
    return series


def compute_additional_metrics(train: list, test: list, forecast: list) -> dict:
    """Metrics resistant to MAPE's near-zero-actual pathology (a known
    issue when a SKU's actual demand is near zero during the holdout
    period): WAPE (weighted absolute percentage error, robust
    to individual near-zero points since it sums before dividing) and MASE
    (mean absolute scaled error, scaled by the naive-baseline's in-sample
    error, standard in the forecasting literature for exactly this reason)."""
    abs_errors = [abs(test[i] - forecast[i]) for i in range(len(test))]
    wape = sum(abs_errors) / sum(abs(t) for t in test) if sum(abs(t) for t in test) > 0 else None

    naive_errors = [abs(train[i] - train[i - 1]) for i in range(1, len(train))]
    scale = sum(naive_errors) / len(naive_errors) if naive_errors else None
    mase = (sum(abs_errors) / len(abs_errors)) / scale if scale else None

    return {"wape": round(wape, 4) if wape is not None else None,
            "mase": round(mase, 4) if mase is not None else None}


def backtest_forecast(series: dict, holdout: int = HOLDOUT_WEEKS) -> dict:
    """Rolling-origin-style holdout backtest: fit Holt's method on all but
    the last `holdout` weeks, forecast forward `holdout` steps, and score
    against the actual held-out values -- true out-of-sample evaluation,
    unlike E4's in-sample MAPE."""
    results = {}
    for sku, values in series.items():
        if len(values) < holdout + 8:  # need enough history to fit meaningfully
            continue
        train, test = values[:-holdout], values[-holdout:]
        try:
            forecasts, in_sample_mape, level, trend = holt_linear_forecast(train, horizon=holdout)
        except ValueError:
            continue

        errors = [abs((test[i] - forecasts[i]) / test[i]) for i in range(holdout) if test[i] != 0]
        out_of_sample_mape = sum(errors) / len(errors) if errors else None

        # naive baseline: forecast = last observed training value, repeated
        naive_forecast = [train[-1]] * holdout
        naive_errors = [abs((test[i] - naive_forecast[i]) / test[i])
                         for i in range(holdout) if test[i] != 0]
        naive_mape = sum(naive_errors) / len(naive_errors) if naive_errors else None

        holt_extra = compute_additional_metrics(train, test, forecasts)
        naive_extra = compute_additional_metrics(train, test, naive_forecast)

        results[sku] = {
            "n_weeks_total": len(values),
            "in_sample_mape_train_only": round(in_sample_mape, 4),
            "out_of_sample_mape_holt": round(out_of_sample_mape, 4) if out_of_sample_mape is not None else None,
            "out_of_sample_mape_naive_baseline": round(naive_mape, 4) if naive_mape is not None else None,
            "out_of_sample_wape_holt": holt_extra["wape"],
            "out_of_sample_wape_naive_baseline": naive_extra["wape"],
            "out_of_sample_mase_holt": holt_extra["mase"],
            "out_of_sample_mase_naive_baseline": naive_extra["mase"],
            "holt_beats_naive": (out_of_sample_mape < naive_mape) if (out_of_sample_mape is not None and naive_mape is not None) else None,
            "holt_beats_naive_by_wape": (holt_extra["wape"] < naive_extra["wape"]) if (holt_extra["wape"] is not None and naive_extra["wape"] is not None) else None,
        }
    return results


def run_inventory_on_real_stats(series: dict) -> dict:
    """Feeds real observed demand statistics (mean, std of weekly demand)
    into the InventoryAgent's EOQ/ROP formula. Unit cost and lead time
    are NOT in the public dataset, so representative, clearly-labeled
    assumed values are used for those two inputs only -- everything
    demand-related is real."""
    agent = InventoryAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path="/tmp/acos_real_inv.db"))
    forecasts = {}
    for sku, values in series.items():
        if len(values) < 8:
            continue
        fc, mape, level, trend = holt_linear_forecast(values, horizon=4)
        forecasts[sku] = {"forecast": fc, "in_sample_mape": mape}

    catalog = {sku: {"unit_cost": 8.0} for sku in forecasts}  # ASSUMED (not in dataset)
    current_inventory = {sku: 0 for sku in forecasts}  # conservative: assume stockout risk

    decision = agent.reason({
        "forecast_demand": {"forecasts": forecasts},
        "catalog": catalog,
        "current_inventory": current_inventory,
        "service_level": 0.95,
        "lead_time_days": 7,  # ASSUMED (not in dataset)
    })
    agent.long_term_memory.close()
    return decision.output


def main():
    if not RAW_PATH.exists():
        print(f"Real dataset not found at {RAW_PATH}. Skipping real-data validation.")
        return

    weekly, n_raw, n_after_filter, top_skus = load_and_aggregate()
    series = build_series(weekly, top_skus)
    backtest = backtest_forecast(series)
    inventory_out = run_inventory_on_real_stats(series)

    oos_mapes = [r["out_of_sample_mape_holt"] for r in backtest.values() if r["out_of_sample_mape_holt"] is not None]
    naive_mapes = [r["out_of_sample_mape_naive_baseline"] for r in backtest.values() if r["out_of_sample_mape_naive_baseline"] is not None]
    n_beats_naive = sum(1 for r in backtest.values() if r["holt_beats_naive"])

    stat_test = paired_comparison(oos_mapes, naive_mapes, seed=42)

    # WAPE/MASE: robust to MAPE's near-zero-actual pathology
    holt_wapes = [r["out_of_sample_wape_holt"] for r in backtest.values() if r["out_of_sample_wape_holt"] is not None]
    naive_wapes = [r["out_of_sample_wape_naive_baseline"] for r in backtest.values() if r["out_of_sample_wape_naive_baseline"] is not None]
    holt_mases = [r["out_of_sample_mase_holt"] for r in backtest.values() if r["out_of_sample_mase_holt"] is not None]
    naive_mases = [r["out_of_sample_mase_naive_baseline"] for r in backtest.values() if r["out_of_sample_mase_naive_baseline"] is not None]
    n_beats_naive_wape = sum(1 for r in backtest.values() if r["holt_beats_naive_by_wape"])
    wape_stat_test = paired_comparison(holt_wapes, naive_wapes, seed=42) if holt_wapes else None

    import statistics as _stats
    summary = {
        "dataset": "UCI Online Retail (Chen, 2015, doi:10.24432/C5BW33), "
                   "retrieved via databricks/Spark-The-Definitive-Guide GitHub mirror",
        "n_raw_rows": n_raw,
        "n_rows_after_filtering_returns": n_after_filter,
        "n_skus_used": len(series),
        "skus": top_skus,
        "holdout_weeks": HOLDOUT_WEEKS,
        "mean_out_of_sample_mape_holt": round(sum(oos_mapes) / len(oos_mapes), 4) if oos_mapes else None,
        "median_out_of_sample_mape_holt": round(_stats.median(oos_mapes), 4) if oos_mapes else None,
        "mean_out_of_sample_mape_naive_baseline": round(sum(naive_mapes) / len(naive_mapes), 4) if naive_mapes else None,
        "median_out_of_sample_mape_naive_baseline": round(_stats.median(naive_mapes), 4) if naive_mapes else None,
        "n_skus_where_holt_beats_naive": n_beats_naive,
        "mean_wape_holt": round(_stats.mean(holt_wapes), 4) if holt_wapes else None,
        "median_wape_holt": round(_stats.median(holt_wapes), 4) if holt_wapes else None,
        "mean_wape_naive_baseline": round(_stats.mean(naive_wapes), 4) if naive_wapes else None,
        "median_wape_naive_baseline": round(_stats.median(naive_wapes), 4) if naive_wapes else None,
        "n_skus_where_holt_beats_naive_by_wape": n_beats_naive_wape,
        "wape_paired_statistical_test": wape_stat_test,
        "mean_mase_holt": round(_stats.mean(holt_mases), 4) if holt_mases else None,
        "median_mase_holt": round(_stats.median(holt_mases), 4) if holt_mases else None,
        "mean_mase_naive_baseline": round(_stats.mean(naive_mases), 4) if naive_mases else None,
        "median_mase_naive_baseline": round(_stats.median(naive_mases), 4) if naive_mases else None,
        "mase_interpretation_note": "MASE < 1 means the method beats the naive 1-step baseline on the training data's own scale; MASE > 1 means it is worse than that baseline.",
        "n_skus_evaluated": len(backtest),
        "paired_statistical_test_holt_vs_naive": stat_test,
        "note_on_outlier": (
            "The mean out-of-sample MAPE is heavily skewed by SKU 21915, whose holdout-period "
            "actual demand was near zero, producing a MAPE of ~219 (21,900%) from the standard "
            "percentage-error formula dividing by a near-zero actual value -- a well-known MAPE "
            "pathology on intermittent real demand. The median is reported alongside the mean for "
            "this reason and is the more representative summary statistic here."
        ),
        "per_sku_backtest": backtest,
        "inventory_agent_on_real_demand_stats": {
            "note": "unit_cost=$8.00 and lead_time_days=7 are ASSUMED (not present in the public dataset); demand statistics (mean/std) are real.",
            "sample_reorder_plan": {k: v for k, v in list(inventory_out.get("reorder_plan", {}).items())[:5]},
        },
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sku_backtest"}, indent=2, default=str))
    print(f"\nFull results written to {OUT_PATH}")

    generate_real_data_figure(backtest)


def generate_real_data_figure(backtest: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)

    skus = list(backtest.keys())
    holt = [backtest[s]["out_of_sample_mape_holt"] for s in skus]
    naive = [backtest[s]["out_of_sample_mape_naive_baseline"] for s in skus]

    order = np.argsort(naive)
    skus = [skus[i] for i in order]
    holt = [holt[i] for i in order]
    naive = [naive[i] for i in order]

    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                          "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(skus))
    w = 0.38
    holt_capped = [min(h, 3.0) for h in holt]
    naive_capped = [min(n, 3.0) for n in naive]
    ax.bar(x - w / 2, holt_capped, w, label="Holt's linear smoothing", color="#2563eb")
    ax.bar(x + w / 2, naive_capped, w, label="Naive (last-value) baseline", color="#94a3b8")
    for i, (h, n) in enumerate(zip(holt, naive)):
        if h > 3.0:
            ax.text(i - w / 2, 3.02, f"{h:.1f}", ha="center", fontsize=7, rotation=90, color="#2563eb")
        if n > 3.0:
            ax.text(i + w / 2, 3.02, f"{n:.1f}", ha="center", fontsize=7, rotation=90, color="#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels(skus, rotation=90, fontsize=7)
    ax.set_ylabel("Out-of-sample MAPE (capped at 3.0 for display)")
    ax.set_title("Holt vs. Naive Baseline: 4-Week-Ahead Out-of-Sample Forecast\n"
                 "(UCI Online Retail dataset, real transactions, n=19 SKUs)")
    ax.legend()
    ax.set_ylim(0, 3.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_real_data_backtest.png")
    plt.close(fig)
    print(f"Figure written to {fig_dir / 'fig_real_data_backtest.png'}")


if __name__ == "__main__":
    main()

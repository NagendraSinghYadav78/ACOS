"""
experiments/rolling_origin_clustered_analysis.py

The rolling-origin backtest (rolling_origin_backtest.py) pools hundreds
of overlapping origin-folds from the same 19-20 series and reports
p-values as if each fold were an independent observation. Folds from
the same series share history and are correlated, so treating fold
count as sample size overstates statistical power.

This script re-runs the same rolling-origin procedure but aggregates
to ONE number per series per method per horizon (the mean MAPE across
that series' own folds) before running any significance test -- a
standard cluster-level / series-level aggregation that respects the
non-independence of folds within a series. The resulting sample size
per test is the number of series (19 UCI, 20 Rossmann), matching the
single-holdout test's sample size, not the fold count.

Because this produces 6 confirmatory tests (2 datasets x 3 horizons,
each testing Holt vs. naive), a Holm-Bonferroni correction is applied
across that family, and Benjamini-Hochberg FDR-adjusted p-values are
reported alongside for comparison.
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
from experiments.stats_utils import paired_comparison, holm_bonferroni, benjamini_hochberg

HORIZONS = [1, 4, 8]
MIN_TRAIN_LENGTH = 16
STEP = 2


def rolling_origin_series_level(series: dict, horizons=HORIZONS, min_train=MIN_TRAIN_LENGTH, step=STEP) -> dict:
    """Same rolling-origin procedure as rolling_origin_backtest.py, but
    tracks per-series fold lists separately, then aggregates each
    series to its own mean MAPE before any statistical test."""
    per_series_per_horizon_holt = {h: {} for h in horizons}
    per_series_per_horizon_naive = {h: {} for h in horizons}
    n_folds_total = 0

    for key, values in series.items():
        n = len(values)
        max_horizon = max(horizons)
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
                    per_series_per_horizon_holt[h].setdefault(key, []).append(sum(holt_errors) / len(holt_errors))
                if naive_errors:
                    per_series_per_horizon_naive[h].setdefault(key, []).append(sum(naive_errors) / len(naive_errors))
            n_folds_total += 1

    # aggregate: one mean MAPE per series per horizon per method
    results = {}
    for h in horizons:
        series_keys = sorted(set(per_series_per_horizon_holt[h]) & set(per_series_per_horizon_naive[h]))
        holt_series_means = [_stats.mean(per_series_per_horizon_holt[h][k]) for k in series_keys]
        naive_series_means = [_stats.mean(per_series_per_horizon_naive[h][k]) for k in series_keys]
        n_folds_this_horizon = sum(len(per_series_per_horizon_holt[h][k]) for k in series_keys)

        entry = {
            "n_series": len(series_keys),
            "n_folds_underlying": n_folds_this_horizon,
            "avg_folds_per_series": round(n_folds_this_horizon / len(series_keys), 1) if series_keys else None,
            "holt_median_of_series_means": round(_stats.median(holt_series_means), 4) if holt_series_means else None,
            "naive_median_of_series_means": round(_stats.median(naive_series_means), 4) if naive_series_means else None,
        }
        if len(holt_series_means) > 1:
            entry["paired_test_series_level"] = paired_comparison(holt_series_means, naive_series_means, seed=h)
        results[f"horizon_{h}w"] = entry

    return {"n_series": len(series), "n_folds_total": n_folds_total, "by_horizon": results}


def main():
    out = {}

    print("=== UCI: series-level clustered analysis ===")
    df_uci, _, _, top_skus_uci = load_uci()
    series_uci = build_series_uci(df_uci, top_skus_uci)
    out["uci"] = rolling_origin_series_level(series_uci)

    print("=== Rossmann: series-level clustered analysis ===")
    df_ross, _, top_stores = load_rossmann()
    series_ross = build_series_rossmann(df_ross, top_stores)
    out["rossmann"] = rolling_origin_series_level(series_ross)

    # Build the family of 6 confirmatory comparisons: 2 datasets x 3 horizons
    family = []
    for dataset_key in ["uci", "rossmann"]:
        for h in HORIZONS:
            entry = out[dataset_key]["by_horizon"][f"horizon_{h}w"]
            test = entry.get("paired_test_series_level")
            if test:
                family.append({
                    "dataset": dataset_key, "horizon": h,
                    "raw_p": test["wilcoxon_p_value"],
                    "effect_size_r": test["matched_pairs_rank_biserial_r"],
                    "n_series": entry["n_series"],
                })

    raw_ps = [f["raw_p"] for f in family]
    holm_adj = holm_bonferroni(raw_ps)
    bh_adj = benjamini_hochberg(raw_ps)
    for i, f in enumerate(family):
        f["holm_adjusted_p"] = round(holm_adj[i], 4)
        f["bh_adjusted_p"] = round(bh_adj[i], 4)
        f["significant_raw_0.05"] = f["raw_p"] < 0.05
        f["significant_holm_0.05"] = holm_adj[i] < 0.05
        f["significant_bh_0.05"] = bh_adj[i] < 0.05

    out["multiple_comparison_correction"] = {
        "family_description": "6 confirmatory tests: {UCI, Rossmann} x {1-week, 4-week, 8-week horizon}, "
                               "each testing Holt vs. naive persistence at the series-cluster level.",
        "family": family,
    }

    out_path = Path(__file__).resolve().parent / "rolling_origin_clustered_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["multiple_comparison_correction"], indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()

"""
experiments/holt_parameter_sensitivity.py

DemandForecastAgent uses fixed Holt smoothing parameters (alpha=0.5,
beta=0.3) with no justification given elsewhere for that specific
choice, and no check that a different fixed choice wouldn't change
the forecasting comparison's conclusions. This script does two things:

1. TRAINING-ONLY GRID SEARCH: for each series, fits alpha/beta by
   grid search minimizing in-sample SSE on the TRAINING portion only
   (never touching the holdout), then forecasts the holdout with the
   fitted parameters -- the standard way to select smoothing
   parameters without leakage. Compares against the fixed
   alpha=0.5/beta=0.3 forecast on the same holdout.

2. SENSITIVITY SWEEP: reports out-of-sample MAPE across a grid of
   fixed (alpha, beta) pairs, so the reader can see how much the
   reported Holt-vs-baseline comparison depends on this specific
   parameter choice rather than treating alpha=0.5/beta=0.3 as
   privileged.

No future/holdout values are used to select any parameter -- fitting
uses train[:origin] only, forecasting evaluates against train[origin:].
"""
from __future__ import annotations

import itertools
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

HOLDOUT = 4
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def fit_alpha_beta_training_only(train: list, holdout: int) -> tuple:
    """Grid search alpha/beta minimizing in-sample SSE using ONLY the
    data up to (but not including) the holdout window -- i.e. we
    further split train into a fit-train/fit-validation split
    internally, fit on fit-train, and score on fit-validation, which
    is itself part of the original training data, never the holdout.
    This avoids any leakage from the actual holdout being forecasted."""
    if len(train) < 8:
        return 0.5, 0.3  # not enough data to tune; fall back to the default

    fit_train = train[:-holdout] if len(train) > holdout + 4 else train[:-2]
    fit_val = train[-holdout:] if len(train) > holdout + 4 else train[-2:]
    if len(fit_train) < 4 or not fit_val:
        return 0.5, 0.3

    best = (0.5, 0.3)
    best_sse = float("inf")
    for alpha, beta in itertools.product(ALPHA_GRID, BETA_GRID):
        try:
            fc, _, _, _ = holt_linear_forecast(fit_train, alpha=alpha, beta=beta, horizon=len(fit_val))
        except ValueError:
            continue
        sse = sum((fit_val[i] - fc[i]) ** 2 for i in range(len(fit_val)))
        if sse < best_sse:
            best_sse = sse
            best = (alpha, beta)
    return best


def evaluate_mape(test: list, forecast: list) -> float:
    terms = [abs((test[i] - forecast[i]) / test[i]) for i in range(len(test)) if test[i] != 0]
    return sum(terms) / len(terms) if terms else None


def backtest_dataset(series: dict, holdout: int = HOLDOUT) -> dict:
    fixed_mapes = []
    tuned_mapes = []
    tuned_params = []
    for key, values in series.items():
        if len(values) < holdout + 8:
            continue
        train, test = values[:-holdout], values[-holdout:]

        fixed_fc, _, _, _ = holt_linear_forecast(train, alpha=0.5, beta=0.3, horizon=holdout)
        fixed_mape = evaluate_mape(test, fixed_fc)

        alpha, beta = fit_alpha_beta_training_only(train, holdout)
        tuned_fc, _, _, _ = holt_linear_forecast(train, alpha=alpha, beta=beta, horizon=holdout)
        tuned_mape = evaluate_mape(test, tuned_fc)

        if fixed_mape is not None:
            fixed_mapes.append(fixed_mape)
        if tuned_mape is not None:
            tuned_mapes.append(tuned_mape)
            tuned_params.append((alpha, beta))

    return {
        "n_series": len(fixed_mapes),
        "fixed_alpha0.5_beta0.3_median_mape": round(_stats.median(fixed_mapes), 4) if fixed_mapes else None,
        "fixed_alpha0.5_beta0.3_mean_mape": round(_stats.mean(fixed_mapes), 4) if fixed_mapes else None,
        "training_only_tuned_median_mape": round(_stats.median(tuned_mapes), 4) if tuned_mapes else None,
        "training_only_tuned_mean_mape": round(_stats.mean(tuned_mapes), 4) if tuned_mapes else None,
        "tuned_alpha_distribution": {
            "mean": round(_stats.mean(p[0] for p in tuned_params), 3) if tuned_params else None,
            "median": round(_stats.median(p[0] for p in tuned_params), 3) if tuned_params else None,
        },
        "tuned_beta_distribution": {
            "mean": round(_stats.mean(p[1] for p in tuned_params), 3) if tuned_params else None,
            "median": round(_stats.median(p[1] for p in tuned_params), 3) if tuned_params else None,
        },
    }


def sensitivity_sweep(series: dict, holdout: int = HOLDOUT) -> dict:
    """Out-of-sample MAPE for every fixed (alpha, beta) pair in the grid,
    averaged across all series, to show how much the fixed-parameter
    choice matters."""
    grid_results = {}
    for alpha, beta in itertools.product(ALPHA_GRID, BETA_GRID):
        mapes = []
        for key, values in series.items():
            if len(values) < holdout + 8:
                continue
            train, test = values[:-holdout], values[-holdout:]
            try:
                fc, _, _, _ = holt_linear_forecast(train, alpha=alpha, beta=beta, horizon=holdout)
            except ValueError:
                continue
            m = evaluate_mape(test, fc)
            if m is not None:
                mapes.append(m)
        if mapes:
            grid_results[f"a{alpha}_b{beta}"] = round(_stats.median(mapes), 4)
    return grid_results


def main():
    out = {}

    print("=== UCI ===")
    df_uci, _, _, top_skus_uci = load_uci()
    series_uci = build_series_uci(df_uci, top_skus_uci)
    out["uci"] = backtest_dataset(series_uci)
    out["uci"]["sensitivity_sweep_median_mape_by_alpha_beta"] = sensitivity_sweep(series_uci)
    print(json.dumps({k: v for k, v in out["uci"].items() if k != "sensitivity_sweep_median_mape_by_alpha_beta"}, indent=2))

    print("\n=== Rossmann ===")
    df_ross, _, top_stores = load_rossmann()
    series_ross = build_series_rossmann(df_ross, top_stores)
    out["rossmann"] = backtest_dataset(series_ross)
    out["rossmann"]["sensitivity_sweep_median_mape_by_alpha_beta"] = sensitivity_sweep(series_ross)
    print(json.dumps({k: v for k, v in out["rossmann"].items() if k != "sensitivity_sweep_median_mape_by_alpha_beta"}, indent=2))

    # summarize sensitivity range
    for dataset_key in ["uci", "rossmann"]:
        sweep = out[dataset_key]["sensitivity_sweep_median_mape_by_alpha_beta"]
        vals = list(sweep.values())
        out[dataset_key]["sensitivity_range"] = {
            "min_median_mape_across_grid": round(min(vals), 4),
            "max_median_mape_across_grid": round(max(vals), 4),
            "default_a0.5_b0.3_median_mape": sweep.get("a0.5_b0.3"),
        }
        print(f"\n{dataset_key} sensitivity range across {len(vals)} (alpha,beta) pairs: "
              f"{min(vals):.4f} to {max(vals):.4f}; default (0.5,0.3) = {sweep.get('a0.5_b0.3')}")

    out_path = Path(__file__).resolve().parent / "holt_parameter_sensitivity_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()

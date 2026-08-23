"""
experiments/forecast_baselines.py

Additional forecasting baselines beyond the naive-persistence baseline
already used in real_data_validation.py / real_data_validation_rossmann.py,
since "Holt vs. one trivial comparator" is too weak a baseline set on
its own:

  - seasonal_naive_forecast: forecasts each future point as the value
    observed `season_length` periods ago (a standard, non-trivial
    baseline for weekly retail data with a plausible ~4-week cycle).
  - croston_forecast: Croston's method (Croston, 1972), the standard
    classical method for INTERMITTENT demand specifically -- directly
    relevant to the UCI Online Retail dataset, where Holt's method
    performed no better than naive persistence, a result initially
    attributed to sparse/intermittent per-SKU demand. If Croston's
    method does noticeably better than Holt's on that specific
    dataset, it supports the intermittency explanation; if it does
    not, that explanation is weaker than it first appeared and this
    experiment says so.
"""
from __future__ import annotations

from typing import List, Tuple


def seasonal_naive_forecast(train: List[float], horizon: int, season_length: int = 4) -> List[float]:
    """forecast(h) = train[-season_length + ((h-1) % season_length)],
    i.e. repeat the last full season's pattern forward."""
    if len(train) < season_length:
        # not enough history for a seasonal baseline; fall back to plain
        # last-value naive
        return [train[-1]] * horizon
    last_season = train[-season_length:]
    return [last_season[h % season_length] for h in range(horizon)]


def croston_forecast(train: List[float], horizon: int, alpha: float = 0.1) -> Tuple[List[float], dict]:
    """Croston's method (Croston, J.D., 1972. Forecasting and stock
    control for intermittent demands. Operational Research Quarterly,
    23(3), 289-303) -- the standard classical method for intermittent
    demand, separately smoothing (a) the non-zero demand SIZE and (b)
    the INTER-DEMAND INTERVAL (periods between non-zero demands), then
    forecasting the demand rate as size/interval. Widely used for
    spare-parts and slow-moving retail SKUs specifically because plain
    exponential smoothing (like Holt's method) is known to perform
    poorly when a large fraction of periods have zero demand.
    """
    non_zero_indices = [i for i, v in enumerate(train) if v > 0]
    if not non_zero_indices:
        return [0.0] * horizon, {"demand_rate": 0.0, "intervals_seen": 0}

    # initialize with the first non-zero observation
    z = train[non_zero_indices[0]]  # smoothed non-zero demand size
    p = non_zero_indices[0] + 1 if non_zero_indices[0] > 0 else 1  # smoothed inter-demand interval
    last_nonzero_idx = non_zero_indices[0]

    for idx in non_zero_indices[1:]:
        interval = idx - last_nonzero_idx
        z = alpha * train[idx] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        last_nonzero_idx = idx

    rate = z / p if p > 0 else 0.0
    forecast = [round(rate, 4)] * horizon
    return forecast, {"demand_rate": round(rate, 4), "smoothed_size": round(z, 4),
                       "smoothed_interval": round(p, 4), "n_nonzero_periods": len(non_zero_indices),
                       "fraction_zero_periods": round(1 - len(non_zero_indices) / len(train), 4)}


def evaluate_forecast(test: List[float], forecast: List[float]) -> dict:
    """Same MAPE/WAPE metrics used elsewhere in this codebase, for a fair
    apples-to-apples comparison with Holt's method and naive persistence."""
    errors = [abs(test[i] - forecast[i]) for i in range(len(test))]
    mape_terms = [abs((test[i] - forecast[i]) / test[i]) for i in range(len(test)) if test[i] != 0]
    mape = sum(mape_terms) / len(mape_terms) if mape_terms else None
    wape = sum(errors) / sum(abs(t) for t in test) if sum(abs(t) for t in test) > 0 else None
    return {"mape": round(mape, 4) if mape is not None else None,
            "wape": round(wape, 4) if wape is not None else None}

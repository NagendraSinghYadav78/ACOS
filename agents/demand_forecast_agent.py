"""
agents/demand_forecast_agent.py

Holt's linear (double) exponential smoothing for near-term demand.
Closed-form update equations, deterministic given the input series:

    level_t = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
    trend_t = beta * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
    forecast(h) = level_t + h * trend_t

Confidence comes from in-sample MAPE -- lower historical error, higher
confidence.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agents.base import BaseAgent, Decision


def holt_linear_forecast(series: List[float], alpha: float = 0.5, beta: float = 0.3,
                          horizon: int = 4):
    if len(series) < 2:
        raise ValueError("Need at least 2 data points for Holt's linear method")

    level = series[0]
    trend = series[1] - series[0]
    fitted = [level]

    for y in series[1:]:
        last_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        fitted.append(level)

    errors = [abs((series[i] - fitted[i]) / series[i]) for i in range(len(series)) if series[i] != 0]
    mape = sum(errors) / len(errors) if errors else 0.0

    forecasts = [round(level + h * trend, 3) for h in range(1, horizon + 1)]
    return forecasts, mape, level, trend


class DemandForecastAgent(BaseAgent):
    name = "demand_forecast_agent"

    def reason(self, context: Dict[str, Any]) -> Decision:
        sales_history: Dict[str, List[float]] = context.get("sales_history", {})
        horizon = context.get("forecast_horizon", 4)
        if not sales_history:
            return Decision(action="forecast", output={"forecasts": {}},
                             confidence=0.0, rationale="No sales history provided.",
                             warnings=["missing sales_history"])

        results: Dict[str, Any] = {}
        mapes = []
        for product_id, series in sales_history.items():
            try:
                forecasts, mape, level, trend = holt_linear_forecast(series, horizon=horizon)
                results[product_id] = {
                    "forecast": forecasts, "trend": round(trend, 4),
                    "level": round(level, 4), "in_sample_mape": round(mape, 4),
                }
                mapes.append(mape)
            except ValueError as exc:
                results[product_id] = {"error": str(exc)}

        avg_mape = sum(mapes) / len(mapes) if mapes else 1.0
        confidence = max(0.05, min(0.98, 1.0 - avg_mape))

        return Decision(
            action="forecast",
            output={"forecasts": results, "avg_in_sample_mape": round(avg_mape, 4)},
            confidence=confidence,
            rationale=(f"Holt's linear exponential smoothing over {len(sales_history)} "
                       f"product series; avg in-sample MAPE={avg_mape:.3f}."),
        )

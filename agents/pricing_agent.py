"""
agents/pricing_agent.py

Constant-elasticity demand model:

    demand(p) = demand(p0) * (p / p0) ^ elasticity        (elasticity < 0)
    revenue(p) = p * demand(p)
    margin(p)  = (p - unit_cost) * demand(p)

Finds the margin-maximizing price via a bounded grid search (400
points) over the allowed price-change range. margin(p) is quasi-concave
on this range for any elasticity < 0, but the optimum sits at an
interior point only when elasticity < -1; for -1 <= elasticity < 0 it's
monotonic and the optimum is the boundary of the allowed range. Either
way the grid search finds it, up to grid resolution. The allowed range
is also checked later by the governance layer, but pre-checked here too
so the agent's own confidence reflects constraint risk.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict

from agents.base import BaseAgent, Decision

DEFAULT_ELASTICITY = -1.4  # typical for moderately elastic retail goods


class PricingAgent(BaseAgent):
    name = "pricing_agent"

    def _optimize_single(self, base_price: float, base_demand: float, unit_cost: float,
                          elasticity: float, max_change_pct: float):
        low = base_price * (1 - max_change_pct)
        high = base_price * (1 + max_change_pct)
        candidates = np.linspace(low, high, 400)
        demand = base_demand * (candidates / base_price) ** elasticity
        margin = (candidates - unit_cost) * demand
        best_idx = int(np.argmax(margin))
        best_price = float(candidates[best_idx])
        best_margin = float(margin[best_idx])
        best_demand = float(demand[best_idx])
        base_margin = (base_price - unit_cost) * base_demand
        margin_lift_pct = ((best_margin - base_margin) / abs(base_margin)) if base_margin else 0.0
        return best_price, best_demand, best_margin, margin_lift_pct

    def reason(self, context: Dict[str, Any]) -> Decision:
        forecast_output = context.get("forecast_demand", {})
        forecasts = forecast_output.get("forecasts", {}) if forecast_output else {}
        catalog = context.get("catalog", {})
        max_change_pct = context.get("max_price_change_pct", 0.25)

        if not forecasts or not catalog:
            return Decision(action="optimize_price", output={"price_plan": {}},
                             confidence=0.0,
                             rationale="Missing forecast or catalog data.",
                             warnings=["missing inputs"])

        plan: Dict[str, Any] = {}
        max_abs_change = 0.0
        min_margin_pct = 1.0

        for product_id, fdata in forecasts.items():
            if product_id not in catalog or "forecast" not in fdata:
                continue
            info = catalog[product_id]
            base_price = info.get("current_price", 20.0)
            unit_cost = info.get("unit_cost", 12.0)
            elasticity = info.get("elasticity", DEFAULT_ELASTICITY)
            base_demand = max(sum(fdata["forecast"]) / len(fdata["forecast"]), 1.0)

            best_price, best_demand, best_margin, lift = self._optimize_single(
                base_price, base_demand, unit_cost, elasticity, max_change_pct)

            change_pct = (best_price - base_price) / base_price if base_price else 0.0
            margin_pct = (best_price - unit_cost) / best_price if best_price else 0.0
            max_abs_change = max(max_abs_change, abs(change_pct))
            min_margin_pct = min(min_margin_pct, margin_pct)

            plan[product_id] = {
                "base_price": round(base_price, 2),
                "recommended_price": round(best_price, 2),
                "price_change_pct": round(change_pct, 4),
                "projected_demand": round(best_demand, 2),
                "projected_margin": round(best_margin, 2),
                "margin_lift_pct": round(lift, 4),
                "resulting_margin_pct": round(margin_pct, 4),
                "elasticity_used": elasticity,
            }

        confidence = max(0.1, min(0.95, 1.0 - max_abs_change))

        return Decision(
            action="optimize_price",
            output={
                "price_plan": plan,
                "price_change_pct": round(max_abs_change, 4),
                "resulting_margin_pct": round(min_margin_pct, 4),
            },
            confidence=confidence,
            rationale=(f"Margin-maximizing price search under constant-elasticity demand "
                       f"for {len(plan)} SKUs, bounded to +/-{max_change_pct:.0%} of current price."),
        )

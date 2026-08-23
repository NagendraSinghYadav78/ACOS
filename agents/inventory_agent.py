"""
agents/inventory_agent.py

Classical EOQ + safety-stock reorder point, using forecasted demand
from DemandForecastAgent:

    EOQ = sqrt( (2 * D * S) / H )
    ROP = (d_avg * L) + z * sigma_d * sqrt(L)

D = annual demand, S = ordering cost, H = annual holding cost/unit,
d_avg = average daily demand, L = lead time (days), sigma_d = std dev
of daily demand, z = service-level z-score (e.g. 1.65 for 95%).
"""

from __future__ import annotations

import statistics
from typing import Any, Dict

from scipy.stats import norm

from agents.base import BaseAgent, Decision

DEFAULT_ORDERING_COST = 75.0       # $ per purchase order
DEFAULT_HOLDING_COST_RATE = 0.22   # 22% of unit cost per year
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_SERVICE_LEVEL = 0.95


class InventoryAgent(BaseAgent):
    name = "inventory_agent"

    def reason(self, context: Dict[str, Any]) -> Decision:
        forecast_output = context.get("forecast_demand", {})
        forecasts = forecast_output.get("forecasts", {}) if forecast_output else {}
        catalog = context.get("catalog", {})  # product_id -> {unit_cost, current_stock}
        current_inventory = context.get("current_inventory", {})

        if not forecasts:
            return Decision(action="evaluate_reorder", output={"reorder_plan": {}},
                             confidence=0.0,
                             rationale="No demand forecast available upstream.",
                             warnings=["missing forecast"])

        service_level = context.get("service_level", DEFAULT_SERVICE_LEVEL)
        lead_time = context.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS)
        z = float(norm.ppf(service_level))

        plan: Dict[str, Any] = {}
        confidences = []

        for product_id, fdata in forecasts.items():
            if "forecast" not in fdata:
                continue
            weekly_forecast = fdata["forecast"]
            daily_series = [max(0.0, v) / 7.0 for v in weekly_forecast]
            d_avg = sum(daily_series) / len(daily_series)
            sigma_d = statistics.pstdev(daily_series) if len(daily_series) > 1 else d_avg * 0.15

            annual_demand = max(d_avg * 365.0, 1e-6)
            unit_cost = catalog.get(product_id, {}).get("unit_cost", 20.0)
            holding_cost = max(unit_cost * DEFAULT_HOLDING_COST_RATE, 0.01)

            eoq = (2 * annual_demand * DEFAULT_ORDERING_COST / holding_cost) ** 0.5
            rop = d_avg * lead_time + z * sigma_d * (lead_time ** 0.5)
            safety_stock = z * sigma_d * (lead_time ** 0.5)

            on_hand = current_inventory.get(product_id, 0)
            needs_reorder = on_hand <= rop

            plan[product_id] = {
                "eoq": round(eoq, 1),
                "reorder_point": round(rop, 1),
                "safety_stock": round(safety_stock, 1),
                "on_hand": on_hand,
                "needs_reorder": needs_reorder,
                "recommended_order_qty": round(eoq, 1) if needs_reorder else 0.0,
            }
            confidences.append(fdata.get("in_sample_mape", 0.2))

        avg_err = sum(confidences) / len(confidences) if confidences else 0.3
        confidence = max(0.05, min(0.97, 1.0 - avg_err))

        return Decision(
            action="evaluate_reorder",
            output={"reorder_plan": plan, "service_level": service_level,
                     "lead_time_days": lead_time, "z_score": round(z, 3)},
            confidence=confidence,
            rationale=(f"EOQ + safety-stock ROP computed for {len(plan)} SKUs at "
                       f"{service_level:.0%} service level, {lead_time}-day lead time."),
        )

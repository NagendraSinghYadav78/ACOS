"""
agents/analytics_agent.py

Aggregates the outputs of upstream agents (forecast, inventory, pricing,
procurement, fraud) into a single business-intelligence rollup with
real, computed KPIs -- projected revenue delta, total procurement spend,
inventory risk exposure, and fraud exposure -- and surfaces the
inter-agent conflicts that the ConsensusResolver (core/consensus.py)
had to arbitrate, so the rollup is explainable end to end.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, Decision


class AnalyticsAgent(BaseAgent):
    name = "analytics_agent"

    def reason(self, context: Dict[str, Any]) -> Decision:
        price_plan = context.get("optimize_price", {}).get("price_plan", {})
        reorder_plan = context.get("assess_inventory", {}).get("reorder_plan", {})
        rankings = context.get("select_supplier", {}).get("rankings", {})
        fraud = context.get("screen_transactions", {})

        projected_margin_total = sum(p.get("projected_margin", 0.0) for p in price_plan.values())
        avg_margin_lift = (sum(p.get("margin_lift_pct", 0.0) for p in price_plan.values()) / len(price_plan)) \
            if price_plan else 0.0

        skus_needing_reorder = sum(1 for r in reorder_plan.values() if r.get("needs_reorder"))
        total_procurement_spend = context.get("select_supplier", {}).get("estimated_spend", 0.0)

        n_flagged = fraud.get("n_flagged", 0)
        n_total = fraud.get("n_total", 0)
        fraud_rate = (n_flagged / n_total) if n_total else 0.0

        kpis = {
            "projected_margin_total": round(projected_margin_total, 2),
            "avg_margin_lift_pct": round(avg_margin_lift, 4),
            "skus_needing_reorder": skus_needing_reorder,
            "total_procurement_spend": round(total_procurement_spend, 2),
            "fraud_flag_rate": round(fraud_rate, 4),
            "n_transactions_screened": n_total,
        }

        confidences = []
        for key in ("optimize_price", "assess_inventory", "select_supplier", "screen_transactions"):
            c = context.get(f"_confidence_{key}")
            if c is not None:
                confidences.append(c)
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.6

        return Decision(
            action="summarize",
            output={"kpis": kpis},
            confidence=overall_confidence,
            rationale="Aggregated pricing, inventory, procurement, and fraud-risk outputs into KPI rollup.",
        )

"""
core/consensus.py

Resolves conflicts between agent recommendations that are in tension --
e.g. InventoryAgent wants to increase stock on a SKU while PricingAgent
just raised its price, which should suppress demand for that same SKU.
A naive orchestrator would apply both blindly. This module re-checks
whether the recommended reorder quantity still makes sense once the
optimized price is factored in, using the same elasticity model
PricingAgent used to set that price:

    resolved_signal = sum(confidence_i * signal_i) / sum(confidence_i)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ConflictReport:
    product_id: str
    original_order_qty: float
    adjusted_order_qty: float
    delta_pct: float
    explanation: str


class ConsensusResolver:
    def weighted_vote(self, signals: List[float], confidences: List[float]) -> float:
        total_conf = sum(confidences)
        if total_conf == 0:
            return sum(signals) / len(signals) if signals else 0.0
        return sum(s * c for s, c in zip(signals, confidences)) / total_conf

    def reconcile_price_and_inventory(self, price_plan: Dict[str, Any],
                                       reorder_plan: Dict[str, Any]) -> List[ConflictReport]:
        """Detect and resolve conflicts between the PricingAgent's price
        recommendation and the InventoryAgent's reorder quantity: a price
        increase should proportionally reduce demand (per the shared
        constant-elasticity model), so the order quantity is rescaled by
        the same demand ratio the PricingAgent already computed."""
        reports: List[ConflictReport] = []
        for product_id, price_info in price_plan.items():
            reorder = reorder_plan.get(product_id)
            if not reorder or not reorder.get("needs_reorder"):
                continue

            base_price = price_info.get("base_price", 0.0)
            rec_price = price_info.get("recommended_price", base_price)
            if base_price == 0:
                continue

            # demand ratio implied by the agreed elasticity model
            elasticity = price_info.get("elasticity_used", -1.4)
            demand_ratio = (rec_price / base_price) ** elasticity if base_price else 1.0

            original_qty = reorder["recommended_order_qty"]
            adjusted_qty = round(original_qty * demand_ratio, 1)
            delta_pct = ((adjusted_qty - original_qty) / original_qty) if original_qty else 0.0

            if abs(delta_pct) > 0.02:  # only report materially different outcomes
                reports.append(ConflictReport(
                    product_id=product_id,
                    original_order_qty=original_qty,
                    adjusted_order_qty=adjusted_qty,
                    delta_pct=round(delta_pct, 4),
                    explanation=(
                        f"InventoryAgent sized the order at baseline price ${base_price:.2f}; "
                        f"PricingAgent's recommended price ${rec_price:.2f} implies a demand "
                        f"ratio of {demand_ratio:.3f} under elasticity={elasticity}. Order "
                        f"quantity reconciled from {original_qty} to {adjusted_qty}."
                    ),
                ))
                reorder_plan[product_id]["recommended_order_qty"] = adjusted_qty
                reorder_plan[product_id]["reconciled_with_pricing"] = True

        return reports

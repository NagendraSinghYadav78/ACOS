"""
agents/procurement_agent.py

Ranks candidate suppliers using TOPSIS (Technique for Order of
Preference by Similarity to Ideal Solution), a standard multi-criteria
decision analysis method:

  1. Normalize the decision matrix (vector normalization)
  2. Weight each normalized criterion
  3. Determine the ideal-best and ideal-worst vectors
  4. Compute each supplier's Euclidean distance to both
  5. Closeness coefficient C_i = D_worst_i / (D_best_i + D_worst_i)

Criteria: unit_price (cost, minimize), lead_time_days (minimize),
quality_score (benefit, maximize), reliability_score (benefit, maximize).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from agents.base import BaseAgent, Decision
from core.knowledge_graph import KnowledgeGraph

CRITERIA = ["unit_price", "lead_time_days", "quality_score", "reliability_score"]
DIRECTIONS = {"unit_price": "min", "lead_time_days": "min",
              "quality_score": "max", "reliability_score": "max"}
DEFAULT_WEIGHTS = {"unit_price": 0.35, "lead_time_days": 0.20,
                    "quality_score": 0.25, "reliability_score": 0.20}


def topsis(matrix: np.ndarray, weights: List[float], directions: List[str]) -> np.ndarray:
    norm = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)
    weighted = norm * np.array(weights)

    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])
    for j, d in enumerate(directions):
        col = weighted[:, j]
        if d == "max":
            ideal_best[j], ideal_worst[j] = col.max(), col.min()
        else:
            ideal_best[j], ideal_worst[j] = col.min(), col.max()

    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-9
    return dist_worst / denom


class ProcurementAgent(BaseAgent):
    name = "procurement_agent"

    def __init__(self, *args, knowledge_graph: KnowledgeGraph = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kg = knowledge_graph

    def reason(self, context: Dict[str, Any]) -> Decision:
        reorder_plan = context.get("assess_inventory", {}).get("reorder_plan", {})
        suppliers: Dict[str, Dict[str, Any]] = context.get("suppliers", {})
        weights = context.get("procurement_weights", DEFAULT_WEIGHTS)

        products_needing_reorder = [pid for pid, p in reorder_plan.items() if p.get("needs_reorder")]

        if not suppliers or not products_needing_reorder:
            return Decision(action="rank_suppliers", output={"rankings": {}, "estimated_spend": 0.0},
                             confidence=0.6 if not products_needing_reorder else 0.0,
                             rationale="No SKUs require reorder this cycle." if not products_needing_reorder
                                       else "No supplier data provided.",
                             warnings=[] if not products_needing_reorder else ["missing suppliers"])

        rankings: Dict[str, Any] = {}
        total_spend = 0.0

        for product_id in products_needing_reorder:
            candidates = suppliers.get(product_id, {})
            if not candidates:
                continue
            names = list(candidates.keys())
            matrix = np.array([[candidates[s][c] for c in CRITERIA] for s in names], dtype=float)
            weight_vec = [weights[c] for c in CRITERIA]
            directions = [DIRECTIONS[c] for c in CRITERIA]
            scores = topsis(matrix, weight_vec, directions)

            ranked = sorted(zip(names, scores.tolist()), key=lambda x: -x[1])
            best_supplier, best_score = ranked[0]

            order_qty = reorder_plan[product_id]["recommended_order_qty"]
            unit_price = candidates[best_supplier]["unit_price"]
            spend = order_qty * unit_price
            total_spend += spend

            # KG-informed risk check: does the top supplier carry propagated risk?
            kg_risk = None
            if self.kg is not None:
                risk_map = self.kg.supply_chain_risk_propagation(best_supplier)
                kg_risk = round(risk_map.get(best_supplier, 0.0), 3)

            rankings[product_id] = {
                "ranking": [{"supplier": n, "topsis_score": round(s, 4)} for n, s in ranked],
                "recommended_supplier": best_supplier,
                "recommended_supplier_score": round(best_score, 4),
                "order_qty": order_qty,
                "estimated_line_spend": round(spend, 2),
                "supplier_risk_score": kg_risk,
            }

        avg_score = (sum(r["recommended_supplier_score"] for r in rankings.values()) / len(rankings)) \
            if rankings else 0.0

        return Decision(
            action="rank_suppliers",
            output={"rankings": rankings, "estimated_spend": round(total_spend, 2)},
            confidence=max(0.1, min(0.95, avg_score)),
            rationale=(f"TOPSIS multi-criteria ranking (price/lead-time/quality/reliability) "
                       f"across {len(rankings)} SKUs requiring reorder."),
        )

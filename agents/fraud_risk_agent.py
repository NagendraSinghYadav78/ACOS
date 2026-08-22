"""
agents/fraud_risk_agent.py

Screens transactions with a robust z-score (median/MAD-based) on
amount, combined with two rule signals: velocity (too many transactions
in a short window) and geo mismatch (shipping country != billing
country).

risk_score = sigmoid(w1*z_amount + w2*velocity_flag + w3*geo_flag)
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List

from agents.base import BaseAgent, Decision

W_AMOUNT, W_VELOCITY, W_GEO = 0.6, 0.9, 0.7
VELOCITY_WINDOW_SECONDS = 3600
VELOCITY_THRESHOLD = 4


def robust_zscore(values: List[float]) -> List[float]:
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev) or 1e-6
    return [(0.6745 * (v - med)) / mad for v in values]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FraudRiskAgent(BaseAgent):
    name = "fraud_risk_agent"

    def reason(self, context: Dict[str, Any]) -> Decision:
        transactions: List[Dict[str, Any]] = context.get("transactions", [])
        if not transactions:
            return Decision(action="score_transactions", output={"scored": [], "max_risk_score": 0.0},
                             confidence=0.5, rationale="No transactions provided this cycle.",
                             warnings=[])

        amounts = [t["amount"] for t in transactions]
        z_scores = robust_zscore(amounts) if len(set(amounts)) > 1 else [0.0] * len(amounts)

        by_customer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in transactions:
            by_customer[t["customer_id"]].append(t)

        scored = []
        for t, z in zip(transactions, z_scores):
            cust_txns = sorted(by_customer[t["customer_id"]], key=lambda x: x["timestamp"])
            window = [x for x in cust_txns
                      if abs(x["timestamp"] - t["timestamp"]) <= VELOCITY_WINDOW_SECONDS]
            velocity_flag = 1.0 if len(window) >= VELOCITY_THRESHOLD else 0.0
            geo_flag = 1.0 if t.get("shipping_country") != t.get("billing_country") else 0.0

            raw = W_AMOUNT * abs(z) + W_VELOCITY * velocity_flag + W_GEO * geo_flag
            risk = sigmoid(raw - 2.0)  # center the sigmoid so typical txns score low

            scored.append({
                "transaction_id": t["transaction_id"],
                "amount_zscore": round(z, 3),
                "velocity_flag": bool(velocity_flag),
                "geo_mismatch_flag": bool(geo_flag),
                "risk_score": round(risk, 4),
                "flagged": risk >= 0.7,
            })

        max_risk = max(s["risk_score"] for s in scored)
        n_flagged = sum(1 for s in scored if s["flagged"])
        confidence = 0.9 if len(transactions) >= 10 else 0.55

        return Decision(
            action="score_transactions",
            output={"scored": scored, "max_risk_score": round(max_risk, 4),
                     "n_flagged": n_flagged, "n_total": len(transactions)},
            confidence=confidence,
            rationale=(f"Robust z-score + velocity/geo rule ensemble over {len(transactions)} "
                       f"transactions; {n_flagged} flagged at risk>=0.70."),
        )

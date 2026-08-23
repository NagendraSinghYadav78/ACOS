"""
agents/governance_agent.py

Final checkpoint: replays every upstream decision through the
PolicyEngine and produces a per-action ruling (approve/reject/escalate)
plus an overall workflow ruling, each with a rationale pointing at the
rule that fired.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent, Decision
from core.policy import PolicyEngine, Ruling


class GovernanceAgent(BaseAgent):
    name = "governance_agent"

    def __init__(self, *args, policy_engine: PolicyEngine = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_engine = policy_engine or PolicyEngine()

    def reason(self, context: Dict[str, Any]) -> Decision:
        rulings: Dict[str, Any] = {}

        price_plan = context.get("optimize_price", {}).get("price_plan", {})
        for product_id, p in price_plan.items():
            result = self.policy_engine.evaluate({
                "action": "optimize_price",
                "price_change_pct": p.get("price_change_pct", 0.0),
                "resulting_margin_pct": p.get("resulting_margin_pct", 1.0),
                "confidence": context.get("_confidence_optimize_price"),
            })
            rulings[f"pricing:{product_id}"] = {
                "ruling": result.ruling.value, "rationale": result.rationale,
                "fired_rules": result.fired_rules,
            }

        fraud = context.get("screen_transactions", {})
        if fraud:
            result = self.policy_engine.evaluate({
                "action": "score_transactions",
                "max_risk_score": fraud.get("max_risk_score", 0.0),
            })
            rulings["fraud_screening"] = {
                "ruling": result.ruling.value, "rationale": result.rationale,
                "fired_rules": result.fired_rules,
            }

        procurement = context.get("select_supplier", {})
        if procurement:
            result = self.policy_engine.evaluate({
                "action": "rank_suppliers",
                "estimated_spend": procurement.get("estimated_spend", 0.0),
            })
            rulings["procurement"] = {
                "ruling": result.ruling.value, "rationale": result.rationale,
                "fired_rules": result.fired_rules,
            }

        # record each ruling as an explicit governed decision for audit
        for key, r in rulings.items():
            self.long_term_memory.record_decision(
                agent=self.name, action=key, rationale=r["rationale"],
                approved=(r["ruling"] == Ruling.APPROVE.value),
                workflow_id=context.get("_workflow_id"),
            )

        n_reject = sum(1 for r in rulings.values() if r["ruling"] == Ruling.REJECT.value)
        n_escalate = sum(1 for r in rulings.values() if r["ruling"] == Ruling.ESCALATE.value)
        if n_reject > 0:
            overall = Ruling.REJECT.value
        elif n_escalate > 0:
            overall = Ruling.ESCALATE.value
        else:
            overall = Ruling.APPROVE.value

        return Decision(
            action="review",
            output={"rulings": rulings, "overall_ruling": overall,
                     "n_rejected": n_reject, "n_escalated": n_escalate,
                     "n_reviewed": len(rulings)},
            confidence=0.95,
            rationale=(f"Governance review of {len(rulings)} agent decisions: "
                       f"{n_reject} rejected, {n_escalate} escalated, "
                       f"{len(rulings) - n_reject - n_escalate} approved."),
        )

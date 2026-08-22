"""
core/policy.py

Checks proposed agent actions against a set of policy rules and
returns a ruling (approve/reject/escalate) with a rationale naming
whichever rule fired. Forward-chaining over an ordered rule list; each
rule is a plain function, so they're each independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Ruling(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass
class PolicyRule:
    name: str
    description: str
    # predicate returns None if rule does not apply, else a Ruling
    check: Callable[[Dict[str, Any]], Optional[Ruling]]
    severity: int = 1  # higher severity rules override lower ones


@dataclass
class PolicyResult:
    ruling: Ruling
    rationale: str
    fired_rules: List[str]
    confidence: float


class PolicyEngine:
    def __init__(self):
        self.rules: List[PolicyRule] = []
        self._register_default_rules()

    def register_rule(self, rule: PolicyRule) -> None:
        self.rules.append(rule)

    def _register_default_rules(self) -> None:
        def price_change_bounds(ctx: Dict[str, Any]) -> Optional[Ruling]:
            if ctx.get("action") != "optimize_price":
                return None
            pct = ctx.get("price_change_pct", 0.0)
            if abs(pct) > 0.35:
                return Ruling.REJECT
            if abs(pct) > 0.20:
                return Ruling.ESCALATE
            return None

        def fraud_risk_gate(ctx: Dict[str, Any]) -> Optional[Ruling]:
            if ctx.get("action") != "score_transactions":
                return None
            max_risk = ctx.get("max_risk_score", 0.0)
            if max_risk >= 0.85:
                return Ruling.ESCALATE
            return None

        def negative_margin_block(ctx: Dict[str, Any]) -> Optional[Ruling]:
            if ctx.get("action") != "optimize_price":
                return None
            if ctx.get("resulting_margin_pct", 1.0) < 0.0:
                return Ruling.REJECT
            return None

        def large_procurement_spend(ctx: Dict[str, Any]) -> Optional[Ruling]:
            if ctx.get("action") != "rank_suppliers":
                return None
            if ctx.get("estimated_spend", 0.0) > 250_000:
                return Ruling.ESCALATE
            return None

        def low_confidence_escalation(ctx: Dict[str, Any]) -> Optional[Ruling]:
            if ctx.get("confidence") is not None and ctx["confidence"] < 0.4:
                return Ruling.ESCALATE
            return None

        self.register_rule(PolicyRule(
            "price_change_bounds",
            "Blocks price changes >35%, escalates changes >20%",
            price_change_bounds, severity=3))
        self.register_rule(PolicyRule(
            "negative_margin_block",
            "Blocks any pricing decision that results in a negative margin",
            negative_margin_block, severity=5))
        self.register_rule(PolicyRule(
            "fraud_risk_gate",
            "Escalates when any transaction risk score >= 0.85",
            fraud_risk_gate, severity=4))
        self.register_rule(PolicyRule(
            "large_procurement_spend",
            "Escalates procurement decisions with estimated spend > $250,000",
            large_procurement_spend, severity=3))
        self.register_rule(PolicyRule(
            "low_confidence_escalation",
            "Escalates any decision made with agent confidence < 0.4",
            low_confidence_escalation, severity=2))

    def evaluate(self, context: Dict[str, Any]) -> PolicyResult:
        fired: List[tuple] = []
        for rule in self.rules:
            outcome = rule.check(context)
            if outcome is not None:
                fired.append((rule, outcome))

        if not fired:
            return PolicyResult(
                ruling=Ruling.APPROVE,
                rationale="No policy rule was triggered; action satisfies default governance constraints.",
                fired_rules=[],
                confidence=1.0,
            )

        fired.sort(key=lambda x: -x[0].severity)
        winning_rule, winning_ruling = fired[0]
        rationale = "; ".join(f"[{r.name}] {r.description}" for r, _ in fired)
        return PolicyResult(
            ruling=winning_ruling,
            rationale=rationale,
            fired_rules=[r.name for r, _ in fired],
            confidence=0.9 if len(fired) == 1 else 0.75,
        )

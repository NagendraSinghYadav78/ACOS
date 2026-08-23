"""
agents/base.py

Common loop every agent follows:

    perceive -> reason -> act -> reflect

- perceive: pull context from SharedMemory / VectorMemory / KG
- reason: agent-specific logic (subclasses implement this), returns a
  Decision with a confidence score
- act: emit an event, persist the decision to LongTermMemory
- reflect: sanity-checks the decision against simple bounds, can lower
  confidence or flag for re-planning

Subclasses implement reason(self, context) -> Decision.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory

logger = logging.getLogger("acos.agent")


@dataclass
class Decision:
    action: str
    output: Dict[str, Any]
    confidence: float
    rationale: str
    warnings: List[str] = field(default_factory=list)


class BaseAgent:
    name: str = "base_agent"

    def __init__(self, event_bus: EventBus, shared_memory: SharedMemory,
                 long_term_memory: LongTermMemory):
        self.event_bus = event_bus
        self.shared_memory = shared_memory
        self.long_term_memory = long_term_memory

    # ---- cognitive loop -------------------------------------------------
    def perceive(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge task params with anything relevant already on the
        shared-memory blackboard (e.g. upstream task outputs)."""
        merged = dict(self.shared_memory.all())
        merged.update(context)
        return merged

    def reason(self, context: Dict[str, Any]) -> Decision:  # pragma: no cover - overridden
        raise NotImplementedError

    def reflect(self, decision: Decision) -> Decision:
        """Simple, real self-evaluation: penalize confidence for
        decisions with extreme or inconsistent outputs, and record the
        rationale for why confidence was adjusted."""
        if decision.confidence < 0 or decision.confidence > 1:
            decision.warnings.append("confidence out of [0,1] bounds; clipped")
            decision.confidence = max(0.0, min(1.0, decision.confidence))
        if decision.confidence < 0.3:
            decision.warnings.append("low-confidence decision flagged for governance review")
        return decision

    def act(self, task_id: str, decision: Decision, workflow_id: Optional[str] = None) -> Decision:
        self.shared_memory.set(task_id, decision.output)
        self.long_term_memory.record_episode(
            agent=self.name, kind="decision",
            content={"task_id": task_id, "action": decision.action,
                     "output": decision.output, "warnings": decision.warnings},
            workflow_id=workflow_id, confidence=decision.confidence,
        )
        self.event_bus.publish(
            topic=f"agent.{self.name}.decision",
            payload={"task_id": task_id, "action": decision.action,
                     "output": decision.output, "confidence": decision.confidence,
                     "rationale": decision.rationale},
            source=self.name, correlation_id=workflow_id,
        )
        return decision

    def run(self, task_id: str, context: Dict[str, Any],
            workflow_id: Optional[str] = None) -> Decision:
        t0 = time.perf_counter()
        ctx = self.perceive(context)
        decision = self.reason(ctx)
        decision = self.reflect(decision)
        decision = self.act(task_id, decision, workflow_id=workflow_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        decision.output["_latency_ms"] = round(elapsed_ms, 3)
        logger.info("%s completed task=%s confidence=%.2f latency_ms=%.2f",
                    self.name, task_id, decision.confidence, elapsed_ms)
        return decision

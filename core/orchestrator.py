"""
core/orchestrator.py

Ties the Planner (task DAG + scheduling), the agent registry,
SharedMemory, LongTermMemory, EventBus, ConsensusResolver, and the
governance checkpoint together into one runnable workflow. Dispatches
tasks to agent instances in dependency order and collects the results.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from core.consensus import ConsensusResolver
from core.event_bus import EventBus
from core.knowledge_graph import KnowledgeGraph
from core.memory import LongTermMemory, SharedMemory
from core.planner import Planner, Task


class WorkflowResult:
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.task_outputs: Dict[str, Any] = {}
        self.task_confidences: Dict[str, float] = {}
        self.conflicts: List[Any] = []
        self.wall_clock_seconds: float = 0.0
        self.governance: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "task_outputs": self.task_outputs,
            "task_confidences": self.task_confidences,
            "conflicts": [c.__dict__ for c in self.conflicts],
            "wall_clock_seconds": self.wall_clock_seconds,
            "governance": self.governance,
        }


class Orchestrator:
    def __init__(self, agents: Dict[str, BaseAgent], event_bus: EventBus,
                 shared_memory: SharedMemory, long_term_memory: LongTermMemory,
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 planner: Optional[Planner] = None,
                 consensus: Optional[ConsensusResolver] = None):
        self.agents = agents
        self.event_bus = event_bus
        self.shared_memory = shared_memory
        self.long_term_memory = long_term_memory
        self.kg = knowledge_graph
        self.planner = planner or Planner()
        self.consensus = consensus or ConsensusResolver()

    def run_workflow(self, goal: str, inputs: Dict[str, Any]) -> WorkflowResult:
        workflow_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        self.shared_memory.clear()
        for k, v in inputs.items():
            self.shared_memory.set(k, v)
        self.shared_memory.set("_workflow_id", workflow_id)

        tasks = self.planner.build_plan(goal)
        waves = self.planner.schedule(tasks)

        result = WorkflowResult(workflow_id)

        self.event_bus.publish("workflow.started", {"goal": goal, "waves": len(waves)},
                                source="orchestrator", correlation_id=workflow_id)

        for wave in waves:
            for task in wave:
                agent = self.agents.get(task.agent)
                if agent is None:
                    raise RuntimeError(f"No agent registered for '{task.agent}'")
                context = dict(task.params)
                decision = agent.run(task.task_id, context, workflow_id=workflow_id)
                result.task_outputs[task.task_id] = decision.output
                result.task_confidences[task.task_id] = decision.confidence
                self.shared_memory.set(f"_confidence_{task.task_id}", decision.confidence)

        # Consensus / conflict resolution between pricing and inventory
        price_plan = result.task_outputs.get("optimize_price", {}).get("price_plan", {})
        reorder_plan = result.task_outputs.get("assess_inventory", {}).get("reorder_plan", {})
        if price_plan and reorder_plan:
            conflicts = self.consensus.reconcile_price_and_inventory(price_plan, reorder_plan)
            result.conflicts = conflicts
            if conflicts:
                self.long_term_memory.record_episode(
                    agent="orchestrator", kind="consensus_resolution",
                    content={"n_conflicts": len(conflicts),
                             "conflicts": [c.__dict__ for c in conflicts]},
                    workflow_id=workflow_id,
                )

        result.governance = result.task_outputs.get("governance_review", {})
        result.wall_clock_seconds = time.perf_counter() - t0

        self.event_bus.publish("workflow.completed",
                                {"workflow_id": workflow_id,
                                 "wall_clock_seconds": result.wall_clock_seconds,
                                 "overall_ruling": result.governance.get("overall_ruling")},
                                source="orchestrator", correlation_id=workflow_id)
        return result

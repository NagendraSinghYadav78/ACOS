"""
core/planner.py

Breaks a goal down into a task DAG, then schedules it via topological
sort (Kahn's algorithm). Detects cycles and raises rather than silently
producing a bad plan; groups independent tasks into waves so the
orchestrator can run them in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx


@dataclass
class Task:
    task_id: str
    agent: str
    action: str
    depends_on: List[str] = field(default_factory=list)
    params: Dict = field(default_factory=dict)


class PlanningError(Exception):
    pass


class Planner:
    """Goal -> Task DAG -> ordered execution waves."""

    # Static goal templates: an inspectable rule table, so task
    # decomposition stays deterministic and auditable.
    GOAL_TEMPLATES: Dict[str, List[Task]] = {
        "quarterly_pricing_and_inventory_review": [
            Task("forecast_demand", "demand_forecast_agent", "forecast", []),
            Task("assess_inventory", "inventory_agent", "evaluate_reorder", ["forecast_demand"]),
            Task("optimize_price", "pricing_agent", "optimize_price", ["forecast_demand"]),
            Task("select_supplier", "procurement_agent", "rank_suppliers", ["assess_inventory"]),
            Task("screen_transactions", "fraud_risk_agent", "score_transactions", []),
            Task("analytics_rollup", "analytics_agent", "summarize", [
                "assess_inventory", "optimize_price", "select_supplier", "screen_transactions"
            ]),
            Task("governance_review", "governance_agent", "review", ["analytics_rollup"]),
        ],
    }

    def build_plan(self, goal: str) -> List[Task]:
        if goal not in self.GOAL_TEMPLATES:
            raise PlanningError(f"Unknown goal: {goal}")
        return list(self.GOAL_TEMPLATES[goal])

    def schedule(self, tasks: List[Task]) -> List[List[Task]]:
        """Return execution waves: list of lists of tasks that can run
        concurrently, respecting dependencies (Kahn's algorithm on a DAG)."""
        g = nx.DiGraph()
        by_id = {t.task_id: t for t in tasks}
        for t in tasks:
            g.add_node(t.task_id)
            for dep in t.depends_on:
                if dep not in by_id:
                    raise PlanningError(f"Task {t.task_id} depends on unknown task {dep}")
                g.add_edge(dep, t.task_id)

        if not nx.is_directed_acyclic_graph(g):
            cycles = list(nx.simple_cycles(g))
            raise PlanningError(f"Cyclic plan detected: {cycles}")

        waves: List[List[Task]] = []
        remaining = set(g.nodes)
        indegree = {n: g.in_degree(n) for n in g.nodes}

        while remaining:
            ready = [n for n in remaining if indegree[n] == 0]
            if not ready:
                raise PlanningError("Deadlock in plan scheduling (should be unreachable for a DAG)")
            waves.append([by_id[n] for n in ready])
            for n in ready:
                remaining.remove(n)
                for succ in g.successors(n):
                    indegree[succ] -= 1
        return waves

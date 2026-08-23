"""
main.py

Wires together every ACOS layer (event bus, shared memory, long-term
memory, vector memory, knowledge graph, planner, consensus resolver,
policy engine, and all seven specialized agents) and runs the
"quarterly_pricing_and_inventory_review" workflow end to end against
the synthetic dataset, printing the actual computed results.

Usage:
    python3 main.py
"""

from __future__ import annotations

import json
import sys

from agents.analytics_agent import AnalyticsAgent
from agents.demand_forecast_agent import DemandForecastAgent
from agents.fraud_risk_agent import FraudRiskAgent
from agents.governance_agent import GovernanceAgent
from agents.inventory_agent import InventoryAgent
from agents.pricing_agent import PricingAgent
from agents.procurement_agent import ProcurementAgent
from core.consensus import ConsensusResolver
from core.event_bus import EventBus
from core.knowledge_graph import KnowledgeGraph
from core.memory import LongTermMemory, SharedMemory
from core.orchestrator import Orchestrator
from core.planner import Planner
from core.policy import PolicyEngine
from core.vector_memory import VectorMemory
from data.synthetic_data import build_full_dataset


def build_knowledge_graph(dataset: dict) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for pid, info in dataset["catalog"].items():
        kg.add_node(pid, "Product", category=info["category"])
    for pid, supplier_map in dataset["suppliers"].items():
        for sname, info in supplier_map.items():
            if sname not in kg.g:
                kg.add_node(sname, "Supplier")
            kg.add_edge(sname, pid, "SUPPLIES", **info)
    return kg


def build_system(db_path: str = "acos_memory.db", seed: int = 42):
    event_bus = EventBus()
    shared_memory = SharedMemory()
    long_term_memory = LongTermMemory(db_path=db_path)
    vector_memory = VectorMemory(dim=32)
    policy_engine = PolicyEngine()
    consensus = ConsensusResolver()
    planner = Planner()

    dataset = build_full_dataset(seed=seed)
    kg = build_knowledge_graph(dataset)

    agent_kwargs = dict(event_bus=event_bus, shared_memory=shared_memory,
                         long_term_memory=long_term_memory)

    agents = {
        "demand_forecast_agent": DemandForecastAgent(**agent_kwargs),
        "inventory_agent": InventoryAgent(**agent_kwargs),
        "pricing_agent": PricingAgent(**agent_kwargs),
        "procurement_agent": ProcurementAgent(**agent_kwargs, knowledge_graph=kg),
        "fraud_risk_agent": FraudRiskAgent(**agent_kwargs),
        "analytics_agent": AnalyticsAgent(**agent_kwargs),
        "governance_agent": GovernanceAgent(**agent_kwargs, policy_engine=policy_engine),
    }

    orchestrator = Orchestrator(agents=agents, event_bus=event_bus,
                                 shared_memory=shared_memory,
                                 long_term_memory=long_term_memory,
                                 knowledge_graph=kg, planner=planner,
                                 consensus=consensus)

    # seed vector memory with policy/procedure documents for semantic retrieval
    for doc in [
        "Pricing policy: price changes above 35 percent require full committee approval.",
        "Inventory policy: safety stock is sized at the 95 percent service level by default.",
        "Fraud policy: any transaction risk score above 0.85 must be escalated to the fraud team.",
        "Procurement policy: any single order over $250,000 requires executive sign-off.",
    ]:
        vector_memory.add(doc, metadata={"type": "policy_doc"})

    return orchestrator, dataset, vector_memory, long_term_memory, kg


def run_demo():
    orchestrator, dataset, vector_memory, long_term_memory, kg = build_system()

    inputs = {
        "sales_history": dataset["sales_history"],
        "catalog": dataset["catalog"],
        "current_inventory": dataset["current_inventory"],
        "suppliers": dataset["suppliers"],
        "transactions": dataset["transactions"],
        "forecast_horizon": 4,
        "service_level": 0.95,
        "lead_time_days": 7,
        "max_price_change_pct": 0.25,
    }

    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)

    print("=" * 70)
    print(f"ACOS workflow completed: {result.workflow_id}")
    print(f"Wall-clock time: {result.wall_clock_seconds:.4f}s")
    print("=" * 70)

    print("\n--- Task confidences ---")
    for task_id, conf in result.task_confidences.items():
        print(f"  {task_id:25s} confidence={conf:.3f}")

    print("\n--- Analytics rollup (KPIs) ---")
    kpis = result.task_outputs.get("analytics_rollup", {}).get("kpis", {})
    print(json.dumps(kpis, indent=2))

    print(f"\n--- Consensus conflicts resolved: {len(result.conflicts)} ---")
    for c in result.conflicts[:5]:
        print(f"  {c.product_id}: {c.original_order_qty} -> {c.adjusted_order_qty} "
              f"({c.delta_pct:+.1%})")

    print("\n--- Governance ruling ---")
    gov = result.governance
    print(f"  Overall: {gov.get('overall_ruling')}  "
          f"(reviewed={gov.get('n_reviewed')}, "
          f"rejected={gov.get('n_rejected')}, escalated={gov.get('n_escalated')})")

    print("\n--- KG stats ---")
    print(f"  {kg.stats()}")

    print("\n--- Long-term memory stats ---")
    print(f"  {long_term_memory.stats()}")

    print("\n--- Vector memory semantic search demo ---")
    query = "what happens if a price change is too large"
    hits = vector_memory.search(query, k=2)
    for h in hits:
        print(f"  score={h['score']:.3f} :: {h['text']}")

    return result


if __name__ == "__main__":
    run_demo()
    sys.exit(0)

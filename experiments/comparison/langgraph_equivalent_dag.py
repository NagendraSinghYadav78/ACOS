"""
experiments/comparison/langgraph_equivalent_dag.py

Builds and executes a LangGraph StateGraph that mirrors ACOS's
quarterly_pricing_and_inventory_review task DAG exactly (same 7 nodes,
same dependency edges, see core/planner.py), with every node a no-op
Python function so the measurement isolates orchestration overhead
from agent computation (comparable to ACOS's own E7 in
experiments/run_experiments.py).

Notably, the naive version of this script (writing each node's output
to a shared "data" dict key) raised langgraph.errors.InvalidUpdateError
because two parallel branches (assess_inventory and optimize_price,
both children of forecast_demand) wrote to the same state key in the
same super-step. This version fixes that with an explicit
Annotated[..., reducer] merge function -- a real architectural
difference from ACOS: LangGraph requires this kind of explicit
state-merge handling for concurrent writes, whereas ACOS's
SharedMemory blackboard avoids the issue by construction (each
task writes to a distinct key).

Run: python3 experiments/comparison/langgraph_equivalent_dag.py
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Annotated
import operator, time

def merge_dicts(a, b):
    merged = dict(a)
    merged.update(b)
    return merged

class WFState(TypedDict):
    data: Annotated[Dict[str, Any], merge_dicts]

def make_node(key):
    def node(state):
        return {"data": {key: {"ok": True}}}
    return node

g = StateGraph(WFState)
for key in ["forecast_demand", "assess_inventory", "optimize_price", "select_supplier",
            "screen_transactions", "analytics_rollup", "governance_review"]:
    g.add_node(key, make_node(key))

g.set_entry_point("forecast_demand")
g.add_edge("forecast_demand", "assess_inventory")
g.add_edge("forecast_demand", "optimize_price")
g.add_edge("assess_inventory", "select_supplier")
g.set_entry_point("screen_transactions")
g.add_edge("select_supplier", "analytics_rollup")
g.add_edge("optimize_price", "analytics_rollup")
g.add_edge("screen_transactions", "analytics_rollup")
g.add_edge("analytics_rollup", "governance_review")
g.add_edge("governance_review", END)

app = g.compile()
t0 = time.perf_counter()
result = app.invoke({"data": {}})
elapsed = time.perf_counter() - t0
print("LangGraph result keys:", list(result["data"].keys()))
print(f"LangGraph latency: {elapsed*1000:.3f} ms")

# warm repeated timing
import statistics
times = []
for _ in range(10):
    t0 = time.perf_counter()
    app.invoke({"data": {}})
    times.append((time.perf_counter() - t0) * 1000)
print(f"LangGraph repeated (n=10): mean={statistics.mean(times):.3f}ms std={statistics.pstdev(times):.3f}ms")

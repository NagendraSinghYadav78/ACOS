"""
experiments/comparison/larger_dag_benchmark.py

The 7-node DAG comparison in langgraph_equivalent_dag.py is too small
to support any general claim about scheduler efficiency. This script
builds a larger, structurally comparable DAG (50 nodes, 5 layers of 10
parallel nodes each feeding into the next layer) in both ACOS's Planner
and LangGraph's StateGraph, and measures scheduling-only overhead
(no-op nodes) at this larger size.

This does NOT resolve the deeper limitation that real per-agent
computation vs. no-op nodes are not the same workload; it only extends
the *scheduling-only* comparison to a size where a scaling observation
is slightly more defensible than at 7 nodes, while still being far
short of a rigorous scaling study (which would need several sizes
across orders of magnitude).
"""
from __future__ import annotations

import sys
import time
import statistics
from pathlib import Path
from typing import TypedDict, Dict, Any, Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

N_LAYERS = 5
NODES_PER_LAYER = 10
N_NODES = N_LAYERS * NODES_PER_LAYER  # 50


def _merge_dicts(a, b):
    merged = dict(a)
    merged.update(b)
    return merged


def build_acos_tasks():
    from core.planner import Task
    tasks = []
    for layer in range(N_LAYERS):
        for i in range(NODES_PER_LAYER):
            tid = f"L{layer}_N{i}"
            deps = [f"L{layer-1}_N{j}" for j in range(NODES_PER_LAYER)] if layer > 0 else []
            tasks.append(Task(tid, "noop_agent", "noop", deps))
    return tasks


def benchmark_acos(n_runs=20):
    from core.planner import Planner
    planner = Planner()
    tasks = build_acos_tasks()
    planner.GOAL_TEMPLATES["large_benchmark_dag"] = tasks

    times = []
    waves = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        built = planner.build_plan("large_benchmark_dag")
        waves = planner.schedule(built)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "n_nodes": N_NODES, "n_waves": len(waves),
        "mean_ms": round(statistics.mean(times), 4),
        "stdev_ms": round(statistics.pstdev(times), 4),
    }


def benchmark_langgraph(n_runs=10):
    from langgraph.graph import StateGraph, END

    class WFState(TypedDict):
        data: Annotated[Dict[str, Any], _merge_dicts]

    def make_node(key):
        def node(state):
            return {"data": {key: {"ok": True}}}
        return node

    g = StateGraph(WFState)
    for layer in range(N_LAYERS):
        for i in range(NODES_PER_LAYER):
            tid = f"L{layer}_N{i}"
            g.add_node(tid, make_node(tid))

    for layer in range(N_LAYERS):
        for i in range(NODES_PER_LAYER):
            tid = f"L{layer}_N{i}"
            if layer == 0:
                g.set_entry_point(tid)
            else:
                for j in range(NODES_PER_LAYER):
                    g.add_edge(f"L{layer-1}_N{j}", tid)
    for i in range(NODES_PER_LAYER):
        g.add_edge(f"L{N_LAYERS-1}_N{i}", END)

    app = g.compile()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        app.invoke({"data": {}})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "n_nodes": N_NODES,
        "mean_ms": round(statistics.mean(times), 4),
        "stdev_ms": round(statistics.pstdev(times), 4),
    }


def main():
    print(f"Benchmarking {N_NODES}-node DAG ({N_LAYERS} layers x {NODES_PER_LAYER} nodes/layer)...")
    acos_result = benchmark_acos()
    print("ACOS Planner:", acos_result)

    try:
        lg_result = benchmark_langgraph()
        print("LangGraph StateGraph:", lg_result)
    except Exception as e:
        lg_result = {"error": str(e)}
        print("LangGraph benchmark failed:", e)

    import json
    out = {"n_nodes": N_NODES, "acos": acos_result, "langgraph": lg_result}
    out_path = Path(__file__).resolve().parent / "larger_dag_benchmark_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

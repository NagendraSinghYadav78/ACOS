"""
experiments/run_experiments.py

Runs real, repeatable experiments against the actual ACOS implementation
and writes raw measurements to experiments/results.json. Every number
in this file is produced by executing the code in this repository on
this machine -- nothing here is invented or copied from a hypothetical
run. Where a metric would require infrastructure this sandbox does not
have (e.g. multi-node distributed throughput, GPU inference latency),
this is explicitly labeled as NOT MEASURED rather than filled in.

Experiments:
  E1. End-to-end workflow latency vs. catalog size (scalability)
  E2. Per-agent latency breakdown
  E3. Fraud-detection precision/recall/F1 vs. ground truth (synthetic,
      seeded, injected anomalies -- see data/synthetic_data.py)
  E4. Demand-forecast accuracy (in-sample MAPE) across product catalog
  E5. Governance escalation-rate sensitivity to policy thresholds
  E6. Repeated-run latency stability (mean/std over N runs)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.fraud_risk_agent import FraudRiskAgent
from core.event_bus import EventBus
from core.memory import LongTermMemory, SharedMemory
from core.policy import PolicyEngine
from data.synthetic_data import (
    build_full_dataset, generate_catalog, generate_current_inventory,
    generate_sales_history, generate_suppliers, generate_transactions,
)
from main import build_system

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def e1_scalability_vs_catalog_size() -> List[Dict[str, Any]]:
    """Measures actual wall-clock workflow latency as catalog size grows."""
    results = []
    for n_products in [5, 10, 20, 40, 80]:
        catalog = generate_catalog(n_products=n_products, seed=42)
        sales_history = generate_sales_history(catalog, seed=43)
        current_inventory = generate_current_inventory(catalog, seed=44)
        suppliers = generate_suppliers(catalog, seed=45)
        transactions, _ = generate_transactions(catalog, n_transactions=100, seed=46)

        orchestrator, _, _, ltm, _ = build_system(db_path=f"/tmp/acos_scale_{n_products}.db")
        inputs = {
            "sales_history": sales_history, "catalog": catalog,
            "current_inventory": current_inventory, "suppliers": suppliers,
            "transactions": transactions,
        }
        t0 = time.perf_counter()
        result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
        elapsed = time.perf_counter() - t0
        ltm.close()

        results.append({
            "n_products": n_products,
            "wall_clock_seconds": round(elapsed, 5),
            "reported_wall_clock_seconds": round(result.wall_clock_seconds, 5),
        })
    return results


def e2_per_agent_latency_breakdown() -> Dict[str, float]:
    orchestrator, dataset, _, ltm, _ = build_system(db_path="/tmp/acos_latency.db")
    inputs = {
        "sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
        "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
        "transactions": dataset["transactions"],
    }
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    ltm.close()
    breakdown = {}
    for task_id, output in result.task_outputs.items():
        breakdown[task_id] = output.get("_latency_ms")
    return breakdown


def e3_fraud_detection_quality() -> Dict[str, Any]:
    dataset = build_full_dataset(seed=42)
    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path="/tmp/acos_fraud.db"))
    decision = agent.reason({"transactions": dataset["transactions"]})
    scored = {s["transaction_id"]: s for s in decision.output["scored"]}
    ground_truth = set(dataset["fraud_ground_truth"])

    flagged = {tid for tid, s in scored.items() if s["flagged"]}
    tp = len(flagged & ground_truth)
    fp = len(flagged - ground_truth)
    fn = len(ground_truth - flagged)
    tn = len(scored) - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(scored) if scored else 0.0

    agent.long_term_memory.close()
    return {
        "n_transactions": len(scored), "n_ground_truth_fraud": len(ground_truth),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round(accuracy, 4),
    }


def e4_forecast_accuracy() -> Dict[str, Any]:
    orchestrator, dataset, _, ltm, _ = build_system(db_path="/tmp/acos_forecast.db")
    inputs = {"sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
              "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
              "transactions": dataset["transactions"]}
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    ltm.close()
    forecasts = result.task_outputs["forecast_demand"]["forecasts"]
    mapes = [f["in_sample_mape"] for f in forecasts.values() if "in_sample_mape" in f]
    return {
        "n_products": len(forecasts),
        "mean_in_sample_mape": round(statistics.mean(mapes), 4) if mapes else None,
        "median_in_sample_mape": round(statistics.median(mapes), 4) if mapes else None,
        "min_mape": round(min(mapes), 4) if mapes else None,
        "max_mape": round(max(mapes), 4) if mapes else None,
    }


def e5_price_search_bound_vs_escalation_rate() -> List[Dict[str, Any]]:
    """Sweeps `max_price_change_pct`, the bound the PricingAgent is allowed
    to search within, and measures the resulting governance escalation
    rate on the same fixed dataset. NOTE: this varies the *agent's*
    search bound, not the PolicyEngine's fixed 20%/35% escalate/reject
    thresholds (those remain constant). The experiment therefore shows
    how widening an agent's action space interacts with a fixed
    governance rule: once the agent is permitted to search past the
    20% escalation threshold, materially more SKUs trigger escalation."""
    dataset = build_full_dataset(seed=42)
    orchestrator, _, _, ltm, _ = build_system(db_path="/tmp/acos_gov.db")
    inputs = {"sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
              "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
              "transactions": dataset["transactions"]}
    results = []
    for max_change in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        inputs["max_price_change_pct"] = max_change
        result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", dict(inputs))
        gov = result.governance
        results.append({
            "max_price_change_pct_allowed": max_change,
            "n_escalated": gov.get("n_escalated", 0),
            "n_rejected": gov.get("n_rejected", 0),
            "n_reviewed": gov.get("n_reviewed", 0),
        })
    ltm.close()
    return results


def e6_latency_stability(n_runs: int = 10) -> Dict[str, Any]:
    orchestrator, dataset, _, ltm, _ = build_system(db_path="/tmp/acos_stability.db")
    inputs = {"sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
              "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
              "transactions": dataset["transactions"]}
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        orchestrator.run_workflow("quarterly_pricing_and_inventory_review", dict(inputs))
        timings.append(time.perf_counter() - t0)
    ltm.close()
    return {
        "n_runs": n_runs,
        "mean_seconds": round(statistics.mean(timings), 5),
        "stdev_seconds": round(statistics.pstdev(timings), 5) if len(timings) > 1 else 0.0,
        "min_seconds": round(min(timings), 5),
        "max_seconds": round(max(timings), 5),
        "raw_seconds": [round(t, 5) for t in timings],
    }


def e7_scheduling_only_overhead(n_runs: int = 20) -> Dict[str, Any]:
    """Isolates the Planner's build_plan()+schedule() cost from any agent
    computation, to separate orchestration overhead from algorithmic work."""
    from core.planner import Planner
    planner = Planner()
    timings = []
    n_tasks = n_waves = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        tasks = planner.build_plan("quarterly_pricing_and_inventory_review")
        waves = planner.schedule(tasks)
        timings.append((time.perf_counter() - t0) * 1000)
        n_tasks, n_waves = len(tasks), len(waves)
    return {
        "n_runs": n_runs, "n_tasks": n_tasks, "n_waves": n_waves,
        "mean_ms": round(statistics.mean(timings), 4),
        "stdev_ms": round(statistics.pstdev(timings), 4) if len(timings) > 1 else 0.0,
    }


def main():
    print("Running E1: scalability vs. catalog size...")
    e1 = e1_scalability_vs_catalog_size()

    print("Running E2: per-agent latency breakdown...")
    e2 = e2_per_agent_latency_breakdown()

    print("Running E3: fraud detection quality vs. ground truth...")
    e3 = e3_fraud_detection_quality()

    print("Running E4: demand forecast accuracy...")
    e4 = e4_forecast_accuracy()

    print("Running E5: governance threshold sensitivity...")
    e5 = e5_price_search_bound_vs_escalation_rate()

    print("Running E6: latency stability over repeated runs...")
    e6 = e6_latency_stability(n_runs=10)

    print("Running E7: scheduling-only overhead (Planner isolated from agents)...")
    e7 = e7_scheduling_only_overhead(n_runs=20)

    all_results = {
        "environment_note": (
            "All experiments executed locally in a single-process sandboxed "
            "container (no GPU, no distributed cluster, no external network "
            "access to third-party APIs). Latencies reflect this environment "
            "and are reported as such, not as production/cloud benchmarks."
        ),
        "E1_scalability_vs_catalog_size": e1,
        "E2_per_agent_latency_ms": e2,
        "E3_fraud_detection_quality": e3,
        "E4_forecast_accuracy": e4,
        "E5_price_search_bound_vs_escalation_rate": e5,
        "E6_latency_stability": e6,
        "E7_scheduling_only_overhead": e7,
    }

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

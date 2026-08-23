"""
experiments/ablation_study_v2.py

A more rigorous follow-up to ablation_study.py, addressing three gaps
in the original version:

1. EQUAL-WORK BASELINE: the original NAIVE_SEQUENTIAL only called 3
   agents (forecast, inventory, pricing), so its latency comparison
   against the full 7-agent ACOS partly measured "less work done", not
   architecture overhead. EQUAL_WORK_SEQUENTIAL here calls all 7 agents
   -- the same computation ACOS_FULL does -- just without Orchestrator,
   EventBus, GovernanceAgent-as-a-gate, or ConsensusResolver. This
   isolates orchestration/governance/reconciliation overhead from
   agent-count differences.

2. MULTI-SEED: the original ablation ran once, on seed=42. This script
   runs every configuration across N independent seeds and reports
   mean, 95% CI, and effect size, not a single-seed point estimate.

3. INDEPENDENT OUTCOME MEASURE: the original experiment showed that
   reconciliation changes order quantities by 1,127 units, and that
   governance rejects policy-violating actions -- both are mechanism
   effects (the components alter outputs), not evidence that the
   altered outputs are BETTER. This script adds a downstream economic
   simulation: extends the synthetic sales-history generator by a few
   weeks beyond the forecasting window, treats those extra weeks as
   "realized demand", and computes stockout units, holding cost, and
   fill rate for the RECONCILED vs UNRECONCILED reorder plan against
   that realized demand -- a real outcome measure, not a proxy.

Governance is evaluated separately across multiple stress levels (not
only one 60% bound) so "governance catches violations" is shown to
hold across a range of conditions, not one cherry-pickable setting.
"""
from __future__ import annotations

import json
import statistics as _stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

from agents.analytics_agent import AnalyticsAgent
from agents.demand_forecast_agent import DemandForecastAgent, holt_linear_forecast
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
from core.planner import Planner, Task
from core.policy import PolicyEngine
from data.synthetic_data import (
    build_full_dataset, generate_catalog, generate_current_inventory,
    generate_sales_history, generate_suppliers, generate_transactions,
)
from main import build_knowledge_graph
from experiments.stats_utils import paired_comparison, holm_bonferroni

OUT_PATH = Path(__file__).resolve().parent / "ablation_v2_results.json"

HOLDING_COST_RATE_WEEKLY = 0.22 / 52.0  # same annual rate InventoryAgent uses, converted to weekly


def build_dataset_with_future(seed: int, future_weeks: int = 4):
    """Builds a synthetic dataset the same way build_full_dataset() does,
    but with sales_history extended by `future_weeks` beyond the window
    used for forecasting, so those extra weeks can serve as ground-truth
    realized demand for the outcome-quality simulation."""
    catalog = generate_catalog(seed=seed)
    full_sales = generate_sales_history(catalog, weeks=12 + future_weeks, seed=seed + 1)
    train_sales = {k: v[:12] for k, v in full_sales.items()}
    future_sales = {k: v[12:12 + future_weeks] for k, v in full_sales.items()}
    current_inventory = generate_current_inventory(catalog, seed=seed + 2)
    suppliers = generate_suppliers(catalog, seed=seed + 3)
    transactions, fraud_ground_truth = generate_transactions(catalog, seed=seed + 4)
    return {
        "catalog": catalog, "sales_history": train_sales, "future_sales": future_sales,
        "current_inventory": current_inventory, "suppliers": suppliers,
        "transactions": transactions, "fraud_ground_truth": fraud_ground_truth,
    }


def make_agents(db_path, kg):
    event_bus = EventBus()
    shared_memory = SharedMemory()
    ltm = LongTermMemory(db_path=db_path)
    agent_kwargs = dict(event_bus=event_bus, shared_memory=shared_memory, long_term_memory=ltm)
    agents = {
        "demand_forecast_agent": DemandForecastAgent(**agent_kwargs),
        "inventory_agent": InventoryAgent(**agent_kwargs),
        "pricing_agent": PricingAgent(**agent_kwargs),
        "procurement_agent": ProcurementAgent(**agent_kwargs, knowledge_graph=kg),
        "fraud_risk_agent": FraudRiskAgent(**agent_kwargs),
        "analytics_agent": AnalyticsAgent(**agent_kwargs),
        "governance_agent": GovernanceAgent(**agent_kwargs, policy_engine=PolicyEngine()),
    }
    return agents, event_bus, shared_memory, ltm


def config_equal_work_sequential(dataset, inputs, seed):
    """All 7 agents' reason() calls, in sequence, with no Orchestrator,
    EventBus, GovernanceAgent gate, or ConsensusResolver -- the same
    computational work as ACOS_FULL, minus the architecture."""
    t0 = time.perf_counter()
    ltm = LongTermMemory(db_path=f"/tmp/acos_v2_equalwork_{seed}.db")
    event_bus = EventBus()
    shared_memory = SharedMemory()
    kwargs = dict(event_bus=event_bus, shared_memory=shared_memory, long_term_memory=ltm)

    forecast_agent = DemandForecastAgent(**kwargs)
    forecast_decision = forecast_agent.reason(inputs)

    inv_agent = InventoryAgent(**kwargs)
    inv_ctx = dict(inputs); inv_ctx["forecast_demand"] = forecast_decision.output
    inv_decision = inv_agent.reason(inv_ctx)

    price_agent = PricingAgent(**kwargs)
    price_ctx = dict(inputs); price_ctx["forecast_demand"] = forecast_decision.output
    price_decision = price_agent.reason(price_ctx)

    kg = build_knowledge_graph({"catalog": dataset["catalog"], "suppliers": dataset["suppliers"]})
    proc_agent = ProcurementAgent(**kwargs, knowledge_graph=kg)
    proc_ctx = dict(inputs)
    proc_ctx["assess_inventory"] = inv_decision.output
    proc_decision = proc_agent.reason(proc_ctx)

    fraud_agent = FraudRiskAgent(**kwargs)
    fraud_decision = fraud_agent.reason(inputs)

    analytics_agent = AnalyticsAgent(**kwargs)
    analytics_ctx = dict(inputs)
    analytics_ctx["optimize_price"] = price_decision.output
    analytics_ctx["assess_inventory"] = inv_decision.output
    analytics_ctx["select_supplier"] = proc_decision.output
    analytics_ctx["screen_transactions"] = fraud_decision.output
    analytics_agent.reason(analytics_ctx)

    # governance agent's reason() is called (same computation cost as
    # ACOS_FULL) but its ruling is NOT used to gate/reject anything --
    # this config computes governance's answer without acting on it,
    # isolating "governance computation cost" from "governance as a gate"
    gov_agent = GovernanceAgent(**kwargs, policy_engine=PolicyEngine())
    gov_ctx = dict(inputs)
    gov_ctx["optimize_price"] = price_decision.output
    gov_ctx["select_supplier"] = proc_decision.output
    gov_ctx["screen_transactions"] = fraud_decision.output
    gov_agent.reason(gov_ctx)

    elapsed = time.perf_counter() - t0
    ltm.close()
    return {
        "config": "EQUAL_WORK_SEQUENTIAL",
        "latency_seconds": elapsed,
        "reorder_plan": inv_decision.output.get("reorder_plan", {}),
        "price_plan": price_decision.output.get("price_plan", {}),
    }


def run_acos_config(dataset, inputs, skip_governance: bool, skip_reconciliation: bool, config_name: str, seed):
    t0 = time.perf_counter()
    kg = build_knowledge_graph(dataset)
    db_path = f"/tmp/acos_v2_{config_name}_{seed}.db"
    agents, event_bus, shared_memory, ltm = make_agents(db_path, kg)
    planner = Planner()
    consensus = ConsensusResolver()
    orchestrator = Orchestrator(agents=agents, event_bus=event_bus, shared_memory=shared_memory,
                                 long_term_memory=ltm, knowledge_graph=kg, planner=planner,
                                 consensus=consensus)

    tasks = planner.build_plan("quarterly_pricing_and_inventory_review")
    if skip_governance:
        tasks = [t for t in tasks if t.task_id != "governance_review"]
    waves = planner.schedule(tasks)
    result = _run_waves_directly(orchestrator, tasks, waves, inputs, skip_reconciliation)

    elapsed = time.perf_counter() - t0
    ltm.close()

    price_plan = result["task_outputs"].get("optimize_price", {}).get("price_plan", {})
    governance_out = result["task_outputs"].get("governance_review")

    return {
        "config": config_name.upper(),
        "latency_seconds": elapsed,
        "reorder_plan": result["task_outputs"].get("assess_inventory", {}).get("reorder_plan", {}),
        "price_plan": price_plan,
        "n_rejected_by_governance": governance_out.get("n_rejected") if governance_out else None,
        "n_escalated_by_governance": governance_out.get("n_escalated") if governance_out else None,
        "n_conflicts_resolved": len(result["conflicts"]),
    }


def _run_waves_directly(orchestrator, tasks, waves, inputs, skip_reconciliation):
    import uuid
    workflow_id = str(uuid.uuid4())
    orchestrator.shared_memory.clear()
    for k, v in inputs.items():
        orchestrator.shared_memory.set(k, v)
    orchestrator.shared_memory.set("_workflow_id", workflow_id)

    task_outputs = {}
    for wave in waves:
        for task in wave:
            agent = orchestrator.agents[task.agent]
            decision = agent.run(task.task_id, dict(task.params), workflow_id=workflow_id)
            task_outputs[task.task_id] = decision.output
            orchestrator.shared_memory.set(f"_confidence_{task.task_id}", decision.confidence)

    conflicts = []
    if not skip_reconciliation:
        price_plan = task_outputs.get("optimize_price", {}).get("price_plan", {})
        reorder_plan = task_outputs.get("assess_inventory", {}).get("reorder_plan", {})
        if price_plan and reorder_plan:
            conflicts = orchestrator.consensus.reconcile_price_and_inventory(price_plan, reorder_plan)

    return {"task_outputs": task_outputs, "conflicts": conflicts}


def compute_unreconciled_deviation(naive_reorder_plan, reconciled_reorder_plan):
    total_deviation = 0.0
    n_skus_compared = 0
    for sku, naive_entry in naive_reorder_plan.items():
        recon_entry = reconciled_reorder_plan.get(sku)
        if recon_entry is None:
            continue
        naive_qty = naive_entry.get("recommended_order_qty", 0.0)
        recon_qty = recon_entry.get("recommended_order_qty", 0.0)
        if naive_qty > 0 or recon_qty > 0:
            total_deviation += abs(naive_qty - recon_qty)
            n_skus_compared += 1
    return round(total_deviation, 1), n_skus_compared


def simulate_outcome(reorder_plan, catalog, current_inventory, future_sales):
    """Independent decision-quality outcome measure: given a reorder plan
    and REALIZED (not forecasted) future demand, computes stockout units,
    holding cost, lost-margin cost, and fill rate. This does not use
    ACOS_FULL as the reference -- it uses actual subsequent demand drawn
    from the synthetic generator's own continuation of each SKU's series,
    which was never shown to any agent."""
    total_stockout_units = 0.0
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_realized_demand = 0.0
    total_fulfilled = 0.0
    n_skus = 0

    for sku, plan in reorder_plan.items():
        if sku not in future_sales or sku not in catalog:
            continue
        order_qty = plan.get("recommended_order_qty", 0.0)
        on_hand = current_inventory.get(sku, 0)
        available = on_hand + order_qty
        realized_demand = sum(future_sales[sku][:1])  # next week's actual realized demand
        unit_cost = catalog[sku]["unit_cost"]
        price = catalog[sku]["current_price"]
        margin_per_unit = max(0.0, price - unit_cost)

        fulfilled = min(available, realized_demand)
        stockout_units = max(0.0, realized_demand - available)
        holding_units = max(0.0, available - realized_demand)

        total_stockout_units += stockout_units
        total_holding_cost += holding_units * unit_cost * HOLDING_COST_RATE_WEEKLY
        total_stockout_cost += stockout_units * margin_per_unit  # lost-margin cost of a stockout
        total_realized_demand += realized_demand
        total_fulfilled += fulfilled
        n_skus += 1

    fill_rate = total_fulfilled / total_realized_demand if total_realized_demand > 0 else None
    return {
        "n_skus": n_skus,
        "total_stockout_units": round(total_stockout_units, 1),
        "total_holding_cost": round(total_holding_cost, 2),
        "total_stockout_cost_lost_margin": round(total_stockout_cost, 2),
        "total_cost": round(total_holding_cost + total_stockout_cost, 2),
        "fill_rate": round(fill_rate, 4) if fill_rate is not None else None,
    }


def simulate_outcome_price_elastic(reorder_plan, price_plan, catalog, current_inventory, future_sales):
    """Same outcome measure as simulate_outcome(), but the realized
    demand is adjusted by the constant-elasticity relationship
    demand(p) = demand(p0)*(p/p0)^elasticity using the ACTUAL price in
    price_plan, i.e. it tests the reconciliation mechanism under the
    assumption its own elasticity model is correct. Both the reconciled
    and unreconciled configurations share the same price_plan (only the
    order quantity differs), so this isolates whether sizing the order
    around the correct post-price demand level helps, under the
    condition where the elasticity assumption actually holds."""
    total_stockout_units = 0.0
    total_holding_cost = 0.0
    total_stockout_cost = 0.0
    total_realized_demand = 0.0
    total_fulfilled = 0.0
    n_skus = 0

    for sku, plan in reorder_plan.items():
        if sku not in future_sales or sku not in catalog:
            continue
        order_qty = plan.get("recommended_order_qty", 0.0)
        on_hand = current_inventory.get(sku, 0)
        available = on_hand + order_qty

        base_organic_demand = sum(future_sales[sku][:1])
        price_info = price_plan.get(sku, {})
        base_price = price_info.get("base_price")
        rec_price = price_info.get("recommended_price")
        elasticity = price_info.get("elasticity_used", catalog[sku].get("elasticity", -1.4))
        if base_price and rec_price and base_price > 0:
            demand_ratio = (rec_price / base_price) ** elasticity
        else:
            demand_ratio = 1.0
        realized_demand = base_organic_demand * demand_ratio

        unit_cost = catalog[sku]["unit_cost"]
        price = catalog[sku]["current_price"]
        margin_per_unit = max(0.0, price - unit_cost)

        fulfilled = min(available, realized_demand)
        stockout_units = max(0.0, realized_demand - available)
        holding_units = max(0.0, available - realized_demand)

        total_stockout_units += stockout_units
        total_holding_cost += holding_units * unit_cost * HOLDING_COST_RATE_WEEKLY
        total_stockout_cost += stockout_units * margin_per_unit
        total_realized_demand += realized_demand
        total_fulfilled += fulfilled
        n_skus += 1

    fill_rate = total_fulfilled / total_realized_demand if total_realized_demand > 0 else None
    return {
        "n_skus": n_skus,
        "total_stockout_units": round(total_stockout_units, 1),
        "total_holding_cost": round(total_holding_cost, 2),
        "total_stockout_cost_lost_margin": round(total_stockout_cost, 2),
        "total_cost": round(total_holding_cost + total_stockout_cost, 2),
        "fill_rate": round(fill_rate, 4) if fill_rate is not None else None,
    }


def governance_stress_sweep(dataset, inputs, seed, levels=(0.25, 0.35, 0.45, 0.60, 0.80)):
    """Governance catch rate across multiple stress levels, not just one
    60% bound, so 'governance catches violations' is shown across a
    range of conditions rather than a single chosen setting."""
    results = {}
    for level in levels:
        stress_inputs = dict(inputs)
        stress_inputs["max_price_change_pct"] = level

        ltm = LongTermMemory(db_path=f"/tmp/acos_v2_govstress_{seed}_{int(level*100)}.db")
        event_bus = EventBus()
        shared_memory = SharedMemory()
        kwargs = dict(event_bus=event_bus, shared_memory=shared_memory, long_term_memory=ltm)
        forecast_agent = DemandForecastAgent(**kwargs)
        forecast_decision = forecast_agent.reason(stress_inputs)
        price_agent = PricingAgent(**kwargs)
        price_ctx = dict(stress_inputs); price_ctx["forecast_demand"] = forecast_decision.output
        price_decision = price_agent.reason(price_ctx)
        naive_extreme = sum(1 for p in price_decision.output.get("price_plan", {}).values()
                             if abs(p.get("price_change_pct", 0.0)) > 0.35)
        ltm.close()

        full_result = run_acos_config(dataset, stress_inputs, skip_governance=False,
                                       skip_reconciliation=False, config_name=f"stress{int(level*100)}", seed=seed)
        results[f"{int(level*100)}pct"] = {
            "max_price_change_pct": level,
            "naive_extreme_would_ship": naive_extreme,
            "acos_full_n_rejected": full_result["n_rejected_by_governance"],
            "acos_full_n_escalated": full_result["n_escalated_by_governance"],
        }
    return results


def main():
    N_SEEDS = 20
    N_TIMING_REPS = 3  # repeated timed trials per configuration per seed
    seeds = list(range(1, N_SEEDS + 1))
    order_rng = random.Random(777)  # controls per-seed config execution order, independent of data seeds

    per_seed_results = []
    for seed in seeds:
        dataset = build_dataset_with_future(seed=seed, future_weeks=4)
        inputs = {
            "sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
            "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
            "transactions": dataset["transactions"],
        }

        # Warm-up: one untimed call before the timed ones, to avoid the first
        # config in a seed always absorbing any first-call/cold-cache cost.
        _ = run_acos_config(dataset, inputs, skip_governance=False, skip_reconciliation=False,
                             config_name="warmup", seed=seed)

        def call_config(name):
            if name == "equal_work":
                return config_equal_work_sequential(dataset, inputs, seed)
            elif name == "no_governance":
                return run_acos_config(dataset, inputs, skip_governance=True, skip_reconciliation=False,
                                        config_name="no_governance", seed=seed)
            elif name == "no_reconciliation":
                return run_acos_config(dataset, inputs, skip_governance=False, skip_reconciliation=True,
                                        config_name="no_reconciliation", seed=seed)
            elif name == "full":
                return run_acos_config(dataset, inputs, skip_governance=False, skip_reconciliation=False,
                                        config_name="full", seed=seed)

        # N_TIMING_REPS repeated timed trials per configuration, fully
        # interleaved across configurations within each seed (not run as
        # blocks of 3-same-config-in-a-row): build the flat list of all
        # (config, repetition) calls for this seed and shuffle it once, so
        # any sandbox load drift over the course of the seed's timing
        # window affects all four configs symmetrically rather than
        # systematically favoring whichever config happens to run first
        # or last. Each config's own elapsed time is still measured
        # internally via time.perf_counter() around only that config's
        # computation; interleaving only changes what runs immediately
        # before/after each individual measurement.
        config_names = ["equal_work", "no_governance", "no_reconciliation", "full"]
        call_plan = config_names * N_TIMING_REPS
        order_rng.shuffle(call_plan)
        latencies_by_config = {name: [] for name in config_names}
        last_result_by_config = {}
        for name in call_plan:
            result = call_config(name)
            latencies_by_config[name].append(result["latency_seconds"])
            last_result_by_config[name] = result  # outputs are deterministic given (dataset, seed); any repetition's non-timing fields are equivalent

        config_results = {}
        for name in config_names:
            result = dict(last_result_by_config[name])
            result["latency_seconds"] = _stats.median(latencies_by_config[name])
            result["latency_seconds_all_reps"] = latencies_by_config[name]
            config_results[name] = result

        equal_work = config_results["equal_work"]
        no_gov = config_results["no_governance"]
        no_recon = config_results["no_reconciliation"]
        full = config_results["full"]

        dev_units, n_compared = compute_unreconciled_deviation(no_recon["reorder_plan"], full["reorder_plan"])

        outcome_reconciled = simulate_outcome(full["reorder_plan"], dataset["catalog"],
                                               dataset["current_inventory"], dataset["future_sales"])
        outcome_unreconciled = simulate_outcome(no_recon["reorder_plan"], dataset["catalog"],
                                                 dataset["current_inventory"], dataset["future_sales"])

        outcome_reconciled_elastic = simulate_outcome_price_elastic(
            full["reorder_plan"], full["price_plan"], dataset["catalog"],
            dataset["current_inventory"], dataset["future_sales"])
        outcome_unreconciled_elastic = simulate_outcome_price_elastic(
            no_recon["reorder_plan"], no_recon["price_plan"], dataset["catalog"],
            dataset["current_inventory"], dataset["future_sales"])

        per_seed_results.append({
            "seed": seed,
            "latency_equal_work_seconds": equal_work["latency_seconds"],
            "latency_no_governance_seconds": no_gov["latency_seconds"],
            "latency_no_reconciliation_seconds": no_recon["latency_seconds"],
            "latency_full_seconds": full["latency_seconds"],
            "unreconciled_deviation_units": dev_units,
            "n_skus_compared": n_compared,
            "outcome_reconciled_total_cost": outcome_reconciled["total_cost"],
            "outcome_unreconciled_total_cost": outcome_unreconciled["total_cost"],
            "outcome_reconciled_fill_rate": outcome_reconciled["fill_rate"],
            "outcome_unreconciled_fill_rate": outcome_unreconciled["fill_rate"],
            "outcome_reconciled_stockout_units": outcome_reconciled["total_stockout_units"],
            "outcome_unreconciled_stockout_units": outcome_unreconciled["total_stockout_units"],
            "outcome_reconciled_elastic_total_cost": outcome_reconciled_elastic["total_cost"],
            "outcome_unreconciled_elastic_total_cost": outcome_unreconciled_elastic["total_cost"],
            "outcome_reconciled_elastic_fill_rate": outcome_reconciled_elastic["fill_rate"],
            "outcome_unreconciled_elastic_fill_rate": outcome_unreconciled_elastic["fill_rate"],
        })
        print(f"seed={seed} done: recon_cost={outcome_reconciled['total_cost']:.1f} "
              f"unrecon_cost={outcome_unreconciled['total_cost']:.1f}")

    # aggregate latency across seeds
    def agg(key):
        vals = [r[key] for r in per_seed_results]
        return {"mean": round(_stats.mean(vals), 5), "sd": round(_stats.pstdev(vals), 5),
                "min": round(min(vals), 5), "max": round(max(vals), 5)}

    latency_summary = {
        "equal_work_sequential": agg("latency_equal_work_seconds"),
        "acos_no_governance": agg("latency_no_governance_seconds"),
        "acos_no_reconciliation": agg("latency_no_reconciliation_seconds"),
        "acos_full": agg("latency_full_seconds"),
    }

    # governance and orchestration overhead, computed per-seed then aggregated (paired)
    gov_overhead = [r["latency_full_seconds"] - r["latency_no_governance_seconds"] for r in per_seed_results]
    orch_overhead = [r["latency_full_seconds"] - r["latency_equal_work_seconds"] for r in per_seed_results]
    recon_overhead = [r["latency_full_seconds"] - r["latency_no_reconciliation_seconds"] for r in per_seed_results]

    # Full paired tests (CI + Wilcoxon p) for the three architecture-latency
    # contrasts, not just mean/SD -- ACOS_FULL vs. EQUAL_WORK_SEQUENTIAL is
    # the primary RQ1 comparison; the other two are secondary/mechanistic.
    full_latency = [r["latency_full_seconds"] for r in per_seed_results]
    equal_work_latency = [r["latency_equal_work_seconds"] for r in per_seed_results]
    no_gov_latency = [r["latency_no_governance_seconds"] for r in per_seed_results]
    no_recon_latency = [r["latency_no_reconciliation_seconds"] for r in per_seed_results]
    orchestration_test = paired_comparison(full_latency, equal_work_latency, seed=201)
    governance_latency_test = paired_comparison(full_latency, no_gov_latency, seed=202)
    reconciliation_latency_test = paired_comparison(full_latency, no_recon_latency, seed=203)

    # paired statistical test: reconciled vs unreconciled outcome cost, across seeds
    recon_costs = [r["outcome_reconciled_total_cost"] for r in per_seed_results]
    unrecon_costs = [r["outcome_unreconciled_total_cost"] for r in per_seed_results]
    cost_test = paired_comparison(recon_costs, unrecon_costs, seed=1)

    recon_fill = [r["outcome_reconciled_fill_rate"] for r in per_seed_results if r["outcome_reconciled_fill_rate"] is not None]
    unrecon_fill = [r["outcome_unreconciled_fill_rate"] for r in per_seed_results if r["outcome_unreconciled_fill_rate"] is not None]
    fill_test = paired_comparison(recon_fill, unrecon_fill, seed=2) if len(recon_fill) == len(unrecon_fill) and len(recon_fill) > 1 else None

    # elastic-demand version: tests reconciliation under its own elasticity assumption
    recon_costs_e = [r["outcome_reconciled_elastic_total_cost"] for r in per_seed_results]
    unrecon_costs_e = [r["outcome_unreconciled_elastic_total_cost"] for r in per_seed_results]
    cost_test_elastic = paired_comparison(recon_costs_e, unrecon_costs_e, seed=3)

    recon_fill_e = [r["outcome_reconciled_elastic_fill_rate"] for r in per_seed_results if r["outcome_reconciled_elastic_fill_rate"] is not None]
    unrecon_fill_e = [r["outcome_unreconciled_elastic_fill_rate"] for r in per_seed_results if r["outcome_unreconciled_elastic_fill_rate"] is not None]
    fill_test_elastic = paired_comparison(recon_fill_e, unrecon_fill_e, seed=4) if len(recon_fill_e) == len(unrecon_fill_e) and len(recon_fill_e) > 1 else None

    # Reconciliation outcome family: the price-independent and price-elastic cost
    # tests are two pre-specified, planned-before-either-was-run tests, so we
    # apply Holm-Bonferroni correction across just this pair (Section 7 of the
    # paper defines this as the reconciliation outcome family).
    recon_family_raw_ps = [cost_test["wilcoxon_p_value"], cost_test_elastic["wilcoxon_p_value"]]
    recon_family_holm = holm_bonferroni(recon_family_raw_ps)
    cost_test["holm_adjusted_p"] = round(recon_family_holm[0], 4)
    cost_test_elastic["holm_adjusted_p"] = round(recon_family_holm[1], 4)

    deviation_vals = [r["unreconciled_deviation_units"] for r in per_seed_results]

    print("\nRunning governance multi-level stress sweep (seed=42)...")
    gov_stress_dataset = build_dataset_with_future(seed=42, future_weeks=4)
    gov_stress_inputs = {
        "sales_history": gov_stress_dataset["sales_history"], "catalog": gov_stress_dataset["catalog"],
        "current_inventory": gov_stress_dataset["current_inventory"], "suppliers": gov_stress_dataset["suppliers"],
        "transactions": gov_stress_dataset["transactions"],
    }
    gov_sweep = governance_stress_sweep(gov_stress_dataset, gov_stress_inputs, seed=42)

    summary = {
        "n_seeds": N_SEEDS,
        "latency_by_configuration_seconds": latency_summary,
        "governance_overhead_seconds": {"mean": round(_stats.mean(gov_overhead), 5), "sd": round(_stats.pstdev(gov_overhead), 5)},
        "orchestration_overhead_seconds_equal_work_vs_full": {"mean": round(_stats.mean(orch_overhead), 5), "sd": round(_stats.pstdev(orch_overhead), 5)},
        "reconciliation_overhead_seconds": {"mean": round(_stats.mean(recon_overhead), 5), "sd": round(_stats.pstdev(recon_overhead), 5)},
        "orchestration_overhead_paired_test": orchestration_test,
        "governance_latency_paired_test": governance_latency_test,
        "reconciliation_latency_paired_test": reconciliation_latency_test,
        "unreconciled_deviation_units": {"mean": round(_stats.mean(deviation_vals), 2), "sd": round(_stats.pstdev(deviation_vals), 2),
                                          "min": min(deviation_vals), "max": max(deviation_vals)},
        "outcome_measure": {
            "description": "Downstream economic simulation: holding cost + lost-margin stockout cost "
                            "against REALIZED demand (a held-out continuation of each SKU's series never "
                            "shown to any agent), reconciled vs unreconciled reorder plan, mean over seeds. "
                            "'price_independent' realized demand ignores any price effect (the generator's "
                            "organic continuation, unmodified). 'price_elastic' realized demand additionally "
                            "applies the SAME constant-elasticity relationship the PricingAgent itself "
                            "assumes, using the actual recommended price -- i.e. it tests reconciliation "
                            "under the condition where its own elasticity assumption is correct.",
            "price_independent": {
                "reconciled_mean_total_cost": round(_stats.mean(recon_costs), 2),
                "unreconciled_mean_total_cost": round(_stats.mean(unrecon_costs), 2),
                "reconciled_mean_fill_rate": round(_stats.mean(recon_fill), 4) if recon_fill else None,
                "unreconciled_mean_fill_rate": round(_stats.mean(unrecon_fill), 4) if unrecon_fill else None,
                "paired_test_total_cost": cost_test,
                "paired_test_fill_rate": fill_test,
            },
            "price_elastic": {
                "reconciled_mean_total_cost": round(_stats.mean(recon_costs_e), 2),
                "unreconciled_mean_total_cost": round(_stats.mean(unrecon_costs_e), 2),
                "reconciled_mean_fill_rate": round(_stats.mean(recon_fill_e), 4) if recon_fill_e else None,
                "unreconciled_mean_fill_rate": round(_stats.mean(unrecon_fill_e), 4) if unrecon_fill_e else None,
                "paired_test_total_cost": cost_test_elastic,
                "paired_test_fill_rate": fill_test_elastic,
            },
        },
        "governance_stress_sweep": gov_sweep,
        "per_seed_results": per_seed_results,
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed_results"}, indent=2, default=str))
    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()

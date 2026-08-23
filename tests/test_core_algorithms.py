import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from agents.demand_forecast_agent import holt_linear_forecast
from agents.fraud_risk_agent import robust_zscore, sigmoid
from agents.procurement_agent import topsis
from core.knowledge_graph import KnowledgeGraph
from core.memory import LongTermMemory, SharedMemory
from core.planner import Planner, PlanningError, Task
from core.policy import PolicyEngine, Ruling
from core.event_bus import EventBus
from experiments.stats_utils import paired_comparison


def test_holt_forecast_flat_series_has_zero_trend():
    series = [100.0] * 10
    forecasts, mape, level, trend = holt_linear_forecast(series, horizon=3)
    assert abs(trend) < 1e-6
    assert all(abs(f - 100.0) < 1e-6 for f in forecasts)
    assert mape < 1e-6


def test_holt_forecast_requires_two_points():
    with pytest.raises(ValueError):
        holt_linear_forecast([5.0])


def test_holt_forecast_upward_trend_extrapolates_up():
    series = [10, 20, 30, 40, 50]
    forecasts, mape, level, trend = holt_linear_forecast(series, alpha=0.9, beta=0.9, horizon=2)
    assert trend > 0
    assert forecasts[1] > forecasts[0] > series[-1]


def test_robust_zscore_flags_outlier():
    values = [10, 11, 9, 10, 10, 500]
    z = robust_zscore(values)
    assert abs(z[-1]) > abs(z[0])
    assert abs(z[-1]) > 5


def test_sigmoid_bounds():
    assert 0.0 < sigmoid(-100) < 0.001
    assert 0.999 < sigmoid(100) <= 1.0
    assert sigmoid(0) == pytest.approx(0.5)


def test_topsis_prefers_cheaper_faster_better_supplier():
    # supplier A dominates on every criterion vs supplier B
    matrix = np.array([
        [10.0, 3.0, 9.0, 0.95],   # supplier A: cheap, fast, high quality, reliable
        [20.0, 10.0, 5.0, 0.60],  # supplier B: expensive, slow, lower quality, less reliable
    ])
    weights = [0.35, 0.20, 0.25, 0.20]
    directions = ["min", "min", "max", "max"]
    scores = topsis(matrix, weights, directions)
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)


def test_knowledge_graph_risk_propagation_decays_with_distance():
    kg = KnowledgeGraph()
    kg.add_node("SUP-1", "Supplier")
    kg.add_node("SKU-1", "Product")
    kg.add_node("SKU-2", "Product")
    kg.add_edge("SUP-1", "SKU-1", "SUPPLIES")
    kg.add_edge("SKU-1", "SKU-2", "SUBSTITUTES")

    risk = kg.supply_chain_risk_propagation("SUP-1", damping=0.5)
    assert risk["SUP-1"] == 1.0
    assert risk["SKU-1"] == pytest.approx(0.5)
    assert risk["SKU-2"] == pytest.approx(0.25)
    assert risk["SKU-1"] > risk["SKU-2"]


def test_knowledge_graph_shortest_path():
    kg = KnowledgeGraph()
    kg.add_node("A", "Supplier")
    kg.add_node("B", "Product")
    kg.add_node("C", "Product")
    kg.add_edge("A", "B", "SUPPLIES")
    kg.add_edge("B", "C", "SUBSTITUTES")
    path = kg.shortest_path("A", "C")
    assert path == ["A", "B", "C"]
    assert kg.shortest_path("C", "A") is None  # directed graph, no reverse path


def test_planner_detects_cycles():
    planner = Planner()
    tasks = [
        Task("t1", "agent_x", "do", depends_on=["t2"]),
        Task("t2", "agent_x", "do", depends_on=["t1"]),
    ]
    with pytest.raises(PlanningError):
        planner.schedule(tasks)


def test_planner_produces_valid_topological_waves():
    planner = Planner()
    tasks = [
        Task("t1", "a", "do", depends_on=[]),
        Task("t2", "a", "do", depends_on=["t1"]),
        Task("t3", "a", "do", depends_on=["t1"]),
        Task("t4", "a", "do", depends_on=["t2", "t3"]),
    ]
    waves = planner.schedule(tasks)
    ids_per_wave = [{t.task_id for t in wave} for wave in waves]
    assert ids_per_wave[0] == {"t1"}
    assert ids_per_wave[1] == {"t2", "t3"}
    assert ids_per_wave[2] == {"t4"}


def test_planner_known_goal_builds_expected_task_count():
    planner = Planner()
    tasks = planner.build_plan("quarterly_pricing_and_inventory_review")
    assert len(tasks) == 7
    assert tasks[-1].task_id == "governance_review"


def test_planner_unknown_goal_raises():
    planner = Planner()
    with pytest.raises(PlanningError):
        planner.build_plan("not_a_real_goal")


def test_policy_engine_blocks_extreme_price_change():
    engine = PolicyEngine()
    result = engine.evaluate({"action": "optimize_price", "price_change_pct": 0.5,
                               "resulting_margin_pct": 0.3})
    assert result.ruling == Ruling.REJECT


def test_policy_engine_escalates_moderate_price_change():
    engine = PolicyEngine()
    result = engine.evaluate({"action": "optimize_price", "price_change_pct": 0.25,
                               "resulting_margin_pct": 0.3})
    assert result.ruling == Ruling.ESCALATE


def test_policy_engine_approves_small_price_change():
    engine = PolicyEngine()
    result = engine.evaluate({"action": "optimize_price", "price_change_pct": 0.05,
                               "resulting_margin_pct": 0.3})
    assert result.ruling == Ruling.APPROVE


def test_policy_engine_negative_margin_always_rejected():
    engine = PolicyEngine()
    result = engine.evaluate({"action": "optimize_price", "price_change_pct": 0.01,
                               "resulting_margin_pct": -0.1})
    assert result.ruling == Ruling.REJECT


def test_shared_memory_roundtrip():
    sm = SharedMemory()
    sm.set("x", {"a": 1})
    assert sm.get("x") == {"a": 1}
    assert sm.get("missing", "default") == "default"
    sm.clear()
    assert sm.get("x") is None


def test_long_term_memory_persists_episodes(tmp_path):
    db_path = str(tmp_path / "test.db")
    ltm = LongTermMemory(db_path=db_path)
    ltm.record_episode("test_agent", "decision", {"foo": "bar"}, workflow_id="wf1", confidence=0.8)
    episodes = ltm.recent_episodes(agent="test_agent")
    assert len(episodes) == 1
    assert episodes[0]["content"] == {"foo": "bar"}
    assert episodes[0]["confidence"] == 0.8
    ltm.close()


def test_event_bus_dispatches_to_subscribers():
    bus = EventBus()
    received = []
    bus.subscribe("test.topic", lambda e: received.append(e.payload))
    bus.publish("test.topic", {"value": 42}, source="tester")
    assert len(received) == 1
    assert received[0]["value"] == 42


def test_event_bus_wildcard_subscriber_receives_all():
    bus = EventBus()
    received = []
    bus.subscribe("*", lambda e: received.append(e.topic))
    bus.publish("a.b", {}, source="x")
    bus.publish("c.d", {}, source="x")
    assert received == ["a.b", "c.d"]


def test_event_bus_isolates_subscriber_errors():
    bus = EventBus()
    calls = []

    def bad_handler(e):
        raise RuntimeError("boom")

    def good_handler(e):
        calls.append(e.topic)

    bus.subscribe("t", bad_handler)
    bus.subscribe("t", good_handler)
    bus.publish("t", {}, source="x")  # should not raise
    assert calls == ["t"]


def test_paired_comparison_detects_consistent_effect():
    # a is uniformly smaller than b -> should be highly significant,
    # maximal effect size, all pairs favor a
    a = [0.10, 0.12, 0.15, 0.11, 0.09, 0.13, 0.14, 0.10]
    b = [0.20, 0.25, 0.22, 0.21, 0.19, 0.23, 0.24, 0.20]
    result = paired_comparison(a, b, n_bootstrap=2000, seed=1)
    assert result["significant_at_0.05"] is True
    assert result["matched_pairs_rank_biserial_r"] == pytest.approx(-1.0, abs=1e-6)
    assert result["n_a_better"] == len(a)
    assert result["n_b_better"] == 0
    assert result["bootstrap_ci_95"][1] < 0  # CI entirely below zero


def test_paired_comparison_detects_no_effect():
    # a and b are randomly interleaved around the same center -> should
    # NOT be significant, effect size near zero
    import random
    rng = random.Random(7)
    center = 0.5
    a = [center + rng.uniform(-0.05, 0.05) for _ in range(20)]
    b = [center + rng.uniform(-0.05, 0.05) for _ in range(20)]
    result = paired_comparison(a, b, n_bootstrap=2000, seed=2)
    assert abs(result["matched_pairs_rank_biserial_r"]) < 0.6
    # bootstrap CI should span (or be very close to) zero for a null effect
    lo, hi = result["bootstrap_ci_95"]
    assert lo < 0.05  # not a strongly one-sided interval


def test_paired_comparison_requires_equal_length():
    with pytest.raises(AssertionError):
        paired_comparison([1.0, 2.0], [1.0])


def test_paired_comparison_handles_all_zero_differences():
    a = [0.1, 0.2, 0.3]
    result = paired_comparison(a, list(a), n_bootstrap=500, seed=3)
    assert result["n_tied"] == 3
    assert result["matched_pairs_rank_biserial_r"] == 0.0


def test_pricing_agent_interior_optimum_for_elastic_epsilon():
    """For eps < -1, the unconstrained stationary point p* = eps*c/(1+eps)
    is margin(p)'s global maximizer on p>0. When p* falls inside the
    allowed search interval, the grid search should land near it; the
    constrained optimum is min(max(p*, low), high) in general, not p*
    itself (see the below-L and above-U cases below for when it isn't)."""
    from agents.pricing_agent import PricingAgent
    agent = PricingAgent.__new__(PricingAgent)  # no I/O deps needed for _optimize_single
    c, p0, d0, eps, delta = 12.0, 20.0, 100.0, -2.2, 0.25
    best_price, _, _, _ = agent._optimize_single(p0, d0, c, eps, delta)
    p_star = eps * c / (1 + eps)
    low, high = p0 * (1 - delta), p0 * (1 + delta)
    assert low < p_star < high  # this case only tests the interior sub-case
    predicted = min(max(p_star, low), high)
    assert best_price == pytest.approx(predicted, abs=(high - low) / 399 + 1e-6)


def test_pricing_agent_clips_to_upper_bound_when_star_above_interval():
    """For eps closer to -1 (still < -1), p* can lie above the upper
    search bound; the constrained optimum is then the upper bound, not
    the unconstrained p*."""
    from agents.pricing_agent import PricingAgent
    agent = PricingAgent.__new__(PricingAgent)
    c, p0, d0, eps, delta = 12.0, 20.0, 100.0, -1.5, 0.25
    best_price, _, _, _ = agent._optimize_single(p0, d0, c, eps, delta)
    p_star = eps * c / (1 + eps)
    low, high = p0 * (1 - delta), p0 * (1 + delta)
    assert p_star > high  # confirms this exercises the above-U sub-case
    assert best_price == pytest.approx(high, abs=1e-6)


def test_pricing_agent_clips_to_lower_bound_when_star_below_interval():
    """For eps very negative (still < -1), p* approaches c and can lie
    below the lower search bound; the constrained optimum is then the
    lower bound, not the unconstrained p*."""
    from agents.pricing_agent import PricingAgent
    agent = PricingAgent.__new__(PricingAgent)
    c, p0, d0, eps, delta = 12.0, 20.0, 100.0, -20.0, 0.25
    best_price, _, _, _ = agent._optimize_single(p0, d0, c, eps, delta)
    p_star = eps * c / (1 + eps)
    low, high = p0 * (1 - delta), p0 * (1 + delta)
    assert p_star < low  # confirms this exercises the below-L sub-case
    assert best_price == pytest.approx(low, abs=1e-6)


def test_pricing_agent_boundary_optimum_for_inelastic_epsilon():
    """For -1 <= eps < 0, margin(p) is monotonically increasing (no
    interior maximum); the grid search should select the upper bound of
    the allowed price range exactly."""
    from agents.pricing_agent import PricingAgent
    agent = PricingAgent.__new__(PricingAgent)
    c, p0, d0, eps, delta = 12.0, 20.0, 100.0, -0.8, 0.25
    best_price, _, _, _ = agent._optimize_single(p0, d0, c, eps, delta)
    assert best_price == pytest.approx(p0 * (1 + delta), abs=1e-6)

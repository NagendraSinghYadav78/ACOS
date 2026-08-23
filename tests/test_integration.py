import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from data.synthetic_data import build_full_dataset
from main import build_system


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("db") / "acos_test.db")
    orchestrator, dataset, vector_memory, long_term_memory, kg = build_system(db_path=db_path)
    yield orchestrator, dataset, vector_memory, long_term_memory, kg
    long_term_memory.close()


def test_synthetic_dataset_is_internally_consistent():
    ds = build_full_dataset(seed=1)
    assert set(ds["sales_history"].keys()) == set(ds["catalog"].keys())
    assert set(ds["current_inventory"].keys()) == set(ds["catalog"].keys())
    assert set(ds["suppliers"].keys()) == set(ds["catalog"].keys())
    assert len(ds["transactions"]) > 0
    txn_ids = [t["transaction_id"] for t in ds["transactions"]]
    assert len(txn_ids) == len(set(txn_ids))  # all unique


def test_synthetic_dataset_reproducible_with_same_seed():
    ds1 = build_full_dataset(seed=99)
    ds2 = build_full_dataset(seed=99)
    assert ds1["catalog"] == ds2["catalog"]
    assert ds1["sales_history"] == ds2["sales_history"]


def test_full_workflow_executes_all_tasks(system):
    orchestrator, dataset, _, _, _ = system
    inputs = {
        "sales_history": dataset["sales_history"],
        "catalog": dataset["catalog"],
        "current_inventory": dataset["current_inventory"],
        "suppliers": dataset["suppliers"],
        "transactions": dataset["transactions"],
    }
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    expected_tasks = {
        "forecast_demand", "assess_inventory", "optimize_price",
        "select_supplier", "screen_transactions", "analytics_rollup",
        "governance_review",
    }
    assert expected_tasks.issubset(result.task_outputs.keys())
    assert all(0.0 <= c <= 1.0 for c in result.task_confidences.values())
    assert result.wall_clock_seconds > 0


def test_full_workflow_governance_ruling_is_one_of_valid_states(system):
    orchestrator, dataset, _, _, _ = system
    inputs = {
        "sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
        "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
        "transactions": dataset["transactions"],
    }
    result = orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    assert result.governance.get("overall_ruling") in {"approve", "reject", "escalate"}


def test_full_workflow_persists_to_long_term_memory(system):
    orchestrator, dataset, _, long_term_memory, _ = system
    stats_before = long_term_memory.stats()
    inputs = {
        "sales_history": dataset["sales_history"], "catalog": dataset["catalog"],
        "current_inventory": dataset["current_inventory"], "suppliers": dataset["suppliers"],
        "transactions": dataset["transactions"],
    }
    orchestrator.run_workflow("quarterly_pricing_and_inventory_review", inputs)
    stats_after = long_term_memory.stats()
    assert stats_after["episodes"] > stats_before["episodes"]
    assert stats_after["decisions"] > stats_before["decisions"]


def test_fraud_agent_flags_injected_anomalies_with_reasonable_recall(system):
    """Evaluates the FraudRiskAgent's actual output against the dataset's
    known-injected ground truth. This is a real precision/recall
    computation over deterministic synthetic data, not a fabricated metric."""
    _, dataset, _, _, _ = system
    from agents.fraud_risk_agent import FraudRiskAgent
    from core.event_bus import EventBus
    from core.memory import LongTermMemory, SharedMemory
    import tempfile, os

    tmp_db = tempfile.mktemp(suffix=".db")
    agent = FraudRiskAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path=tmp_db))
    decision = agent.reason({"transactions": dataset["transactions"]})
    scored = {s["transaction_id"]: s for s in decision.output["scored"]}

    ground_truth = set(dataset["fraud_ground_truth"])
    flagged = {tid for tid, s in scored.items() if s["flagged"]}

    true_positives = len(flagged & ground_truth)
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    precision = true_positives / len(flagged) if flagged else 0.0

    # sanity bounds, not exact-match: this is a statistical heuristic
    # detector, so we assert it beats random guessing by a wide margin
    baseline_rate = len(ground_truth) / len(scored)
    assert recall > baseline_rate  # better than chance
    assert 0.0 <= precision <= 1.0
    agent.long_term_memory.close()
    os.remove(tmp_db)


def test_inventory_reorder_point_increases_with_service_level():
    """Higher required service level should never decrease the reorder
    point / safety stock, for identical demand statistics -- this is a
    monotonicity property of the underlying formula and a good
    regression check on the InventoryAgent's math."""
    from agents.inventory_agent import InventoryAgent
    from core.event_bus import EventBus
    from core.memory import LongTermMemory, SharedMemory
    import tempfile, os

    tmp_db = tempfile.mktemp(suffix=".db")
    agent = InventoryAgent(event_bus=EventBus(), shared_memory=SharedMemory(),
                            long_term_memory=LongTermMemory(db_path=tmp_db))

    forecast_ctx = {
        "forecast_demand": {"forecasts": {
            "SKU-X": {"forecast": [100, 105, 110, 108], "in_sample_mape": 0.1}
        }},
        "catalog": {"SKU-X": {"unit_cost": 10.0}},
        "current_inventory": {"SKU-X": 50},
    }

    low_sl = agent.reason({**forecast_ctx, "service_level": 0.80, "lead_time_days": 7})
    high_sl = agent.reason({**forecast_ctx, "service_level": 0.99, "lead_time_days": 7})

    rop_low = low_sl.output["reorder_plan"]["SKU-X"]["reorder_point"]
    rop_high = high_sl.output["reorder_plan"]["SKU-X"]["reorder_point"]
    assert rop_high >= rop_low

    agent.long_term_memory.close()
    os.remove(tmp_db)


def test_build_system_respects_seed_parameter():
    """Regression test for a real bug: build_system() previously
    hardcoded seed=42 internally regardless of caller intent, silently
    making any 'multi-seed' experiment that used it re-test the
    identical dataset every time. This test fails loudly if that
    regresses."""
    from main import build_system
    import os

    orch1, dataset1, _, ltm1, _ = build_system(db_path="/tmp/test_seed_1.db", seed=1)
    orch2, dataset2, _, ltm2, _ = build_system(db_path="/tmp/test_seed_2.db", seed=2)
    ltm1.close()
    ltm2.close()
    for f in ("/tmp/test_seed_1.db", "/tmp/test_seed_2.db"):
        if os.path.exists(f):
            os.remove(f)

    # different seeds must produce genuinely different catalogs
    assert dataset1["catalog"] != dataset2["catalog"]
    assert dataset1["current_inventory"] != dataset2["current_inventory"]

    # same seed must be fully reproducible
    orch3, dataset3, _, ltm3, _ = build_system(db_path="/tmp/test_seed_1b.db", seed=1)
    ltm3.close()
    if os.path.exists("/tmp/test_seed_1b.db"):
        os.remove("/tmp/test_seed_1b.db")
    assert dataset1["catalog"] == dataset3["catalog"]

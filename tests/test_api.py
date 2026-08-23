import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from api.main import app, DEMO_API_TOKEN

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {DEMO_API_TOKEN}"}


def test_health_check_no_auth_required():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_protected_endpoint_rejects_missing_auth():
    r = client.get("/agents")
    assert r.status_code == 401


def test_protected_endpoint_rejects_bad_token():
    r = client.get("/agents", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_list_agents_returns_seven_registered_agents():
    r = client.get("/agents", headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()["agents"]) == 7


def test_run_workflow_returns_full_result():
    r = client.post("/workflows/run",
                     json={"goal": "quarterly_pricing_and_inventory_review"},
                     headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "task_outputs" in body
    assert body["governance"]["overall_ruling"] in {"approve", "reject", "escalate"}


def test_run_workflow_rejects_unknown_goal():
    r = client.post("/workflows/run", json={"goal": "not_a_real_goal"}, headers=AUTH)
    assert r.status_code == 400


def test_semantic_search_endpoint():
    r = client.post("/memory/search", json={"query": "large price change approval", "k": 1},
                     headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_knowledge_graph_stats_endpoint():
    r = client.get("/knowledge-graph/stats", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["nodes"] > 0 and body["edges"] > 0

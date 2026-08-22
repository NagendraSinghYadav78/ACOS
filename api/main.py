"""
api/main.py

FastAPI service exposing ACOS over REST, backed by the same
Orchestrator used in main.py. Run with:

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Interactive OpenAPI/Swagger docs at /docs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from data.synthetic_data import build_full_dataset
from main import build_system

app = FastAPI(
    title="ACOS — Agentic Commerce Operating System",
    description="Enterprise multi-agent decision intelligence platform.",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)
DEMO_API_TOKEN = "acos-demo-token"  # placeholder-free but intentionally simple for a research prototype

_orchestrator, _dataset, _vector_memory, _long_term_memory, _kg = build_system(db_path="acos_api.db")
_start_time = time.time()


def require_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Simple bearer-token auth. In a production deployment this would
    delegate to an enterprise IdP (OAuth2/OIDC); this project implements
    a minimal, real bearer-token check suitable for a research
    prototype rather than a mocked auth layer."""
    if credentials is None or credentials.credentials != DEMO_API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                             detail="Invalid or missing bearer token")
    return True


class WorkflowRequest(BaseModel):
    goal: str = "quarterly_pricing_and_inventory_review"
    use_synthetic_data: bool = True
    overrides: Dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok", "uptime_seconds": round(time.time() - _start_time, 2)}


@app.get("/agents")
def list_agents(_: bool = Depends(require_auth)) -> Dict[str, Any]:
    return {"agents": list(_orchestrator.agents.keys())}


@app.post("/workflows/run")
def run_workflow(req: WorkflowRequest, _: bool = Depends(require_auth)) -> Dict[str, Any]:
    if req.goal not in _orchestrator.planner.GOAL_TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Unknown goal '{req.goal}'")

    inputs = {
        "sales_history": _dataset["sales_history"],
        "catalog": _dataset["catalog"],
        "current_inventory": _dataset["current_inventory"],
        "suppliers": _dataset["suppliers"],
        "transactions": _dataset["transactions"],
    }
    inputs.update(req.overrides)

    result = _orchestrator.run_workflow(req.goal, inputs)
    return result.to_dict()


@app.get("/memory/episodes")
def get_episodes(agent: Optional[str] = None, limit: int = 20,
                  _: bool = Depends(require_auth)) -> Dict[str, Any]:
    return {"episodes": _long_term_memory.recent_episodes(agent=agent, limit=limit)}


@app.get("/memory/stats")
def memory_stats(_: bool = Depends(require_auth)) -> Dict[str, Any]:
    return _long_term_memory.stats()


@app.post("/memory/search")
def semantic_search(req: SearchRequest, _: bool = Depends(require_auth)) -> Dict[str, Any]:
    return {"results": _vector_memory.search(req.query, k=req.k)}


@app.get("/knowledge-graph/stats")
def kg_stats(_: bool = Depends(require_auth)) -> Dict[str, Any]:
    return _kg.stats()


@app.get("/knowledge-graph/risk/{supplier_id}")
def kg_risk(supplier_id: str, _: bool = Depends(require_auth)) -> Dict[str, Any]:
    risk = _kg.supply_chain_risk_propagation(supplier_id)
    if not risk:
        raise HTTPException(status_code=404, detail=f"Unknown supplier '{supplier_id}'")
    return {"supplier_id": supplier_id, "propagated_risk": risk}


@app.get("/knowledge-graph/centrality")
def kg_centrality(node_type: Optional[str] = None, _: bool = Depends(require_auth)) -> Dict[str, Any]:
    ranking = _kg.centrality_ranking(node_type=node_type)
    return {"ranking": [{"node": n, "score": s} for n, s in ranking[:20]]}

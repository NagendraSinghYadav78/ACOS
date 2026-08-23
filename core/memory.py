"""
core/memory.py

Two memory layers:

- SharedMemory: working-memory blackboard agents read/write during a
  single workflow run. In-process dict, thread-safe.
- LongTermMemory: durable episodic memory in SQLite. Every decision,
  workflow outcome, and governance ruling gets persisted here so it
  can be queried and audited later. Semantic recall lives separately
  in vector_memory.py.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SharedMemory:
    """Thread-safe in-process blackboard for a single workflow run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class LongTermMemory:
    """SQLite-backed episodic / enterprise memory.

    Schema:
      episodes(id, ts, agent, workflow_id, kind, content_json, confidence)
      decisions(id, ts, agent, action, rationale, approved, workflow_id)
    """

    def __init__(self, db_path: str = "acos_memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    agent TEXT NOT NULL,
                    workflow_id TEXT,
                    kind TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    confidence REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    rationale TEXT,
                    approved INTEGER,
                    workflow_id TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decisions_workflow ON decisions(workflow_id)"
            )

    def record_episode(self, agent: str, kind: str, content: Dict[str, Any],
                        workflow_id: Optional[str] = None,
                        confidence: Optional[float] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO episodes (ts, agent, workflow_id, kind, content_json, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), agent, workflow_id, kind, json.dumps(content), confidence),
            )
            return cur.lastrowid

    def record_decision(self, agent: str, action: str, rationale: str,
                         approved: bool, workflow_id: Optional[str] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO decisions (ts, agent, action, rationale, approved, workflow_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), agent, action, rationale, int(approved), workflow_id),
            )
            return cur.lastrowid

    def recent_episodes(self, agent: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            if agent:
                rows = self._conn.execute(
                    "SELECT id, ts, agent, workflow_id, kind, content_json, confidence "
                    "FROM episodes WHERE agent=? ORDER BY id DESC LIMIT ?",
                    (agent, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, ts, agent, workflow_id, kind, content_json, confidence "
                    "FROM episodes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "id": r[0], "ts": r[1], "agent": r[2], "workflow_id": r[3],
                "kind": r[4], "content": json.loads(r[5]), "confidence": r[6],
            }
            for r in rows
        ]

    def decisions_for_workflow(self, workflow_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, agent, action, rationale, approved FROM decisions "
                "WHERE workflow_id=? ORDER BY id ASC",
                (workflow_id,),
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "agent": r[2], "action": r[3],
             "rationale": r[4], "approved": bool(r[5])}
            for r in rows
        ]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            n_ep = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            n_dec = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        return {"episodes": n_ep, "decisions": n_dec}

    def close(self) -> None:
        self._conn.close()

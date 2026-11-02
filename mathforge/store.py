"""Persistent knowledge base.

Only the head process writes here; workers return results over the executor.
SQLite in WAL mode handles millions of rows comfortably and survives crashes.

Tables
------
discoveries  verified results (deduplicated by a canonical ``key``)
traces       verified problem -> solution pairs used as training data
rollouts     every sampled attempt with its verifier reward, grouped per prompt
             (reinforcement learning; only the verifier's verdict is stored as reward)
metrics      per-round statistics, for plotting learning curves
state        JSON blobs for engine/curriculum checkpoints
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS discoveries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    engine     TEXT NOT NULL,
    key        TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL,          -- theorem | conjecture | law | solution
    statement  TEXT NOT NULL,          -- human-readable result
    payload    TEXT NOT NULL,          -- JSON with everything needed to re-verify
    score      REAL NOT NULL DEFAULT 0,
    round      INTEGER NOT NULL,
    created    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_disc_engine ON discoveries(engine, score DESC);

CREATE TABLE IF NOT EXISTS traces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    engine     TEXT NOT NULL,
    problem    TEXT NOT NULL,
    solution   TEXT NOT NULL,
    meta       TEXT NOT NULL DEFAULT '{}',
    round      INTEGER NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0,   -- consumed by a training step
    UNIQUE(engine, problem, solution)
);

CREATE TABLE IF NOT EXISTS rollouts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    engine     TEXT NOT NULL,
    round      INTEGER NOT NULL,
    grp        TEXT NOT NULL,          -- samples compared with each other (same prompt)
    prompt     TEXT NOT NULL,
    completion TEXT NOT NULL,
    reward     REAL NOT NULL,          -- assigned by OUR verifier, never by the model
    used       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics (
    round      INTEGER NOT NULL,
    engine     TEXT NOT NULL,
    name       TEXT NOT NULL,
    value      REAL NOT NULL,
    t          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated    REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------------------------------------------------------------- discoveries
    def add_discovery(self, engine: str, key: str, kind: str, statement: str,
                      payload: dict, score: float, round_: int) -> bool:
        """Insert a verified result. Returns True if it was new."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO discoveries(engine,key,kind,statement,payload,score,round,created)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (engine, key, kind, statement, json.dumps(payload, default=str), float(score),
             round_, time.time()),
        )
        self.db.commit()
        return cur.rowcount > 0

    def has_key(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM discoveries WHERE key=?", (key,)).fetchone() is not None

    def discoveries(self, engine: str | None = None, limit: int = 50,
                    order: str = "score") -> list[dict]:
        col = {"score": "score DESC", "new": "id DESC", "old": "id ASC"}[order]
        q = "SELECT id,engine,kind,statement,payload,score,round,created FROM discoveries"
        args: tuple = ()
        if engine:
            q += " WHERE engine=?"
            args = (engine,)
        q += f" ORDER BY {col} LIMIT ?"
        rows = self.db.execute(q, args + (limit,)).fetchall()
        keys = ["id", "engine", "kind", "statement", "payload", "score", "round", "created"]
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def count(self, engine: str | None = None) -> int:
        if engine:
            return self.db.execute("SELECT COUNT(*) FROM discoveries WHERE engine=?", (engine,)).fetchone()[0]
        return self.db.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]

    # ---------------------------------------------------------------- traces
    def add_traces(self, engine: str, rows: Iterable[tuple[str, str, dict]], round_: int) -> int:
        n = 0
        for problem, solution, meta in rows:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO traces(engine,problem,solution,meta,round) VALUES (?,?,?,?,?)",
                (engine, problem, solution, json.dumps(meta, default=str), round_),
            )
            n += cur.rowcount
        self.db.commit()
        return n

    def unused_trace_count(self, engine: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM traces WHERE engine=? AND used=0", (engine,)).fetchone()[0]

    def traces(self, engine: str, limit: int = 100000, only_unused: bool = False) -> list[dict]:
        q = "SELECT id,problem,solution,meta FROM traces WHERE engine=?"
        if only_unused:
            q += " AND used=0"
        q += " ORDER BY id DESC LIMIT ?"
        return [{"id": i, "problem": p, "solution": s, "meta": json.loads(m)}
                for i, p, s, m in self.db.execute(q, (engine, limit)).fetchall()]

    def mark_used(self, ids: list[int]) -> None:
        self.db.executemany("UPDATE traces SET used=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    # ---------------------------------------------------------------- rollouts (RL)
    def add_rollouts(self, engine: str, rows: Iterable[tuple[str, str, str, float]], round_: int) -> int:
        """rows: (group, prompt, completion, reward)."""
        rows = list(rows)
        self.db.executemany(
            "INSERT INTO rollouts(engine,round,grp,prompt,completion,reward) VALUES (?,?,?,?,?,?)",
            [(engine, round_, g, p, c, float(r)) for g, p, c, r in rows])
        self.db.commit()
        return len(rows)

    def rollouts(self, engine: str, only_unused: bool = True, limit: int = 100000) -> list[dict]:
        q = "SELECT id,grp,prompt,completion,reward FROM rollouts WHERE engine=?"
        if only_unused:
            q += " AND used=0"
        q += " ORDER BY id DESC LIMIT ?"
        return [{"id": i, "group": g, "prompt": p, "completion": c, "reward": r}
                for i, g, p, c, r in self.db.execute(q, (engine, limit)).fetchall()]

    def mark_rollouts_used(self, ids: list[int]) -> None:
        self.db.executemany("UPDATE rollouts SET used=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    # ---------------------------------------------------------------- metrics
    def log_metrics(self, round_: int, engine: str, values: dict[str, float]) -> None:
        t = time.time()
        self.db.executemany(
            "INSERT INTO metrics(round,engine,name,value,t) VALUES (?,?,?,?,?)",
            [(round_, engine, k, float(v), t) for k, v in values.items()
             if isinstance(v, (int, float))],
        )
        self.db.commit()

    def metric_series(self, engine: str, name: str) -> list[tuple[int, float]]:
        return self.db.execute(
            "SELECT round,value FROM metrics WHERE engine=? AND name=? ORDER BY round",
            (engine, name)).fetchall()

    # ---------------------------------------------------------------- state
    def save_state(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO state(key,value,updated) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (key, json.dumps(value, default=str), time.time()),
        )
        self.db.commit()

    def load_state(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def close(self) -> None:
        self.db.close()

"""Async SQLite helpers for the eval results database (eval/data/eval.db).

Separate from the app's production database. Eval runners write here;
the observability API reads from here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

EVAL_DB_PATH = Path(__file__).parent / "data" / "eval.db"

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    eval_type TEXT NOT NULL,
    pipeline_variant TEXT DEFAULT 'default',
    created_at TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_eval_runs_type ON eval_runs(eval_type);",
    "CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at);",
]


async def init_eval_db(conn: aiosqlite.Connection) -> None:
    await conn.execute(_CREATE_TABLE)
    for idx in _CREATE_INDEXES:
        await conn.execute(idx)
    await conn.commit()


async def get_eval_connection() -> aiosqlite.Connection:
    EVAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(EVAL_DB_PATH))
    await init_eval_db(conn)
    return conn


async def save_eval_run(
    conn: aiosqlite.Connection,
    run_id: str,
    eval_type: str,
    pipeline_variant: str,
    summary: dict,
    detail: list[dict],
    metadata: dict,
) -> None:
    await conn.execute(
        "INSERT INTO eval_runs (run_id, eval_type, pipeline_variant, created_at, summary_json, detail_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            eval_type,
            pipeline_variant,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(summary),
            json.dumps(detail),
            json.dumps(metadata),
        ),
    )
    await conn.commit()


async def list_eval_runs(
    conn: aiosqlite.Connection,
    eval_type: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> list[dict]:
    if eval_type:
        cursor = await conn.execute(
            "SELECT run_id, eval_type, pipeline_variant, created_at, summary_json, metadata_json FROM eval_runs WHERE eval_type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (eval_type, limit, offset),
        )
    else:
        cursor = await conn.execute(
            "SELECT run_id, eval_type, pipeline_variant, created_at, summary_json, metadata_json FROM eval_runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    rows = await cursor.fetchall()
    return [
        {
            "run_id": r[0],
            "eval_type": r[1],
            "pipeline_variant": r[2],
            "created_at": r[3],
            "summary": json.loads(r[4]),
            "metadata": json.loads(r[5]),
        }
        for r in rows
    ]


async def get_eval_run(conn: aiosqlite.Connection, run_id: str) -> dict | None:
    cursor = await conn.execute(
        "SELECT run_id, eval_type, pipeline_variant, created_at, summary_json, detail_json, metadata_json FROM eval_runs WHERE run_id = ?",
        (run_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {
        "run_id": row[0],
        "eval_type": row[1],
        "pipeline_variant": row[2],
        "created_at": row[3],
        "summary": json.loads(row[4]),
        "detail": json.loads(row[5]),
        "metadata": json.loads(row[6]),
    }

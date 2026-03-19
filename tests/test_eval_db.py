import asyncio
import json
import pytest
import aiosqlite

from eval.db import init_eval_db, save_eval_run, list_eval_runs, get_eval_run


@pytest.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    await init_eval_db(conn)
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_init_creates_table(db):
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_runs'")
    row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_save_and_get_run(db):
    await save_eval_run(
        conn=db,
        run_id="rq_123",
        eval_type="response_quality",
        pipeline_variant="default",
        summary={"overall": 4.2},
        detail=[{"turn": 1, "scores": {}}],
        metadata={"model": "gemini-2.5-pro"},
    )
    run = await get_eval_run(db, "rq_123")
    assert run is not None
    assert run["run_id"] == "rq_123"
    assert run["eval_type"] == "response_quality"
    assert run["summary"]["overall"] == 4.2
    assert len(run["detail"]) == 1


@pytest.mark.asyncio
async def test_list_runs_filtered(db):
    await save_eval_run(db, "rq_1", "response_quality", "default", {}, [], {})
    await save_eval_run(db, "tu_1", "tool_use", "default", {}, [], {})
    await save_eval_run(db, "rq_2", "response_quality", "v2", {}, [], {})

    all_runs = await list_eval_runs(db)
    assert len(all_runs) == 3

    rq_runs = await list_eval_runs(db, eval_type="response_quality")
    assert len(rq_runs) == 2
    assert all(r["eval_type"] == "response_quality" for r in rq_runs)


@pytest.mark.asyncio
async def test_list_runs_pagination(db):
    for i in range(5):
        await save_eval_run(db, f"run_{i}", "response_quality", "default", {}, [], {})

    page = await list_eval_runs(db, offset=1, limit=2)
    assert len(page) == 2


@pytest.mark.asyncio
async def test_get_nonexistent_run(db):
    run = await get_eval_run(db, "nonexistent")
    assert run is None

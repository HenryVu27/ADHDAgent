"""Tests for eval observability API endpoints.

Uses the eval.db functions directly since the API endpoints
depend on full app lifespan wiring.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite

from eval.db import init_eval_db, save_eval_run, list_eval_runs, get_eval_run


@pytest.fixture
async def eval_db():
    """In-memory eval database for testing."""
    conn = await aiosqlite.connect(":memory:")
    await init_eval_db(conn)
    # Seed test data
    await save_eval_run(conn, "rq_1", "response_quality", "default",
                        {"overall": 4.2}, [{"turn": 1, "scores": {}}], {})
    await save_eval_run(conn, "tu_1", "tool_use", "default",
                        {"mean_f1": 0.8}, [], {})
    yield conn
    await conn.close()


@pytest.mark.asyncio
async def test_list_eval_runs_all(eval_db):
    runs = await list_eval_runs(eval_db)
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_list_eval_runs_filtered(eval_db):
    runs = await list_eval_runs(eval_db, eval_type="response_quality")
    assert len(runs) == 1
    assert runs[0]["eval_type"] == "response_quality"


@pytest.mark.asyncio
async def test_get_eval_run_found(eval_db):
    run = await get_eval_run(eval_db, "rq_1")
    assert run is not None
    assert run["summary"]["overall"] == 4.2
    assert len(run["detail"]) == 1


@pytest.mark.asyncio
async def test_get_eval_run_not_found(eval_db):
    run = await get_eval_run(eval_db, "nonexistent")
    assert run is None

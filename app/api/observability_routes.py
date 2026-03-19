"""Observability API routes for the admin page.

Endpoints:
- GET  /api/observability/sessions                     — List all sessions with overview stats
- GET  /api/observability/sessions/{session_id}        — Full session detail
- GET  /api/observability/sessions/{session_id}/events — Filtered event log
- POST /api/observability/sessions/{session_id}/analyze — On-demand re-analysis
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from app.agent.store_protocol import SessionStoreBase
from app.api.deps import get_analyzer, get_eval_db, get_event_bus, get_session_store
from app.auth.dependencies import get_current_user
from app.models.schemas import (
    EvalListResponse,
    EvalRunDetail,
    EvalRunSummary,
    SessionDetailResponse,
    SessionListResponse,
    SessionOverview,
    UserRow,
)
from eval.db import get_eval_run, list_eval_runs

obs_router = APIRouter(prefix="/observability")


@obs_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session_store: SessionStoreBase = Depends(get_session_store),
    event_bus=Depends(get_event_bus),
    current_user: UserRow = Depends(get_current_user),
):
    """List all sessions with overview stats."""
    # Collect session IDs from store and event bus
    session_ids: set[str] = set()

    all_sessions = await session_store.get_all_sessions()
    for s in all_sessions:
        session_ids.add(s.session_id)

    if event_bus:
        session_ids.update(await event_bus.get_all_session_ids())

    overviews = []
    for sid in session_ids:
        state = await session_store.get(sid)
        traces = await session_store.get_traces(sid)
        analyses = await session_store.get_analyses(sid)

        total_flags = sum(len(a.flags) for a in analyses)
        avg_quality = (
            sum(a.quality_score for a in analyses) / len(analyses)
            if analyses else 1.0
        )
        tool_calls_count = sum(len(t.tool_calls) for t in traces)
        blocked_count = sum(1 for t in traces if t.input_blocked)

        created_at, updated_at = await session_store.get_session_timestamps(sid)

        overviews.append(SessionOverview(
            session_id=sid,
            turn_count=state.turn_count,
            created_at=created_at,
            updated_at=updated_at,
            total_flags=total_flags,
            avg_quality_score=round(avg_quality, 2),
            tool_calls_count=tool_calls_count,
            blocked_count=blocked_count,
        ))

    # Sort by turn count descending (most active first)
    overviews.sort(key=lambda o: o.turn_count, reverse=True)
    total = len(overviews)
    paginated = overviews[offset:offset + limit]
    return SessionListResponse(sessions=paginated, total=total, offset=offset, limit=limit)


@obs_router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    session_store: SessionStoreBase = Depends(get_session_store),
    event_bus=Depends(get_event_bus),
    current_user: UserRow = Depends(get_current_user),
):
    """Full session detail: messages, traces, analyses, events."""
    if not await session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await session_store.get_messages(session_id)
    traces = await session_store.get_traces(session_id)
    analyses = await session_store.get_analyses(session_id)
    events = await event_bus.get_events(session_id) if event_bus else []

    return SessionDetailResponse(
        session_id=session_id,
        messages=messages,
        traces=[t.model_dump() for t in traces],
        analyses=[a.model_dump() for a in analyses],
        events=[e.model_dump() for e in events],
    )


@obs_router.get("/sessions/{session_id}/events")
async def get_session_events(
    session_id: str,
    category: str | None = Query(None),
    session_store: SessionStoreBase = Depends(get_session_store),
    event_bus=Depends(get_event_bus),
    current_user: UserRow = Depends(get_current_user),
):
    """Filtered event log for a session."""
    if not await session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    if not event_bus:
        return {"events": []}

    events = await event_bus.get_events(session_id, category=category)
    return {"events": [e.model_dump() for e in events]}


@obs_router.post("/sessions/{session_id}/analyze")
async def analyze_session(
    session_id: str,
    session_store: SessionStoreBase = Depends(get_session_store),
    analyzer=Depends(get_analyzer),
    current_user: UserRow = Depends(get_current_user),
):
    """On-demand re-analysis of all turns in a session."""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not available")

    if not await session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await session_store.get_messages(session_id)
    traces = await session_store.get_traces(session_id)

    # Group messages by turn
    turns: dict[int, dict] = {}
    for msg in messages:
        turn = msg.get("turn", 0)
        if turn not in turns:
            turns[turn] = {"user": "", "assistant": ""}
        if msg["role"] == "user":
            turns[turn]["user"] = msg["content"]
        elif msg["role"] == "assistant":
            turns[turn]["assistant"] = msg["content"]

    # Build trace lookup by turn
    trace_by_turn = {t.turn: t for t in traces}

    tasks = []
    for turn_num in sorted(turns.keys()):
        turn_data = turns[turn_num]
        trace = trace_by_turn.get(turn_num)
        if turn_data["user"] and turn_data["assistant"]:
            from app.models.schemas import EnrichedTrace
            t = trace or EnrichedTrace(session_id=session_id, turn=turn_num)
            tasks.append(analyzer.analyze_turn(
                session_id=session_id,
                turn=turn_num,
                user_message=turn_data["user"],
                assistant_response=turn_data["assistant"],
                enriched_trace=t,
            ))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    analyses = await session_store.get_analyses(session_id)
    return {
        "status": "ok",
        "turns_analyzed": len(tasks),
        "total_flags": sum(len(a.flags) for a in analyses),
    }


@obs_router.get("/evals", response_model=EvalListResponse)
async def list_evals(
    eval_type: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    eval_db=Depends(get_eval_db),
):
    """List eval runs, optionally filtered by eval_type."""
    if eval_db is None:
        return EvalListResponse()
    runs = await list_eval_runs(eval_db, eval_type=eval_type, offset=offset, limit=limit)
    # Get total count for pagination (list_eval_runs returns paginated results)
    all_runs = await list_eval_runs(eval_db, eval_type=eval_type, offset=0, limit=10000)
    return EvalListResponse(
        runs=[EvalRunSummary(**r) for r in runs],
        total=len(all_runs),
        offset=offset,
        limit=limit,
    )


@obs_router.get("/evals/compare")
async def compare_eval_runs(
    run_a: str = Query(...),
    run_b: str = Query(...),
    eval_db=Depends(get_eval_db),
):
    """Compare two eval runs side-by-side."""
    if eval_db is None:
        raise HTTPException(status_code=503, detail="Eval database not available")
    a = await get_eval_run(eval_db, run_a)
    b = await get_eval_run(eval_db, run_b)
    if not a:
        raise HTTPException(status_code=404, detail=f"Run {run_a} not found")
    if not b:
        raise HTTPException(status_code=404, detail=f"Run {run_b} not found")
    return {"run_a": a, "run_b": b}


@obs_router.get("/evals/{run_id}", response_model=EvalRunDetail)
async def get_eval_detail(
    run_id: str,
    eval_db=Depends(get_eval_db),
):
    """Full detail for a single eval run."""
    if eval_db is None:
        raise HTTPException(status_code=503, detail="Eval database not available")
    run = await get_eval_run(eval_db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return EvalRunDetail(**run)

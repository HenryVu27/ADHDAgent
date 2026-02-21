"""Observability API routes for the admin page.

Endpoints:
- GET  /api/observability/sessions                     — List all sessions with overview stats
- GET  /api/observability/sessions/{session_id}        — Full session detail
- GET  /api/observability/sessions/{session_id}/events — Filtered event log
- POST /api/observability/sessions/{session_id}/analyze — On-demand re-analysis
"""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    ObservabilityEvent,
    SessionDetailResponse,
    SessionListResponse,
    SessionOverview,
)

obs_router = APIRouter(prefix="/observability")

# Set during app startup
_session_store = None
_event_bus = None
_analyzer = None


def set_observability_deps(session_store, event_bus, analyzer):
    global _session_store, _event_bus, _analyzer
    _session_store = session_store
    _event_bus = event_bus
    _analyzer = analyzer


@obs_router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all sessions with overview stats."""
    if not _session_store:
        raise HTTPException(status_code=503, detail="Not initialized")

    # Collect session IDs from store and event bus
    session_ids: set[str] = set()

    if hasattr(_session_store, "get_all_sessions"):
        all_sessions = _session_store.get_all_sessions()
        if isinstance(all_sessions, list):
            for s in all_sessions:
                if hasattr(s, "session_id"):
                    session_ids.add(s.session_id)
                elif isinstance(s, dict):
                    session_ids.add(s["session_id"])

    if _event_bus:
        session_ids.update(_event_bus.get_all_session_ids())

    overviews = []
    for sid in session_ids:
        state = _session_store.get(sid)
        traces = _session_store.get_traces(sid)
        analyses = _session_store.get_analyses(sid)

        total_flags = sum(len(a.flags) for a in analyses)
        avg_quality = (
            sum(a.quality_score for a in analyses) / len(analyses)
            if analyses else 1.0
        )
        tool_calls_count = sum(len(t.tool_calls) for t in traces)
        blocked_count = sum(1 for t in traces if t.input_blocked)

        # Try to get timestamps
        created_at = ""
        updated_at = ""
        if hasattr(state, "session_id") and isinstance(state, object):
            # For SQLite store, get_all_sessions returns dicts with timestamps
            if hasattr(_session_store, "_conn"):
                try:
                    row = _session_store._conn.execute(
                        "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    if row:
                        created_at = row["created_at"] or ""
                        updated_at = row["updated_at"] or ""
                except Exception:
                    pass

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
    return SessionListResponse(sessions=overviews)


@obs_router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(session_id: str):
    """Full session detail: messages, traces, analyses, events."""
    if not _session_store:
        raise HTTPException(status_code=503, detail="Not initialized")

    messages = _session_store.get_messages(session_id)
    traces = _session_store.get_traces(session_id)
    analyses = _session_store.get_analyses(session_id)
    events = _event_bus.get_events(session_id) if _event_bus else []

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
):
    """Filtered event log for a session."""
    if not _event_bus:
        return {"events": []}

    events = _event_bus.get_events(session_id, category=category)
    return {"events": [e.model_dump() for e in events]}


@obs_router.post("/sessions/{session_id}/analyze")
async def analyze_session(session_id: str):
    """On-demand re-analysis of all turns in a session."""
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not available")
    if not _session_store:
        raise HTTPException(status_code=503, detail="Not initialized")

    messages = _session_store.get_messages(session_id)
    traces = _session_store.get_traces(session_id)

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
            tasks.append(_analyzer.analyze_turn(
                session_id=session_id,
                turn=turn_num,
                user_message=turn_data["user"],
                assistant_response=turn_data["assistant"],
                enriched_trace=t,
            ))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    analyses = _session_store.get_analyses(session_id)
    return {
        "status": "ok",
        "turns_analyzed": len(tasks),
        "total_flags": sum(len(a.flags) for a in analyses),
    }

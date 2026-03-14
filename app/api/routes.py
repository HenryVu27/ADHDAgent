"""
API routes for ADHDAgent.

Endpoints:
- POST /api/chat/stream — main pipeline (SSE)
- GET  /api/session/{id}  — session state
- GET  /api/session/{id}/outcomes — outcome tracking
- GET  /api/health      — health check
- GET  /api/knowledge/topics — approved topic boundaries
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import AgentOrchestrator
from app.api.deps import get_knowledge_base, get_orchestrator
from app.api.rate_limit import limiter
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    MessagesResponse,
    OutcomesResponse,
    SeedSessionRequest,
    SessionResponse,
    SessionsResponse,
)
from app.rag.knowledge_store import KnowledgeStore

router = APIRouter()


@router.post("/chat/stream")
@limiter.limit(lambda: settings.RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """
    Streaming chat endpoint. Emits Server-Sent Events:
      - status  { text }              — pipeline stage updates
      - token   { text }              — one LLM response chunk
      - replace { text }              — output gate replaced the response
      - done    { session_id, agent_used, phase, pipeline_trace, response? }
      - error   { message }           — timeout or unhandled exception

    Design: uses LangGraph astream_events(version="v2") to stream tokens
    from the ReAct agent's final response in real time. See spec at
    docs/superpowers/specs/2026-03-13-streaming-design.md.
    """
    async def event_generator():
        async for event_type, data in orchestrator.process_stream(
            message=body.message,
            session_id=body.session_id,
        ):
            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/session/seed")
@limiter.limit(lambda: settings.RATE_LIMIT_DEFAULT)
async def seed_session(
    request: Request,
    body: SeedSessionRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Pre-populate a session with onboarding data so the agent has family context from the start."""
    await orchestrator.seed_session(body)
    return {"status": "ok", "session_id": body.session_id}


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Delete all data for a session (right-to-erasure)."""
    store = orchestrator.get_session_store()
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    await store.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Returns current conversation state for a session."""
    store = orchestrator.get_session_store()
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    state = await orchestrator.get_session(session_id)
    phase = await orchestrator.infer_phase(session_id)
    return SessionResponse(
        session_id=state.session_id,
        phase=phase,
        turn_count=state.turn_count,
        family_profile=state.family_profile,
        active_strategies=state.active_strategies,
        goals=state.goals,
    )


@router.get("/session/{session_id}/outcomes", response_model=OutcomesResponse)
async def get_outcomes(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Returns outcome tracking data for a session."""
    store = orchestrator.get_session_store()
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    state = await orchestrator.get_session(session_id)
    return OutcomesResponse(
        session_id=state.session_id,
        outcomes=state.outcomes,
        recommended_strategies=state.recommended_strategies,
        goals=state.goals,
    )


@router.get("/sessions", response_model=SessionsResponse)
async def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Returns all sessions ordered by most recently updated."""
    store = orchestrator.get_session_store()
    items, total = await store.get_all_sessions_paginated(offset=offset, limit=limit)
    return SessionsResponse(sessions=items, total=total, offset=offset, limit=limit)


@router.get("/session/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(
    session_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Returns all messages for a session."""
    store = orchestrator.get_session_store()
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    messages, total = await store.get_messages_paginated(session_id, offset=offset, limit=limit)
    return MessagesResponse(session_id=session_id, messages=messages, total=total, offset=offset, limit=limit)


@router.get("/health")
async def health(
    knowledge_base: KnowledgeStore = Depends(get_knowledge_base),
):
    return {
        "status": "ok",
        "index_built": knowledge_base.is_indexed if knowledge_base else False,
    }


@router.get("/knowledge/topics")
async def knowledge_topics(
    knowledge_base: KnowledgeStore = Depends(get_knowledge_base),
):
    """Returns the approved topic boundaries from the knowledge base."""
    return {
        "topics": knowledge_base.get_all_topics(),
        "document_count": len(knowledge_base.documents),
    }


@router.get("/knowledge/documents")
async def knowledge_documents(
    knowledge_base: KnowledgeStore = Depends(get_knowledge_base),
):
    """Returns all knowledge documents for the Resource Library."""
    return {"documents": [doc for doc in knowledge_base.documents]}

"""
API routes for ADHDAgent.

Endpoints:
- POST /api/chat       — main pipeline
- GET  /api/session/{id}  — session state
- GET  /api/session/{id}/outcomes — outcome tracking
- GET  /api/health      — health check
- GET  /api/knowledge/topics — approved topic boundaries
"""

import asyncio

from fastapi import APIRouter, HTTPException

from app.config import settings

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MessagesResponse,
    OutcomesResponse,
    SeedSessionRequest,
    SessionListItem,
    SessionResponse,
    SessionsResponse,
)

router = APIRouter()

# These get set during app startup (see main.py)
_orchestrator = None
_knowledge_base = None


def set_orchestrator(orchestrator):
    global _orchestrator
    _orchestrator = orchestrator


def set_knowledge_base(kb):
    global _knowledge_base
    _knowledge_base = kb


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Processes parent input through the agent pipeline:
    1. Input gate: crisis + jailbreak classification
    2. Context assembly: system prompt + conversation trimming
    3. ReAct loop: Gemini reasons and calls tools (search, profile, goals, outcomes)
    4. Output gate: medication + diagnosis + scope classification

    Returns full PipelineTrace for frontend visualization.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return await asyncio.wait_for(
            _orchestrator.process(
                message=request.message,
                session_id=request.session_id,
            ),
            timeout=settings.CHAT_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")


@router.post("/session/seed")
async def seed_session(request: SeedSessionRequest):
    """Pre-populate a session with onboarding data so the agent has family context from the start."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    _orchestrator.seed_session(request)
    return {"status": "ok", "session_id": request.session_id}


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Returns current conversation state for a session."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    state = _orchestrator.get_session(session_id)
    # Phase is inferred from session state, not stored
    phase = _orchestrator.infer_phase(session_id)
    return SessionResponse(
        session_id=state.session_id,
        phase=phase,
        turn_count=state.turn_count,
        family_profile=state.family_profile,
        active_strategies=state.active_strategies,
        goals=state.goals,
    )


@router.get("/session/{session_id}/outcomes", response_model=OutcomesResponse)
async def get_outcomes(session_id: str):
    """Returns outcome tracking data for a session."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    state = _orchestrator.get_session(session_id)
    return OutcomesResponse(
        session_id=state.session_id,
        outcomes=state.outcomes,
        recommended_strategies=state.recommended_strategies,
        goals=state.goals,
    )


@router.get("/sessions", response_model=SessionsResponse)
async def list_sessions():
    """Returns all sessions ordered by most recently updated."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    store = _orchestrator.get_session_store()
    items = store.get_all_sessions()

    # Sort by turn_count descending (most active first)
    items.sort(key=lambda x: x.turn_count, reverse=True)
    return SessionsResponse(sessions=items)


@router.get("/session/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(session_id: str):
    """Returns all messages for a session."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    messages = _orchestrator.get_session_store().get_messages(session_id)
    return MessagesResponse(session_id=session_id, messages=messages)


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "index_built": _knowledge_base.is_indexed if _knowledge_base else False,
    }


@router.get("/knowledge/topics")
async def knowledge_topics():
    """Returns the approved topic boundaries from the knowledge base."""
    if not _knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    return {
        "topics": _knowledge_base.get_all_topics(),
        "document_count": len(_knowledge_base.documents),
    }


@router.get("/knowledge/documents")
async def knowledge_documents():
    """Returns all knowledge documents for the Resource Library."""
    if not _knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")

    return {"documents": [doc for doc in _knowledge_base.documents]}

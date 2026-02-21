"""
API routes for ADHDAgent.

Endpoints:
- POST /api/chat       — main pipeline
- GET  /api/session/{id}  — session state
- GET  /api/session/{id}/outcomes — outcome tracking
- GET  /api/health      — health check
- GET  /api/knowledge/topics — approved topic boundaries
"""

from fastapi import APIRouter, HTTPException

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
    Main chat endpoint. Processes parent input through the ReAct agent:
    1. pre_model_hook: NeMo input guardrails + context injection
    2. ReAct loop: Gemini reasons and calls tools (search, profile, goals, outcomes)
    3. post_model_hook: NeMo output guardrails

    Returns full PipelineTrace for frontend visualization.
    """
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await _orchestrator.process(
        message=request.message,
        session_id=request.session_id,
    )


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
    phase = _orchestrator._infer_phase(session_id)
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

    store = _orchestrator._session_store
    all_sessions = store.get_all_sessions()

    items = []
    for s in all_sessions:
        if isinstance(s, dict):
            items.append(SessionListItem(
                session_id=s["session_id"],
                turn_count=s.get("turn_count", 0),
                phase=s.get("phase", "intake"),
                created_at=s.get("created_at", ""),
                updated_at=s.get("updated_at", ""),
            ))
        else:
            # InMemorySessionStore returns SessionState objects
            items.append(SessionListItem(
                session_id=s.session_id,
                turn_count=s.turn_count,
                phase=s.phase.value if hasattr(s.phase, "value") else str(s.phase),
            ))

    # Sort by turn_count descending (most active first)
    items.sort(key=lambda x: x.turn_count, reverse=True)
    return SessionsResponse(sessions=items)


@router.get("/session/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(session_id: str):
    """Returns all messages for a session."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    messages = _orchestrator._session_store.get_messages(session_id)
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

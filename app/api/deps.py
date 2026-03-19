"""FastAPI dependency injection functions.

Replaces module-level globals with proper Depends() dependencies.
All dependencies are stored on app.state during lifespan startup.
"""

import aiosqlite
from fastapi import HTTPException, Request

from app.agent.orchestrator import AgentOrchestrator
from app.agent.store_protocol import SessionStoreBase
from app.rag.knowledge_store import KnowledgeStore


def get_orchestrator(request: Request) -> AgentOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if not orch:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return orch


def get_knowledge_base(request: Request) -> KnowledgeStore:
    kb = getattr(request.app.state, "knowledge_base", None)
    if not kb:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    return kb


def get_session_store(request: Request) -> SessionStoreBase:
    store = getattr(request.app.state, "session_store", None)
    if not store:
        raise HTTPException(status_code=503, detail="Not initialized")
    return store


def get_event_bus(request: Request):
    return getattr(request.app.state, "event_bus", None)


def get_analyzer(request: Request):
    return getattr(request.app.state, "analyzer", None)


def get_db(request: Request) -> aiosqlite.Connection:
    conn = getattr(request.app.state, "db_conn", None)
    if not conn:
        raise HTTPException(status_code=503, detail="Database not available")
    return conn


def get_eval_db(request: Request):
    """Return eval database connection (None if no eval runs yet)."""
    return getattr(request.app.state, "eval_db", None)

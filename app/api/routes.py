"""
API routes for ADHDAgent.

Endpoints:
- POST /api/chat/stream — main pipeline (SSE)
- POST /api/upload — file upload for multimodal attachments
- GET  /api/attachments/{id}/thumbnail — serve attachment thumbnail
- GET  /api/session/{id}  — session state
- GET  /api/session/{id}/outcomes — outcome tracking
- GET  /api/health      — health check
- GET  /api/knowledge/topics — approved topic boundaries
"""

import json
import os
import re
import shutil

import aiosqlite
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.agent.orchestrator import AgentOrchestrator
from app.api.deps import get_db, get_knowledge_base, get_orchestrator
from app.api.rate_limit import limiter
from app.auth.dependencies import get_current_user
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    MessagesResponse,
    OutcomesResponse,
    SeedSessionRequest,
    SessionResponse,
    SessionsResponse,
    UploadBlockedResponse,
    UploadResponse,
    UserRow,
)
from app.rag.knowledge_store import KnowledgeStore
from app.services.file_upload import (
    delete_gemini_file,
    extract_text_from_file,
    generate_attachment_id,
    generate_thumbnail,
    upload_to_gemini,
    validate_magic_bytes,
)

router = APIRouter()


async def _check_session_owner(
    session_id: str,
    user_id: int,
    conn: aiosqlite.Connection,
) -> None:
    """Raises 404 if session not found, 403 if owned by a different user."""
    async with conn.execute(
        "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")


async def _assert_not_other_users_session(
    session_id: str,
    user_id: int,
    conn: aiosqlite.Connection,
) -> None:
    """For create-or-access paths: only checks ownership if the session exists.

    New sessions (row is None) are allowed through — they will be created
    with the correct owner. Existing sessions owned by a different user get 403.
    """
    async with conn.execute(
        "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
    ) as cur:
        row = await cur.fetchone()
    if row is not None and row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")


@router.post("/chat/stream")
@limiter.limit(lambda: settings.RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
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
    await _assert_not_other_users_session(body.session_id, current_user.id, conn)

    async def event_generator():
        async for event_type, data in orchestrator.process_stream(
            message=body.message,
            session_id=body.session_id,
            user_id=current_user.id,
            attachment_ids=body.attachment_ids or None,
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
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Pre-populate a session with onboarding data so the agent has family context from the start."""
    await _assert_not_other_users_session(body.session_id, current_user.id, conn)
    await orchestrator.seed_session(body, user_id=current_user.id)
    return {"status": "ok", "session_id": body.session_id}


@router.post("/upload")
@limiter.limit(lambda: settings.UPLOAD_RATE_LIMIT)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Upload a file for attachment to a chat message."""
    # Validate session_id format (prevent path traversal in thumbnail dir)
    if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    await _assert_not_other_users_session(session_id, current_user.id, conn)

    # Validate content type
    if file.content_type not in settings.UPLOAD_ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    # Read file data
    data = await file.read()

    # Validate size
    if len(data) > settings.UPLOAD_MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # Validate magic bytes
    if not validate_magic_bytes(data, file.content_type):
        raise HTTPException(status_code=415, detail="File content does not match declared type")

    # Upload to Gemini File API
    try:
        gemini_file_name, gemini_file_uri = await upload_to_gemini(
            data, file.filename or "upload", file.content_type
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini upload failed: {e}")

    # Extract text and scan through input gate (fail-closed)
    input_gate = getattr(request.app.state, "input_gate", None)
    if input_gate:
        try:
            extracted_text = await extract_text_from_file(gemini_file_uri, file.content_type)
            if extracted_text.strip():
                check = await input_gate.check(extracted_text)
                if not check.is_allowed:
                    await delete_gemini_file(gemini_file_name)
                    return UploadBlockedResponse(
                        reason=check.blocked_reason or "content",
                        response=check.override_response or "",
                    )
        except Exception as e:
            # Fail closed: reject upload if extraction fails
            await delete_gemini_file(gemini_file_name)
            raise HTTPException(status_code=502, detail=f"File content scanning failed: {e}")

    # Generate thumbnail (images only)
    attachment_id = generate_attachment_id()
    thumbnail_path = generate_thumbnail(data, file.content_type, session_id, attachment_id)

    # Store metadata
    store = orchestrator.get_session_store()
    await store.save_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        gemini_file_name=gemini_file_name,
        gemini_file_uri=gemini_file_uri,
        filename=file.filename or "upload",
        content_type=file.content_type,
        size_bytes=len(data),
        thumbnail_path=thumbnail_path,
    )
    await store.commit()

    thumbnail_url = f"/api/attachments/{attachment_id}/thumbnail" if thumbnail_path else None

    return UploadResponse(
        id=attachment_id,
        filename=file.filename or "upload",
        content_type=file.content_type,
        thumbnail_url=thumbnail_url,
    )


@router.get("/attachments/{attachment_id}/thumbnail")
async def get_thumbnail(
    attachment_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Serve a thumbnail image for an attachment."""
    store = orchestrator.get_session_store()
    attachments = await store.get_attachments([attachment_id])
    if not attachments:
        raise HTTPException(status_code=404, detail="Attachment not found")

    thumb_path = attachments[0].get("thumbnail_path")
    if not thumb_path or not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path, media_type="image/webp")


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Delete all data for a session (right-to-erasure)."""
    await _check_session_owner(session_id, current_user.id, conn)

    store = orchestrator.get_session_store()

    # Clean up attachments: Gemini files + local thumbnails
    attachments = await store.get_attachments_by_session(session_id)
    for att in attachments:
        if att.get("gemini_file_name"):
            await delete_gemini_file(att["gemini_file_name"])
        if att.get("thumbnail_path") and os.path.exists(att["thumbnail_path"]):
            os.remove(att["thumbnail_path"])

    # Delete thumbnail directory if empty
    thumb_dir = os.path.join(settings.UPLOAD_THUMBNAIL_DIR, session_id)
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir, ignore_errors=True)

    await store.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Returns current conversation state for a session."""
    await _check_session_owner(session_id, current_user.id, conn)

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
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Returns outcome tracking data for a session."""
    await _check_session_owner(session_id, current_user.id, conn)

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
    current_user: UserRow = Depends(get_current_user),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Returns all sessions ordered by most recently updated."""
    store = orchestrator.get_session_store()
    items, total = await store.get_all_sessions_paginated(offset=offset, limit=limit, user_id=current_user.id)
    return SessionsResponse(sessions=items, total=total, offset=offset, limit=limit)


@router.get("/session/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(
    session_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    current_user: UserRow = Depends(get_current_user),
    conn: aiosqlite.Connection = Depends(get_db),
):
    """Returns all messages for a session."""
    await _check_session_owner(session_id, current_user.id, conn)

    store = orchestrator.get_session_store()
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

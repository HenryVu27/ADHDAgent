"""API endpoint tests — uses AgentOrchestrator with mocked agent.

Uses httpx.AsyncClient with ASGITransport so the async aiosqlite-backed
session store shares the same event loop as the test.
"""

import asyncio
import json as _json

import httpx
import pytest
from httpx import ASGITransport
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.orchestrator import AgentOrchestrator
from app.agent.session_store import create_in_memory_store
from app.api.deps import get_knowledge_base, get_orchestrator
from app.main import app
from app.rag.knowledge_store import KnowledgeStore


def _make_mock_agent():
    """Mock agent whose astream_events yields a minimal successful event sequence."""
    from langchain_core.messages import AIMessageChunk

    response_text = "I'm here to help with ADHD parenting strategies."

    async def mock_astream_events(*args, **kwargs):
        yield {
            "event": "on_chain_start",
            "name": "LangGraph",
            "metadata": {
                "langgraph_node": "pro_react_agent",
                "langgraph_checkpoint_ns": "pro_react_agent:test",
            },
            "data": {},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {
                "langgraph_node": "agent",
                "langgraph_checkpoint_ns": "pro_react_agent:test",
            },
            "data": {"chunk": AIMessageChunk(content=response_text)},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "", "langgraph_checkpoint_ns": ""},
            "data": {
                "output": {
                    "messages": [
                        HumanMessage(content="test message"),
                        AIMessage(content=response_text),
                    ],
                    "input_blocked": False,
                    "trace_steps": [],
                }
            },
        }

    agent = AsyncMock()
    agent.astream_events = mock_astream_events
    return agent


async def _consume_stream(response) -> dict:
    """Parse a streaming /api/chat/stream response and return the done payload."""
    assert response.status_code == 200
    done_data = {}
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode()
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event_type = ""
            data_str = ""
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]
            if event_type == "done" and data_str:
                done_data = _json.loads(data_str)
    return done_data


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    from app.api.rate_limit import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
async def client():
    store = KnowledgeStore()
    session_store = await create_in_memory_store()
    agent = _make_mock_agent()
    orchestrator = AgentOrchestrator(agent=agent, session_store=session_store)

    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_knowledge_base] = lambda: store

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_health_endpoint(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


async def test_chat_endpoint(client):
    response = await client.post("/api/chat/stream", json={
        "message": "Hi, I need help with my child",
        "session_id": "api_test",
    })
    data = await _consume_stream(response)
    assert data["session_id"] == "api_test"
    assert "agent_used" in data
    assert "phase" in data
    assert "pipeline_trace" in data


async def test_chat_returns_pipeline_trace(client):
    response = await client.post("/api/chat/stream", json={
        "message": "My 7 year old won't do homework",
        "session_id": "trace_test",
    })
    data = await _consume_stream(response)
    trace = data["pipeline_trace"]
    assert len(trace["steps"]) > 0
    assert trace["total_duration_ms"] >= 0


async def test_session_endpoint(client):
    # Send a message first to create session
    stream_resp = await client.post("/api/chat/stream", json={"message": "Hi", "session_id": "session_api_test"})
    await _consume_stream(stream_resp)

    response = await client.get("/api/session/session_api_test")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session_api_test"
    assert data["phase"] == "intake"
    assert data["turn_count"] >= 1


async def test_outcomes_endpoint(client):
    stream_resp = await client.post("/api/chat/stream", json={"message": "Hi", "session_id": "outcomes_test"})
    await _consume_stream(stream_resp)

    response = await client.get("/api/session/outcomes_test/outcomes")
    assert response.status_code == 200
    data = response.json()
    assert "outcomes" in data
    assert "recommended_strategies" in data
    assert "goals" in data


async def test_knowledge_topics_endpoint(client):
    response = await client.get("/api/knowledge/topics")
    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert data["document_count"] > 0


async def test_seed_session(client):
    response = await client.post("/api/session/seed", json={
        "session_id": "seed_test",
        "child_name": "Kai",
        "child_age": "7",
        "challenges": ["homework"],
        "goals": ["Better homework routine"],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

    # Verify session was seeded
    response = await client.get("/api/session/seed_test")
    data = response.json()
    assert data["family_profile"]["child_name"] == "Kai"


# --- Step 2: Message validation tests ---

async def test_chat_rejects_empty_message(client):
    response = await client.post("/api/chat/stream", json={
        "message": "",
        "session_id": "val_test",
    })
    assert response.status_code == 422


async def test_chat_rejects_whitespace_message(client):
    response = await client.post("/api/chat/stream", json={
        "message": "   \n\t  ",
        "session_id": "val_test",
    })
    assert response.status_code == 422


async def test_chat_rejects_oversized_message(client):
    response = await client.post("/api/chat/stream", json={
        "message": "x" * 5001,
        "session_id": "val_test",
    })
    assert response.status_code == 422


# --- Step 3: Timeout test ---

async def test_chat_timeout(client):
    from app.config import settings

    async def slow_astream_events(*args, **kwargs):
        await asyncio.sleep(60)
        yield {}  # never reached

    slow_agent = AsyncMock()
    slow_agent.astream_events = slow_astream_events
    store = await create_in_memory_store()
    slow_orchestrator = AgentOrchestrator(agent=slow_agent, session_store=store)
    app.dependency_overrides[get_orchestrator] = lambda: slow_orchestrator

    original = settings.CHAT_TIMEOUT_S
    settings.CHAT_TIMEOUT_S = 0.05
    try:
        response = await client.post("/api/chat/stream", json={
            "message": "Hello",
            "session_id": "timeout_test",
        })
        assert response.status_code == 200
        content = response.content.decode()
        assert 'event: error' in content
    finally:
        settings.CHAT_TIMEOUT_S = original


# --- Step 4: Auth tests ---

async def test_auth_required_when_configured(client):
    from app.config import settings
    original = settings.API_KEY
    settings.API_KEY = "test-secret-key"
    try:
        # Without key -> 401 on non-public endpoint
        response = await client.get("/api/sessions")
        assert response.status_code == 401

        # With key -> 200
        response = await client.get("/api/sessions", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200

        # /api/health is still public even with auth configured
        response = await client.get("/api/health")
        assert response.status_code == 200
    finally:
        settings.API_KEY = original


async def test_health_is_public_without_auth(client):
    """Health is public when no API_KEY is configured (dev mode)."""
    response = await client.get("/api/health")
    assert response.status_code == 200


async def test_observability_requires_admin_key(client):
    from app.config import settings
    from app.api.deps import get_session_store, get_event_bus

    orig_key = settings.API_KEY
    orig_admin = settings.ADMIN_API_KEY
    settings.API_KEY = "regular-key"
    settings.ADMIN_API_KEY = "admin-key"

    # Override session_store and event_bus for observability endpoint
    obs_store = await create_in_memory_store()
    app.dependency_overrides[get_session_store] = lambda: obs_store
    app.dependency_overrides[get_event_bus] = lambda: None

    try:
        # Regular key -> 403 on observability
        response = await client.get(
            "/api/observability/sessions",
            headers={"X-API-Key": "regular-key"},
        )
        assert response.status_code == 403

        # Admin key -> 200
        response = await client.get(
            "/api/observability/sessions",
            headers={"X-API-Key": "admin-key"},
        )
        assert response.status_code == 200
    finally:
        settings.API_KEY = orig_key
        settings.ADMIN_API_KEY = orig_admin


# --- Step 5: Rate limiting test ---

async def test_rate_limit_on_chat(client):
    from app.api.rate_limit import limiter
    from app.config import settings

    original = settings.RATE_LIMIT_CHAT
    settings.RATE_LIMIT_CHAT = "2/minute"
    limiter.enabled = True
    limiter.reset()
    try:
        # First 2 should succeed
        for _ in range(2):
            response = await client.post("/api/chat/stream", json={
                "message": "Hello",
                "session_id": "rate_test",
            })
            assert response.status_code == 200
            await _consume_stream(response)

        # Third should be rate limited
        response = await client.post("/api/chat/stream", json={
            "message": "Hello",
            "session_id": "rate_test",
        })
        assert response.status_code == 429
    finally:
        settings.RATE_LIMIT_CHAT = original
        limiter.enabled = False


# --- Step 6: Session 404 tests ---

async def test_get_nonexistent_session_returns_404(client):
    response = await client.get("/api/session/nonexistent_session_xyz")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


async def test_get_nonexistent_session_messages_returns_404(client):
    response = await client.get("/api/session/nonexistent_session_xyz/messages")
    assert response.status_code == 404


async def test_get_nonexistent_session_outcomes_returns_404(client):
    response = await client.get("/api/session/nonexistent_session_xyz/outcomes")
    assert response.status_code == 404


# --- Step 7: Pagination tests ---

async def test_sessions_list_pagination(client):
    # Create a few sessions
    for i in range(3):
        r = await client.post("/api/chat/stream", json={"message": "Hi", "session_id": f"page_{i}"})
        await _consume_stream(r)

    response = await client.get("/api/sessions?offset=0&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) <= 2
    assert data["total"] == 3
    assert data["offset"] == 0
    assert data["limit"] == 2


async def test_messages_pagination(client):
    # Create a session with messages
    for i in range(3):
        r = await client.post("/api/chat/stream", json={"message": f"msg {i}", "session_id": "paginate_msgs"})
        await _consume_stream(r)

    response = await client.get("/api/session/paginate_msgs/messages?offset=0&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) <= 2
    assert data["total"] >= 3  # at least 3 user messages + 3 assistant
    assert data["offset"] == 0
    assert data["limit"] == 2


async def test_pagination_rejects_invalid_limit(client):
    response = await client.get("/api/sessions?limit=0")
    assert response.status_code == 422

    response = await client.get("/api/sessions?limit=201")
    assert response.status_code == 422


# --- Step 8: Delete session tests ---

async def test_delete_nonexistent_session_returns_404(client):
    response = await client.delete("/api/session/test-delete-session")
    assert response.status_code == 404


async def test_delete_session(client):
    # Create the session first via seed
    await client.post("/api/session/seed", json={
        "session_id": "test-delete-session",
        "child_name": "Test",
    })
    response = await client.delete("/api/session/test-delete-session")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    # Verify it's gone (GET should return 404)
    response = await client.get("/api/session/test-delete-session")
    assert response.status_code == 404

"""API endpoint tests — uses AgentOrchestrator with mocked agent."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.orchestrator import AgentOrchestrator
from app.agent.session_store import SessionStateStore
from app.api.deps import get_knowledge_base, get_orchestrator
from app.main import app
from app.rag.knowledge_store import KnowledgeStore


def _make_mock_agent():
    """Create a mock agent that returns a simple AI response."""
    agent = AsyncMock()
    agent.ainvoke = AsyncMock(return_value={
        "messages": [
            HumanMessage(content="test message"),
            AIMessage(content="I'm here to help with ADHD parenting strategies."),
        ],
        "session_id": "test",
        "input_blocked": False,
        "block_response": "",
        "trace_steps": [],
    })
    return agent


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    from app.api.rate_limit import limiter
    limiter.enabled = False
    yield
    limiter.enabled = True


@pytest.fixture
def client():
    store = KnowledgeStore()
    session_store = SessionStateStore()
    agent = _make_mock_agent()
    orchestrator = AgentOrchestrator(agent=agent, session_store=session_store)

    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_knowledge_base] = lambda: store
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_chat_endpoint(client):
    response = client.post("/api/chat", json={
        "message": "Hi, I need help with my child",
        "session_id": "api_test",
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "agent_used" in data
    assert "phase" in data
    assert "pipeline_trace" in data
    assert data["session_id"] == "api_test"


def test_chat_returns_pipeline_trace(client):
    response = client.post("/api/chat", json={
        "message": "My 7 year old won't do homework",
        "session_id": "trace_test",
    })
    data = response.json()
    trace = data["pipeline_trace"]
    assert len(trace["steps"]) > 0
    assert trace["total_duration_ms"] >= 0


def test_session_endpoint(client):
    # Send a message first to create session
    client.post("/api/chat", json={"message": "Hi", "session_id": "session_api_test"})

    response = client.get("/api/session/session_api_test")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session_api_test"
    assert data["phase"] == "intake"
    assert data["turn_count"] >= 1


def test_outcomes_endpoint(client):
    client.post("/api/chat", json={"message": "Hi", "session_id": "outcomes_test"})

    response = client.get("/api/session/outcomes_test/outcomes")
    assert response.status_code == 200
    data = response.json()
    assert "outcomes" in data
    assert "recommended_strategies" in data
    assert "goals" in data


def test_knowledge_topics_endpoint(client):
    response = client.get("/api/knowledge/topics")
    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert data["document_count"] > 0


def test_seed_session(client):
    response = client.post("/api/session/seed", json={
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
    response = client.get("/api/session/seed_test")
    data = response.json()
    assert data["family_profile"]["child_name"] == "Kai"


# --- Step 2: Message validation tests ---

def test_chat_rejects_empty_message(client):
    response = client.post("/api/chat", json={
        "message": "",
        "session_id": "val_test",
    })
    assert response.status_code == 422


def test_chat_rejects_whitespace_message(client):
    response = client.post("/api/chat", json={
        "message": "   \n\t  ",
        "session_id": "val_test",
    })
    assert response.status_code == 422


def test_chat_rejects_oversized_message(client):
    response = client.post("/api/chat", json={
        "message": "x" * 5001,
        "session_id": "val_test",
    })
    assert response.status_code == 422


# --- Step 3: Timeout test ---

def test_chat_timeout(client):
    async def slow_process(**kwargs):
        await asyncio.sleep(10)

    # Get the orchestrator from the override
    orchestrator = app.dependency_overrides[get_orchestrator]()
    orchestrator.process = slow_process

    with patch.object(type(app.state), "CHAT_TIMEOUT_S", 0.01, create=True):
        from app.config import settings
        original = settings.CHAT_TIMEOUT_S
        settings.CHAT_TIMEOUT_S = 0.01
        try:
            response = client.post("/api/chat", json={
                "message": "Hello",
                "session_id": "timeout_test",
            })
            assert response.status_code == 504
            assert "timed out" in response.json()["detail"]
        finally:
            settings.CHAT_TIMEOUT_S = original


# --- Step 4: Auth tests ---

def test_auth_required_when_configured(client):
    from app.config import settings
    original = settings.API_KEY
    settings.API_KEY = "test-secret-key"
    try:
        # Without key -> 401 on non-public endpoint
        response = client.get("/api/sessions")
        assert response.status_code == 401

        # With key -> 200
        response = client.get("/api/sessions", headers={"X-API-Key": "test-secret-key"})
        assert response.status_code == 200

        # /api/health is still public even with auth configured
        response = client.get("/api/health")
        assert response.status_code == 200
    finally:
        settings.API_KEY = original


def test_health_is_public_without_auth(client):
    """Health is public when no API_KEY is configured (dev mode)."""
    response = client.get("/api/health")
    assert response.status_code == 200


def test_observability_requires_admin_key(client):
    from app.config import settings
    from app.api.deps import get_session_store, get_event_bus
    from app.agent.session_store import SessionStateStore

    orig_key = settings.API_KEY
    orig_admin = settings.ADMIN_API_KEY
    settings.API_KEY = "regular-key"
    settings.ADMIN_API_KEY = "admin-key"

    # Override session_store and event_bus for observability endpoint
    obs_store = SessionStateStore()
    app.dependency_overrides[get_session_store] = lambda: obs_store
    app.dependency_overrides[get_event_bus] = lambda: None

    try:
        # Regular key -> 403 on observability
        response = client.get(
            "/api/observability/sessions",
            headers={"X-API-Key": "regular-key"},
        )
        assert response.status_code == 403

        # Admin key -> 200
        response = client.get(
            "/api/observability/sessions",
            headers={"X-API-Key": "admin-key"},
        )
        assert response.status_code == 200
    finally:
        settings.API_KEY = orig_key
        settings.ADMIN_API_KEY = orig_admin


# --- Step 5: Rate limiting test ---

def test_rate_limit_on_chat(client):
    from app.api.rate_limit import limiter
    from app.config import settings

    original = settings.RATE_LIMIT_CHAT
    settings.RATE_LIMIT_CHAT = "2/minute"
    limiter.enabled = True
    # Clear any prior rate limit state
    limiter.reset()
    try:
        # First 2 should succeed
        for _ in range(2):
            response = client.post("/api/chat", json={
                "message": "Hello",
                "session_id": "rate_test",
            })
            assert response.status_code == 200

        # Third should be rate limited
        response = client.post("/api/chat", json={
            "message": "Hello",
            "session_id": "rate_test",
        })
        assert response.status_code == 429
    finally:
        settings.RATE_LIMIT_CHAT = original
        limiter.enabled = False


# --- Step 6: Session 404 tests ---

def test_get_nonexistent_session_returns_404(client):
    response = client.get("/api/session/nonexistent_session_xyz")
    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]


def test_get_nonexistent_session_messages_returns_404(client):
    response = client.get("/api/session/nonexistent_session_xyz/messages")
    assert response.status_code == 404


def test_get_nonexistent_session_outcomes_returns_404(client):
    response = client.get("/api/session/nonexistent_session_xyz/outcomes")
    assert response.status_code == 404


# --- Step 7: Pagination tests ---

def test_sessions_list_pagination(client):
    # Create a few sessions
    for i in range(3):
        client.post("/api/chat", json={"message": "Hi", "session_id": f"page_{i}"})

    response = client.get("/api/sessions?offset=0&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) <= 2
    assert data["total"] == 3
    assert data["offset"] == 0
    assert data["limit"] == 2


def test_messages_pagination(client):
    # Create a session with messages
    for i in range(3):
        client.post("/api/chat", json={"message": f"msg {i}", "session_id": "paginate_msgs"})

    response = client.get("/api/session/paginate_msgs/messages?offset=0&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) <= 2
    assert data["total"] >= 3  # at least 3 user messages + 3 assistant
    assert data["offset"] == 0
    assert data["limit"] == 2


def test_pagination_rejects_invalid_limit(client):
    response = client.get("/api/sessions?limit=0")
    assert response.status_code == 422

    response = client.get("/api/sessions?limit=201")
    assert response.status_code == 422


# --- Step 8: Delete session tests ---

def test_delete_nonexistent_session_returns_404(client):
    response = client.delete("/api/session/test-delete-session")
    assert response.status_code == 404


def test_delete_session(client):
    # Create the session first via seed
    client.post("/api/session/seed", json={
        "session_id": "test-delete-session",
        "child_name": "Test",
    })
    response = client.delete("/api/session/test-delete-session")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    # Verify it's gone (GET should return 404)
    response = client.get("/api/session/test-delete-session")
    assert response.status_code == 404

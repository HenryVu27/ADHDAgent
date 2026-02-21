"""API endpoint tests — uses AgentOrchestrator with mocked agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.orchestrator import AgentOrchestrator
from app.agent.session_store import SessionStateStore
from app.api.routes import set_knowledge_base, set_orchestrator
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


@pytest.fixture
def client():
    store = KnowledgeStore()
    session_store = SessionStateStore()
    agent = _make_mock_agent()
    orchestrator = AgentOrchestrator(agent=agent, session_store=session_store)
    set_orchestrator(orchestrator)
    set_knowledge_base(store)
    return TestClient(app, raise_server_exceptions=False)


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

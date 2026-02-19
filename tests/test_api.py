"""API endpoint tests."""

import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.agents.intake import IntakeAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.progress import ProgressAgent
from app.agents.strategy import StrategyAgent
from app.api.routes import set_knowledge_base, set_orchestrator
from app.guardrails.validator import GuardrailsValidator
from app.main import app
from app.models.schemas import InputCheckResult, OutputCheckResult
from app.phase_manager import PhaseManager
from app.rag.knowledge_store import KnowledgeStore
from app.rag.retriever import HybridRetriever


def _make_mock_guardrails():
    guardrails = AsyncMock(spec=GuardrailsValidator)
    guardrails.check_input = AsyncMock(return_value=InputCheckResult(is_allowed=True))
    guardrails.check_output = AsyncMock(return_value=OutputCheckResult(is_valid=True))
    return guardrails


@pytest.fixture
def client():
    # Test client with fully wired orchestrator (no Gemini)
    store = KnowledgeStore()
    orchestrator = AgentOrchestrator(
        guardrails=_make_mock_guardrails(),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(
            knowledge_store=store,
            gemini_client=None,
        ),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )
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
    assert trace["total_duration_ms"] > 0


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


def test_frontend_serves(client):
    response = client.get("/")
    assert response.status_code == 200

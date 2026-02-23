"""Tests for observability API endpoints."""

import pytest
from unittest.mock import AsyncMock

import httpx
from httpx import ASGITransport
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.event_bus import EventBus
from app.agent.orchestrator import AgentOrchestrator
from app.agent.session_store import create_in_memory_store
from app.api.deps import get_analyzer, get_event_bus, get_knowledge_base, get_orchestrator, get_session_store
from app.main import app
from app.models.schemas import EnrichedTrace, TurnAnalysis, AnalysisFlag
from app.rag.knowledge_store import KnowledgeStore


def _make_mock_agent():
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
async def setup():
    """Set up store, event_bus, and client with observability deps wired."""
    store = await create_in_memory_store()
    event_bus = EventBus(buffer_size=100)
    kb = KnowledgeStore()
    agent = _make_mock_agent()
    orchestrator = AgentOrchestrator(
        agent=agent, session_store=store, event_bus=event_bus,
    )

    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_knowledge_base] = lambda: kb
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_analyzer] = lambda: None

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield store, event_bus, client
    app.dependency_overrides.clear()


class TestObservabilityAPI:

    async def test_list_sessions_empty(self, setup):
        _, _, client = setup
        response = await client.get("/api/observability/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []

    async def test_list_sessions_with_data(self, setup):
        store, event_bus, client = setup
        # Create a session with some data
        await store.get("obs_test_1")
        await store.increment_turn("obs_test_1")
        await store.add_message("obs_test_1", "user", "Hello", 1)
        await store.add_message("obs_test_1", "assistant", "Hi there!", 1)

        # Add a trace
        trace = EnrichedTrace(session_id="obs_test_1", turn=1, total_duration_ms=500)
        await store.save_trace("obs_test_1", trace)

        # Emit an event so the session shows up in event_bus too
        await event_bus.emit("agent", "turn_start", "obs_test_1", turn=1)

        await store.commit()

        response = await client.get("/api/observability/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) >= 1
        session = next(s for s in data["sessions"] if s["session_id"] == "obs_test_1")
        assert session["turn_count"] == 1

    async def test_session_detail(self, setup):
        store, event_bus, client = setup
        await store.get("detail_test")
        await store.increment_turn("detail_test")
        await store.add_message("detail_test", "user", "Hello", 1)
        await store.add_message("detail_test", "assistant", "Hi!", 1)

        trace = EnrichedTrace(session_id="detail_test", turn=1, total_duration_ms=300)
        await store.save_trace("detail_test", trace)

        analysis = TurnAnalysis(
            session_id="detail_test", turn=1,
            quality_score=0.8, summary="Decent turn",
            flags=[AnalysisFlag(flag_type="tone_issue", severity="info", description="Minor tone issue")],
        )
        await store.save_analysis("detail_test", analysis)

        await event_bus.emit("agent", "turn_start", "detail_test", turn=1)

        await store.commit()

        response = await client.get("/api/observability/sessions/detail_test")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "detail_test"
        assert len(data["messages"]) == 2
        assert len(data["traces"]) == 1
        assert len(data["analyses"]) == 1
        assert len(data["events"]) >= 1

    async def test_session_events_filtered(self, setup):
        store, event_bus, client = setup
        # Create the session so it passes 404 check
        await store.get("events_test")
        await store.commit()

        await event_bus.emit("guardrails", "input_check_passed", "events_test")
        await event_bus.emit("agent", "turn_start", "events_test")
        await event_bus.emit("memory", "summary_updated", "events_test")

        # All events
        response = await client.get("/api/observability/sessions/events_test/events")
        assert response.status_code == 200
        assert len(response.json()["events"]) == 3

        # Filtered by category
        response = await client.get("/api/observability/sessions/events_test/events?category=guardrails")
        assert response.status_code == 200
        events = response.json()["events"]
        assert len(events) == 1
        assert events[0]["category"] == "guardrails"

    async def test_session_detail_nonexistent_returns_404(self, setup):
        _, _, client = setup
        response = await client.get("/api/observability/sessions/nonexistent")
        assert response.status_code == 404

    async def test_list_sessions_includes_event_only_sessions(self, setup):
        _, event_bus, client = setup
        # Session exists only in event bus, not in store
        await event_bus.emit("agent", "turn_start", "event_only_session")

        response = await client.get("/api/observability/sessions")
        data = response.json()
        ids = [s["session_id"] for s in data["sessions"]]
        assert "event_only_session" in ids

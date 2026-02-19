"""Integration tests for the orchestrator pipeline."""

import pytest
from unittest.mock import AsyncMock

from app.agents.intake import IntakeAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.progress import ProgressAgent
from app.agents.safety import SafetyMonitor
from app.agents.strategy import StrategyAgent
from app.predicates.extractor import PredicateExtractor
from app.rag.bm25_index import BM25Index
from app.rag.knowledge_store import KnowledgeStore
from app.rag.retriever import HybridRetriever
from app.rules.python_engine import PythonRulesEngine


def _make_safety_gemini():
    """Mock Gemini that classifies safety based on the user message in the prompt."""
    client = AsyncMock()

    async def _classify(prompt: str):
        # Extract just the user message from the prompt template
        # Format: "...Parent message: {message}\n\nReturn JSON..."
        msg = prompt.split("Parent message: ", 1)[-1].split("\n\nReturn JSON")[0].lower()
        if any(w in msg for w in ["harm", "suicide", "kill", "hurt"]):
            return {"level": "crisis", "detected_topic": "harm"}
        if any(w in msg for w in ["medication", "dosage", "ritalin", "adderall"]):
            return {"level": "out_of_scope", "detected_topic": "medication"}
        return {"level": "safe", "detected_topic": None}

    client.extract_json = _classify
    return client


@pytest.fixture
def orchestrator():
    """Full pipeline orchestrator with mock safety Gemini, other agents use fallbacks."""
    store = KnowledgeStore()
    bm25 = BM25Index()
    bm25.build(store.chunks)
    return AgentOrchestrator(
        extractor=PredicateExtractor(gemini_client=None),
        safety=SafetyMonitor(gemini_client=_make_safety_gemini()),
        rules_engine=PythonRulesEngine(),
        retriever=HybridRetriever(
            knowledge_store=store,
            bm25_index=bm25,
            gemini_client=None,
        ),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
        guardrails_validator=None,
    )


@pytest.mark.asyncio
async def test_first_message_triggers_intake(orchestrator):
    result = await orchestrator.process("Hi, I need help with my kid", "test_session")
    assert result.agent_used == "intake"
    assert result.phase.value == "intake"
    assert result.pipeline_trace is not None
    assert len(result.pipeline_trace.steps) > 0


@pytest.mark.asyncio
async def test_pipeline_trace_has_all_steps(orchestrator):
    result = await orchestrator.process("My 7 year old won't do homework", "test_trace")
    step_names = [s.name for s in result.pipeline_trace.steps]
    assert "predicate_extraction" in step_names
    assert "safety_check" in step_names
    assert "rules_engine" in step_names
    assert "response_generation" in step_names
    assert "response_validation" in step_names


@pytest.mark.asyncio
async def test_safety_override_mid_conversation(orchestrator):
    # Normal message first
    await orchestrator.process("Hi there", "safety_test")
    # Crisis message
    result = await orchestrator.process("I'm worried about harm to my child", "safety_test")
    assert result.agent_used == "safety"
    assert "988" in result.response


@pytest.mark.asyncio
async def test_out_of_scope_deflection(orchestrator):
    result = await orchestrator.process("Should I try medication for ADHD?", "scope_test")
    assert result.agent_used == "safety"
    assert "healthcare provider" in result.response.lower() or "outside" in result.response.lower()


@pytest.mark.asyncio
async def test_session_state_isolation(orchestrator):
    await orchestrator.process("Hi", "session_a")
    await orchestrator.process("Hi", "session_b")
    state_a = orchestrator.get_session("session_a")
    state_b = orchestrator.get_session("session_b")
    assert state_a.session_id != state_b.session_id


@pytest.mark.asyncio
async def test_conversation_history_recorded(orchestrator):
    await orchestrator.process("Hello", "history_test")
    await orchestrator.process("My child is 8 years old", "history_test")
    state = orchestrator.get_session("history_test")
    assert len(state.conversation_history) == 2
    assert state.conversation_history[0]["user_message"] == "Hello"


@pytest.mark.asyncio
async def test_pipeline_timing(orchestrator):
    result = await orchestrator.process("My son won't do homework", "timing_test")
    assert result.pipeline_trace.total_duration_ms > 0
    for step in result.pipeline_trace.steps:
        assert step.duration_ms >= 0


def test_format_rag_context_xml_structure():
    """RAG context should use structured XML with metadata."""
    from app.models.schemas import RetrievalResult

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    results = [
        RetrievalResult(
            document_id="visual_schedule",
            document_name="Visual Schedule",
            content="Create a visual chart showing daily activities.",
            score=0.92,
            match_type="vector+tag",
            source="AAP Guidelines",
            tags=["daily_routines"],
            evidence_level="strong",
            document_type="strategy",
        ),
    ]
    ctx = orch._format_rag_context(results)
    assert "<retrieval_results" in ctx
    assert 'evidence_level="strong"' in ctx
    assert 'score="0.92"' in ctx
    assert "<system_instruction>" in ctx
    assert "<name>Visual Schedule</name>" in ctx
    assert "<source>AAP Guidelines</source>" in ctx
    assert "<tags>daily_routines</tags>" in ctx


def test_format_rag_context_empty():
    """Empty results should return empty string."""
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    assert orch._format_rag_context([]) == ""


# --- Context window tests ---

from app.agents.context import format_conversation_window


def test_format_conversation_window_empty():
    assert format_conversation_window([]) == ""


def test_format_conversation_window_formats_turns():
    history = [
        {"user_message": "Hello", "agent_response": "Hi there!"},
        {"user_message": "My son is 7", "agent_response": "Thank you for sharing."},
    ]
    result = format_conversation_window(history, max_turns=5)
    assert "Parent: Hello" in result
    assert "Coach: Hi there!" in result
    assert "Parent: My son is 7" in result


def test_format_conversation_window_respects_max_turns():
    history = [
        {"user_message": f"msg {i}", "agent_response": f"resp {i}"}
        for i in range(10)
    ]
    result = format_conversation_window(history, max_turns=3)
    assert "msg 7" in result
    assert "msg 9" in result
    assert "msg 0" not in result


# --- Conversation history injection tests ---

from tests.conftest import MockGeminiClient
from app.models.schemas import SeedSessionRequest


@pytest.mark.asyncio
async def test_strategy_prompt_includes_conversation_history():
    """Response generation should include formatted conversation history."""
    mock = MockGeminiClient()
    store = KnowledgeStore()
    bm25 = BM25Index()
    bm25.build(store.chunks)
    orchestrator = AgentOrchestrator(
        extractor=PredicateExtractor(gemini_client=None),
        safety=SafetyMonitor(gemini_client=None),
        rules_engine=PythonRulesEngine(),
        retriever=HybridRetriever(knowledge_store=store, bm25_index=bm25, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=mock),
        progress=ProgressAgent(gemini_client=None),
    )
    orchestrator.seed_session(SeedSessionRequest(
        session_id="hist_test", child_name="Jamie", child_age="7",
        challenges=["homework"], goals=["finish homework"]
    ))
    await orchestrator.process("My son won't do homework", "hist_test")
    await orchestrator.process("What about a timer technique?", "hist_test")
    assert len(mock.generate_calls) >= 1
    last_prompt = mock.generate_calls[-1]
    assert "Parent:" in last_prompt
    assert "Coach:" in last_prompt


@pytest.mark.asyncio
async def test_strategy_prompt_includes_child_name():
    """Strategy response prompt should include the child's name from profile."""
    mock = MockGeminiClient()
    store = KnowledgeStore()
    bm25 = BM25Index()
    bm25.build(store.chunks)
    orchestrator = AgentOrchestrator(
        extractor=PredicateExtractor(gemini_client=None),
        safety=SafetyMonitor(gemini_client=None),
        rules_engine=PythonRulesEngine(),
        retriever=HybridRetriever(knowledge_store=store, bm25_index=bm25, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=mock),
        progress=ProgressAgent(gemini_client=None),
    )
    orchestrator.seed_session(SeedSessionRequest(
        session_id="name_test", child_name="Jamie", child_age="7",
        challenges=["homework"], goals=["finish homework"]
    ))
    await orchestrator.process("What strategies work for homework?", "name_test")
    prompt = mock.generate_calls[-1]
    assert "Jamie" in prompt


@pytest.mark.asyncio
async def test_strategy_prompt_fallback_when_no_child_name():
    """Without child_name, prompt should use 'your child' fallback."""
    mock = MockGeminiClient()
    store = KnowledgeStore()
    bm25 = BM25Index()
    bm25.build(store.chunks)
    orchestrator = AgentOrchestrator(
        extractor=PredicateExtractor(gemini_client=None),
        safety=SafetyMonitor(gemini_client=None),
        rules_engine=PythonRulesEngine(),
        retriever=HybridRetriever(knowledge_store=store, bm25_index=bm25, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=mock),
        progress=ProgressAgent(gemini_client=None),
    )
    orchestrator.seed_session(SeedSessionRequest(
        session_id="noname_test", child_age="7",
        challenges=["homework"], goals=["finish homework"]
    ))
    await orchestrator.process("What strategies work for homework?", "noname_test")
    prompt = mock.generate_calls[-1]
    assert "your child" in prompt

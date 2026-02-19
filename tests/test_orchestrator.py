"""Integration tests for the orchestrator pipeline."""

import pytest
from unittest.mock import AsyncMock

from app.agents.intake import IntakeAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.progress import ProgressAgent
from app.agents.strategy import StrategyAgent
from app.guardrails.validator import GuardrailsValidator
from app.models.schemas import GuardrailsError, InputCheckResult, OutputCheckResult
from app.phase_manager import PhaseManager
from app.rag.knowledge_store import KnowledgeStore
from app.rag.retriever import HybridRetriever


def _make_mock_guardrails(
    input_allowed=True,
    input_reason=None,
    input_response=None,
    output_valid=True,
    output_violation=None,
):
    """Create a mock GuardrailsValidator."""
    guardrails = AsyncMock(spec=GuardrailsValidator)
    guardrails.check_input = AsyncMock(return_value=InputCheckResult(
        is_allowed=input_allowed,
        blocked_reason=input_reason,
        override_response=input_response,
    ))
    guardrails.check_output = AsyncMock(return_value=OutputCheckResult(
        is_valid=output_valid,
        violation_type=output_violation,
    ))
    return guardrails


@pytest.fixture
def orchestrator():
    """Full pipeline orchestrator with mock guardrails, agents use fallbacks."""
    store = KnowledgeStore()

    return AgentOrchestrator(
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
    assert "input_rails" in step_names
    assert "phase_manager" in step_names
    assert "response_generation" in step_names
    assert "output_rails" in step_names


@pytest.mark.asyncio
async def test_input_rails_block_crisis():
    """Crisis messages should be blocked by input rails with override response."""
    store = KnowledgeStore()

    orch = AgentOrchestrator(
        guardrails=_make_mock_guardrails(
            input_allowed=False,
            input_reason="crisis",
            input_response="Call 988 for help.",
        ),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )
    result = await orch.process("I'm worried about harm", "crisis_test")
    assert result.agent_used == "guardrails"
    assert result.response == "Call 988 for help."


@pytest.mark.asyncio
async def test_input_rails_block_out_of_scope():
    """Out-of-scope messages should be deflected."""
    store = KnowledgeStore()

    orch = AgentOrchestrator(
        guardrails=_make_mock_guardrails(
            input_allowed=False,
            input_reason="out_of_scope",
            input_response="Ask your healthcare provider.",
        ),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )
    result = await orch.process("Should I try medication?", "scope_test")
    assert result.agent_used == "guardrails"


@pytest.mark.asyncio
async def test_output_rails_block_violation():
    """Output rails should replace response when violation detected."""
    store = KnowledgeStore()

    orch = AgentOrchestrator(
        guardrails=_make_mock_guardrails(output_valid=False, output_violation="medication"),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )
    result = await orch.process("Hi", "output_test")
    assert "helpful, appropriate guidance" in result.response


@pytest.mark.asyncio
async def test_guardrails_error_doesnt_crash_pipeline():
    """GuardrailsError in input rails should not crash — pipeline continues."""
    store = KnowledgeStore()


    guardrails = AsyncMock(spec=GuardrailsValidator)
    guardrails.check_input = AsyncMock(side_effect=GuardrailsError("NeMo down"))
    guardrails.check_output = AsyncMock(return_value=OutputCheckResult(is_valid=True))

    orch = AgentOrchestrator(
        guardrails=guardrails,
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )
    result = await orch.process("Hi there", "error_test")
    assert result.agent_used == "intake"
    assert result.response != ""


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

    orchestrator = AgentOrchestrator(
        guardrails=_make_mock_guardrails(),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
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

    orchestrator = AgentOrchestrator(
        guardrails=_make_mock_guardrails(),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
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

    orchestrator = AgentOrchestrator(
        guardrails=_make_mock_guardrails(),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
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


# --- Progressive profiling tests ---


def _make_profiling_orchestrator():
    """Helper to build orchestrator for profiling tests."""
    store = KnowledgeStore()

    return AgentOrchestrator(
        guardrails=_make_mock_guardrails(),
        phase_manager=PhaseManager(),
        retriever=HybridRetriever(knowledge_store=store, gemini_client=None),
        intake=IntakeAgent(gemini_client=None),
        strategy=StrategyAgent(gemini_client=None),
        progress=ProgressAgent(gemini_client=None),
    )


@pytest.mark.asyncio
async def test_progressive_profiling_updates_child_age():
    """Profile should learn child_age from conversation message."""
    orch = _make_profiling_orchestrator()
    orch.seed_session(SeedSessionRequest(
        session_id="profile_test", challenges=["homework"], goals=["finish homework"]
    ))
    state = orch.get_session("profile_test")
    assert state.family_profile.child_age is None
    await orch.process("My 8 year old just can't focus", "profile_test")
    assert state.family_profile.child_age == "8"


@pytest.mark.asyncio
async def test_progressive_profiling_adds_new_challenges():
    """Profile should accumulate new challenge areas from message keywords."""
    orch = _make_profiling_orchestrator()
    orch.seed_session(SeedSessionRequest(
        session_id="challenge_test", challenges=["homework"], goals=["finish homework"]
    ))
    state = orch.get_session("challenge_test")
    assert "emotion" not in state.family_profile.challenge_areas
    await orch.process("He also has terrible meltdowns and tantrums", "challenge_test")
    assert "emotion" in state.family_profile.challenge_areas


@pytest.mark.asyncio
async def test_progressive_profiling_does_not_overwrite_existing_age():
    """Profile should not overwrite existing child_age."""
    orch = _make_profiling_orchestrator()
    orch.seed_session(SeedSessionRequest(
        session_id="no_overwrite", child_age="7",
        challenges=["homework"], goals=["finish homework"]
    ))
    state = orch.get_session("no_overwrite")
    assert state.family_profile.child_age == "7"
    await orch.process("My 9 year old nephew also has ADHD", "no_overwrite")
    assert state.family_profile.child_age == "7"

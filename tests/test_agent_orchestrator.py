"""Tests for AgentOrchestrator — session management and phase inference."""

import json
from unittest.mock import AsyncMock

import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.prompts import SAFE_OUTPUT_FALLBACK
from app.agent.session_store import SessionStateStore
from app.models.schemas import ConversationPhase, SeedSessionRequest


class TestPhaseInference:

    def setup_method(self):
        self.store = SessionStateStore()
        self.orchestrator = AgentOrchestrator(agent=None, session_store=self.store)

    def test_new_session_is_intake(self):
        assert self.orchestrator._infer_phase("new") == ConversationPhase.intake

    def test_profile_data_means_strategy(self):
        self.store.update_profile("p1", child_age="7", challenge_areas=["homework"])
        assert self.orchestrator._infer_phase("p1") == ConversationPhase.strategy

    def test_active_strategies_means_strategy(self):
        state = self.store.get("s1")
        state.active_strategies = ["visual timer"]
        assert self.orchestrator._infer_phase("s1") == ConversationPhase.strategy

    def test_outcomes_mean_progress(self):
        self.store.add_outcome("o1", "timer", "positive")
        assert self.orchestrator._infer_phase("o1") == ConversationPhase.progress


class TestSessionManagement:

    def setup_method(self):
        self.store = SessionStateStore()
        self.orchestrator = AgentOrchestrator(agent=None, session_store=self.store)

    def test_get_session(self):
        state = self.orchestrator.get_session("test_session")
        assert state.session_id == "test_session"

    def test_seed_session(self):
        request = SeedSessionRequest(
            session_id="seeded",
            child_name="Kai",
            child_age="7",
            challenges=["homework"],
            goals=["Better homework routine"],
        )
        self.orchestrator.seed_session(request)
        state = self.orchestrator.get_session("seeded")
        assert state.family_profile.child_name == "Kai"
        assert len(state.goals) == 1


def _make_output_gate(*, is_valid: bool, violation_type: str | None = None):
    """Create a mock OutputGate returning a fixed result."""
    from app.guardrails.validator import OutputGate
    from app.models.schemas import OutputCheckResult

    gate = AsyncMock(spec=OutputGate)
    gate.check = AsyncMock(return_value=OutputCheckResult(
        is_valid=is_valid,
        violation_type=violation_type,
    ))
    return gate


class TestGateCheckOnFallbacks:
    """Verify _gate_check runs the output gate on fallback-synthesized responses."""

    def setup_method(self):
        self.store = SessionStateStore()

    @pytest.mark.asyncio
    async def test_gate_check_passes_safe_response(self):
        gate = _make_output_gate(is_valid=True)
        orch = AgentOrchestrator(agent=None, session_store=self.store, output_gate=gate)
        result = await orch._gate_check("Try a visual timer.")
        assert result == "Try a visual timer."
        gate.check.assert_awaited_once_with("Try a visual timer.")

    @pytest.mark.asyncio
    async def test_gate_check_replaces_violation_with_fallback(self):
        gate = _make_output_gate(is_valid=False, violation_type="medication")
        orch = AgentOrchestrator(agent=None, session_store=self.store, output_gate=gate)
        result = await orch._gate_check("You should try Adderall.")
        assert result == SAFE_OUTPUT_FALLBACK

    @pytest.mark.asyncio
    async def test_gate_check_noop_without_gate(self):
        orch = AgentOrchestrator(agent=None, session_store=self.store, output_gate=None)
        result = await orch._gate_check("anything")
        assert result == "anything"

    @pytest.mark.asyncio
    async def test_gate_check_allows_on_exception(self):
        gate = _make_output_gate(is_valid=True)
        gate.check = AsyncMock(side_effect=RuntimeError("API down"))
        orch = AgentOrchestrator(agent=None, session_store=self.store, output_gate=gate)
        result = await orch._gate_check("some response")
        assert result == "some response"

    @pytest.mark.asyncio
    async def test_synthesize_from_tool_results_gated(self):
        """_synthesize_from_tool_results should run generated text through the output gate."""
        gate = _make_output_gate(is_valid=False, violation_type="medication")
        gemini = AsyncMock()
        gemini.generate = AsyncMock(return_value="Take Ritalin daily.")
        orch = AgentOrchestrator(
            agent=None, session_store=self.store,
            gemini_client=gemini, output_gate=gate,
        )
        # Ensure session exists with a profile
        self.store.get("s1")
        result = await orch._synthesize_from_tool_results("help my child", ["[1] strategy..."], "s1")
        assert result == SAFE_OUTPUT_FALLBACK

    @pytest.mark.asyncio
    async def test_search_and_synthesize_gated(self):
        """_search_and_synthesize should run generated text through the output gate."""
        gate = _make_output_gate(is_valid=False, violation_type="diagnosis")
        gemini = AsyncMock()
        gemini.generate = AsyncMock(return_value="Your child has ADHD.")
        orch = AgentOrchestrator(
            agent=None, session_store=self.store,
            gemini_client=gemini, output_gate=gate,
        )
        self.store.get("s1")
        result = await orch._search_and_synthesize("is my child ok", "s1")
        assert result == SAFE_OUTPUT_FALLBACK

    @pytest.mark.asyncio
    async def test_synthesize_acknowledgment_gated(self):
        """_synthesize_acknowledgment should run generated text through the output gate."""
        gate = _make_output_gate(is_valid=False, violation_type="scope")
        gemini = AsyncMock()
        gemini.generate = AsyncMock(return_value="Here's legal advice.")
        orch = AgentOrchestrator(
            agent=None, session_store=self.store,
            gemini_client=gemini, output_gate=gate,
        )
        self.store.get("s1")
        tool_calls = [{"name": "update_family_profile", "args": {}}]
        result = await orch._synthesize_acknowledgment("my child is 7", tool_calls, [], "s1")
        assert result == SAFE_OUTPUT_FALLBACK

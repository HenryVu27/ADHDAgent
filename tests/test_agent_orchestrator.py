"""Tests for AgentOrchestrator — session management and phase inference."""

import pytest

from app.agent.orchestrator import AgentOrchestrator
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

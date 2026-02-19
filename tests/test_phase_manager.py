"""Tests for the PhaseManager."""

import pytest

from app.models.schemas import (
    ConversationPhase,
    FamilyProfile,
    SessionState,
)
from app.phase_manager import PhaseManager


@pytest.fixture
def pm():
    return PhaseManager()


@pytest.fixture
def intake_session():
    return SessionState(session_id="test", phase=ConversationPhase.intake)


@pytest.fixture
def strategy_session():
    return SessionState(
        session_id="test",
        phase=ConversationPhase.strategy,
        family_profile=FamilyProfile(
            child_age="7",
            challenge_areas=["homework"],
            attempted_strategies=["timeout"],
        ),
        active_strategies=["timer_technique"],
    )


@pytest.fixture
def progress_session():
    return SessionState(
        session_id="test",
        phase=ConversationPhase.progress,
        active_strategies=["timer_technique"],
    )


class TestPhaseRouting:
    def test_routes_intake_to_intake_agent(self, pm, intake_session):
        decision = pm.decide("Hello", intake_session)
        assert decision.agent == "intake"
        assert decision.phase == ConversationPhase.intake

    def test_routes_strategy_to_strategy_agent(self, pm, strategy_session):
        decision = pm.decide("Help with homework", strategy_session)
        assert decision.agent == "strategy"
        assert decision.phase == ConversationPhase.strategy

    def test_routes_progress_to_progress_agent(self, pm, progress_session):
        decision = pm.decide("How are things going", progress_session)
        assert decision.agent == "progress"

    def test_routes_followup_to_strategy_agent(self, pm):
        state = SessionState(session_id="test", phase=ConversationPhase.followup)
        decision = pm.decide("Checking in", state)
        assert decision.agent == "strategy"


class TestPhaseTransitions:
    def test_transitions_intake_to_strategy(self, pm):
        state = SessionState(
            session_id="test",
            phase=ConversationPhase.intake,
            family_profile=FamilyProfile(
                child_age="7",
                challenge_areas=["homework"],
                attempted_strategies=["timeout"],
            ),
        )
        decision = pm.decide("My child needs help", state)
        assert decision.phase == ConversationPhase.strategy
        assert decision.agent == "strategy"
        assert decision.phase_changed is True
        assert "transition_to_strategy" in decision.directives

    def test_requires_profile_for_transition(self, pm, intake_session):
        decision = pm.decide("Hello", intake_session)
        assert decision.phase == ConversationPhase.intake
        assert decision.phase_changed is False

    def test_no_transition_without_child_age(self, pm):
        state = SessionState(
            session_id="test",
            phase=ConversationPhase.intake,
            family_profile=FamilyProfile(
                challenge_areas=["homework"],
                attempted_strategies=["timeout"],
            ),
        )
        decision = pm.decide("Help me", state)
        assert decision.phase == ConversationPhase.intake

    def test_strategy_to_progress_on_progress_signals(self, pm, strategy_session):
        decision = pm.decide("I want to check on our progress", strategy_session)
        assert decision.phase == ConversationPhase.progress
        assert decision.phase_changed is True

    def test_no_strategy_to_progress_without_active_strategies(self, pm):
        state = SessionState(
            session_id="test",
            phase=ConversationPhase.strategy,
            active_strategies=[],  # no active strategies
        )
        decision = pm.decide("How is progress going?", state)
        assert decision.phase == ConversationPhase.strategy

    def test_progress_to_strategy_on_new_strategy_request(self, pm, progress_session):
        decision = pm.decide("I want to try something different", progress_session)
        assert decision.phase == ConversationPhase.strategy
        assert decision.phase_changed is True


class TestDirectivesAndConstraints:
    def test_generates_gather_info_for_intake(self, pm, intake_session):
        decision = pm.decide("Hello", intake_session)
        assert "gather_info" in decision.directives

    def test_generates_recommend_strategy(self, pm, strategy_session):
        decision = pm.decide("Help with homework", strategy_session)
        assert "recommend_strategy" in decision.directives
        assert "must_have_action_steps" in decision.constraints

    def test_always_empathetic(self, pm, intake_session):
        decision = pm.decide("Hello", intake_session)
        assert "tone_empathetic" in decision.constraints

    def test_frustration_constraint(self, pm, intake_session):
        decision = pm.decide("I'm so frustrated with everything", intake_session)
        assert "must_acknowledge_frustration" in decision.constraints

    def test_track_progress_directive(self, pm, progress_session):
        decision = pm.decide("Let me check in", progress_session)
        assert "track_progress" in decision.directives


class TestIntakeCompletion:
    def test_incomplete_without_age(self, pm):
        state = SessionState(
            session_id="test",
            family_profile=FamilyProfile(
                challenge_areas=["homework"],
                attempted_strategies=["timeout"],
            ),
        )
        assert pm.check_intake_complete(state) is False

    def test_incomplete_without_challenges(self, pm):
        state = SessionState(
            session_id="test",
            family_profile=FamilyProfile(
                child_age="7",
                attempted_strategies=["timeout"],
            ),
        )
        assert pm.check_intake_complete(state) is False

    def test_incomplete_without_strategies(self, pm):
        state = SessionState(
            session_id="test",
            family_profile=FamilyProfile(
                child_age="7",
                challenge_areas=["homework"],
            ),
        )
        assert pm.check_intake_complete(state) is False

    def test_complete_with_all_fields(self, pm):
        state = SessionState(
            session_id="test",
            family_profile=FamilyProfile(
                child_age="7",
                challenge_areas=["homework"],
                attempted_strategies=["timeout"],
            ),
        )
        assert pm.check_intake_complete(state) is True


class TestUpdateProfile:
    def test_update_profile_extracts_age(self, pm):
        state = SessionState(session_id="test")
        pm.update_profile("My 8 year old won't listen", state)
        assert state.family_profile.child_age == "8"

    def test_update_profile_extracts_challenges(self, pm):
        state = SessionState(session_id="test")
        pm.update_profile("He has terrible meltdowns at homework time", state)
        assert "emotion" in state.family_profile.challenge_areas
        assert "homework" in state.family_profile.challenge_areas

    def test_update_profile_does_not_overwrite_age(self, pm):
        state = SessionState(
            session_id="test",
            family_profile=FamilyProfile(child_age="7"),
        )
        pm.update_profile("My 9 year old nephew has ADHD too", state)
        assert state.family_profile.child_age == "7"

    def test_update_profile_extracts_situations(self, pm):
        state = SessionState(session_id="test")
        pm.update_profile("Bedtime is the hardest part of our day", state)
        assert "bedtime" in state.family_profile.hardest_situations

"""Tests for AgentOrchestrator — session management and phase inference."""

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
        self.store.add_active_strategy("s1", "visual timer")
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


class TestBuildToolCallsSummary:

    def test_search_knowledge_base(self):
        tool_calls = [{"name": "search_knowledge_base", "args": {"query": "homework strategies"}}]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert result == 'search_knowledge_base(query="homework strategies")'

    def test_update_family_profile(self):
        tool_calls = [{"name": "update_family_profile", "args": {"child_name": "Kai", "child_age": "7"}}]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert result == "update_family_profile(child_name, child_age)"

    def test_track_outcome(self):
        tool_calls = [{"name": "track_outcome", "args": {"strategy_name": "visual timer", "outcome": "positive"}}]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert result == "track_outcome(visual timer: positive)"

    def test_manage_goals(self):
        tool_calls = [{"name": "manage_goals", "args": {"action": "add", "description": "Homework by 6pm"}}]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert result == "manage_goals(add: Homework by 6pm)"

    def test_multiple_tools(self):
        tool_calls = [
            {"name": "search_knowledge_base", "args": {"query": "bedtime"}},
            {"name": "update_family_profile", "args": {"child_name": "Kai"}},
        ]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert "search_knowledge_base" in result
        assert "update_family_profile" in result
        assert "; " in result

    def test_empty_list(self):
        assert AgentOrchestrator._build_tool_calls_summary([]) == ""

    def test_unknown_tool(self):
        tool_calls = [{"name": "some_new_tool", "args": {}}]
        result = AgentOrchestrator._build_tool_calls_summary(tool_calls)
        assert result == "some_new_tool"

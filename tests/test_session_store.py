"""Tests for SessionStateStore — profile, outcomes, goals, seeding."""

import pytest

from app.agent.session_store import SessionStateStore
from app.models.schemas import SeedSessionRequest


class TestSessionStateStore:

    def setup_method(self):
        self.store = SessionStateStore()

    def test_get_creates_new_session(self):
        state = self.store.get("new_session")
        assert state.session_id == "new_session"
        assert state.turn_count == 0
        assert state.family_profile.child_name is None

    def test_get_returns_independent_copy(self):
        self.store.increment_turn("s1")
        s1 = self.store.get("s1")
        assert s1.turn_count == 1
        # Mutating the copy should not affect internal state
        s1.turn_count = 999
        s2 = self.store.get("s1")
        assert s2.turn_count == 1  # still 1, not 999

    def test_get_returns_different_objects(self):
        s1 = self.store.get("s1")
        s2 = self.store.get("s1")
        assert s1 is not s2

    def test_increment_turn(self):
        assert self.store.increment_turn("t1") == 1
        assert self.store.increment_turn("t1") == 2
        assert self.store.increment_turn("t1") == 3

    def test_update_profile_scalar_fields(self):
        profile = self.store.update_profile("p1", child_name="Kai", child_age="7")
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    def test_update_profile_list_append(self):
        self.store.update_profile("p1", challenge_areas=["homework"])
        profile = self.store.update_profile("p1", challenge_areas=["bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    def test_update_profile_list_dedup(self):
        self.store.update_profile("p1", challenge_areas=["homework"])
        profile = self.store.update_profile("p1", challenge_areas=["homework", "bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    def test_update_profile_ignores_none(self):
        self.store.update_profile("p1", child_name="Kai")
        profile = self.store.update_profile("p1", child_name=None, child_age="8")
        assert profile.child_name == "Kai"
        assert profile.child_age == "8"

    def test_add_outcome(self):
        outcome = self.store.add_outcome("o1", "visual timer", "positive", "worked great")
        assert outcome.goal_description == "visual timer"
        assert outcome.signal == "positive"
        assert outcome.detail == "worked great"

        state = self.store.get("o1")
        assert len(state.outcomes) == 1

    def test_manage_goal_add(self):
        goals = self.store.manage_goal("g1", "add", "Homework done by 6pm")
        assert len(goals) == 1
        assert goals[0].description == "Homework done by 6pm"
        assert goals[0].status == "active"

    def test_manage_goal_complete(self):
        self.store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = self.store.manage_goal("g1", "complete", "Homework done by 6pm")
        assert goals[0].status == "completed"

    def test_manage_goal_complete_case_insensitive(self):
        self.store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = self.store.manage_goal("g1", "complete", "homework done by 6pm")
        assert goals[0].status == "completed"

    def test_manage_goal_list(self):
        self.store.manage_goal("g1", "add", "Goal A")
        self.store.manage_goal("g1", "add", "Goal B")
        goals = self.store.manage_goal("g1", "list")
        assert len(goals) == 2

    def test_seed_session(self):
        request = SeedSessionRequest(
            session_id="seed1",
            child_name="Kai",
            child_age="8",
            challenges=["homework", "bedtime"],
            tried_strategies=["timer"],
            goals=["Homework by 6pm"],
        )
        self.store.seed_session(request)

        state = self.store.get("seed1")
        assert state.family_profile.child_name == "Kai"
        assert state.family_profile.child_age == "8"
        assert state.family_profile.challenge_areas == ["homework", "bedtime"]
        assert state.family_profile.attempted_strategies == ["timer"]
        assert len(state.goals) == 1
        assert state.goals[0].description == "Homework by 6pm"

    def test_sessions_isolated(self):
        self.store.update_profile("s1", child_name="Kai")
        self.store.update_profile("s2", child_name="Alex")
        assert self.store.get("s1").family_profile.child_name == "Kai"
        assert self.store.get("s2").family_profile.child_name == "Alex"

    def test_tool_calls_summary_stored_and_retrieved(self):
        self.store.add_message("s1", "user", "Help", turn=1)
        self.store.add_message(
            "s1", "assistant", "Sure!", turn=1,
            tool_calls_summary='search_knowledge_base(query="homework")',
        )
        messages = self.store.get_messages("s1")
        assert messages[0].get("tool_calls_summary", "") == ""
        assert messages[1]["tool_calls_summary"] == 'search_knowledge_base(query="homework")'

    def test_tool_calls_summary_default_empty(self):
        self.store.add_message("s1", "assistant", "Hi", turn=1)
        messages = self.store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""

    def test_session_exists_false_before_creation(self):
        assert self.store.session_exists("nonexistent") is False

    def test_session_exists_true_after_get(self):
        self.store.get("new_session")
        assert self.store.session_exists("new_session") is True

    def test_get_all_sessions_paginated(self):
        for i in range(5):
            self.store.get(f"s{i}")
        items, total = self.store.get_all_sessions_paginated(offset=0, limit=3)
        assert total == 5
        assert len(items) == 3

    def test_get_all_sessions_paginated_offset(self):
        for i in range(5):
            self.store.get(f"s{i}")
        items, total = self.store.get_all_sessions_paginated(offset=3, limit=10)
        assert total == 5
        assert len(items) == 2

    def test_get_messages_paginated(self):
        for i in range(5):
            self.store.add_message("s1", "user", f"msg {i}", turn=i)
        messages, total = self.store.get_messages_paginated("s1", offset=0, limit=3)
        assert total == 5
        assert len(messages) == 3

    def test_get_messages_paginated_offset(self):
        for i in range(5):
            self.store.add_message("s1", "user", f"msg {i}", turn=i)
        messages, total = self.store.get_messages_paginated("s1", offset=3, limit=10)
        assert total == 5
        assert len(messages) == 2

    def test_delete_session(self):
        self.store.increment_turn("del-test")
        self.store.add_message("del-test", "user", "hello", 1)
        self.store.update_profile("del-test", child_name="Test")
        self.store.delete_session("del-test")
        # get() creates a fresh empty session
        state = self.store.get("del-test")
        assert state.turn_count == 0
        assert state.family_profile.child_name is None
        assert self.store.get_messages("del-test") == []


    def test_save_and_get_tool_results(self):
        self.store.save_tool_result("tr1", "search_knowledge_base", "homework strategies", "Result: visual timer...", turn=1)
        self.store.save_tool_result("tr1", "search_knowledge_base", "bedtime routines", "Result: bedtime routine...", turn=2)
        results = self.store.get_recent_tool_results("tr1", limit=5)
        assert len(results) == 2
        assert results[0].tool_name == "search_knowledge_base"
        assert results[0].query == "homework strategies"
        assert results[0].turn == 1
        assert results[1].turn == 2

    def test_get_recent_tool_results_respects_limit(self):
        for i in range(5):
            self.store.save_tool_result("tr2", "search_knowledge_base", f"query {i}", f"result {i}", turn=i)
        results = self.store.get_recent_tool_results("tr2", limit=2)
        assert len(results) == 2
        # Should be the most recent 2
        assert results[0].turn == 3
        assert results[1].turn == 4

    def test_get_recent_tool_results_empty(self):
        results = self.store.get_recent_tool_results("nonexistent")
        assert results == []

    def test_delete_session_removes_tool_results(self):
        self.store.save_tool_result("del-tr", "search_knowledge_base", "q", "r", turn=1)
        self.store.delete_session("del-tr")
        results = self.store.get_recent_tool_results("del-tr")
        assert results == []
    def test_delete_nonexistent_session_is_noop(self):
        self.store.delete_session("does-not-exist")  # Should not raise

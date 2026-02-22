"""Tests for SQLiteSessionStore — mirrors test_session_store.py + message/blocked/strategy tests."""

import sqlite3

import pytest

from app.agent.sqlite_store import SQLiteSessionStore
from app.db import init_db
from app.models.schemas import SeedSessionRequest


def _make_store() -> SQLiteSessionStore:
    """Create a SQLiteSessionStore backed by an in-memory database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return SQLiteSessionStore(conn)


class TestSQLiteSessionStore:

    def setup_method(self):
        self.store = _make_store()

    def test_get_creates_new_session(self):
        state = self.store.get("new_session")
        assert state.session_id == "new_session"
        assert state.turn_count == 0
        assert state.family_profile.child_name is None

    def test_get_returns_consistent_data(self):
        """SQLite store materializes fresh objects; data should match what was persisted."""
        self.store.increment_turn("s1")
        self.store.increment_turn("s1")
        s = self.store.get("s1")
        assert s.turn_count == 2

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


class TestMessagePersistence:

    def setup_method(self):
        self.store = _make_store()

    def test_add_and_get_messages(self):
        self.store.add_message("s1", "user", "Hello", turn=1)
        self.store.add_message("s1", "assistant", "Hi there!", turn=1)
        messages = self.store.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_tool_calls_summary_stored_and_retrieved(self):
        self.store.add_message("s1", "user", "Help", turn=1)
        self.store.add_message(
            "s1", "assistant", "Sure!", turn=1,
            tool_calls_summary='search_knowledge_base(query="homework")',
        )
        messages = self.store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""
        assert messages[1]["tool_calls_summary"] == 'search_knowledge_base(query="homework")'

    def test_tool_calls_summary_default_empty(self):
        self.store.add_message("s1", "assistant", "Hi", turn=1)
        messages = self.store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""

    def test_tool_calls_summary_in_messages_range(self):
        self.store.add_message("s1", "assistant", "Response", turn=1, tool_calls_summary="update_family_profile(child_name)")
        msgs = self.store.get_messages_range("s1", start_turn=1)
        assert msgs[0]["tool_calls_summary"] == "update_family_profile(child_name)"

    def test_blocked_messages_recorded(self):
        self.store.add_message(
            "s1", "user", "bad message", turn=1,
            blocked=True, blocked_reason="crisis",
        )
        self.store.add_message(
            "s1", "assistant", "Please call 988.", turn=1,
            blocked=True, blocked_reason="crisis",
        )
        messages = self.store.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["blocked"] is True
        assert messages[0]["blocked_reason"] == "crisis"

    def test_get_messages_limit(self):
        for i in range(10):
            self.store.add_message("s1", "user", f"msg {i}", turn=i)
        messages = self.store.get_messages("s1", limit=3)
        assert len(messages) == 3
        # Should be the LAST 3 messages
        assert messages[0]["content"] == "msg 7"
        assert messages[2]["content"] == "msg 9"

    def test_get_excludes_blocked_in_session_state(self):
        """SessionState.conversation_history should exclude blocked messages."""
        self.store.add_message("s1", "user", "normal", turn=1)
        self.store.add_message("s1", "assistant", "reply", turn=1)
        self.store.add_message("s1", "user", "blocked", turn=2, blocked=True, blocked_reason="crisis")
        self.store.add_message("s1", "assistant", "Please call 988.", turn=2, blocked=True, blocked_reason="crisis")
        self.store.add_message("s1", "user", "ok normal again", turn=3)
        self.store.add_message("s1", "assistant", "reply 2", turn=3)

        state = self.store.get("s1")
        # conversation_history should only have non-blocked messages
        assert len(state.conversation_history) == 4
        assert all(msg["content"] != "blocked" for msg in state.conversation_history)

    def test_messages_across_sessions_isolated(self):
        self.store.add_message("s1", "user", "hello s1", turn=1)
        self.store.add_message("s2", "user", "hello s2", turn=1)
        assert len(self.store.get_messages("s1")) == 1
        assert len(self.store.get_messages("s2")) == 1


class TestActiveStrategies:

    def setup_method(self):
        self.store = _make_store()

    def test_add_and_get_strategies(self):
        self.store.add_active_strategy("s1", "visual timer")
        self.store.add_active_strategy("s1", "reward chart")
        strategies = self.store.get_active_strategies("s1")
        assert "visual timer" in strategies
        assert "reward chart" in strategies

    def test_dedup_strategies(self):
        self.store.add_active_strategy("s1", "visual timer")
        self.store.add_active_strategy("s1", "visual timer")
        strategies = self.store.get_active_strategies("s1")
        assert len(strategies) == 1

    def test_strategies_in_session_state(self):
        self.store.add_active_strategy("s1", "visual timer")
        state = self.store.get("s1")
        assert "visual timer" in state.active_strategies


class TestSchemaInit:

    def test_init_db_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "sessions" in table_names
        assert "family_profiles" in table_names
        assert "messages" in table_names
        assert "goals" in table_names
        assert "outcomes" in table_names
        assert "active_strategies" in table_names
        assert "schema_version" in table_names

    def test_init_db_idempotent(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        init_db(conn)
        init_db(conn)  # Should not raise
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version >= 1  # Current max version

"""Tests for async SQLiteSessionStore — mirrors test_session_store.py + message/blocked/strategy tests."""

import sqlite3

import pytest

from app.agent.session_store import create_in_memory_store
from app.db import init_db
from app.models.schemas import SeedSessionRequest


async def _make_store():
    """Create a SQLiteSessionStore backed by an in-memory database."""
    return await create_in_memory_store()


class TestSQLiteSessionStore:

    async def test_get_creates_new_session(self):
        store = await _make_store()
        state = await store.get("new_session")
        assert state.session_id == "new_session"
        assert state.turn_count == 0
        assert state.family_profile.child_name is None

    async def test_get_returns_consistent_data(self):
        """SQLite store materializes fresh objects; data should match what was persisted."""
        store = await _make_store()
        await store.increment_turn("s1")
        await store.increment_turn("s1")
        await store.commit()
        s = await store.get("s1")
        assert s.turn_count == 2

    async def test_increment_turn(self):
        store = await _make_store()
        assert await store.increment_turn("t1") == 1
        assert await store.increment_turn("t1") == 2
        assert await store.increment_turn("t1") == 3

    async def test_update_profile_scalar_fields(self):
        store = await _make_store()
        profile = await store.update_profile("p1", child_name="Kai", child_age="7")
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    async def test_update_profile_list_append(self):
        store = await _make_store()
        await store.update_profile("p1", challenge_areas=["homework"])
        profile = await store.update_profile("p1", challenge_areas=["bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    async def test_update_profile_list_dedup(self):
        store = await _make_store()
        await store.update_profile("p1", challenge_areas=["homework"])
        profile = await store.update_profile("p1", challenge_areas=["homework", "bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    async def test_update_profile_ignores_none(self):
        store = await _make_store()
        await store.update_profile("p1", child_name="Kai")
        profile = await store.update_profile("p1", child_name=None, child_age="8")
        assert profile.child_name == "Kai"
        assert profile.child_age == "8"

    async def test_add_outcome(self):
        store = await _make_store()
        outcome = await store.add_outcome("o1", "visual timer", "positive", "worked great")
        assert outcome.strategy_name == "visual timer"
        assert outcome.signal == "positive"
        assert outcome.detail == "worked great"

        await store.commit()
        state = await store.get("o1")
        assert len(state.outcomes) == 1

    async def test_manage_goal_add(self):
        store = await _make_store()
        goals = await store.manage_goal("g1", "add", "Homework done by 6pm")
        assert len(goals) == 1
        assert goals[0].description == "Homework done by 6pm"
        assert goals[0].status == "active"

    async def test_manage_goal_complete(self):
        store = await _make_store()
        await store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = await store.manage_goal("g1", "complete", "Homework done by 6pm")
        assert goals[0].status == "completed"

    async def test_manage_goal_complete_case_insensitive(self):
        store = await _make_store()
        await store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = await store.manage_goal("g1", "complete", "homework done by 6pm")
        assert goals[0].status == "completed"

    async def test_manage_goal_list(self):
        store = await _make_store()
        await store.manage_goal("g1", "add", "Goal A")
        await store.manage_goal("g1", "add", "Goal B")
        goals = await store.manage_goal("g1", "list")
        assert len(goals) == 2

    async def test_seed_session(self):
        store = await _make_store()
        request = SeedSessionRequest(
            session_id="seed1",
            child_name="Kai",
            child_age="8",
            challenges=["homework", "bedtime"],
            tried_strategies=["timer"],
            goals=["Homework by 6pm"],
        )
        await store.seed_session(request)

        state = await store.get("seed1")
        assert state.family_profile.child_name == "Kai"
        assert state.family_profile.child_age == "8"
        assert state.family_profile.challenge_areas == ["homework", "bedtime"]
        assert state.family_profile.attempted_strategies == ["timer"]
        assert len(state.goals) == 1
        assert state.goals[0].description == "Homework by 6pm"

    async def test_sessions_isolated(self):
        store = await _make_store()
        await store.update_profile("s1", child_name="Kai")
        await store.update_profile("s2", child_name="Alex")
        await store.commit()
        assert (await store.get("s1")).family_profile.child_name == "Kai"
        assert (await store.get("s2")).family_profile.child_name == "Alex"

    async def test_session_exists_false_before_creation(self):
        store = await _make_store()
        assert await store.session_exists("nonexistent") is False

    async def test_session_exists_true_after_get(self):
        store = await _make_store()
        await store.get("new_session")
        await store.commit()
        assert await store.session_exists("new_session") is True

    async def test_get_all_sessions_paginated(self):
        store = await _make_store()
        for i in range(5):
            await store.get(f"s{i}")
        await store.commit()
        items, total = await store.get_all_sessions_paginated(offset=0, limit=3)
        assert total == 5
        assert len(items) == 3

    async def test_get_all_sessions_paginated_offset(self):
        store = await _make_store()
        for i in range(5):
            await store.get(f"s{i}")
        await store.commit()
        items, total = await store.get_all_sessions_paginated(offset=3, limit=10)
        assert total == 5
        assert len(items) == 2

    async def test_get_messages_paginated(self):
        store = await _make_store()
        for i in range(5):
            await store.add_message("s1", "user", f"msg {i}", turn=i)
        await store.commit()
        messages, total = await store.get_messages_paginated("s1", offset=0, limit=3)
        assert total == 5
        assert len(messages) == 3

    async def test_get_messages_paginated_offset(self):
        store = await _make_store()
        for i in range(5):
            await store.add_message("s1", "user", f"msg {i}", turn=i)
        await store.commit()
        messages, total = await store.get_messages_paginated("s1", offset=3, limit=10)
        assert total == 5
        assert len(messages) == 2

    async def test_delete_session(self):
        store = await _make_store()
        await store.increment_turn("del-test")
        await store.add_message("del-test", "user", "hello", 1)
        await store.update_profile("del-test", child_name="Test")
        await store.commit()
        await store.delete_session("del-test")
        # get() creates a fresh empty session
        state = await store.get("del-test")
        assert state.turn_count == 0
        assert state.family_profile.child_name is None
        assert await store.get_messages("del-test") == []

    async def test_save_and_get_tool_results(self):
        store = await _make_store()
        await store.save_tool_result("tr1", "search_knowledge_base", "homework strategies", "Result: visual timer...", turn=1)
        await store.save_tool_result("tr1", "search_knowledge_base", "bedtime routines", "Result: bedtime routine...", turn=2)
        await store.commit()
        results = await store.get_recent_tool_results("tr1", limit=5)
        assert len(results) == 2
        assert results[0].tool_name == "search_knowledge_base"
        assert results[0].query == "homework strategies"
        assert results[0].turn == 1
        assert results[1].turn == 2

    async def test_get_recent_tool_results_respects_limit(self):
        store = await _make_store()
        for i in range(5):
            await store.save_tool_result("tr2", "search_knowledge_base", f"query {i}", f"result {i}", turn=i)
        await store.commit()
        results = await store.get_recent_tool_results("tr2", limit=2)
        assert len(results) == 2
        assert results[0].turn == 3
        assert results[1].turn == 4

    async def test_get_recent_tool_results_empty(self):
        store = await _make_store()
        results = await store.get_recent_tool_results("nonexistent")
        assert results == []

    async def test_delete_session_removes_tool_results(self):
        store = await _make_store()
        await store.save_tool_result("del-tr", "search_knowledge_base", "q", "r", turn=1)
        await store.commit()
        await store.delete_session("del-tr")
        results = await store.get_recent_tool_results("del-tr")
        assert results == []

    async def test_delete_nonexistent_session_is_noop(self):
        store = await _make_store()
        await store.delete_session("does-not-exist")  # Should not raise


class TestMessagePersistence:

    async def test_add_and_get_messages(self):
        store = await _make_store()
        await store.add_message("s1", "user", "Hello", turn=1)
        await store.add_message("s1", "assistant", "Hi there!", turn=1)
        await store.commit()
        messages = await store.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    async def test_tool_calls_summary_stored_and_retrieved(self):
        store = await _make_store()
        await store.add_message("s1", "user", "Help", turn=1)
        await store.add_message(
            "s1", "assistant", "Sure!", turn=1,
            tool_calls_summary='search_knowledge_base(query="homework")',
        )
        await store.commit()
        messages = await store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""
        assert messages[1]["tool_calls_summary"] == 'search_knowledge_base(query="homework")'

    async def test_tool_calls_summary_default_empty(self):
        store = await _make_store()
        await store.add_message("s1", "assistant", "Hi", turn=1)
        await store.commit()
        messages = await store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""

    async def test_tool_calls_summary_in_messages_range(self):
        store = await _make_store()
        await store.add_message("s1", "assistant", "Response", turn=1, tool_calls_summary="update_family_profile(child_name)")
        await store.commit()
        msgs = await store.get_messages_range("s1", start_turn=1)
        assert msgs[0]["tool_calls_summary"] == "update_family_profile(child_name)"

    async def test_blocked_messages_recorded(self):
        store = await _make_store()
        await store.add_message(
            "s1", "user", "bad message", turn=1,
            blocked=True, blocked_reason="crisis",
        )
        await store.add_message(
            "s1", "assistant", "Please call 988.", turn=1,
            blocked=True, blocked_reason="crisis",
        )
        await store.commit()
        messages = await store.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["blocked"] is True
        assert messages[0]["blocked_reason"] == "crisis"

    async def test_get_messages_limit(self):
        store = await _make_store()
        for i in range(10):
            await store.add_message("s1", "user", f"msg {i}", turn=i)
        await store.commit()
        messages = await store.get_messages("s1", limit=3)
        assert len(messages) == 3
        # Should be the LAST 3 messages
        assert messages[0]["content"] == "msg 7"
        assert messages[2]["content"] == "msg 9"

    async def test_messages_across_sessions_isolated(self):
        store = await _make_store()
        await store.add_message("s1", "user", "hello s1", turn=1)
        await store.add_message("s2", "user", "hello s2", turn=1)
        await store.commit()
        assert len(await store.get_messages("s1")) == 1
        assert len(await store.get_messages("s2")) == 1


class TestActiveStrategies:

    async def test_add_and_get_strategies(self):
        store = await _make_store()
        await store.add_active_strategy("s1", "visual timer")
        await store.add_active_strategy("s1", "reward chart")
        await store.commit()
        strategies = await store.get_active_strategies("s1")
        assert "visual timer" in strategies
        assert "reward chart" in strategies

    async def test_dedup_strategies(self):
        store = await _make_store()
        await store.add_active_strategy("s1", "visual timer")
        await store.add_active_strategy("s1", "visual timer")
        await store.commit()
        strategies = await store.get_active_strategies("s1")
        assert len(strategies) == 1

    async def test_strategies_in_session_state(self):
        store = await _make_store()
        await store.add_active_strategy("s1", "visual timer")
        await store.commit()
        state = await store.get("s1")
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

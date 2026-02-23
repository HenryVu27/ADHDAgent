"""Tests for async in-memory session store (create_in_memory_store)."""

import pytest

from app.agent.session_store import create_in_memory_store
from app.models.schemas import SeedSessionRequest


class TestSessionStateStore:

    @pytest.mark.asyncio
    async def test_get_creates_new_session(self):
        store = await create_in_memory_store()
        state = await store.get("new_session")
        assert state.session_id == "new_session"
        assert state.turn_count == 0
        assert state.family_profile.child_name is None

    @pytest.mark.asyncio
    async def test_get_returns_independent_copy(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        s1 = await store.get("s1")
        assert s1.turn_count == 1
        # Mutating the copy should not affect internal state
        s1.turn_count = 999
        s2 = await store.get("s1")
        assert s2.turn_count == 1  # still 1, not 999

    @pytest.mark.asyncio
    async def test_get_returns_different_objects(self):
        store = await create_in_memory_store()
        s1 = await store.get("s1")
        s2 = await store.get("s1")
        assert s1 is not s2

    @pytest.mark.asyncio
    async def test_increment_turn(self):
        store = await create_in_memory_store()
        assert await store.increment_turn("t1") == 1
        assert await store.increment_turn("t1") == 2
        assert await store.increment_turn("t1") == 3

    @pytest.mark.asyncio
    async def test_update_profile_scalar_fields(self):
        store = await create_in_memory_store()
        profile = await store.update_profile("p1", child_name="Kai", child_age="7")
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    @pytest.mark.asyncio
    async def test_update_profile_list_append(self):
        store = await create_in_memory_store()
        await store.update_profile("p1", challenge_areas=["homework"])
        await store.commit()
        profile = await store.update_profile("p1", challenge_areas=["bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    @pytest.mark.asyncio
    async def test_update_profile_list_dedup(self):
        store = await create_in_memory_store()
        await store.update_profile("p1", challenge_areas=["homework"])
        await store.commit()
        profile = await store.update_profile("p1", challenge_areas=["homework", "bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    @pytest.mark.asyncio
    async def test_update_profile_ignores_none(self):
        store = await create_in_memory_store()
        await store.update_profile("p1", child_name="Kai")
        await store.commit()
        profile = await store.update_profile("p1", child_name=None, child_age="8")
        assert profile.child_name == "Kai"
        assert profile.child_age == "8"

    @pytest.mark.asyncio
    async def test_add_outcome(self):
        store = await create_in_memory_store()
        outcome = await store.add_outcome("o1", "visual timer", "positive", "worked great")
        assert outcome.strategy_name == "visual timer"
        assert outcome.signal == "positive"
        assert outcome.detail == "worked great"

        await store.commit()
        state = await store.get("o1")
        assert len(state.outcomes) == 1

    @pytest.mark.asyncio
    async def test_manage_goal_add(self):
        store = await create_in_memory_store()
        goals = await store.manage_goal("g1", "add", "Homework done by 6pm")
        assert len(goals) == 1
        assert goals[0].description == "Homework done by 6pm"
        assert goals[0].status == "active"

    @pytest.mark.asyncio
    async def test_manage_goal_complete(self):
        store = await create_in_memory_store()
        await store.manage_goal("g1", "add", "Homework done by 6pm")
        await store.commit()
        goals = await store.manage_goal("g1", "complete", "Homework done by 6pm")
        assert goals[0].status == "completed"

    @pytest.mark.asyncio
    async def test_manage_goal_complete_case_insensitive(self):
        store = await create_in_memory_store()
        await store.manage_goal("g1", "add", "Homework done by 6pm")
        await store.commit()
        goals = await store.manage_goal("g1", "complete", "homework done by 6pm")
        assert goals[0].status == "completed"

    @pytest.mark.asyncio
    async def test_manage_goal_list(self):
        store = await create_in_memory_store()
        await store.manage_goal("g1", "add", "Goal A")
        await store.manage_goal("g1", "add", "Goal B")
        await store.commit()
        goals = await store.manage_goal("g1", "list")
        assert len(goals) == 2

    @pytest.mark.asyncio
    async def test_seed_session(self):
        store = await create_in_memory_store()
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

    @pytest.mark.asyncio
    async def test_sessions_isolated(self):
        store = await create_in_memory_store()
        await store.update_profile("s1", child_name="Kai")
        await store.update_profile("s2", child_name="Alex")
        await store.commit()
        assert (await store.get("s1")).family_profile.child_name == "Kai"
        assert (await store.get("s2")).family_profile.child_name == "Alex"

    @pytest.mark.asyncio
    async def test_tool_calls_summary_stored_and_retrieved(self):
        store = await create_in_memory_store()
        await store.add_message("s1", "user", "Help", turn=1)
        await store.add_message(
            "s1", "assistant", "Sure!", turn=1,
            tool_calls_summary='search_knowledge_base(query="homework")',
        )
        await store.commit()
        messages = await store.get_messages("s1")
        assert messages[0].get("tool_calls_summary", "") == ""
        assert messages[1]["tool_calls_summary"] == 'search_knowledge_base(query="homework")'

    @pytest.mark.asyncio
    async def test_tool_calls_summary_default_empty(self):
        store = await create_in_memory_store()
        await store.add_message("s1", "assistant", "Hi", turn=1)
        await store.commit()
        messages = await store.get_messages("s1")
        assert messages[0]["tool_calls_summary"] == ""

    @pytest.mark.asyncio
    async def test_session_exists_false_before_creation(self):
        store = await create_in_memory_store()
        assert await store.session_exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_session_exists_true_after_get(self):
        store = await create_in_memory_store()
        await store.get("new_session")
        assert await store.session_exists("new_session") is True

    @pytest.mark.asyncio
    async def test_get_all_sessions_paginated(self):
        store = await create_in_memory_store()
        for i in range(5):
            await store.get(f"s{i}")
        await store.commit()
        items, total = await store.get_all_sessions_paginated(offset=0, limit=3)
        assert total == 5
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_get_all_sessions_paginated_offset(self):
        store = await create_in_memory_store()
        for i in range(5):
            await store.get(f"s{i}")
        await store.commit()
        items, total = await store.get_all_sessions_paginated(offset=3, limit=10)
        assert total == 5
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_get_messages_paginated(self):
        store = await create_in_memory_store()
        for i in range(5):
            await store.add_message("s1", "user", f"msg {i}", turn=i)
        await store.commit()
        messages, total = await store.get_messages_paginated("s1", offset=0, limit=3)
        assert total == 5
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_get_messages_paginated_offset(self):
        store = await create_in_memory_store()
        for i in range(5):
            await store.add_message("s1", "user", f"msg {i}", turn=i)
        await store.commit()
        messages, total = await store.get_messages_paginated("s1", offset=3, limit=10)
        assert total == 5
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self):
        store = await create_in_memory_store()
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

    @pytest.mark.asyncio
    async def test_save_and_get_tool_results(self):
        store = await create_in_memory_store()
        await store.save_tool_result("tr1", "search_knowledge_base", "homework strategies", "Result: visual timer...", turn=1)
        await store.save_tool_result("tr1", "search_knowledge_base", "bedtime routines", "Result: bedtime routine...", turn=2)
        await store.commit()
        results = await store.get_recent_tool_results("tr1", limit=5)
        assert len(results) == 2
        assert results[0].tool_name == "search_knowledge_base"
        assert results[0].query == "homework strategies"
        assert results[0].turn == 1
        assert results[1].turn == 2

    @pytest.mark.asyncio
    async def test_get_recent_tool_results_respects_limit(self):
        store = await create_in_memory_store()
        for i in range(5):
            await store.save_tool_result("tr2", "search_knowledge_base", f"query {i}", f"result {i}", turn=i)
        await store.commit()
        results = await store.get_recent_tool_results("tr2", limit=2)
        assert len(results) == 2
        # Should be the most recent 2
        assert results[0].turn == 3
        assert results[1].turn == 4

    @pytest.mark.asyncio
    async def test_get_recent_tool_results_empty(self):
        store = await create_in_memory_store()
        results = await store.get_recent_tool_results("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_session_removes_tool_results(self):
        store = await create_in_memory_store()
        await store.save_tool_result("del-tr", "search_knowledge_base", "q", "r", turn=1)
        await store.commit()
        await store.delete_session("del-tr")
        results = await store.get_recent_tool_results("del-tr")
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session_is_noop(self):
        store = await create_in_memory_store()
        await store.delete_session("does-not-exist")  # Should not raise

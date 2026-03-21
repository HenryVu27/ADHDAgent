"""Tests for prepare_context (context assembly for the ReAct agent)."""

import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.hooks import create_prepare_context
from app.agent.session_store import create_in_memory_store


class TestPrepareContext:

    @pytest.mark.asyncio
    async def test_injects_system_prompt(self):
        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="My child needs help with homework")],
            "session_id": "test1",
        }

        result = await prepare(state)
        assert "llm_input_messages" in result
        msgs = result["llm_input_messages"]
        assert isinstance(msgs[0], SystemMessage)
        assert "ADHD parenting coach" in msgs[0].content

    @pytest.mark.asyncio
    async def test_injects_profile_context(self):
        store = await create_in_memory_store()
        await store.update_profile("ctx1", child_name="Kai", child_age="7")
        await store.commit()
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="Help with homework")],
            "session_id": "ctx1",
        }

        result = await prepare(state)
        system_msg = result["llm_input_messages"][0]
        assert "Kai" in system_msg.content
        assert "7" in system_msg.content

    @pytest.mark.asyncio
    async def test_trims_long_conversation(self):
        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"Turn {i}"))
            messages.append(AIMessage(content=f"Response {i}"))
        messages.append(HumanMessage(content="Latest message"))

        state = {
            "messages": messages,
            "session_id": "trim1",
        }

        result = await prepare(state)
        llm_msgs = result["llm_input_messages"]
        non_system = [m for m in llm_msgs if not isinstance(m, SystemMessage)]
        assert len(non_system) <= 12  # CONTEXT_WINDOW_TURNS * 2

    @pytest.mark.asyncio
    async def test_includes_conversation_state_block(self):
        """Conversation state block is appended to the last HumanMessage."""
        store = await create_in_memory_store()
        await store.increment_turn("state1")
        await store.increment_turn("state1")
        await store.commit()
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="Tell me about homework strategies")],
            "session_id": "state1",
        }

        result = await prepare(state)
        # conversation_state is appended to the last HumanMessage
        last_human = result["llm_input_messages"][-1]
        assert isinstance(last_human, HumanMessage)
        assert "<conversation_state>" in last_human.content
        assert "<turn>2</turn>" in last_human.content
        assert "<phase>" in last_human.content
        assert "<focus>" in last_human.content

    @pytest.mark.asyncio
    async def test_char_budget_trims_long_messages(self, monkeypatch):
        """Small CONTEXT_MAX_CHARS should force trimming of long messages."""
        from app import config
        monkeypatch.setattr(config.settings, "CONTEXT_MAX_CHARS", 500)

        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        messages = []
        for i in range(5):
            messages.append(HumanMessage(content=f"Message {i} " + "x" * 200))
            messages.append(AIMessage(content=f"Response {i} " + "y" * 200))

        state = {"messages": messages, "session_id": "chartest"}
        result = await prepare(state)
        non_system = [m for m in result["llm_input_messages"] if not isinstance(m, SystemMessage)]
        # Should have trimmed — fewer messages than original
        assert len(non_system) < 10
        # At least 2 messages always preserved
        assert len(non_system) >= 2

    @pytest.mark.asyncio
    async def test_large_char_budget_defers_to_count_cap(self, monkeypatch):
        """Large CONTEXT_MAX_CHARS should not trim — count cap takes precedence."""
        from app import config
        monkeypatch.setattr(config.settings, "CONTEXT_MAX_CHARS", 999999)

        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"Turn {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        state = {"messages": messages, "session_id": "bigbudget"}
        result = await prepare(state)
        non_system = [m for m in result["llm_input_messages"] if not isinstance(m, SystemMessage)]
        # Count cap (CONTEXT_WINDOW_TURNS * 2 = 12) should apply
        assert len(non_system) <= 12

    @pytest.mark.asyncio
    async def test_char_budget_preserves_at_least_two_messages(self, monkeypatch):
        """Even with tiny budget, at least 2 messages must be preserved."""
        from app import config
        monkeypatch.setattr(config.settings, "CONTEXT_MAX_CHARS", 1)

        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        messages = [
            HumanMessage(content="A" * 500),
            AIMessage(content="B" * 500),
            HumanMessage(content="C" * 500),
        ]
        state = {"messages": messages, "session_id": "tinybudget"}
        result = await prepare(state)
        non_system = [m for m in result["llm_input_messages"] if not isinstance(m, SystemMessage)]
        assert len(non_system) >= 2

    @pytest.mark.asyncio
    async def test_latest_message_always_preserved(self, monkeypatch):
        """The latest user message should always be in the output."""
        from app import config
        monkeypatch.setattr(config.settings, "CONTEXT_MAX_CHARS", 500)

        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        messages = [
            HumanMessage(content="Old " + "x" * 200),
            AIMessage(content="Old response " + "y" * 200),
            HumanMessage(content="Latest question"),
        ]
        state = {"messages": messages, "session_id": "latest"}
        result = await prepare(state)
        non_system = [m for m in result["llm_input_messages"] if not isinstance(m, SystemMessage)]
        assert any("Latest question" in m.content for m in non_system)


    @pytest.mark.asyncio
    async def test_injects_prior_search_evidence(self):
        store = await create_in_memory_store()
        await store.save_tool_result("ev1", "search_knowledge_base", "homework tips", "Strategy: visual timer with 15-min chunks", turn=1)
        await store.commit()
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="Remind me of those homework tips")],
            "session_id": "ev1",
        }

        result = await prepare(state)
        system_msg = result["llm_input_messages"][0]
        assert "<prior-search-evidence>" in system_msg.content
        assert "visual timer" in system_msg.content
        assert 'query: "homework tips"' in system_msg.content

    @pytest.mark.asyncio
    async def test_passes_through_tool_results(self):
        """prepare_context should work even when last message is a ToolMessage."""
        store = await create_in_memory_store()
        prepare = create_prepare_context(store)

        state = {
            "messages": [
                HumanMessage(content="Help with homework"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "search", "args": {}}]),
                ToolMessage(content="results...", tool_call_id="1"),
            ],
            "session_id": "tool_loop",
        }

        result = await prepare(state)
        assert "llm_input_messages" in result

    @pytest.mark.asyncio
    async def test_prepare_context_caches_system_prompt(self):
        """System prompt should be cached within a single turn's ReAct loop."""
        from app.agent.hooks import _prompt_cache
        _prompt_cache.clear()

        store = await create_in_memory_store()
        await store.increment_turn("cache-test")  # Set turn to 1
        await store.commit()
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="hello")],
            "session_id": "cache-test",
        }

        result1 = await prepare(state)
        prompt1 = result1["llm_input_messages"][0].content

        # Call again with same state (simulates next ReAct iteration)
        result2 = await prepare(state)
        prompt2 = result2["llm_input_messages"][0].content

        assert prompt1 == prompt2, "Prompt should be identical (cached)"

    @pytest.mark.asyncio
    async def test_cache_invalidated_after_mutating_tool(self):
        """Cache should be invalidated when a state-mutating tool runs."""
        from app.agent.hooks import _prompt_cache
        _prompt_cache.clear()

        store = await create_in_memory_store()
        await store.increment_turn("mut-test")
        await store.commit()
        prepare = create_prepare_context(store)

        state1 = {
            "messages": [HumanMessage(content="hello")],
            "session_id": "mut-test",
        }
        result1 = await prepare(state1)
        prompt1 = result1["llm_input_messages"][0].content

        # Simulate a tool call that mutates state
        await store.update_profile("mut-test", child_name="Kai")
        await store.commit()
        state2 = {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="", tool_calls=[{"id": "tc1", "name": "update_family_profile", "args": {"child_name": "Kai"}}]),
                ToolMessage(content="Profile updated", tool_call_id="tc1", name="update_family_profile"),
            ],
            "session_id": "mut-test",
        }
        result2 = await prepare(state2)
        prompt2 = result2["llm_input_messages"][0].content

        # Prompt should now include "Kai" since cache was invalidated
        assert "Kai" in prompt2

    @pytest.mark.asyncio
    async def test_memory_usage_event_emitted(self):
        """Context assembly should emit a memory_usage event listing injected profile fields."""
        from unittest.mock import AsyncMock
        store = await create_in_memory_store()
        await store.update_profile("s1", child_name="Kai", child_age="8")
        await store.commit()

        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()

        prepare_context = create_prepare_context(store, event_bus=event_bus)
        state = {"messages": [HumanMessage(content="Hello")], "session_id": "s1"}
        await prepare_context(state)

        event_bus.emit.assert_called()
        calls_flat = " ".join(str(c) for c in event_bus.emit.call_args_list)
        assert "memory_usage" in calls_flat


class TestGraphitiContextAssembly:

    @pytest.mark.asyncio
    async def test_prepare_context_queries_graphiti(self):
        """When graphiti_client is provided, context assembly should query it."""
        from datetime import datetime
        from unittest.mock import MagicMock

        mock_graphiti = AsyncMock()
        edge = MagicMock()
        edge.fact = "Alex struggles with homework initiation"
        edge.valid_at = datetime(2026, 1, 1)
        edge.invalid_at = None
        mock_graphiti.search = AsyncMock(return_value=[edge])

        store = await create_in_memory_store()

        prepare = create_prepare_context(
            session_store=store,
            graphiti_client=mock_graphiti,
        )

        state = {
            "messages": [HumanMessage(content="Help with homework struggles")],
            "session_id": "test-session",
            "user_id": 1,
        }

        result = await prepare(state)
        mock_graphiti.search.assert_called_once()
        system_msg = result["llm_input_messages"][0]
        assert isinstance(system_msg, SystemMessage)
        assert "homework initiation" in system_msg.content

    @pytest.mark.asyncio
    async def test_prepare_context_without_graphiti_uses_default(self):
        """When graphiti_client is None, context assembly uses default summary text."""
        store = await create_in_memory_store()
        prepare = create_prepare_context(session_store=store, graphiti_client=None)

        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "test-session",
        }

        result = await prepare(state)
        system_msg = result["llm_input_messages"][0]
        assert isinstance(system_msg, SystemMessage)
        assert "beginning of the conversation" in system_msg.content

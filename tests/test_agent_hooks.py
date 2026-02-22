"""Tests for prepare_context (context assembly for the ReAct agent)."""

import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.hooks import create_prepare_context
from app.agent.session_store import InMemorySessionStore


class TestPrepareContext:

    @pytest.mark.asyncio
    async def test_injects_system_prompt(self):
        store = InMemorySessionStore()
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
        store = InMemorySessionStore()
        store.update_profile("ctx1", child_name="Kai", child_age="7")
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
        store = InMemorySessionStore()
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
        """System prompt should start with <conversation_state> block."""
        store = InMemorySessionStore()
        store.increment_turn("state1")
        store.increment_turn("state1")
        prepare = create_prepare_context(store)

        state = {
            "messages": [HumanMessage(content="Tell me about homework strategies")],
            "session_id": "state1",
        }

        result = await prepare(state)
        system_content = result["llm_input_messages"][0].content
        assert "<conversation_state>" in system_content
        assert "<turn>2</turn>" in system_content
        assert "<phase>" in system_content
        assert "<focus>" in system_content

    @pytest.mark.asyncio
    async def test_char_budget_trims_long_messages(self, monkeypatch):
        """Small CONTEXT_MAX_CHARS should force trimming of long messages."""
        from app import config
        monkeypatch.setattr(config.settings, "CONTEXT_MAX_CHARS", 500)

        store = InMemorySessionStore()
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

        store = InMemorySessionStore()
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

        store = InMemorySessionStore()
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

        store = InMemorySessionStore()
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
    async def test_passes_through_tool_results(self):
        """prepare_context should work even when last message is a ToolMessage."""
        store = InMemorySessionStore()
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

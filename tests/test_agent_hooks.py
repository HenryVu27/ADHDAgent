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

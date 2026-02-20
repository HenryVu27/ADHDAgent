"""Tests for pre_model_hook and post_model_hook."""

import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from app.agent.hooks import create_hooks
from app.agent.prompts import SAFE_OUTPUT_FALLBACK
from app.agent.session_store import SessionStateStore
from app.models.schemas import InputCheckResult, OutputCheckResult


def _make_mock_guardrails(
    input_allowed=True,
    input_reason=None,
    input_override=None,
    output_valid=True,
    output_violation=None,
):
    guardrails = AsyncMock()
    guardrails.check_input = AsyncMock(return_value=InputCheckResult(
        is_allowed=input_allowed,
        blocked_reason=input_reason,
        override_response=input_override,
    ))
    guardrails.check_output = AsyncMock(return_value=OutputCheckResult(
        is_valid=output_valid,
        violation_type=output_violation,
    ))
    return guardrails


class TestPreModelHook:

    @pytest.mark.asyncio
    async def test_allows_normal_message(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        pre_hook, _ = create_hooks(guardrails, store)

        state = {
            "messages": [HumanMessage(content="My child needs help with homework")],
            "session_id": "test1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await pre_hook(state)
        # Should return llm_input_messages with system prompt
        assert "llm_input_messages" in result
        msgs = result["llm_input_messages"]
        assert isinstance(msgs[0], SystemMessage)
        assert "ADHD parenting coach" in msgs[0].content

    @pytest.mark.asyncio
    async def test_blocks_crisis_message(self):
        override = "Please call 988."
        guardrails = _make_mock_guardrails(
            input_allowed=False,
            input_reason="crisis",
            input_override=override,
        )
        store = SessionStateStore()
        pre_hook, _ = create_hooks(guardrails, store)

        state = {
            "messages": [HumanMessage(content="I want to hurt myself")],
            "session_id": "crisis1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await pre_hook(state)
        assert isinstance(result, Command)
        assert result.goto == "__end__"
        assert result.update["input_blocked"] is True
        assert result.update["block_response"] == override

    @pytest.mark.asyncio
    async def test_skips_guardrails_on_tool_result(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        pre_hook, _ = create_hooks(guardrails, store)

        state = {
            "messages": [
                HumanMessage(content="Help with homework"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "search", "args": {}}]),
                ToolMessage(content="results...", tool_call_id="1"),
            ],
            "session_id": "tool_loop",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await pre_hook(state)
        # Should NOT have called check_input (last message is ToolMessage, not HumanMessage)
        guardrails.check_input.assert_not_called()
        assert "llm_input_messages" in result

    @pytest.mark.asyncio
    async def test_injects_profile_context(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        store.update_profile("ctx1", child_name="Kai", child_age="7")
        pre_hook, _ = create_hooks(guardrails, store)

        state = {
            "messages": [HumanMessage(content="Help with homework")],
            "session_id": "ctx1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await pre_hook(state)
        system_msg = result["llm_input_messages"][0]
        assert "Kai" in system_msg.content
        assert "7" in system_msg.content

    @pytest.mark.asyncio
    async def test_trims_long_conversation(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        pre_hook, _ = create_hooks(guardrails, store)

        # Build a conversation with 20 turns (40 messages)
        messages = []
        for i in range(20):
            messages.append(HumanMessage(content=f"Turn {i}"))
            messages.append(AIMessage(content=f"Response {i}"))
        messages.append(HumanMessage(content="Latest message"))

        state = {
            "messages": messages,
            "session_id": "trim1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await pre_hook(state)
        llm_msgs = result["llm_input_messages"]
        # System prompt + trimmed messages (CONTEXT_WINDOW_TURNS * 2 = 12)
        non_system = [m for m in llm_msgs if not isinstance(m, SystemMessage)]
        assert len(non_system) <= 12


class TestPostModelHook:

    @pytest.mark.asyncio
    async def test_allows_safe_response(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        _, post_hook = create_hooks(guardrails, store)

        state = {
            "messages": [
                HumanMessage(content="Help with homework"),
                AIMessage(content="Try a visual timer."),
            ],
            "session_id": "safe1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await post_hook(state)
        assert "trace_steps" in result
        assert result["trace_steps"][0]["detail"]["is_valid"] is True

    @pytest.mark.asyncio
    async def test_replaces_medication_response(self):
        guardrails = _make_mock_guardrails(
            output_valid=False,
            output_violation="medication",
        )
        store = SessionStateStore()
        _, post_hook = create_hooks(guardrails, store)

        state = {
            "messages": [
                HumanMessage(content="What about meds?"),
                AIMessage(content="You should try Adderall."),
            ],
            "session_id": "med1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await post_hook(state)
        # Should have replaced the response
        assert "messages" in result
        last_ai = result["messages"][-1]
        assert last_ai.content == SAFE_OUTPUT_FALLBACK

    @pytest.mark.asyncio
    async def test_ignores_tool_call_messages(self):
        guardrails = _make_mock_guardrails()
        store = SessionStateStore()
        _, post_hook = create_hooks(guardrails, store)

        state = {
            "messages": [
                HumanMessage(content="Help"),
                AIMessage(content="", tool_calls=[{"id": "1", "name": "search", "args": {}}]),
            ],
            "session_id": "tool1",
            "input_blocked": False,
            "block_response": "",
            "trace_steps": [],
        }

        result = await post_hook(state)
        # No final AI message found, should return messages as-is
        assert "messages" in result

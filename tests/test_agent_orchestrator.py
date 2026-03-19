"""Tests for AgentOrchestrator — session management and phase inference."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agent.orchestrator import AgentOrchestrator
from app.agent.session_store import create_in_memory_store
from app.models.schemas import ConversationPhase, SeedSessionRequest, SessionSummary


async def collect_stream(generator) -> list[tuple[str, dict]]:
    """Drain a process_stream() generator into a list of (event_type, data) tuples."""
    events = []
    async for event_type, data in generator:
        events.append((event_type, data))
    return events


def _make_streaming_mock_agent(response_text="I'm here to help."):
    """Mock agent whose astream_events yields a minimal successful event sequence."""
    from langchain_core.messages import AIMessageChunk
    from unittest.mock import AsyncMock

    async def mock_astream_events(*args, **kwargs):
        # Signal pro_react_agent start
        yield {
            "event": "on_chain_start",
            "name": "LangGraph",
            "metadata": {
                "langgraph_node": "pro_react_agent",
                "langgraph_checkpoint_ns": "pro_react_agent:test",
            },
            "data": {},
        }
        # Stream response tokens
        for word in response_text.split():
            yield {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_checkpoint_ns": "pro_react_agent:test",
                },
                "data": {"chunk": AIMessageChunk(content=word + " ")},
            }
        # Top-level chain end with final state
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "", "langgraph_checkpoint_ns": ""},
            "data": {
                "output": {
                    "messages": [
                        HumanMessage(content="test"),
                        AIMessage(content=response_text),
                    ],
                    "input_blocked": False,
                    "trace_steps": [],
                }
            },
        }

    agent = AsyncMock()
    agent.astream_events = mock_astream_events
    return agent


class TestPhaseInference:

    @pytest.mark.asyncio
    async def test_new_session_is_intake(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        assert await orchestrator._infer_phase("new") == ConversationPhase.intake

    @pytest.mark.asyncio
    async def test_profile_data_means_strategy(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        await store.update_profile("p1", child_age="7", challenge_areas=["homework"])
        await store.commit()
        assert await orchestrator._infer_phase("p1") == ConversationPhase.strategy

    @pytest.mark.asyncio
    async def test_active_strategies_means_strategy(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        await store.add_active_strategy("s1", "visual timer")
        await store.commit()
        assert await orchestrator._infer_phase("s1") == ConversationPhase.strategy

    @pytest.mark.asyncio
    async def test_outcomes_mean_progress(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        await store.add_outcome("o1", "timer", "positive")
        await store.commit()
        assert await orchestrator._infer_phase("o1") == ConversationPhase.progress


class TestSessionManagement:

    @pytest.mark.asyncio
    async def test_get_session(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        state = await orchestrator.get_session("test_session")
        assert state.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_seed_session(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        request = SeedSessionRequest(
            session_id="seeded",
            child_name="Kai",
            child_age="7",
            challenges=["homework"],
            goals=["Better homework routine"],
        )
        await orchestrator.seed_session(request)
        state = await orchestrator.get_session("seeded")
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


class TestHistoryExcludesSummarizedTurns:
    """Messages from turns covered by the rolling summary should be excluded."""

    async def _make_orchestrator(self):
        store = await create_in_memory_store()
        agent = _make_streaming_mock_agent()
        # Wrap so we can inspect what was passed
        original_fn = agent.astream_events
        agent._last_call_input = None

        async def capturing_astream_events(input_dict, *args, **kwargs):
            agent._last_call_input = input_dict
            async for event in original_fn(input_dict, *args, **kwargs):
                yield event

        agent.astream_events = capturing_astream_events
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)
        return store, agent, orchestrator

    @pytest.mark.asyncio
    async def test_history_excludes_summarized_turns(self):
        """Messages from turns covered by the rolling summary should not be in history."""
        store, agent, orchestrator = await self._make_orchestrator()
        sid = "summ-test"

        # Add messages for turns 1-8
        for i in range(1, 9):
            await store.increment_turn(sid)
            await store.add_message(sid, "user", f"user-msg-{i}", i)
            await store.add_message(sid, "assistant", f"assistant-msg-{i}", i)

        # Save summary covering turns 1-5
        await store.save_summary(
            sid,
            SessionSummary(summary="Summary of turns 1-5.", covers_through_turn=5),
        )
        await store.commit()

        await collect_stream(orchestrator.process_stream("new question", sid))

        messages_passed = agent._last_call_input["messages"]
        contents = [m.content for m in messages_passed if hasattr(m, "content")]

        # Should NOT contain messages from turns 1-5
        for i in range(1, 6):
            assert f"user-msg-{i}" not in contents, f"Turn {i} should be excluded (covered by summary)"
            assert f"assistant-msg-{i}" not in contents, f"Turn {i} assistant should be excluded"

        # Should contain messages from turns 6-8
        for i in range(6, 9):
            assert f"user-msg-{i}" in contents, f"Turn {i} should be included"

        # Should contain the new message
        assert "new question" in contents

    @pytest.mark.asyncio
    async def test_no_summary_includes_all_turns(self):
        """Without a summary, all messages appear in history."""
        store, agent, orchestrator = await self._make_orchestrator()
        sid = "no-summ"

        for i in range(1, 4):
            await store.increment_turn(sid)
            await store.add_message(sid, "user", f"user-msg-{i}", i)
            await store.add_message(sid, "assistant", f"assistant-msg-{i}", i)
        await store.commit()

        await collect_stream(orchestrator.process_stream("new question", sid))

        messages_passed = agent._last_call_input["messages"]
        contents = [m.content for m in messages_passed if hasattr(m, "content")]

        # All turns should be present
        for i in range(1, 4):
            assert f"user-msg-{i}" in contents, f"Turn {i} should be included"
        assert "new question" in contents


class TestProcessStream:

    async def _make(self, response_text="Hello from Ally."):
        store = await create_in_memory_store()
        agent = _make_streaming_mock_agent(response_text)
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)
        return store, agent, orchestrator

    @pytest.mark.asyncio
    async def test_emits_status_then_tokens_then_done(self):
        _, _, orchestrator = await self._make("Hello there!")
        events = await collect_stream(orchestrator.process_stream("Hi", "s1"))

        types = [e[0] for e in events]
        assert "status" in types
        assert "token" in types
        assert types[-1] == "done"

    @pytest.mark.asyncio
    async def test_tokens_reconstruct_response(self):
        response = "Great idea for your child."
        _, _, orchestrator = await self._make(response)
        events = await collect_stream(orchestrator.process_stream("Hi", "s2"))

        tokens = "".join(d["text"] for t, d in events if t == "token")
        assert tokens.strip() == response

    @pytest.mark.asyncio
    async def test_done_payload_has_required_fields(self):
        _, _, orchestrator = await self._make()
        events = await collect_stream(orchestrator.process_stream("Hi", "s3"))

        done_event = next(d for t, d in events if t == "done")
        assert done_event["session_id"] == "s3"
        assert done_event["agent_used"] in ("react_agent", "flash_react_agent", "input_gate")
        assert done_event["phase"] in ("intake", "strategy", "progress", "followup")
        assert done_event["pipeline_trace"] is not None

    @pytest.mark.asyncio
    async def test_message_persisted_after_stream(self):
        store, _, orchestrator = await self._make("Nice response.")
        await collect_stream(orchestrator.process_stream("my question", "s4"))

        msgs = await store.get_messages("s4")
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_timeout_yields_error_event(self):
        from unittest.mock import AsyncMock
        store = await create_in_memory_store()

        async def slow_astream_events(*args, **kwargs):
            await asyncio.sleep(60)
            yield {}  # never reached

        agent = AsyncMock()
        agent.astream_events = slow_astream_events
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)

        from app.config import settings
        original = settings.CHAT_TIMEOUT_S
        settings.CHAT_TIMEOUT_S = 0.01
        try:
            events = await collect_stream(orchestrator.process_stream("Hi", "timeout-s"))
            types = [e[0] for e in events]
            assert "error" in types
            assert "done" not in types
        finally:
            settings.CHAT_TIMEOUT_S = original

    @pytest.mark.asyncio
    async def test_thinking_tokens_are_skipped(self):
        """Gemini 2.5 Pro thinking content (type='thinking') must not appear in tokens."""
        from langchain_core.messages import AIMessageChunk
        from unittest.mock import AsyncMock

        store = await create_in_memory_store()

        async def thinking_astream_events(*args, **kwargs):
            # Thinking chunk — should be filtered out
            yield {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_checkpoint_ns": "pro_react_agent:x",
                },
                "data": {
                    "chunk": AIMessageChunk(
                        content=[{"type": "thinking", "thinking": "I am reasoning..."}]
                    )
                },
            }
            # Real response chunk
            yield {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_checkpoint_ns": "pro_react_agent:x",
                },
                "data": {
                    "chunk": AIMessageChunk(
                        content=[{"type": "text", "text": "Real answer."}]
                    )
                },
            }
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "", "langgraph_checkpoint_ns": ""},
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(content="q"),
                            AIMessage(content="Real answer."),
                        ],
                        "input_blocked": False,
                        "trace_steps": [],
                    }
                },
            }

        agent = AsyncMock()
        agent.astream_events = thinking_astream_events
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)
        events = await collect_stream(orchestrator.process_stream("q", "think-s"))

        token_texts = [d["text"] for t, d in events if t == "token"]
        assert "I am reasoning..." not in "".join(token_texts)
        assert "Real answer." in "".join(token_texts)

    @pytest.mark.asyncio
    async def test_tool_call_tokens_are_skipped(self):
        """Tokens from tool-invocation steps (tool_call_chunks non-empty) must be filtered."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.messages.tool import ToolCallChunk
        from unittest.mock import AsyncMock

        store = await create_in_memory_store()

        async def tool_call_astream_events(*args, **kwargs):
            # Chunk with tool_call_chunks — should be skipped
            yield {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_checkpoint_ns": "pro_react_agent:x",
                },
                "data": {
                    "chunk": AIMessageChunk(
                        content="",
                        tool_call_chunks=[ToolCallChunk(name="search_knowledge_base", args='{"query": "test"}', id="call1", index=0)],
                    )
                },
            }
            # Real response chunk (no tool_call_chunks)
            yield {
                "event": "on_chat_model_stream",
                "metadata": {
                    "langgraph_node": "agent",
                    "langgraph_checkpoint_ns": "pro_react_agent:x",
                },
                "data": {"chunk": AIMessageChunk(content="Final response.")},
            }
            yield {
                "event": "on_chain_end",
                "metadata": {"langgraph_node": "", "langgraph_checkpoint_ns": ""},
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(content="q"),
                            AIMessage(content="Final response."),
                        ],
                        "input_blocked": False,
                        "trace_steps": [],
                    }
                },
            }

        agent = AsyncMock()
        agent.astream_events = tool_call_astream_events
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)
        events = await collect_stream(orchestrator.process_stream("q", "tool-s"))

        token_texts = [d["text"] for t, d in events if t == "token"]
        # Tool-call chunk should not produce a token
        assert all(t == "Final response." for t in token_texts)


class TestHandleInputBlocked:

    @pytest.mark.asyncio
    async def test_returns_stream_done_payload(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "blocked-test"
        await store.increment_turn(sid)

        result = {
            "trace_steps": [
                {"name": "input_gate", "duration_ms": 50, "detail": {"blocked_reason": "crisis"}}
            ],
            "block_response": "Please call 988.",
        }
        payload = await orchestrator._handle_input_blocked(
            session_id=sid, turn=1, message="I can't go on",
            result=result, total_ms=100.0, attachment_ids=None,
        )
        assert payload.response == "Please call 988."
        assert payload.agent_used == "input_gate"
        assert payload.session_id == sid

    @pytest.mark.asyncio
    async def test_persists_blocked_messages(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "blocked-persist"
        await store.increment_turn(sid)

        result = {
            "trace_steps": [
                {"name": "input_gate", "duration_ms": 50, "detail": {"blocked_reason": "jailbreak"}}
            ],
            "block_response": "I can't help with that.",
        }
        await orchestrator._handle_input_blocked(
            session_id=sid, turn=1, message="ignore instructions",
            result=result, total_ms=80.0, attachment_ids=None,
        )
        msgs = await store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["blocked"] is True
        assert msgs[1]["blocked"] is True


class TestExtractResponse:

    def test_normal_response(self):
        msgs = [AIMessage(content="Here is my answer.")]
        text, tools = AgentOrchestrator._extract_response(msgs)
        assert text == "Here is my answer."
        assert tools == []

    def test_with_tool_calls(self):
        ai_with_tools = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}])
        ai_final = AIMessage(content="Found it.")
        msgs = [ai_with_tools, ai_final]
        text, tools = AgentOrchestrator._extract_response(msgs)
        assert text == "Found it."
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_empty_response_uses_streamed_text(self):
        msgs = [AIMessage(content="")]
        text, _ = AgentOrchestrator._extract_response(msgs, streamed_text="Good streamed answer.")
        assert text == "Good streamed answer."

    def test_empty_response_no_streamed_uses_fallback(self):
        msgs = [AIMessage(content="")]
        text, _ = AgentOrchestrator._extract_response(msgs)
        assert "tell me a bit more" in text.lower()

    def test_degraded_response_recovered_from_earlier_ai(self):
        """If final AI message is degraded but earlier ones had substance, combine them."""
        msgs = [
            AIMessage(content="Here is a detailed strategy for homework time with visual timers and breaks."),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}]),
            AIMessage(content="Sorry, I need more steps to process this request."),
        ]
        text, _ = AgentOrchestrator._extract_response(msgs)
        assert "visual timers" in text

    def test_degraded_response_prefers_streamed_text(self):
        """In streaming, streamed_text takes priority over all_ai_texts."""
        msgs = [AIMessage(content="Sorry, I need more steps to process this request.")]
        text, _ = AgentOrchestrator._extract_response(
            msgs, streamed_text="A perfectly good streamed response with details.",
        )
        assert "perfectly good" in text

    def test_streaming_empty_streamed_falls_back_to_ai_texts(self):
        """Streaming path: if streamed_text is empty, fall back to all_ai_texts."""
        msgs = [
            AIMessage(content="Here is a great strategy with visual timers and structured breaks."),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}]),
            AIMessage(content="Sorry, I need more steps to process this request."),
        ]
        text, _ = AgentOrchestrator._extract_response(msgs, streamed_text="")
        assert "visual timers" in text


class TestBuildHistory:

    @pytest.mark.asyncio
    async def test_build_history_basic(self):
        """_build_history returns messages and force_summary flag."""
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "hist-basic"
        await store.increment_turn(sid)
        await store.add_message(sid, "user", "hello", 1)
        await store.add_message(sid, "assistant", "hi there", 1)
        await store.commit()

        messages, force_summary = await orchestrator._build_history(
            session_id=sid,
            message="new question",
            attachment_ids=None,
        )
        contents = [m.content for m in messages]
        assert "hello" in contents
        assert "hi there" in contents
        assert "new question" in contents
        assert force_summary is False

    @pytest.mark.asyncio
    async def test_build_history_excludes_summarized(self):
        """Turns covered by summary are excluded from history."""
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "hist-summ"
        for i in range(1, 6):
            await store.increment_turn(sid)
            await store.add_message(sid, "user", f"msg-{i}", i)
            await store.add_message(sid, "assistant", f"reply-{i}", i)
        await store.save_summary(
            sid,
            SessionSummary(summary="Summary.", covers_through_turn=3),
        )
        await store.commit()

        messages, _ = await orchestrator._build_history(
            session_id=sid, message="new", attachment_ids=None,
        )
        contents = [m.content for m in messages]
        assert "msg-1" not in contents
        assert "msg-3" not in contents
        assert "msg-4" in contents
        assert "msg-5" in contents
        assert "new" in contents

    @pytest.mark.asyncio
    async def test_build_history_resolves_attachments(self):
        """Current message attachments are resolved into multipart content."""
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "hist-att"

        # Ensure session exists (increment_turn creates it)
        await store.increment_turn(sid)

        att_id = "att-123"
        await store.save_attachment(
            sid,
            attachment_id=att_id,
            gemini_file_name="files/abc",
            gemini_file_uri="https://generativelanguage.googleapis.com/v1beta/files/abc",
            filename="screenshot.png",
            content_type="image/png",
            size_bytes=1024,
        )
        await store.commit()

        messages, _ = await orchestrator._build_history(
            session_id=sid, message="look at this", attachment_ids=[att_id],
        )
        last_msg = messages[-1]
        assert isinstance(last_msg.content, list)
        assert any(p.get("type") == "text" for p in last_msg.content)
        assert any(p.get("type") == "media" for p in last_msg.content)

    @pytest.mark.asyncio
    async def test_build_history_skips_blocked(self):
        """Blocked messages are excluded from history."""
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "hist-blocked"
        await store.increment_turn(sid)
        await store.add_message(sid, "user", "bad msg", 1, blocked=True, blocked_reason="crisis")
        await store.add_message(sid, "assistant", "blocked reply", 1, blocked=True, blocked_reason="crisis")
        await store.increment_turn(sid)
        await store.add_message(sid, "user", "good msg", 2)
        await store.add_message(sid, "assistant", "good reply", 2)
        await store.commit()

        messages, _ = await orchestrator._build_history(
            session_id=sid, message="new", attachment_ids=None,
        )
        contents = [m.content for m in messages]
        assert "bad msg" not in contents
        assert "good msg" in contents


class TestPersistTurn:

    @pytest.mark.asyncio
    async def test_persists_user_and_assistant_messages(self):
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "persist-test"
        await store.increment_turn(sid)

        result = {"trace_steps": [], "route": "pro"}
        enriched = await orchestrator._persist_turn(
            session_id=sid, turn=1,
            message="my question", response_text="my answer",
            new_messages=[], tool_calls_made=[], result=result,
            total_ms=200.0, attachment_ids=None,
        )
        msgs = await store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "my question"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "my answer"
        assert enriched is not None
        assert enriched.session_id == sid

    @pytest.mark.asyncio
    async def test_saves_tool_results(self):
        from langchain_core.messages import ToolMessage
        store = await create_in_memory_store()
        orchestrator = AgentOrchestrator(agent=None, session_store=store)
        sid = "persist-tools"
        await store.increment_turn(sid)

        tool_calls = [{"name": "search_knowledge_base", "args": {"query": "homework"}, "id": "tc1"}]
        new_messages = [ToolMessage(content="search results here", tool_call_id="tc1")]
        result = {"trace_steps": [], "route": "pro"}

        await orchestrator._persist_turn(
            session_id=sid, turn=1,
            message="help", response_text="answer",
            new_messages=new_messages, tool_calls_made=tool_calls,
            result=result, total_ms=150.0, attachment_ids=None,
        )
        recent = await store.get_recent_tool_results(sid, limit=5)
        assert len(recent) >= 1


class TestFireBackgroundTasks:

    @pytest.mark.asyncio
    async def test_fires_memory_and_analyzer(self):
        from unittest.mock import AsyncMock
        store = await create_in_memory_store()
        memory = AsyncMock()
        memory.post_turn_tasks = AsyncMock()
        analyzer = AsyncMock()
        analyzer.analyze_turn = AsyncMock()
        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()

        orchestrator = AgentOrchestrator(
            agent=None, session_store=store,
            memory_manager=memory, analyzer=analyzer, event_bus=event_bus,
        )

        from app.models.schemas import EnrichedTrace
        enriched = EnrichedTrace(
            session_id="bg-test", turn=1,
            timestamp="2026-01-01T00:00:00Z",
            pipeline_steps=[], total_duration_ms=100.0,
        )

        await orchestrator._fire_background_tasks(
            session_id="bg-test", turn=1,
            message="hello", response_text="hi",
            tool_calls_made=[], enriched=enriched,
            force_summary=False, total_ms=100.0,
        )
        # Let background tasks complete
        await orchestrator.shutdown(timeout=2.0)

        memory.post_turn_tasks.assert_called_once()
        analyzer.analyze_turn.assert_called_once()
        event_bus.emit.assert_called_once()

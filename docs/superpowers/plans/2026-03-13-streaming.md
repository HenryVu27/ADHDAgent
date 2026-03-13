# SSE Streaming Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking `POST /api/chat` request/response with true SSE token streaming via `POST /api/chat/stream`, using LangGraph `astream_events(version="v2")` to stream tokens from the Gemini ReAct agent as they arrive.

**Architecture:** The orchestrator gains a `process_stream()` async generator that drives `astream_events`, filters events by checkpoint namespace and chunk type, and yields `(event_type, data)` tuples. The FastAPI endpoint wraps this in a `StreamingResponse`. The frontend consumes SSE via `fetch` + `ReadableStream`, feeds tokens into a `useTypewriter` hook that uses `requestAnimationFrame` to display them at a smooth, human-readable pace.

**Tech Stack:** Python asyncio.timeout (3.11+), LangGraph astream_events v2, FastAPI StreamingResponse, React 19 requestAnimationFrame, Tailwind CSS, TypeScript

**Spec:** `docs/superpowers/specs/2026-03-13-streaming-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `app/models/schemas.py` | Modify | Remove `ChatResponse`, add `StreamDonePayload` |
| `app/agent/orchestrator.py` | Modify | Add `process_stream()`, remove `process()` |
| `app/api/routes.py` | Modify | Replace `POST /api/chat` with `POST /api/chat/stream` |
| `tests/test_agent_orchestrator.py` | Modify | Rewrite history-exclusion tests; add `process_stream` tests |
| `tests/test_api.py` | Modify | Update all `/api/chat` calls to use streaming endpoint |
| `frontend-react/src/types/index.ts` | Modify | Add `StreamEvent` union; remove unused `ChatResponse` interface |
| `frontend-react/src/lib/api.ts` | Modify | Replace `chat()` with `chatStream()` async generator |
| `frontend-react/src/hooks/use-typewriter.ts` | **Create** | rAF-based adaptive typewriter with `reset()` |
| `frontend-react/src/hooks/use-chat.ts` | Modify | Consume SSE, manage streaming/loading/status state |
| `frontend-react/src/components/chat/ChatContainer.tsx` | Modify | Status line, in-progress streaming bubble, smart scroll |
| `frontend-react/src/components/chat/ChatBubble.tsx` | Modify | Streaming variant using `useTypewriter` |
| `frontend-react/src/components/chat/ChatInput.tsx` | Modify | Stop button while `isStreaming` |

---

## Chunk 1: Backend

### Task 1: Schema changes

**Files:**
- Modify: `app/models/schemas.py`

- [ ] **Step 1: Add `StreamDonePayload`, remove `ChatResponse`**

In `app/models/schemas.py`, replace the `ChatResponse` class with `StreamDonePayload`:

```python
# Remove this:
class ChatResponse(BaseModel):
    response: str
    agent_used: str
    phase: ConversationPhase
    pipeline_trace: PipelineTrace
    session_id: str

# Add this:
class StreamDonePayload(BaseModel):
    session_id: str
    agent_used: str                        # "react_agent" | "flash_react_agent" | "input_gate"
    phase: str                             # ConversationPhase value
    pipeline_trace: PipelineTrace | None = None
    response: str | None = None            # Only populated on input-blocked path
```

- [ ] **Step 2: Check for any other imports of `ChatResponse` and update them**

```bash
grep -r "ChatResponse" /Users/vuducdung/personal/ADHDAgent/app /Users/vuducdung/personal/ADHDAgent/tests
```

Expected output: references in `orchestrator.py`, `routes.py`, `test_api.py`. These will all be addressed in subsequent tasks.

- [ ] **Step 3: Commit**

```bash
git add app/models/schemas.py
git commit -m "Replace ChatResponse with StreamDonePayload in schemas"
```

---

### Task 2: `process_stream()` on the orchestrator

**Files:**
- Modify: `app/agent/orchestrator.py`

This is the most complex task. `process_stream()` is an async generator that:
1. Loads history and builds the input (same as `process()`)
2. Calls `astream_events(version="v2")`
3. Filters events to emit `status`, `token`, `replace`, `done`, and `error` events
4. Handles timeout via `asyncio.timeout`
5. Does post-stream persistence before emitting `done`

- [ ] **Step 1: Add imports at the top of `orchestrator.py`**

Add to the import block:
```python
import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
```

(Most are already present — just verify `AsyncGenerator` is imported from `collections.abc`.)

- [ ] **Step 2: Add the `_STATUS_MAP` and `_STATIC_FALLBACK` constants inside `AgentOrchestrator`**

Add as class-level constants at the top of `AgentOrchestrator`:

```python
_STATIC_FALLBACK = (
    "I want to make sure I give you the best help. "
    "Could you tell me a bit more about what you'd like to focus on?"
)

_TOOL_STATUS_MAP = {
    "search_knowledge_base": "Looking up strategies...",
    "get_family_profile": "Reading your profile...",
    "update_family_profile": "Updating your profile...",
    "track_outcome": "Recording outcome...",
    "manage_goals": "Managing goals...",
}
```

- [ ] **Step 3: Add the `_extract_token_text` static method**

This handles Gemini's dual content format (plain str vs list[dict] with type="text"/"thinking"):

```python
@staticmethod
def _extract_token_text(chunk) -> str:
    """Extract streamable text from an AIMessageChunk, skipping thinking parts."""
    content = getattr(chunk, "content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content if isinstance(content, str) else ""
```

- [ ] **Step 4: Write `process_stream()` — the main generator**

Add this method to `AgentOrchestrator`, after `process()` (which will be removed in the next step):

```python
async def process_stream(
    self, message: str, session_id: str
) -> AsyncGenerator[tuple[str, dict], None]:
    """Stream the ReAct agent response as SSE (event_type, data) tuples.

    Event sequence:
        status  — input gate passed; each tool invocation
        token   — one text chunk from the final LLM response
        replace — output gate replaced the streamed response
        done    — stream complete, session committed
        error   — timeout or unhandled exception

    Design notes (see docs/superpowers/specs/2026-03-13-streaming-design.md):
    - Tokens filtered by: (1) checkpoint namespace starting with pro/flash_react_agent,
      (2) no tool_call_chunks, (3) skip thinking content parts (type="thinking").
    - Route captured from first on_chain_start with langgraph_node == pro/flash_react_agent.
    - Output gate replacement detected from trace_steps, not string comparison.
    - On error/timeout: persists fallback text, yields error event, returns.
    - Memory and analyzer run as background tasks after done is emitted.
    """
    turn = await self._session_store.increment_turn(session_id)
    logger.info(
        "[agent:stream] === START === session=%s, turn=%d, message=%.80s",
        session_id, turn, message,
    )

    if self._event_bus:
        await self._event_bus.emit(
            "agent", "turn_start", session_id, turn,
            detail={"message_preview": message[:80]},
        )

    from app.config import settings

    # Build history (same logic as process())
    stored_messages = await self._session_store.get_messages(session_id)
    latest_summary = await self._session_store.get_latest_summary(session_id)
    summary_through_turn = latest_summary.covers_through_turn if latest_summary else 0

    history_messages = []
    unsummarized_chars = 0
    for entry in stored_messages:
        if entry.get("blocked"):
            continue
        msg_turn = entry.get("turn", 0)
        if summary_through_turn > 0 and msg_turn <= summary_through_turn:
            continue
        if entry["role"] == "user":
            history_messages.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            history_messages.append(AIMessage(content=entry["content"]))
        unsummarized_chars += len(entry.get("content", ""))

    # Trigger rolling summary if unsummarized history fills context budget
    context_utilization = unsummarized_chars / settings.CONTEXT_MAX_CHARS
    force_summary = context_utilization >= 0.8

    config = {
        "configurable": {"session_id": session_id},
        "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
    }

    start = time.time()
    route = "pro"
    accumulated_text = ""
    result_state: dict = {}

    try:
        async with asyncio.timeout(settings.CHAT_TIMEOUT_S):
            async for event in self._agent.astream_events(
                {
                    "messages": history_messages + [HumanMessage(content=message)],
                    "session_id": session_id,
                },
                config=config,
                version="v2",
            ):
                etype = event["event"]
                meta = event.get("metadata", {})
                node = meta.get("langgraph_node", "")
                ns = meta.get("langgraph_checkpoint_ns", "")

                # Detect which react agent ran (sets route for agent_used label)
                if etype == "on_chain_start" and node in ("pro_react_agent", "flash_react_agent"):
                    route = "flash" if node == "flash_react_agent" else "pro"
                    yield ("status", {"text": "Thinking..."})

                # Named status per tool call
                elif etype == "on_tool_start" and node == "tools":
                    tool_name = event.get("name", "")
                    yield ("status", {"text": self._TOOL_STATUS_MAP.get(tool_name, "Working on it...")})

                # Stream final-response tokens only
                elif (
                    etype == "on_chat_model_stream"
                    and node == "agent"
                    and (ns.startswith("pro_react_agent") or ns.startswith("flash_react_agent"))
                ):
                    chunk = event["data"]["chunk"]
                    if getattr(chunk, "tool_call_chunks", None):
                        continue  # tool-invocation step — skip
                    text = self._extract_token_text(chunk)
                    if text:
                        accumulated_text += text
                        yield ("token", {"text": text})

                # Capture final graph state from top-level on_chain_end
                elif etype == "on_chain_end" and not node and not ns:
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        result_state = output

    except asyncio.TimeoutError:
        logger.warning("[agent:stream] Timeout session=%s", session_id)
        await self._session_store.add_message(session_id, "user", message, turn)
        await self._session_store.add_message(session_id, "assistant", self._STATIC_FALLBACK, turn)
        await self._session_store.commit()
        yield ("error", {"message": "Request timed out. Please try again."})
        return

    except Exception as exc:
        logger.warning("[agent:stream] Exception session=%s: %s", session_id, exc)
        await self._session_store.add_message(session_id, "user", message, turn)
        await self._session_store.add_message(session_id, "assistant", self._STATIC_FALLBACK, turn)
        await self._session_store.commit()
        yield ("error", {"message": "Something went wrong. Please try again."})
        return

    total_ms = (time.time() - start) * 1000

    # --- Input-blocked path ---
    if result_state.get("input_blocked"):
        response_text = result_state.get("block_response", "")
        blocked_reason = ""
        for step in result_state.get("trace_steps", []):
            if step.get("name") == "input_gate":
                blocked_reason = step.get("detail", {}).get("blocked_reason", "")

        await self._session_store.add_message(
            session_id, "user", message, turn,
            blocked=True, blocked_reason=blocked_reason,
        )
        await self._session_store.add_message(
            session_id, "assistant", response_text, turn,
            blocked=True, blocked_reason=blocked_reason,
        )
        enriched = EnrichedTrace(
            session_id=session_id,
            turn=turn,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_steps=[PipelineStep(**s) for s in result_state.get("trace_steps", [])],
            total_duration_ms=total_ms,
            input_blocked=True,
            blocked_reason=blocked_reason,
            agent_used="input_gate",
        )
        await self._session_store.save_trace(session_id, enriched)
        await self._session_store.commit()

        if self._event_bus:
            await self._event_bus.emit(
                "agent", "turn_blocked", session_id, turn, total_ms,
                detail={"reason": blocked_reason},
            )

        trace = self._build_trace(result_state, total_ms)
        phase = await self._infer_phase(session_id)
        yield ("done", {
            "session_id": session_id,
            "agent_used": "input_gate",
            "phase": phase.value,
            "pipeline_trace": trace.model_dump(),
            "response": response_text,
        })
        return

    # --- Normal path: extract final response from result state ---
    all_messages = result_state.get("messages", [])
    last_human_idx = -1
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            last_human_idx = i
            break
    new_messages = all_messages[last_human_idx + 1:] if last_human_idx >= 0 else []

    final_response = ""
    tool_calls_made = []
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_calls_made.extend(msg.tool_calls)
            elif msg.content:
                final_response = _extract_text(msg.content)

    if not final_response.strip():
        final_response = self._STATIC_FALLBACK

    # Emit replace event if output gate replaced the streamed content
    output_gate_replaced = any(
        s.get("name") == "output_gate" and not s.get("detail", {}).get("is_valid", True)
        for s in result_state.get("trace_steps", [])
    )
    if output_gate_replaced:
        yield ("replace", {"text": final_response})

    # Post-processing — these block done to keep trace accurate
    tool_summary = self._build_tool_calls_summary(tool_calls_made)
    await self._session_store.add_message(session_id, "user", message, turn)
    await self._session_store.add_message(
        session_id, "assistant", final_response, turn,
        tool_calls_summary=tool_summary,
    )

    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            tc_name = tc_query = ""
            for tc in tool_calls_made:
                if tc.get("id") == msg.tool_call_id:
                    tc_name = tc.get("name", "")
                    tc_query = str(tc.get("args", {}).get("query", ""))
                    break
            if tc_name == "search_knowledge_base":
                result_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                await self._session_store.save_tool_result(
                    session_id, tc_name, tc_query, result_text, turn,
                )

    enriched = self._build_enriched_trace(
        session_id, turn, result_state, new_messages, tool_calls_made, total_ms,
    )
    await self._session_store.save_trace(session_id, enriched)
    await self._session_store.commit()

    # Background tasks — non-blocking, do not delay done
    if self._memory:
        self._track_task(
            self._memory.post_turn_tasks(
                session_id=session_id,
                turn=turn,
                user_message=message,
                assistant_response=final_response,
                tool_calls=list(tool_calls_made),
                force_summary=force_summary,
            ),
            "memory",
        )

    if self._analyzer:
        self._track_task(
            self._analyzer.analyze_turn(
                session_id=session_id,
                turn=turn,
                user_message=message,
                assistant_response=final_response,
                enriched_trace=enriched,
            ),
            "analyzer",
        )

    if self._event_bus:
        await self._event_bus.emit(
            "agent", "turn_end", session_id, turn, total_ms,
            detail={"tools": len(tool_calls_made), "model_tier": enriched.model_tier},
        )

    agent_label = "flash_react_agent" if route == "flash" else "react_agent"
    trace = self._build_trace(result_state, total_ms, tool_calls_made)
    phase = await self._infer_phase(session_id)

    logger.info(
        "[agent:stream] === END === session=%s, tools=%d, duration=%.0fms",
        session_id, len(tool_calls_made), total_ms,
    )

    yield ("done", {
        "session_id": session_id,
        "agent_used": agent_label,
        "phase": phase.value,
        "pipeline_trace": trace.model_dump(),
        "response": None,
    })
```

- [ ] **Step 5: Remove `process()` from `AgentOrchestrator`**

Delete the entire `process()` method (lines 63–284 in the current file). The method signature is:
```python
async def process(self, message: str, session_id: str) -> ChatResponse:
```

- [ ] **Step 6: Remove the now-unused `ChatResponse` import from `orchestrator.py`**

In the schemas import at the top of `orchestrator.py`, remove `ChatResponse` from the import list:
```python
# Before:
from app.models.schemas import (
    AgentReasoningStep,
    ChatResponse,
    ...
)

# After:
from app.models.schemas import (
    AgentReasoningStep,
    ...
)
```

- [ ] **Step 7: Commit**

```bash
git add app/agent/orchestrator.py
git commit -m "Add process_stream() streaming generator, remove blocking process()"
```

---

### Task 3: Replace `/api/chat` with `/api/chat/stream`

**Files:**
- Modify: `app/api/routes.py`

- [ ] **Step 1: Update the imports in `routes.py`**

Remove `ChatResponse` from the schemas import. Remove `asyncio` (no longer needed for `wait_for`). Add `StreamingResponse` and `json` imports:

```python
import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import AgentOrchestrator
from app.api.deps import get_knowledge_base, get_orchestrator
from app.api.rate_limit import limiter
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    MessagesResponse,
    OutcomesResponse,
    SeedSessionRequest,
    SessionResponse,
    SessionsResponse,
)
from app.rag.knowledge_store import KnowledgeStore
```

- [ ] **Step 2: Replace the `chat` endpoint with `chat_stream`**

Remove the existing `@router.post("/chat", response_model=ChatResponse)` function and replace with:

```python
@router.post("/chat/stream")
@limiter.limit(lambda: settings.RATE_LIMIT_CHAT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """
    Streaming chat endpoint. Emits Server-Sent Events:
      - status  { text }              — pipeline stage updates
      - token   { text }              — one LLM response chunk
      - replace { text }              — output gate replaced the response
      - done    { session_id, agent_used, phase, pipeline_trace, response? }
      - error   { message }           — timeout or unhandled exception

    Design: uses LangGraph astream_events(version="v2") to stream tokens
    from the ReAct agent's final response in real time. See spec at
    docs/superpowers/specs/2026-03-13-streaming-design.md.
    """
    async def event_generator():
        async for event_type, data in orchestrator.process_stream(
            message=body.message,
            session_id=body.session_id,
        ):
            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 3: Commit**

```bash
git add app/api/routes.py
git commit -m "Replace /api/chat with /api/chat/stream SSE endpoint"
```

---

### Task 4: Rewrite `test_agent_orchestrator.py`

**Files:**
- Modify: `tests/test_agent_orchestrator.py`

The history-exclusion tests called `orchestrator.process()` which is removed. They need to call `process_stream()` instead and consume the generator to trigger the agent. The `TestBuildToolCallsSummary` and phase/session tests don't call `process()` and remain unchanged.

- [ ] **Step 1: Write a helper to collect all events from `process_stream()`**

Add this helper at the top of the test file after imports:

```python
async def collect_stream(generator) -> list[tuple[str, dict]]:
    """Drain a process_stream() generator into a list of (event_type, data) tuples."""
    events = []
    async for event_type, data in generator:
        events.append((event_type, data))
    return events
```

- [ ] **Step 2: Create a mock agent that returns events for a successful turn**

Add this factory function:

```python
def _make_streaming_mock_agent(response_text="I'm here to help."):
    """Mock agent whose astream_events yields a minimal successful event sequence."""
    from langchain_core.messages import AIMessageChunk

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
```

- [ ] **Step 3: Update `TestHistoryExcludesSummarizedTurns` to use `process_stream()`**

Replace the `_make_orchestrator` helper and both test methods. The tests verify the same invariant (history content) but capture it via the `astream_events` call args instead of `ainvoke` call args:

```python
class TestHistoryExcludesSummarizedTurns:

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

    async def test_history_excludes_summarized_turns(self):
        store, agent, orchestrator = await self._make_orchestrator()
        sid = "summ-test"

        for i in range(1, 9):
            await store.increment_turn(sid)
            await store.add_message(sid, "user", f"user-msg-{i}", i)
            await store.add_message(sid, "assistant", f"assistant-msg-{i}", i)

        await store.save_summary(
            sid,
            SessionSummary(summary="Summary of turns 1-5.", covers_through_turn=5),
        )
        await store.commit()

        await collect_stream(orchestrator.process_stream("new question", sid))

        messages_passed = agent._last_call_input["messages"]
        contents = [m.content for m in messages_passed if hasattr(m, "content")]

        for i in range(1, 6):
            assert f"user-msg-{i}" not in contents
            assert f"assistant-msg-{i}" not in contents

        for i in range(6, 9):
            assert f"user-msg-{i}" in contents

        assert "new question" in contents

    async def test_no_summary_includes_all_turns(self):
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

        for i in range(1, 4):
            assert f"user-msg-{i}" in contents
        assert "new question" in contents
```

- [ ] **Step 4: Add `TestProcessStream` class**

```python
class TestProcessStream:

    async def _make(self, response_text="Hello from Ally."):
        store = await create_in_memory_store()
        agent = _make_streaming_mock_agent(response_text)
        orchestrator = AgentOrchestrator(agent=agent, session_store=store)
        return store, agent, orchestrator

    async def test_emits_status_then_tokens_then_done(self):
        _, _, orchestrator = await self._make("Hello there!")
        events = await collect_stream(orchestrator.process_stream("Hi", "s1"))

        types = [e[0] for e in events]
        assert "status" in types
        assert "token" in types
        assert types[-1] == "done"

    async def test_tokens_reconstruct_response(self):
        response = "Great idea for your child."
        _, _, orchestrator = await self._make(response)
        events = await collect_stream(orchestrator.process_stream("Hi", "s2"))

        tokens = "".join(d["text"] for t, d in events if t == "token")
        assert tokens.strip() == response

    async def test_done_payload_has_required_fields(self):
        _, _, orchestrator = await self._make()
        events = await collect_stream(orchestrator.process_stream("Hi", "s3"))

        done_event = next(d for t, d in events if t == "done")
        assert done_event["session_id"] == "s3"
        assert done_event["agent_used"] in ("react_agent", "flash_react_agent", "input_gate")
        assert done_event["phase"] in ("intake", "strategy", "progress", "followup")
        assert done_event["pipeline_trace"] is not None

    async def test_message_persisted_after_stream(self):
        store, _, orchestrator = await self._make("Nice response.")
        await collect_stream(orchestrator.process_stream("my question", "s4"))

        msgs = await store.get_messages("s4")
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    async def test_timeout_yields_error_event(self):
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

    async def test_thinking_tokens_are_skipped(self):
        """Gemini 2.5 Pro thinking content (type='thinking') must not appear in tokens."""
        from langchain_core.messages import AIMessageChunk

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

    async def test_tool_call_tokens_are_skipped(self):
        """Tokens from tool-invocation steps (tool_call_chunks non-empty) must be filtered."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.messages.tool import ToolCallChunk

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
```

- [ ] **Step 5: Run the orchestrator tests**

```bash
./adhd312/Scripts/python.exe -m pytest tests/test_agent_orchestrator.py -v
```

Expected: all tests pass (phase/session/summary tests still work, new stream tests pass).

- [ ] **Step 6: Commit**

```bash
git add tests/test_agent_orchestrator.py
git commit -m "Rewrite orchestrator tests for process_stream()"
```

---

### Task 5: Update `test_api.py`

**Files:**
- Modify: `tests/test_api.py`

All calls to `POST /api/chat` must change to `POST /api/chat/stream`. The tests need a helper that consumes the SSE stream and returns the `done` payload as a dict (mimicking the old JSON response shape for backward test compatibility).

- [ ] **Step 1: Update the mock agent in `_make_mock_agent()`**

The mock agent now needs `astream_events` instead of `ainvoke`:

```python
def _make_mock_agent():
    """Mock agent whose astream_events yields a minimal successful event sequence."""
    from langchain_core.messages import AIMessageChunk

    response_text = "I'm here to help with ADHD parenting strategies."

    async def mock_astream_events(*args, **kwargs):
        yield {
            "event": "on_chain_start",
            "name": "LangGraph",
            "metadata": {
                "langgraph_node": "pro_react_agent",
                "langgraph_checkpoint_ns": "pro_react_agent:test",
            },
            "data": {},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {
                "langgraph_node": "agent",
                "langgraph_checkpoint_ns": "pro_react_agent:test",
            },
            "data": {"chunk": AIMessageChunk(content=response_text)},
        }
        yield {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "", "langgraph_checkpoint_ns": ""},
            "data": {
                "output": {
                    "messages": [
                        HumanMessage(content="test message"),
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
```

- [ ] **Step 2: Add an SSE consumer helper**

```python
import json as _json

async def _consume_stream(response) -> dict:
    """Parse a streaming /api/chat/stream response and return the done payload."""
    assert response.status_code == 200
    done_data = {}
    buffer = ""
    async for chunk in response.aiter_bytes():
        buffer += chunk.decode()
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event_type = ""
            data_str = ""
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]
            if event_type == "done" and data_str:
                done_data = _json.loads(data_str)
    return done_data
```

- [ ] **Step 3: Update all `/api/chat` calls to `/api/chat/stream`**

For each test that calls `/api/chat`, change to:
```python
# Before:
response = await client.post("/api/chat", json={...})
assert response.status_code == 200
data = response.json()
assert "response" in data

# After:
response = await client.post("/api/chat/stream", json={...})
data = await _consume_stream(response)
assert "session_id" in data
```

Specifically update these tests:
- `test_chat_endpoint` — check all `StreamDonePayload` fields present in done payload:
  ```python
  async def test_chat_endpoint(client):
      response = await client.post("/api/chat/stream", json={
          "message": "Hi, I need help with my child",
          "session_id": "api_test",
      })
      data = await _consume_stream(response)
      assert data["session_id"] == "api_test"
      assert "agent_used" in data
      assert "phase" in data
      assert "pipeline_trace" in data
  ```
- `test_chat_returns_pipeline_trace` — parse done payload, check `pipeline_trace`
- `test_session_endpoint` — call `/api/chat/stream` first to create session, then GET session
- `test_outcomes_endpoint` — same pattern
- `test_chat_rejects_empty_message` — still hits `/api/chat/stream` (FastAPI validates before streaming), expects 422
- `test_chat_rejects_whitespace_message` — same
- `test_chat_rejects_oversized_message` — same
- `test_chat_timeout` — build a slow `astream_events` mock agent and use the real `process_stream()` so `asyncio.timeout` fires; verify `error` event in stream response body
- `test_rate_limit_on_chat` — use `/api/chat/stream`
- `test_sessions_list_pagination` — use streaming endpoint to create sessions
- `test_messages_pagination` — same

The timeout test must use the real `process_stream()` with a slow mock agent (not patch `process_stream` directly, which would bypass the `asyncio.timeout` logic inside it). Replace with:

```python
async def test_chat_timeout(client):
    from app.config import settings

    # Build a mock agent whose astream_events sleeps indefinitely.
    # Use the real process_stream() so asyncio.timeout() fires correctly.
    async def slow_astream_events(*args, **kwargs):
        await asyncio.sleep(60)
        yield {}  # never reached

    slow_agent = AsyncMock()
    slow_agent.astream_events = slow_astream_events
    store = await create_in_memory_store()
    slow_orchestrator = AgentOrchestrator(agent=slow_agent, session_store=store)

    # Temporarily replace the orchestrator for this test
    app.dependency_overrides[get_orchestrator] = lambda: slow_orchestrator

    original = settings.CHAT_TIMEOUT_S
    settings.CHAT_TIMEOUT_S = 0.05
    try:
        response = await client.post("/api/chat/stream", json={
            "message": "Hello",
            "session_id": "timeout_test",
        })
        assert response.status_code == 200
        content = response.content.decode()
        assert '"error"' in content
    finally:
        settings.CHAT_TIMEOUT_S = original
        # Fixture teardown calls app.dependency_overrides.clear() — no manual restore needed
```

- [ ] **Step 4: Run the API tests**

```bash
./adhd312/Scripts/python.exe -m pytest tests/test_api.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the full test suite**

```bash
./adhd312/Scripts/python.exe -m pytest tests/ -v -m "not integration"
```

Expected: all non-integration tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_api.py
git commit -m "Update API tests to use /api/chat/stream SSE endpoint"
```

---

## Chunk 2: Frontend

### Task 6: TypeScript streaming types

**Files:**
- Modify: `frontend-react/src/types/index.ts`

- [ ] **Step 1: Remove `ChatResponse` interface**

Delete lines 81–87:
```typescript
export interface ChatResponse {
  response: string
  agent_used: string
  phase: ConversationPhase
  pipeline_trace: PipelineTrace
  session_id: string
}
```

- [ ] **Step 2: Add streaming event types after `ChatRequest`**

```typescript
// SSE streaming events from POST /api/chat/stream
export interface StreamStatusEvent {
  type: "status"
  text: string
}

export interface StreamTokenEvent {
  type: "token"
  text: string
}

export interface StreamReplaceEvent {
  type: "replace"
  text: string
}

export interface StreamDoneEvent {
  type: "done"
  session_id: string
  agent_used: string
  phase: ConversationPhase
  pipeline_trace: PipelineTrace | null
  response: string | null  // only set on input-blocked path
}

export interface StreamErrorEvent {
  type: "error"
  message: string
}

export type StreamEvent =
  | StreamStatusEvent
  | StreamTokenEvent
  | StreamReplaceEvent
  | StreamDoneEvent
  | StreamErrorEvent
```

- [ ] **Step 3: Check TypeScript compilation**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build 2>&1 | head -30
```

Expected: `ChatResponse` is unused — confirm there are no other references to it.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/types/index.ts
git commit -m "Add StreamEvent types, remove ChatResponse TypeScript interface"
```

---

### Task 7: `chatStream()` in `api.ts`

**Files:**
- Modify: `frontend-react/src/lib/api.ts`

- [ ] **Step 1: Remove the `chat()` method and add `chatStream()`**

Replace `chat(data: ChatRequest)` with an async generator:

```typescript
async *chatStream(
  data: ChatRequest,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    signal,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `Request failed: ${res.status}`)
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // Split on SSE message boundaries (\n\n)
      // Keep the trailing incomplete chunk in buffer
      const blocks = buffer.split("\n\n")
      buffer = blocks.pop() ?? ""

      for (const block of blocks) {
        if (!block.trim()) continue
        let eventType = ""
        let dataStr = ""
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) eventType = line.slice(7).trim()
          else if (line.startsWith("data: ")) dataStr = line.slice(6)
        }
        if (!eventType || !dataStr) continue
        try {
          const parsed = JSON.parse(dataStr)
          yield { type: eventType, ...parsed } as StreamEvent
        } catch {
          // malformed JSON — skip
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
},
```

Update the import in `api.ts` to include `StreamEvent`:
```typescript
import type {
  ChatRequest,
  StreamEvent,
  SessionResponse,
  // ... rest unchanged
} from "@/types"
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build 2>&1 | head -30
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/lib/api.ts
git commit -m "Replace chat() with chatStream() async generator in api.ts"
```

---

### Task 8: `useTypewriter` hook

**Files:**
- Create: `frontend-react/src/hooks/use-typewriter.ts`

This hook decouples token receipt from display. Text accumulates in a ref (no re-render on receipt); a `requestAnimationFrame` loop advances a display pointer at an adaptive rate; React state updates are batched every 30 characters to avoid thrashing `dangerouslySetInnerHTML` markdown rendering at 60fps.

- [ ] **Step 1: Create the hook file**

```typescript
import { useRef, useState, useCallback, useEffect } from "react"

const STEP_NORMAL = 12          // chars per frame at 60fps (~720 chars/sec)
const BATCH_THRESHOLD = 30      // re-render every N chars advanced
const CATCHUP_THRESHOLD = 100   // if this far behind, catch up faster

interface TypewriterControls {
  displayedText: string
  reset: (newText: string) => void
}

/**
 * Adaptive typewriter for streaming LLM responses.
 *
 * Problem: LLM tokens arrive in irregular bursts (1–50+ chars). Rendering
 * each chunk directly produces stuttery text.
 *
 * Solution: Decouple receipt from display via requestAnimationFrame:
 *   - fullRef (ref, not state) accumulates received tokens without re-renders
 *   - rAF loop advances shownRef at STEP_NORMAL chars/frame (≈720 chars/sec)
 *   - When buffer is CATCHUP_THRESHOLD+ chars ahead, step = ceil(behind/10)
 *   - React state only updates every BATCH_THRESHOLD chars, keeping markdown
 *     parsing at ~20fps instead of 60fps (avoids layout thrash)
 *
 * reset(newText): used by use-chat when a replace event arrives (output gate
 * replaced the streamed response). Resets fullRef and shownRef to the new text
 * and restarts the rAF loop.
 */
export function useTypewriter(fullText: string): TypewriterControls {
  const fullRef = useRef("")
  const shownRef = useRef(0)
  const lastRenderedRef = useRef(0)
  const rafRef = useRef<number | null>(null)
  const [displayedText, setDisplayedText] = useState("")

  // Keep fullRef in sync with incoming fullText
  useEffect(() => {
    fullRef.current = fullText
    scheduleFrame()
  }, [fullText]) // eslint-disable-line react-hooks/exhaustive-deps

  const scheduleFrame = useCallback(() => {
    if (rafRef.current !== null) return  // already scheduled
    if (shownRef.current >= fullRef.current.length) return
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const tick = useCallback(() => {
    rafRef.current = null
    const full = fullRef.current
    const shown = shownRef.current

    if (shown >= full.length) return

    const behind = full.length - shown
    const step = behind > CATCHUP_THRESHOLD ? Math.ceil(behind / 10) : STEP_NORMAL
    shownRef.current = Math.min(shown + step, full.length)

    const advanced = shownRef.current - lastRenderedRef.current
    if (advanced >= BATCH_THRESHOLD || shownRef.current === full.length) {
      setDisplayedText(full.slice(0, shownRef.current))
      lastRenderedRef.current = shownRef.current
    }

    if (shownRef.current < full.length) {
      rafRef.current = requestAnimationFrame(tick)
    }
  }, [])

  const reset = useCallback((newText: string) => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    fullRef.current = newText
    shownRef.current = 0
    lastRenderedRef.current = 0
    setDisplayedText("")
    scheduleFrame()
  }, [scheduleFrame])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return { displayedText, reset }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build 2>&1 | head -30
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/hooks/use-typewriter.ts
git commit -m "Add useTypewriter hook with adaptive rAF animation"
```

---

### Task 9: Refactor `use-chat.ts`

**Files:**
- Modify: `frontend-react/src/hooks/use-chat.ts`

- [ ] **Step 1: Rewrite `use-chat.ts`**

Two stale-closure hazards exist in a naive streaming hook:
- `isLoading` state read inside `for await` would always see the value from when `sendMessage` was memoized. Fix: track first-token with a local `let firstToken` variable.
- `streamingContent` state read in `_finalize` for the done payload would see the initial empty value. Fix: accumulate tokens in `accumulatedRef` (a ref, updated on every token event), and read from it in `_finalize`.

```typescript
import { useState, useCallback, useRef } from "react"
import type { ChatMessage, PipelineTrace, StreamDoneEvent } from "@/types"
import { api } from "@/lib/api"

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)       // waiting for first token
  const [isStreaming, setIsStreaming] = useState(false)    // tokens arriving
  const [statusText, setStatusText] = useState("")         // current pipeline stage label
  const [streamingContent, setStreamingContent] = useState("") // drives typewriter display
  const [latestTrace, setLatestTrace] = useState<PipelineTrace | null>(null)
  const idCounter = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  // accumulatedRef: source of truth for token accumulation (avoids stale closure on done)
  const accumulatedRef = useRef("")
  // Exposed so StreamingBubble can call typewriter.reset() on replace events
  const typewriterResetRef = useRef<((text: string) => void) | null>(null)

  const sendMessage = useCallback(async (content: string) => {
    const userMsg: ChatMessage = {
      id: `msg_${++idCounter.current}`,
      role: "user",
      content,
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setStreamingContent("")
    setStatusText("")
    accumulatedRef.current = ""

    const controller = new AbortController()
    abortRef.current = controller

    // Local variable to detect first token without stale closure on isLoading state
    let firstToken = true

    try {
      for await (const event of api.chatStream(
        { message: content, session_id: sessionId },
        controller.signal,
      )) {
        if (event.type === "status") {
          setStatusText(event.text)

        } else if (event.type === "token") {
          if (firstToken) {
            firstToken = false
            setIsLoading(false)
            setIsStreaming(true)
            setStatusText("")
          }
          accumulatedRef.current += event.text
          setStreamingContent(prev => prev + event.text)

        } else if (event.type === "replace") {
          // Output gate replaced the streamed content — swap both the display ref and typewriter
          accumulatedRef.current = event.text
          setStreamingContent(event.text)
          typewriterResetRef.current?.(event.text)

        } else if (event.type === "done") {
          _finalize(event)

        } else if (event.type === "error") {
          setMessages(prev => [...prev, {
            id: `msg_${++idCounter.current}`,
            role: "assistant",
            content: "Something went wrong. Please try again.",
            timestamp: new Date(),
          }])
          _clearStreamState()
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        // User cancelled — discard partial content, do not commit to messages
        _clearStreamState()
      } else {
        setMessages(prev => [...prev, {
          id: `msg_${++idCounter.current}`,
          role: "assistant",
          content: "Something went wrong. Please try again.",
          timestamp: new Date(),
        }])
        _clearStreamState()
      }
    } finally {
      abortRef.current = null
    }

    function _finalize(done: StreamDoneEvent) {
      // On input-blocked path, response text comes in done.response (no tokens streamed).
      // Otherwise, use accumulatedRef which has the real token content — NOT streamingContent
      // state, which would be stale (captured at useCallback memoization time).
      const finalContent = done.response ?? accumulatedRef.current
      const assistantMsg: ChatMessage = {
        id: `msg_${++idCounter.current}`,
        role: "assistant",
        content: finalContent,
        timestamp: new Date(),
        agentUsed: done.agent_used,
        pipelineTrace: done.pipeline_trace ?? undefined,
      }
      setMessages(prev => [...prev, assistantMsg])
      if (done.pipeline_trace) setLatestTrace(done.pipeline_trace)
      _clearStreamState()
    }

    function _clearStreamState() {
      setIsLoading(false)
      setIsStreaming(false)
      setStatusText("")
      setStreamingContent("")
      accumulatedRef.current = ""
    }
  }, [sessionId])  // sessionId only — no state in deps (local vars + refs used instead)

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    setLatestTrace(null)
    setStreamingContent("")
    setStatusText("")
    accumulatedRef.current = ""
    idCounter.current = 0
  }, [])

  const loadMessages = useCallback(async () => {
    try {
      const data = await api.getSessionMessages(sessionId)
      if (data.messages.length > 0) {
        const loaded: ChatMessage[] = data.messages
          .filter(m => !m.blocked)
          .map((m) => ({
            id: `msg_${++idCounter.current}`,
            role: m.role as "user" | "assistant",
            content: m.content,
            timestamp: new Date(),
          }))
        setMessages(loaded)
        return loaded.length
      }
      return 0
    } catch {
      return 0
    }
  }, [sessionId])

  return {
    messages,
    isLoading,
    isStreaming,
    statusText,
    streamingContent,
    latestTrace,
    typewriterResetRef,
    sendMessage,
    stopStreaming,
    clearMessages,
    loadMessages,
  }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build 2>&1 | head -40
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/hooks/use-chat.ts
git commit -m "Refactor use-chat to consume SSE with streaming/status/abort state"
```

---

### Task 10: Update chat components

**Files:**
- Modify: `frontend-react/src/components/chat/ChatContainer.tsx`
- Modify: `frontend-react/src/components/chat/ChatBubble.tsx`
- Modify: `frontend-react/src/components/chat/ChatInput.tsx`

Also update any callers of `useChat` (likely `ChatPage.tsx`) to pass the new props.

#### ChatContainer.tsx

The container gains:
- A status line (pulsing dot + status text) shown while `isLoading && statusText`
- A streaming bubble driven by `useTypewriter` while `isStreaming`
- Smart scroll: only auto-scroll when within 120px of bottom

- [ ] **Step 1: Rewrite `ChatContainer.tsx`**

```tsx
import { useRef, useEffect } from "react"
import { motion } from "framer-motion"
import { Sprout } from "lucide-react"
import { ChatBubble } from "./ChatBubble"
import { StreamingBubble } from "./ChatBubble"
import type { ChatMessage } from "@/types"

const SCROLL_THRESHOLD = 120  // px from bottom before auto-scroll disengages

interface Props {
  messages: ChatMessage[]
  isLoading: boolean
  isStreaming: boolean
  statusText: string
  streamingContent: string
  typewriterResetRef: React.MutableRefObject<((text: string) => void) | null>
}

export function ChatContainer({
  messages,
  isLoading,
  isStreaming,
  statusText,
  streamingContent,
  typewriterResetRef,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const scrollToBottomIfNear = () => {
    const el = containerRef.current
    if (!el) return
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distFromBottom < SCROLL_THRESHOLD) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }

  // Scroll on new committed messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Smart scroll during streaming
  useEffect(() => {
    scrollToBottomIfNear()
  }, [streamingContent])

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && !isLoading && !isStreaming && (
        <div className="flex h-full flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-coach/15">
            <Sprout className="h-8 w-8 text-coach" />
          </div>
          <h3 className="mb-1 text-lg font-semibold tracking-tight">
            Hi! I'm Ally, your ADHD parenting coach.
          </h3>
          <p className="max-w-sm text-sm text-muted-foreground leading-relaxed">
            Tell me what's going on with your family, and we'll figure it out together.
          </p>
        </div>
      )}

      {messages.map((msg) => (
        <ChatBubble key={msg.id} message={msg} />
      ))}

      {/* Status line: shown while waiting for first token */}
      {isLoading && statusText && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="rounded-2xl bg-card px-4 py-3 shadow-sm border border-border/30">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-coach" />
              <span className="text-sm text-muted-foreground">{statusText}</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Fallback dots while loading with no status yet */}
      {isLoading && !statusText && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="rounded-2xl bg-card px-4 py-3 shadow-sm border border-border/30">
            <div className="flex items-center gap-1">
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:0ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:150ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:300ms]" />
            </div>
          </div>
        </motion.div>
      )}

      {/* Streaming bubble: typewriter animation */}
      {isStreaming && (
        <StreamingBubble
          content={streamingContent}
          typewriterResetRef={typewriterResetRef}
        />
      )}

      <div ref={bottomRef} />
    </div>
  )
}
```

#### ChatBubble.tsx

Add a `StreamingBubble` export that uses `useTypewriter`:

- [ ] **Step 2: Update `ChatBubble.tsx`**

```tsx
import { useRef } from "react"
import { motion } from "framer-motion"
import { Sprout, User } from "lucide-react"
import type { ChatMessage } from "@/types"
import { useTypewriter } from "@/hooks/use-typewriter"

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(.+)$/s, "<p>$1</p>")
}

interface Props {
  message: ChatMessage
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user"

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-primary" : "bg-coach"
        }`}
      >
        {isUser ? (
          <User className="h-4 w-4 text-primary-foreground" />
        ) : (
          <Sprout className="h-4 w-4 text-coach-foreground" />
        )}
      </div>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-card shadow-sm border border-border/30"
        }`}
      >
        {!isUser && (
          <div className="mb-1 text-xs font-medium text-coach">Ally</div>
        )}
        <div
          className="text-sm leading-relaxed [&_ul]:ml-4 [&_ul]:list-disc [&_ul]:space-y-1 [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
        />
      </div>
    </motion.div>
  )
}

interface StreamingBubbleProps {
  content: string
  typewriterResetRef: React.MutableRefObject<((text: string) => void) | null>
}

/**
 * In-progress assistant bubble during SSE streaming.
 * Uses useTypewriter to animate tokens at a smooth, human-readable pace.
 * Exposes reset() via typewriterResetRef so use-chat can swap content
 * when an output-gate replace event arrives.
 */
export function StreamingBubble({ content, typewriterResetRef }: StreamingBubbleProps) {
  const { displayedText, reset } = useTypewriter(content)

  // Keep typewriterResetRef in sync so use-chat.ts can call reset() on replace events.
  // Direct ref mutation during render is safe here (no state, no side effects).
  if (typewriterResetRef.current !== reset) {
    typewriterResetRef.current = reset
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="flex gap-3"
    >
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
        <Sprout className="h-4 w-4 text-coach-foreground" />
      </div>
      <div className="max-w-[80%] rounded-2xl bg-card px-4 py-3 shadow-sm border border-border/30">
        <div className="mb-1 text-xs font-medium text-coach">Ally</div>
        <div
          className="text-sm leading-relaxed [&_ul]:ml-4 [&_ul]:list-disc [&_ul]:space-y-1 [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(displayedText) }}
        />
      </div>
    </motion.div>
  )
}
```

#### ChatInput.tsx

Add a stop button while `isStreaming`. The textarea is disabled during both loading and streaming.

- [ ] **Step 3: Update `ChatInput.tsx`**

```tsx
import { useState, useRef } from "react"
import { Send, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface Props {
  onSend: (message: string) => void
  onStop: () => void
  isLoading: boolean
  isStreaming: boolean
}

export function ChatInput({ onSend, onStop, isLoading, isStreaming }: Props) {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const busy = isLoading || isStreaming

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || busy) return
    onSend(trimmed)
    setValue("")
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex items-end gap-2 border-t border-border/50 bg-background p-4">
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Share what's going on with your family..."
        className="min-h-[44px] max-h-32 resize-none"
        rows={1}
        disabled={busy}
      />
      {isStreaming ? (
        <Button
          onClick={onStop}
          size="icon"
          variant="outline"
          className="shrink-0"
          title="Stop generating"
        >
          <Square className="h-4 w-4" />
        </Button>
      ) : (
        <Button
          onClick={handleSend}
          disabled={!value.trim() || isLoading}
          size="icon"
          className="shrink-0"
        >
          <Send className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Update `ChatPage.tsx` to pass new props and fix `handleSend`**

`sendMessage` is now `void` — it no longer returns the response data. The existing `handleSend` uses `const data = await sendMessage(content)` to get `data.pipeline_trace.retrieval_results` for the stats update. This needs to change.

The cleanest fix: move the post-send side effects (refresh + stats) into a `useEffect` that fires when `isStreaming` transitions from `true` to `false`. `latestTrace` is already updated by the `done` event handler, so it will be current when the effect fires.

Replace the full `ChatPage.tsx` usage section as follows:

```tsx
// 1. Destructure new fields from useChat (replace existing destructure on line 25):
const {
  messages,
  isLoading,
  isStreaming,
  statusText,
  streamingContent,
  typewriterResetRef,
  latestTrace,
  sendMessage,
  stopStreaming,
  clearMessages,
  loadMessages,
} = useChat(sessionId)

// 2. Track previous isStreaming to detect transition → false
const wasStreamingRef = useRef(false)
useEffect(() => {
  if (wasStreamingRef.current && !isStreaming) {
    // Stream just completed — refresh session state and update stats
    refresh()
    const stats = getSessionStats()
    updateSessionStats({
      sessions: stats.sessions + 1,
      strategies: (latestTrace?.retrieval_results?.length || 0) + stats.strategies,
      streak: stats.streak || 1,
    })
    setShowChips(true)
  }
  wasStreamingRef.current = isStreaming
}, [isStreaming, latestTrace, refresh])

// 3. Simplify handleSend — no longer uses return value:
const handleSend = useCallback(async (content: string) => {
  if (pendingSeed.current) {
    await pendingSeed.current
    pendingSeed.current = null
  }
  setShowChips(false)
  sendMessage(content)
}, [sendMessage])

// 4. Update QuickReplyChips visibility (add isStreaming):
<QuickReplyChips
  phase={phase}
  onSelect={handleChipSelect}
  visible={showChips && !isLoading && !isStreaming}
/>

// 5. Pass to ChatContainer:
<ChatContainer
  messages={messages}
  isLoading={isLoading}
  isStreaming={isStreaming}
  statusText={statusText}
  streamingContent={streamingContent}
  typewriterResetRef={typewriterResetRef}
/>

// 6. Pass to ChatInput:
<ChatInput
  onSend={handleSend}
  onStop={stopStreaming}
  isLoading={isLoading}
  isStreaming={isStreaming}
/>
```

**Note on `latestTrace?.retrieval_results`:** The existing code accesses `data.pipeline_trace.retrieval_results` which is a field on `PipelineTrace`. Verify that `PipelineTrace` in `types/index.ts` has a `retrieval_results` field (it does — it's defined as `retrieval_results: RetrievalResult[]`). `latestTrace` is `PipelineTrace | null`, hence the optional chain.

- [ ] **Step 5: Run TypeScript build**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build 2>&1
```

Expected: clean build with no TypeScript errors.

- [ ] **Step 6: Commit**

```bash
git add frontend-react/src/components/chat/ChatContainer.tsx \
        frontend-react/src/components/chat/ChatBubble.tsx \
        frontend-react/src/components/chat/ChatInput.tsx \
        frontend-react/src/pages/ChatPage.tsx
git commit -m "Add streaming UI: status line, typewriter bubble, stop button"
```

---

## Final verification

- [ ] **Run the full test suite**

```bash
./adhd312/Scripts/python.exe -m pytest tests/ -v -m "not integration"
```

Expected: all non-integration tests pass.

- [ ] **Run the frontend build**

```bash
cd /Users/vuducdung/personal/ADHDAgent/frontend-react && npm run build
```

Expected: clean build.

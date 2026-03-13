# Streaming Design — ADHDAgent

**Date:** 2026-03-13
**Status:** Approved
**Context:** Replacing the blocking `POST /api/chat` request-response with true SSE token streaming. Inspired by the DatDat sibling project which uses the same Gemini + RAG stack and demonstrates noticeably smoother UX through progressive disclosure and typewriter animation.

---

## Problem

The existing `/api/chat` endpoint runs the full ReAct pipeline (`ainvoke`) before returning anything. The user waits 5–15 seconds with no feedback. There is no indication of what the agent is doing, and the full response appears all at once — which feels abrupt for long coaching answers.

---

## Goal

Stream the agent's response token-by-token, with named status updates at each pipeline stage. The user should see progress within ~1 second of sending a message, and the final text should appear character-by-character rather than all at once.

---

## Key Design Decisions

### Decision 1: Use `astream_events(version="v2")` instead of `ainvoke()`

**Alternatives considered:**
- **`ainvoke()` + re-stream**: run the full ReAct loop as-is, then make a separate direct Gemini streaming call to re-generate the response. Simple to implement but doubles LLM cost and adds latency — the user still waits for the full loop before any tokens appear.
- **`astream()` (state-level)**: streams graph state updates but not token-level output. Gives tool call events but not individual characters.
- **`astream_events(version="v2")`**: emits fine-grained events including `on_chat_model_stream` (token-level), `on_tool_start`, `on_tool_end`. This is the only approach that gives true token streaming from the actual agent response.

**Chosen:** `astream_events(version="v2")`. The increase in implementation complexity (event filtering) is worth the latency reduction.

---

### Decision 2: Token filtering strategy — checkpoint namespace + tool_call_chunks + thinking parts

`astream_events` fires `on_chat_model_stream` for every LLM iteration in the ReAct loop, including tool-invocation reasoning steps. Naively forwarding all tokens would show internal reasoning to the user.

**Key observation:** `input_gate` and `output_gate` are plain async functions that call Gemini directly via `app/llm/client.py` — they bypass LangGraph's model nodes entirely. So they emit **no** `on_chat_model_stream` events. This simplifies filtering.

**LangGraph subgraph node naming:** The outer graph has two nodes — `pro_react_agent` and `flash_react_agent` — each of which is a compiled `create_react_agent` subgraph. Inside that subgraph, LangGraph names the LLM node `"agent"`. With `astream_events(version="v2")`, each event carries:
- `metadata["langgraph_node"]`: the node name within the current subgraph (e.g., `"agent"`)
- `metadata["langgraph_checkpoint_ns"]`: the full namespace path (e.g., `"pro_react_agent:abc123"`)

So the secondary node filter is: `metadata["langgraph_node"] == "agent"` AND `metadata["langgraph_checkpoint_ns"]` starts with `"pro_react_agent"` or `"flash_react_agent"`.

**Remaining sources of spurious tokens:**
- Tool-calling steps inside the ReAct loop (agent deciding which tool to call)
- Gemini thinking tokens (Gemini 2.5 Pro with `budget_tokens` emits thinking content before response content)

**Token filter applied (belt and suspenders):**

1. **Checkpoint namespace check**: only process `on_chat_model_stream` events where `metadata["langgraph_checkpoint_ns"]` starts with `"pro_react_agent"` or `"flash_react_agent"`, AND `metadata["langgraph_node"] == "agent"`.

2. **Skip tool-invocation steps**: skip if `chunk.tool_call_chunks` is non-empty.

3. **Skip thinking tokens**: Gemini 2.5 Pro with `thinking_budget > 0` emits thinking content as parts with `type == "thinking"` in `chunk.content`. The LangChain-Google adapter surfaces `AIMessageChunk.content` as either:
   - A plain `str` when thinking is not active (emit directly)
   - A `list[dict]` where each dict has a `type` key: `"thinking"` (skip) or `"text"` (emit the `"text"` value)

   Filter: if `chunk.content` is a list, extract only dicts where `type == "text"`, join their `"text"` values. If `chunk.content` is a str, use directly.

4. Emit only if the resulting text is non-empty after filtering.

---

### Decision 3: `replace` event for output gate violations

The output gate runs **after** the ReAct loop completes. By that point, tokens have already been streamed to the client. If the output gate replaces the response with a safe fallback, we cannot un-send those tokens.

**Alternatives considered:**
- **Buffer all tokens, run output gate, then stream**: defeats the purpose (user waits for full generation before seeing anything).
- **Ignore the mismatch**: user sees the unsafe response. Not acceptable.
- **`replace` event**: client receives all tokens, then receives a `replace` event with the safe fallback text, and swaps the entire displayed content. Slightly jarring but correct, and violations are rare in practice.

**Chosen:** `replace` event. On the frontend, the `replace` event resets both `streamingContent` (React state) and the typewriter hook's internal `fullRef` buffer to the new text, so the typewriter picks up the replacement immediately. The `done` event always reflects the final committed response text. The `useTypewriter` hook must expose a `reset(newText: string)` function for this purpose.

---

### Decision 4: Remove non-streaming endpoint entirely

**Alternatives considered:**
- Keep both `/api/chat` and `/api/chat/stream`: adds maintenance surface, tests need to cover both paths, confusing for future contributors.
- Streaming only: single code path, simpler.

**Chosen:** Replace `POST /api/chat` with `POST /api/chat/stream`. The non-streaming endpoint is removed. All existing tests that hit `/api/chat` are updated to use a test SSE consumer helper.

---

### Decision 5: No `EventSource` on the frontend — use `fetch` + `ReadableStream`

`EventSource` (browser native) only supports `GET` requests with no body. Our endpoint needs a `POST` body (`message`, `session_id`).

**Chosen:** `fetch()` with `response.body.getReader()`. SSE parsing is done manually with a buffer-remainder pattern (handles chunks that split across SSE message boundaries). This is the same approach used in DatDat and is well-understood.

---

### Decision 6: `useTypewriter` hook with adaptive step and batched markdown rendering

Raw token delivery from Gemini is irregular — chunks vary from 1 to 50+ characters. Displaying them directly produces a stuttery effect.

**Approach:** Decouple token receipt from display using a `requestAnimationFrame` loop:
- Tokens accumulate in `fullRef` (a ref, not state — no re-render on receipt)
- An rAF loop advances a `shownRef` pointer through `fullRef.length`
- Adaptive step: if buffer is >100 chars ahead, step = `ceil(behind / 10)` to catch up; otherwise step = 12 chars/frame (~720 chars/sec at 60fps — comfortable reading speed)
- React state (`displayedText`) is updated only every 30 chars advanced, or when fully caught up → markdown parsing runs at ~20fps instead of 60fps, avoiding layout thrash

**Why refs not state for the buffer:** React state updates batch and may be deferred. Using refs for `fullRef` and `shownRef` ensures the rAF loop always sees the latest token and is not delayed by React's scheduler.

**`reset(newText)` function:** The hook exposes `reset(newText: string)` for the `replace` event. It sets `fullRef.current = newText`, `shownRef.current = 0`, clears displayed state, and re-triggers the rAF loop.

---

### Decision 7: Smart scroll — only when within 120px of bottom

Auto-scrolling that fights the user's intentional scroll is a common annoyance. The chat container only scrolls to bottom when `scrollHeight - scrollTop - clientHeight < 120`. This threshold gives a ~2-line buffer before the auto-scroll kicks in.

---

### Decision 8: Post-stream processing — what blocks `done`

After `astream_events` exhausts, `process_stream()` performs post-processing before emitting `done`:

**Blocks `done` (awaited):**
- Session message persistence (`add_message` × 2)
- Enriched trace building and saving (`save_trace`)
- `session_store.commit()`

**Does not block `done` (fire-and-forget background tasks):**
- Memory manager post-turn tasks (`_memory.post_turn_tasks`) — involves an LLM call, could take seconds
- Conversation analyzer (`_analyzer.analyze_turn`) — also LLM-backed

This matches the existing behavior in `process()` and keeps the gap between last token and `done` to ~100–200ms (persistence only).

---

### Decision 9: Retrieving `agent_used` / `route` from the event stream

With `ainvoke()`, route was read from `result.get("route")` after completion. With `astream_events`, the route is determined by watching for `on_chain_start` events where `metadata["langgraph_node"]` is `"pro_react_agent"` or `"flash_react_agent"`. Note: use `metadata["langgraph_node"]`, **not** `event["name"]` — in LangGraph v2, `event["name"]` reflects the runnable's internal name (often `"LangGraph"`), while the graph node name is always in `metadata["langgraph_node"]`. The first such `on_chain_start` event seen sets the route variable that will populate `agent_used` in the `done` payload.

---

### Decision 10: Timeout handling and exception path for the streaming endpoint

`asyncio.wait_for()` cannot wrap a `StreamingResponse` generator. Instead, `process_stream()` uses `asyncio.timeout(settings.CHAT_TIMEOUT_S)` (Python 3.11+ context manager) around the `astream_events` loop. On timeout, `asyncio.TimeoutError` is caught, and the generator yields `("error", {"message": "Request timed out."})` before exiting.

**Exception path behavior (parity with existing `process()`):** On any unhandled exception (including timeout), `process_stream()`:
1. Persists the user message and a static fallback response ("I want to make sure I give you the best help. Could you tell me a bit more about what you'd like to focus on?") to session history — same text used in the current `process()` fallback.
2. Calls `session_store.commit()`.
3. Yields `("error", {"message": "..."})` so the frontend shows an error state.
4. The turn counter has already been incremented at the start of `process_stream()` (same as `process()`), so it is not rolled back. The persisted fallback response maintains a consistent turn record.

---

### Decision 11: Rate limiting on the streaming endpoint

The streaming endpoint uses the same `@limiter.limit(lambda: settings.RATE_LIMIT_CHAT)` decorator as the old `/api/chat`. The `Request` object is available as a dependency parameter in the route function, which is what `slowapi` requires for identification.

---

## Architecture

### Backend

```
POST /api/chat/stream
  └── StreamingResponse(text/event-stream)
        └── AgentOrchestrator.process_stream()   [async generator]
              └── asyncio.timeout(CHAT_TIMEOUT_S)
                    └── agent.astream_events(version="v2")
                          ├── on_chain_start (pro/flash_react_agent) → capture route
                          ├── on_tool_start  → yield ("status", { text: named_status })
                          ├── on_chat_model_stream (filtered) → yield ("token", { text: chunk })
                          └── [exhausted]
                                ├── post-processing (save, trace) — blocking
                                ├── fire background tasks (memory, analyzer) — non-blocking
                                ├── if output_gate replaced → yield ("replace", { text: fallback })
                                └── yield ("done", { session_id, agent_used, phase, pipeline_trace })

  [input_blocked path]
  └── agent.astream_events exhausts after input_gate
        └── detect input_blocked in accumulated state
              └── yield ("done", { session_id, response: block_response, agent_used: "input_gate", ... })
              (no tokens emitted — block response is returned in done payload, not streamed)
```

**SSE wire format:**
```
event: status\ndata: {"text": "Thinking..."}\n\n
event: token\ndata: {"text": "Here "}\n\n
event: token\ndata: {"text": "are some"}\n\n
event: done\ndata: {"session_id": "...", "pipeline_trace": {...}}\n\n
```

**Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no   ← prevents nginx from buffering the stream
```

---

### Frontend

```
ChatPage
  └── useChat(sessionId)
        ├── streamingContent: string   ← accumulated tokens, cleared on done
        ├── statusText: string         ← current stage label, cleared on first token
        ├── isLoading: boolean         ← true until first token arrives
        ├── isStreaming: boolean       ← true while tokens arriving
        └── abortRef: AbortController ← stored in ref for user cancellation

  └── ChatContainer
        ├── StatusLine (pulsing dot + statusText, shown while isLoading && statusText)
        ├── ChatBubble (streaming, in-progress)
        │     └── useTypewriter(streamingContent)
        │           ├── displayedText: string  ← what the bubble renders
        │           └── reset(newText)         ← called on replace event
        └── ChatInput (shows stop button while isStreaming, send button otherwise)
```

**`api.ts`**: `chatStream(data: ChatRequest): AsyncGenerator<StreamEvent>`. The existing `chat()` function is removed. `chatStream` handles SSE parsing with a buffer-remainder approach.

**`use-chat.ts`** event handling:
1. `status` → set `statusText`, keep `isLoading: true`
2. First `token` → set `isLoading: false`, `isStreaming: true`, clear `statusText`
3. Subsequent `token` → append to `streamingContent`
4. `replace` → overwrite `streamingContent`, call `typewriter.reset(newText)`
5. `done` → finalize: push `streamingContent` into `messages`, clear streaming state, store `pipeline_trace`
6. `error` → push a static fallback message ("Something went wrong. Please try again."), clear streaming state. No partial content is committed.
7. Abort (user clicks stop) → `controller.abort()`, catch `AbortError` in reader. Partial content is **discarded** — not pushed into `messages`. The in-progress bubble disappears. This avoids backend/frontend sync issues: since the backend never emitted `done`, it never persisted the partial response, so discarding on the frontend keeps both sides consistent.

---

## Event Taxonomy

| Event | Payload | Emitted when |
|-------|---------|--------------|
| `status` | `{ text: string }` | Input gate passes ("Thinking..."); each tool invocation starts |
| `token` | `{ text: string }` | Each text chunk from the final LLM response |
| `replace` | `{ text: string }` | Output gate replaces the response with safe fallback |
| `done` | `StreamDonePayload` | Stream complete, session committed |
| `error` | `{ message: string }` | Unhandled exception or timeout in the generator |

**`StreamDonePayload` (Python Pydantic / TypeScript interface):**
```
session_id:      str
agent_used:      str           # "react_agent" | "flash_react_agent" | "input_gate"
phase:           str           # ConversationPhase value
pipeline_trace:  PipelineTrace | null
response:        str | null    # populated only on input-blocked path; null otherwise
```

**Status text map** (all 5 tools):
```
(input gate passed)     → "Thinking..."
search_knowledge_base   → "Looking up strategies..."
get_family_profile      → "Reading your profile..."
update_family_profile   → "Updating your profile..."
track_outcome           → "Recording outcome..."
manage_goals            → "Managing goals..."
(unknown tool)          → "Working on it..."   ← generic fallback
```

---

## Files Changed

| File | Change |
|------|--------|
| `app/api/routes.py` | Remove `POST /api/chat`, add `POST /api/chat/stream` with same rate limiter |
| `app/agent/orchestrator.py` | Add `process_stream()` async generator, remove `process()` |
| `app/models/schemas.py` | Remove `ChatResponse`, add `StreamDonePayload` |
| `frontend-react/src/lib/api.ts` | Remove `chat()`, add `chatStream()` async generator |
| `frontend-react/src/hooks/use-chat.ts` | Refactor to consume SSE via `chatStream()` |
| `frontend-react/src/hooks/use-typewriter.ts` | New hook — rAF adaptive typewriter with `reset()` |
| `frontend-react/src/components/chat/ChatBubble.tsx` | Use `useTypewriter` for streaming bubbles |
| `frontend-react/src/components/chat/ChatContainer.tsx` | Status line, smart scroll |
| `frontend-react/src/components/chat/ChatInput.tsx` | Add stop button shown while `isStreaming` |
| `frontend-react/src/types/index.ts` | Add `StreamEvent` union types, remove unused `ChatResponse` TypeScript interface |
| `tests/test_api.py` | Update to use `/api/chat/stream` with SSE consumer helper |
| `tests/test_agent_orchestrator.py` | Rewrite to test `process_stream()` instead of removed `process()` |

---

## What Is Not Changing

- The ReAct agent graph (`graph.py`) — no changes to pipeline topology
- Tool implementations (`tools.py`)
- Memory, analyzer, event bus — same background task pattern
- Session store, SQLite persistence
- Guardrails logic (`validator.py`)
- Observability routes

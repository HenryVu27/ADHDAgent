# Thinking UI & Contextual Streaming Status

## Problem

The streaming experience shows static, hardcoded status messages ("Understanding your message...", "Preparing a response...", "Looking up strategies...") regardless of conversation context. The loading animation (pulsing dot + bouncing dots) feels dated. Assistant messages are wrapped in card bubbles that feel heavy compared to modern chat UIs (Claude, ChatGPT).

## Solution

Three changes:
1. A parallel Flash call generates a contextual one-line summary from the user's message, shown as a persistent header above the response.
2. Tool status messages use actual tool input args instead of hardcoded strings.
3. UI refresh: shimmer animation replaces pulsing/bouncing dots, assistant messages go borderless.

## Design

### Backend: Summary generation

A new async function fires in parallel with the main agent pipeline when `process_stream` starts:

- Uses `asyncio.create_task` to run a Gemini Flash call concurrently.
- Prompt: generate a present-participle summary of what a parenting coach would focus on, under 10 words. Example output: "Exploring bedtime routine strategies for a 7-year-old".
- Yields a new `summary` SSE event: `event: summary\ndata: {"text": "..."}\n\n`.
- Non-blocking: if Flash is slow or fails, no summary emitted. The response streams without one.
- The summary is also included in the `done` event payload so the frontend can persist it on the finalized message.

### Backend: Contextual tool status

Replace the hardcoded `tool_status` dict in `process_stream` with dynamic messages derived from `event["data"]["input"]`:

| Tool | Current (static) | New (contextual) |
|------|------------------|-------------------|
| `search_knowledge_base` | "Looking up strategies..." | "Searching for '{query}'..." |
| `update_family_profile` | "Updating your profile..." | "Noting that {first words}..." |
| `track_outcome` | "Recording progress..." | "Recording how {strategy_name} went..." |
| `manage_goals` | "Managing goals..." | "{action}ing goal: {summary}..." |
| `get_family_profile` | "Reviewing your family's info..." | Unchanged (no meaningful input) |

Remove the static initial messages:
- Remove "Understanding your message..." (the summary line replaces this).
- Remove "Preparing a response..." (the shimmer animation covers this gap).

### Backend: Event types

New SSE event type:

```
event: summary
data: {"text": "Exploring bedtime routine strategies for your 7-year-old"}
```

The `done` event payload gains a `summary` field:

```
event: done
data: {"response": "...", "summary": "Exploring bedtime...", ...}
```

### Frontend: Types

New type added to `StreamEvent` union in `types/index.ts`:

```typescript
interface StreamSummaryEvent {
  type: "summary"
  text: string
}
```

`StreamDoneEvent` gains `summary: string | null`.

`ChatMessage` gains `summary?: string` to persist the summary on finalized messages.

### Frontend: State (use-chat.ts)

New state:
- `summaryText: string` -- set when `summary` event arrives, cleared on next send.

Event handling flow:
1. User sends message: `isLoading = true`. Shimmer animation starts.
2. `summary` event: `summaryText` set. Summary line appears above shimmer.
3. `status` events: `statusText` updates with contextual tool messages, shown below summary.
4. First `token`: `isLoading = false`, `isStreaming = true`. Shimmer and status fade out. Response streams below summary line.
5. `done`: summary line stays as a header on the finalized `ChatMessage`.

Remove the bouncing dots fallback (`isLoading && !statusText` branch).

### Frontend: Animation

Replace pulsing dot and bouncing dots with a single shimmer bar:
- CSS `@keyframes` with a translucent gradient sliding left-to-right on `background-position`.
- Narrow horizontal bar, subtle, lives below the summary line.
- Framer-motion `opacity` fade-in on mount, fade-out when first token arrives.
- No dots anywhere.

### Frontend: Borderless assistant messages

Strip card styling from assistant bubbles in both `ChatBubble` and `StreamingBubble`:

- **Remove** from assistant messages: `bg-card`, `shadow-sm`, `border border-border/30`, `rounded-2xl`.
- **Keep** padding for text alignment and the coach avatar + "Ally" label.
- **User messages**: unchanged (keep bubble styling).
- The summary line renders as small muted text above the response content, no card wrapper.
- The loading shimmer also renders directly on the page background, no card.

### Frontend: ChatContainer layout

Loading state:

```
[Coach avatar]  Exploring bedtime routine strategies...     <- summary (muted, small)
                ━━━━━━━━━ (shimmer)                         <- smooth animation
                Searching for 'executive function aids'...  <- contextual status (muted)
```

Streaming state (shimmer and status fade out):

```
[Coach avatar]  Exploring bedtime routine strategies...     <- summary stays
                Here are some approaches that might help     <- streaming response
                with homework meltdowns...
```

Finalized state (identical to streaming, just no typewriter):

```
[Coach avatar]  Exploring bedtime routine strategies...     <- summary header
                Here are some approaches that might help
                with homework meltdowns...
```

## Files to modify

| File | Changes |
|------|---------|
| `app/agent/orchestrator.py` | Summary generation task, contextual tool status, new `summary` event, add summary to `done` payload |
| `app/models/schemas.py` | Add `summary` field to `StreamDonePayload` |
| `frontend-react/src/types/index.ts` | Add `StreamSummaryEvent`, update `StreamDoneEvent` and `ChatMessage` |
| `frontend-react/src/hooks/use-chat.ts` | Add `summaryText` state, handle `summary` event, pass summary to finalized message |
| `frontend-react/src/components/chat/ChatContainer.tsx` | Summary line component, shimmer animation, remove bouncing dots, remove card from loading state |
| `frontend-react/src/components/chat/ChatBubble.tsx` | Borderless assistant styling in both `ChatBubble` and `StreamingBubble`, render summary header |

## Out of scope

- Expandable thinking block (parents don't need raw reasoning).
- "Done" indicator / checkmark.
- Summary for input-blocked messages (crisis/jailbreak -- these skip the agent).
- Persisting summary server-side in session history.

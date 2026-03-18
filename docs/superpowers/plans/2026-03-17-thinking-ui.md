# Thinking UI & Contextual Streaming Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static streaming status messages with a contextual LLM-generated summary line, make tool statuses dynamic, add shimmer animation, and make assistant messages borderless.

**Architecture:** A parallel Gemini Flash call generates a one-line summary when the stream starts. Tool status messages are derived from actual tool input args. The frontend renders a new `summary` event type as a persistent header above the response, with a CSS shimmer animation replacing the old pulsing/bouncing dots. Assistant message cards are stripped to borderless text.

**Tech Stack:** Python/FastAPI (backend), Gemini Flash (summary generation), React/TypeScript/Tailwind/Framer Motion (frontend)

---

### Task 1: Add `summary` field to StreamDonePayload

**Files:**
- Modify: `app/models/schemas.py:242-247`
- Modify: `frontend-react/src/types/index.ts:82-116`

- [ ] **Step 1: Add summary field to StreamDonePayload**

In `app/models/schemas.py`, add `summary` to `StreamDonePayload`:

```python
class StreamDonePayload(BaseModel):
    session_id: str
    agent_used: str
    phase: str
    pipeline_trace: PipelineTrace | None = None
    response: str | None = None
    summary: str | None = None  # Contextual one-liner from parallel Flash call
```

- [ ] **Step 2: Add StreamSummaryEvent and update frontend types**

In `frontend-react/src/types/index.ts`, add `StreamSummaryEvent`, update `StreamDoneEvent`, update `ChatMessage`, and update `StreamEvent` union:

```typescript
// After StreamReplaceEvent (line ~95):
export interface StreamSummaryEvent {
  type: "summary"
  text: string
}

// Update StreamDoneEvent to include summary:
export interface StreamDoneEvent {
  type: "done"
  session_id: string
  agent_used: string
  phase: ConversationPhase
  pipeline_trace: PipelineTrace | null
  response: string | null
  summary: string | null
}

// Update StreamEvent union:
export type StreamEvent =
  | StreamStatusEvent
  | StreamTokenEvent
  | StreamReplaceEvent
  | StreamSummaryEvent
  | StreamDoneEvent
  | StreamErrorEvent
```

Add `summary` to the `ChatMessage` interface:

```typescript
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  agentUsed?: string
  pipelineTrace?: PipelineTrace
  summary?: string
}
```

- [ ] **Step 3: Commit**

```bash
git add app/models/schemas.py frontend-react/src/types/index.ts
git commit -m "Add summary field to StreamDonePayload and frontend types"
```

---

### Task 2: Backend summary generation and contextual tool status

**Files:**
- Modify: `app/agent/orchestrator.py:41-47` (add `gemini_client` to `__init__`)
- Modify: `app/agent/orchestrator.py:297-540` (process_stream changes)
- Modify: `app/main.py:130-137` (wire `gemini_client` to orchestrator)

- [ ] **Step 1: Add GeminiClient dependency to AgentOrchestrator**

In `app/agent/orchestrator.py`, update `__init__` to accept and store a `gemini_client`:

```python
def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None, analyzer=None, event_bus=None, output_gate=None, gemini_client=None):
    self._agent = agent
    self._session_store = session_store
    self._memory = memory_manager
    self._analyzer = analyzer
    self._event_bus = event_bus
    self._output_gate = output_gate
    self._gemini = gemini_client
    self._pending_tasks: set[asyncio.Task] = set()
```

In `app/main.py`, pass the existing `gemini` client to the orchestrator (line ~130):

```python
orchestrator = AgentOrchestrator(
    agent=agent,
    session_store=session_store,
    memory_manager=memory_manager,
    analyzer=analyzer,
    event_bus=event_bus,
    output_gate=output_gate,
    gemini_client=gemini,
)
```

- [ ] **Step 2: Add summary generation method to AgentOrchestrator**

Add a private method to `AgentOrchestrator` that calls Gemini Flash to generate a contextual summary. Place it near the top of the class (after `__init__`):

```python
async def _generate_summary(self, message: str) -> str | None:
    """Generate a contextual one-line summary from the user's message via Flash."""
    if not self._gemini:
        return None
    try:
        prompt = (
            "You are an ADHD parenting coach's internal narrator. "
            "Summarize in under 10 words what you would focus on for this parent's message. "
            "Use present participle form. Do NOT include quotes or punctuation at the end.\n"
            "Examples:\n"
            "- Exploring bedtime routine strategies for a 7-year-old\n"
            "- Considering ways to handle homework meltdowns\n"
            "- Thinking about morning routine structure\n"
            "- Looking into positive reinforcement approaches\n\n"
            f"Parent's message: {message}"
        )
        result = await self._gemini.generate(
            prompt,
            temperature=0.3,
            max_output_tokens=30,
            timeout=5.0,
        )
        summary = result.strip().rstrip(".")
        return summary if summary else None
    except Exception:
        logger.debug("[agent] Summary generation failed — skipping")
        return None
```

- [ ] **Step 3: Update process_stream to fire summary in parallel and use contextual tool status**

In `process_stream`, replace the initial status yield and add the parallel summary task. Replace the hardcoded `tool_status` dict with dynamic messages from tool input args.

Replace the section from the initial `yield ("status", ...)` through the tool status handling (lines ~354-382):

```python
        try:
            # Fire summary generation in parallel — non-blocking
            summary_task = asyncio.create_task(self._generate_summary(message))
            summary_text: str | None = None
            summary_sent = False

            token_count = 0
            token_start = None
            async with asyncio.timeout(settings.CHAT_TIMEOUT_S):
                async for event in self._agent.astream_events(
                    input_data, config=config, version="v2"
                ):
                    # Check if summary is ready and hasn't been sent yet
                    if not summary_sent and summary_task.done():
                        try:
                            summary_text = summary_task.result()
                        except Exception:
                            summary_text = None
                        if summary_text:
                            yield ("summary", {"text": summary_text})
                        summary_sent = True

                    kind = event["event"]
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node")

                    # Tool invocation status updates — contextual
                    if kind == "on_tool_start":
                        tool_name = event.get("name", "")
                        tool_input = event.get("data", {}).get("input", {})

                        if tool_name == "search_knowledge_base":
                            query = tool_input.get("query", "strategies")
                            status = f"Searching for '{query}'..."
                        elif tool_name == "get_document_details":
                            status = "Reading document details..."
                        elif tool_name == "get_related_documents":
                            status = "Finding related strategies..."
                        elif tool_name == "update_family_profile":
                            # Summarize what's being updated
                            fields = [k for k, v in tool_input.items()
                                      if v is not None and k != "config"]
                            if fields:
                                status = f"Noting {', '.join(fields[:2])}..."
                            else:
                                status = "Updating your profile..."
                        elif tool_name == "track_outcome":
                            strategy = tool_input.get("strategy_name", "a strategy")
                            status = f"Recording how {strategy} went..."
                        elif tool_name == "manage_goals":
                            action = tool_input.get("action", "managing")
                            desc = tool_input.get("description", "")
                            if desc:
                                short_desc = desc[:40].rstrip()
                                status = f"{action.capitalize()}ing goal: {short_desc}..."
                            else:
                                status = f"Reviewing goals..."
                        elif tool_name == "get_family_profile":
                            status = "Reviewing your family's info..."
                        else:
                            status = "Working on it..."
                        yield ("status", {"text": status})
```

Remove the entire `on_chain_start` handler block (the `if kind == "on_chain_start" and node:` section that yields "Understanding your message..." and "Preparing a response..."). Keep the rest of the streaming logic (on_chat_model_stream, on_chain_end) unchanged.

- [ ] **Step 4: Send summary before first token if not yet sent**

In the `on_chat_model_stream` handler, right before yielding the first token, check if summary needs to be sent:

```python
                            if text:
                                if token_count == 0:
                                    # Ensure summary is sent before first token
                                    if not summary_sent:
                                        if not summary_task.done():
                                            # Give it a brief grace period
                                            try:
                                                await asyncio.wait_for(
                                                    asyncio.shield(summary_task), timeout=0.5
                                                )
                                            except (TimeoutError, asyncio.TimeoutError):
                                                pass
                                        if summary_task.done():
                                            try:
                                                summary_text = summary_task.result()
                                            except Exception:
                                                summary_text = None
                                            if summary_text:
                                                yield ("summary", {"text": summary_text})
                                        summary_sent = True

                                token_count += 1
                                # ... rest of existing token yielding logic
```

- [ ] **Step 5: Include summary in done payload**

In the normal response processing section (after `process_stream`'s streaming loop), where `StreamDonePayload` is constructed (around line ~530 for blocked path, ~610 for normal path, ~470 for error path), add `summary=summary_text`:

For the normal done path:
```python
            yield (
                "done",
                StreamDonePayload(
                    response=response_text,
                    agent_used=agent_label,
                    phase=await self._infer_phase(session_id),
                    pipeline_trace=trace,
                    session_id=session_id,
                    summary=summary_text,
                ).model_dump(),
            )
```

For the input-blocked and error paths, set `summary=None`.

- [ ] **Step 6: Cancel summary task on error/timeout**

In the `except TimeoutError` and `except Exception` blocks, cancel the summary task if still running:

```python
        except TimeoutError:
            if not summary_task.done():
                summary_task.cancel()
            yield ("error", {"message": f"Response timed out after {settings.CHAT_TIMEOUT_S}s"})
            return

        except Exception as e:
            if not summary_task.done():
                summary_task.cancel()
            # ... rest of existing error handling
```

- [ ] **Step 7: Commit**

```bash
git add app/agent/orchestrator.py app/main.py
git commit -m "Add parallel summary generation and contextual tool status"
```

---

### Task 3: Frontend state and event handling

**Files:**
- Modify: `frontend-react/src/hooks/use-chat.ts`

- [ ] **Step 1: Add summaryText state and handle summary event**

Add `summaryText` state and update the event loop:

```typescript
export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [statusText, setStatusText] = useState("")
  const [streamingContent, setStreamingContent] = useState("")
  const [summaryText, setSummaryText] = useState("")  // NEW
  const [latestTrace, setLatestTrace] = useState<PipelineTrace | null>(null)
  // ... refs unchanged
```

- [ ] **Step 2: Handle the summary event in the event loop**

In the `for await` loop, add a handler for the `summary` event. The full updated event handlers:

```typescript
      for await (const event of api.chatStream(
        { message: content, session_id: sessionId },
        controller.signal,
      )) {
        if (event.type === "summary") {
          setSummaryText(event.text)

        } else if (event.type === "status") {
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
          accumulatedRef.current = event.text
          setStreamingContent(event.text)
          typewriterResetRef.current?.(event.text)

        } else if (event.type === "done") {
          // ... existing done logic unchanged
```

- [ ] **Step 3: Persist summary on finalized message**

Update `_finalize` to include summary:

```typescript
    function _finalize(done: StreamDoneEvent) {
      const finalContent = done.response ?? accumulatedRef.current
      const assistantMsg: ChatMessage = {
        id: `msg_${++idCounter.current}`,
        role: "assistant",
        content: finalContent,
        timestamp: new Date(),
        agentUsed: done.agent_used,
        pipelineTrace: done.pipeline_trace ?? undefined,
        summary: done.summary ?? undefined,
      }
      setMessages(prev => [...prev, assistantMsg])
      if (done.pipeline_trace) setLatestTrace(done.pipeline_trace)
      _clearStreamState()
    }
```

- [ ] **Step 4: Clear summaryText in _clearStreamState**

```typescript
    function _clearStreamState() {
      setIsLoading(false)
      setIsStreaming(false)
      setStatusText("")
      setStreamingContent("")
      setSummaryText("")
      accumulatedRef.current = ""
    }
```

- [ ] **Step 5: Expose summaryText in the return value**

```typescript
  return {
    messages,
    isLoading,
    isStreaming,
    statusText,
    summaryText,
    streamingContent,
    latestTrace,
    typewriterResetRef,
    onStreamComplete,
    sendMessage,
    stopStreaming,
    clearMessages,
    loadMessages,
  }
```

- [ ] **Step 6: Commit**

```bash
git add frontend-react/src/hooks/use-chat.ts
git commit -m "Handle summary event and expose summaryText state"
```

---

### Task 4: Borderless assistant messages

**Files:**
- Modify: `frontend-react/src/components/chat/ChatBubble.tsx`

- [ ] **Step 1: Update ChatBubble for borderless assistant messages with summary header**

Replace the `ChatBubble` component to remove card styling from assistant messages and add a summary header:

```tsx
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
        className={`max-w-[80%] ${
          isUser
            ? "rounded-2xl px-4 py-3 bg-primary text-primary-foreground"
            : "px-1 py-1"
        }`}
      >
        {!isUser && (
          <div className="mb-1 text-xs font-medium text-coach">Ally</div>
        )}
        {!isUser && message.summary && (
          <div className="mb-2 text-xs text-muted-foreground">
            {message.summary}
          </div>
        )}
        <div
          className="text-sm leading-relaxed [&_ul]:ml-4 [&_ul]:list-disc [&_ul]:space-y-1 [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
        />
      </div>
    </motion.div>
  )
}
```

- [ ] **Step 2: Rename StreamingBubble to StreamingContent (content-only, no wrapper)**

The avatar/wrapper will now be provided by `ChatContainer.tsx` in Task 5. Replace `StreamingBubble` with a content-only `StreamingContent` component. Also update the export:

```tsx
export function StreamingContent({ content, typewriterResetRef, onComplete }: StreamingBubbleProps) {
  const { displayedText, reset } = useTypewriter(content, onComplete)

  if (typewriterResetRef.current !== reset) {
    typewriterResetRef.current = reset
  }

  return (
    <div
      className="text-sm leading-relaxed [&_ul]:ml-4 [&_ul]:list-disc [&_ul]:space-y-1 [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold"
      dangerouslySetInnerHTML={{ __html: formatMarkdown(displayedText) }}
    />
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/components/chat/ChatBubble.tsx
git commit -m "Make assistant messages borderless and add summary header"
```

---

### Task 5: Shimmer animation and ChatContainer updates

**Files:**
- Modify: `frontend-react/src/components/chat/ChatContainer.tsx`

- [ ] **Step 1: Add props for summaryText**

Update the component props to accept `summaryText`:

```tsx
interface ChatContainerProps {
  messages: ChatMessage[]
  isLoading: boolean
  isStreaming: boolean
  statusText: string
  summaryText: string
  streamingContent: string
  typewriterResetRef: React.MutableRefObject<((text: string) => void) | null>
  onStreamComplete: () => void
}
```

Update the destructuring in the component signature to include `summaryText`.

- [ ] **Step 2: Replace loading indicators with shimmer + summary**

Remove the bouncing dots block (`isLoading && !statusText`) entirely.

Replace the status line block (`isLoading && statusText`) with a new loading state that shows the summary line and shimmer animation:

```tsx
      {/* Loading state: summary + shimmer + contextual status */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="px-1 py-1">
            <div className="mb-1 text-xs font-medium text-coach">Ally</div>
            {summaryText && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mb-2 text-xs text-muted-foreground"
              >
                {summaryText}
              </motion.div>
            )}
            <div className="mb-2 h-1 w-32 overflow-hidden rounded-full bg-muted">
              <div className="h-full w-1/2 animate-shimmer rounded-full bg-gradient-to-r from-transparent via-coach/30 to-transparent" />
            </div>
            {statusText && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xs text-muted-foreground"
              >
                {statusText}
              </motion.div>
            )}
          </div>
        </motion.div>
      )}
```

- [ ] **Step 3: Add summary line above streaming content**

Update the streaming section. `StreamingContent` (renamed in Task 4) is now content-only, so wrap it with the avatar, label, and summary here. Update the import at the top of `ChatContainer.tsx` from `StreamingBubble` to `StreamingContent`:

```tsx
import { ChatBubble, StreamingContent } from "./ChatBubble"
```

Replace the streaming bubble section:

```tsx
      {/* Streaming: summary header + streaming response */}
      {isStreaming && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="max-w-[80%] px-1 py-1">
            <div className="mb-1 text-xs font-medium text-coach">Ally</div>
            {summaryText && (
              <div className="mb-2 text-xs text-muted-foreground">
                {summaryText}
              </div>
            )}
            <StreamingContent
              content={streamingContent}
              typewriterResetRef={typewriterResetRef}
              onComplete={onStreamComplete}
            />
          </div>
        </motion.div>
      )}
```

- [ ] **Step 4: Add shimmer keyframe animation in CSS**

This project uses Tailwind CSS v4 with CSS-based config (no `tailwind.config.js`). Add the shimmer animation in `frontend-react/src/index.css`, inside the `@theme inline` block:

```css
@theme inline {
    /* ... existing theme variables ... */

    --animate-shimmer: shimmer 1.5s ease-in-out infinite;
}
```

And add the keyframes after the `@layer base` block:

```css
@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(300%); }
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend-react/src/components/chat/ChatContainer.tsx frontend-react/src/index.css
git commit -m "Add shimmer animation, summary line, and remove bouncing dots"
```

---

### Task 6: Wire summaryText through ChatPage

**Files:**
- Modify: `frontend-react/src/pages/ChatPage.tsx`

- [ ] **Step 1: Pass summaryText to ChatContainer**

Read `ChatPage.tsx` and update the `ChatContainer` usage to pass `summaryText`:

```tsx
const { messages, isLoading, isStreaming, statusText, summaryText, streamingContent, ... } = useChat(sessionId)

// In the JSX:
<ChatContainer
  messages={messages}
  isLoading={isLoading}
  isStreaming={isStreaming}
  statusText={statusText}
  summaryText={summaryText}
  streamingContent={streamingContent}
  typewriterResetRef={typewriterResetRef}
  onStreamComplete={onStreamComplete}
/>
```

- [ ] **Step 2: Commit**

```bash
git add frontend-react/src/pages/ChatPage.tsx
git commit -m "Wire summaryText from useChat to ChatContainer"
```

---

### Task 7: Verify routes.py forwards new event type

**Files:**
- Verify: `app/api/routes.py:53-58`

- [ ] **Step 1: Verify routes.py generically forwards events**

The route at lines 53-58 uses a generic `event_generator()` that forwards any `(event_type, data)` tuple as `event: {event_type}\ndata: {json}\n\n`. No allowlist — the new `summary` event type will be forwarded automatically. **No changes needed.**

- [ ] **Step 2: Commit (no-op, verification only)**

No commit needed for this task.

---

### Task 8: Manual smoke test

- [ ] **Step 1: Start the backend**

```bash
./adhd312/Scripts/python.exe -m uvicorn app.main:app --reload
```

- [ ] **Step 2: Start the frontend**

```bash
cd frontend-react && npm run dev
```

- [ ] **Step 3: Verify the following in the browser**

1. Send a message -> shimmer animation appears (no bouncing dots, no pulsing dot)
2. Summary line appears above shimmer (e.g., "Exploring homework strategies...")
3. If a tool is called, contextual status appears (e.g., "Searching for 'executive function homework'...")
4. Response streams below the summary line, borderless (no card/shadow/border)
5. Finalized message has summary as a subtle muted header
6. User messages still have their bubble styling
7. Abort/stop works cleanly
8. If summary Flash call is slow, response still streams without blocking

import { useState, useCallback, useRef } from "react"
import type { Attachment, ChatMessage, PipelineTrace, StreamDoneEvent } from "@/types"
import { api } from "@/lib/api"

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)       // waiting for first token
  const [isStreaming, setIsStreaming] = useState(false)    // tokens arriving
  const [statusText, setStatusText] = useState("")         // current pipeline stage label
  const [summaryText, setSummaryText] = useState("")  // Contextual summary from parallel Flash call
  const [streamingContent, setStreamingContent] = useState("") // drives typewriter display
  const [latestTrace, setLatestTrace] = useState<PipelineTrace | null>(null)
  const idCounter = useRef(0)
  const abortRef = useRef<AbortController | null>(null)
  // accumulatedRef: source of truth for token accumulation (avoids stale closure on done)
  const accumulatedRef = useRef("")
  // Exposed so ChatContainer can call typewriter.reset() on replace events
  const typewriterResetRef = useRef<((text: string) => void) | null>(null)
  // Buffer the done payload so finalization waits for the typewriter to catch up
  const pendingDoneRef = useRef<{ event: StreamDoneEvent; finalize: (e: StreamDoneEvent) => void } | null>(null)

  // Called by useTypewriter when displayed text catches up to all received tokens.
  // If a done event is already buffered, finalize now.
  const onStreamComplete = useCallback(() => {
    const pending = pendingDoneRef.current
    if (pending) {
      pendingDoneRef.current = null
      pending.finalize(pending.event)
    }
  }, [])

  const sendMessage = useCallback(async (content: string, attachments?: Attachment[]) => {
    const userMsg: ChatMessage = {
      id: `msg_${++idCounter.current}`,
      role: "user",
      content,
      timestamp: new Date(),
      attachments,
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)
    setStreamingContent("")
    setStatusText("")
    accumulatedRef.current = ""

    const controller = new AbortController()
    abortRef.current = controller

    let firstToken = true

    try {
      for await (const event of api.chatStream(
        { message: content, session_id: sessionId, attachment_ids: attachments?.map(a => a.id) },
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
            // Yield to event loop so React renders the streaming block
            // before processing subsequent tokens/done in the same SSE chunk
            await new Promise(resolve => setTimeout(resolve, 0))
          }
          accumulatedRef.current += event.text
          setStreamingContent(prev => prev + event.text)

        } else if (event.type === "reset") {
          // Agent is calling a tool — clear intermediate reasoning so only
          // the final post-tool response is displayed
          accumulatedRef.current = ""
          setStreamingContent("")
          typewriterResetRef.current?.("")
          firstToken = true
          setIsStreaming(false)
          setIsLoading(true)

        } else if (event.type === "replace") {
          accumulatedRef.current = event.text
          setStreamingContent(event.text)
          typewriterResetRef.current?.(event.text)

        } else if (event.type === "done") {
          // If no tokens were streamed (e.g. blocked input), finalize immediately.
          // Otherwise buffer the done event and let the typewriter's onComplete trigger it.
          if (!accumulatedRef.current) {
            _finalize(event)
          } else {
            pendingDoneRef.current = { event, finalize: _finalize }
            // Safety: finalize after 5s even if typewriter hasn't caught up
            setTimeout(() => {
              if (pendingDoneRef.current) {
                const p = pendingDoneRef.current
                pendingDoneRef.current = null
                p.finalize(p.event)
              }
            }, 5000)
          }

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

    function _clearStreamState() {
      setIsLoading(false)
      setIsStreaming(false)
      setStatusText("")
      setStreamingContent("")
      setSummaryText("")
      accumulatedRef.current = ""
      pendingDoneRef.current = null
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
    setSummaryText("")
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
}

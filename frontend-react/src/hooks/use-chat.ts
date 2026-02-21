import { useState, useCallback, useRef } from "react"
import type { ChatMessage, PipelineTrace } from "@/types"
import { api } from "@/lib/api"

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [latestTrace, setLatestTrace] = useState<PipelineTrace | null>(null)
  const idCounter = useRef(0)

  const sendMessage = useCallback(async (content: string) => {
    const userMsg: ChatMessage = {
      id: `msg_${++idCounter.current}`,
      role: "user",
      content,
      timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const data = await api.chat({ message: content, session_id: sessionId })
      const assistantMsg: ChatMessage = {
        id: `msg_${++idCounter.current}`,
        role: "assistant",
        content: data.response,
        timestamp: new Date(),
        agentUsed: data.agent_used,
        pipelineTrace: data.pipeline_trace,
      }
      setMessages(prev => [...prev, assistantMsg])
      setLatestTrace(data.pipeline_trace)
      return data
    } catch (error) {
      const errorMsg: ChatMessage = {
        id: `msg_${++idCounter.current}`,
        role: "assistant",
        content: "Sorry, something went wrong. Please try again.",
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMsg])
      throw error
    } finally {
      setIsLoading(false)
    }
  }, [sessionId])

  const clearMessages = useCallback(() => {
    setMessages([])
    setLatestTrace(null)
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

  return { messages, isLoading, latestTrace, sendMessage, clearMessages, loadMessages }
}

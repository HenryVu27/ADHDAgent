import { useState, useCallback, useEffect, useRef } from "react"
import { Code2, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ChatContainer } from "@/components/chat/ChatContainer"
import { ChatInput } from "@/components/chat/ChatInput"
import { QuickReplyChips } from "@/components/chat/QuickReplyChips"
import { SessionProgress } from "@/components/chat/SessionProgress"
import { DevPanel } from "@/components/chat/DevPanel"
import { useChat } from "@/hooks/use-chat"
import { useSession } from "@/hooks/use-session"
import { useAuth } from "@/hooks/use-auth"
import { getActiveSessionId, setActiveSessionId, updateSessionStats, getSessionStats } from "@/lib/auth"
import { api } from "@/lib/api"
import type { ConversationPhase } from "@/types"

function createSessionId() {
  return `session_${Date.now()}`
}

export function ChatPage() {
  const [sessionId, setSessionId] = useState<string>(() => {
    return getActiveSessionId() || createSessionId()
  })
  const { messages, isLoading, latestTrace, sendMessage, clearMessages, loadMessages } = useChat(sessionId)
  const { session, refresh } = useSession(sessionId)
  const { getOnboarding } = useAuth()
  const [devOpen, setDevOpen] = useState(false)
  const [showChips, setShowChips] = useState(true)
  const initialized = useRef(false)

  // On mount: persist session ID and load existing messages if resuming
  useEffect(() => {
    if (initialized.current) return
    initialized.current = true

    setActiveSessionId(sessionId)

    const existing = getActiveSessionId()
    if (existing === sessionId) {
      // Try to load existing messages from backend
      loadMessages().then((count) => {
        if (count === 0) {
          // New session — seed with onboarding data
          const onboarding = getOnboarding()
          if (onboarding) {
            api.seedSession(sessionId, onboarding).catch(() => {})
          }
        } else {
          // Resuming — hide chips since conversation is already going
          setShowChips(false)
          refresh()
        }
      })
    }
  }, [sessionId, getOnboarding, loadMessages, refresh])

  const handleNewChat = useCallback(() => {
    const newId = createSessionId()
    setSessionId(newId)
    setActiveSessionId(newId)
    clearMessages()
    setShowChips(true)
    initialized.current = false

    // Seed new session with onboarding
    const onboarding = getOnboarding()
    if (onboarding) {
      api.seedSession(newId, onboarding).catch(() => {})
    }
    initialized.current = true
  }, [clearMessages, getOnboarding])

  const handleSend = useCallback(async (content: string) => {
    setShowChips(false)
    const data = await sendMessage(content)
    if (data) {
      await refresh()
      const stats = getSessionStats()
      updateSessionStats({
        sessions: stats.sessions + 1,
        strategies: (data.pipeline_trace.retrieval_results?.length || 0) + stats.strategies,
        streak: stats.streak || 1,
      })
      setShowChips(true)
    }
  }, [sendMessage, refresh])

  const handleChipSelect = useCallback((text: string) => {
    handleSend(text)
  }, [handleSend])

  const phase = (session?.phase || latestTrace?.phase_decision?.phase || "intake") as ConversationPhase

  return (
    <>
      <Card className="flex h-[calc(100vh-10rem)] flex-col overflow-hidden">
        {/* Header with progress bar + new chat */}
        <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
          <SessionProgress currentPhase={phase} />
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleNewChat}
              className="gap-2 text-xs text-muted-foreground"
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDevOpen(!devOpen)}
              className="gap-2 text-xs text-muted-foreground"
            >
              <Code2 className="h-3.5 w-3.5" />
              Pipeline
            </Button>
          </div>
        </div>

        {/* Chat area */}
        <ChatContainer messages={messages} isLoading={isLoading} />

        {/* Quick reply chips */}
        <QuickReplyChips
          phase={phase}
          onSelect={handleChipSelect}
          visible={showChips && !isLoading}
        />

        {/* Input */}
        <ChatInput onSend={handleSend} isLoading={isLoading} />
      </Card>

      <DevPanel trace={latestTrace} isOpen={devOpen} onClose={() => setDevOpen(false)} />
    </>
  )
}

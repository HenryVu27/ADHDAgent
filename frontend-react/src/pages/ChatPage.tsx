import { useState, useMemo, useCallback, useEffect, useRef } from "react"
import { Code2 } from "lucide-react"
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
import { updateSessionStats, getSessionStats } from "@/lib/auth"
import { api } from "@/lib/api"
import type { ConversationPhase } from "@/types"

export function ChatPage() {
  const sessionId = useMemo(() => `session_${Date.now()}`, [])
  const { messages, isLoading, latestTrace, sendMessage } = useChat(sessionId)
  const { session, refresh } = useSession(sessionId)
  const { getOnboarding } = useAuth()
  const [devOpen, setDevOpen] = useState(false)
  const [showChips, setShowChips] = useState(true)
  const seeded = useRef(false)

  useEffect(() => {
    if (seeded.current) return
    seeded.current = true
    const onboarding = getOnboarding()
    if (onboarding) {
      api.seedSession(sessionId, onboarding).catch(() => {})
    }
  }, [sessionId, getOnboarding])

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
        {/* Header with progress bar */}
        <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
          <SessionProgress currentPhase={phase} />
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

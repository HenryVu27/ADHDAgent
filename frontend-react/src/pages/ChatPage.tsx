import { useState, useCallback, useEffect, useRef } from "react"
import { PanelLeft, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ChatContainer } from "@/components/chat/ChatContainer"
import { ChatInput } from "@/components/chat/ChatInput"
import { QuickReplyChips } from "@/components/chat/QuickReplyChips"
import { SessionProgress } from "@/components/chat/SessionProgress"
import { ChatSidebar } from "@/components/chat/ChatSidebar"
import { useChat } from "@/hooks/use-chat"
import { useSession } from "@/hooks/use-session"
import { useAuth } from "@/hooks/use-auth"
import { getActiveSessionId, setActiveSessionId, updateSessionStats, getSessionStats } from "@/lib/auth"
import { api } from "@/lib/api"
import type { Attachment, ConversationPhase } from "@/types"

function createSessionId() {
  const now = new Date()
  const pad = (n: number, len = 2) => String(n).padStart(len, "0")
  const date = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}`
  const time = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  const ms = pad(now.getMilliseconds(), 3)
  return `session_${date}_${time}_${ms}`
}

export function ChatPage() {
  const [sessionId, setSessionId] = useState<string>(() => {
    return getActiveSessionId() || createSessionId()
  })
  const {
    messages, isLoading, isStreaming, statusText, streamingContent,
    latestTrace, typewriterResetRef, onStreamComplete, sendMessage, stopStreaming, clearMessages, loadMessages, summaryText,
  } = useChat(sessionId)
  const { session, refresh } = useSession(sessionId)
  const { getOnboarding } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [showChips, setShowChips] = useState(true)
  const initialized = useRef(false)
  const pendingSeed = useRef<Promise<unknown> | null>(null)

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
            pendingSeed.current = api.seedSession(sessionId, onboarding)
              .catch((err) => console.warn("[seed] Failed to seed session:", err))
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
    const prevSessionId = sessionId
    const newId = createSessionId()
    setSessionId(newId)
    setActiveSessionId(newId)
    clearMessages()
    setShowChips(true)
    initialized.current = false

    // Seed new session: prefer live profile from previous session (reflects agent
    // updates like age changes), fall back to localStorage onboarding data.
    pendingSeed.current = api.getSession(prevSessionId)
      .then((prev) => {
        const p = prev.family_profile
        return api.seedSession(newId, {
          childName: p.child_name || "",
          childAge: p.child_age || "",
          diagnosisStatus: p.diagnosis_status || "",
          adhdSubtype: p.adhd_subtype || "",
          challenges: p.challenge_areas,
          triedStrategies: p.attempted_strategies,
          goals: prev.goals.filter((g) => g.status === "active").map((g) => g.description),
        })
      })
      .catch(() => {
        // Previous session not found (e.g. first ever session) — fall back to onboarding
        const onboarding = getOnboarding()
        if (onboarding) {
          return api.seedSession(newId, onboarding)
        }
      })
      .catch((err) => console.warn("[seed] Failed to seed session:", err))
    initialized.current = true
  }, [sessionId, clearMessages, getOnboarding])

  const handleSelectSession = useCallback((selectedId: string) => {
    if (selectedId === sessionId) return
    setSessionId(selectedId)
    setActiveSessionId(selectedId)
    clearMessages()
    setShowChips(false)
    initialized.current = false
  }, [sessionId, clearMessages])

  // Post-stream side effects: refresh session state and update stats
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

  const handleSend = useCallback(async (content: string, attachments?: Attachment[]) => {
    // Wait for any pending seed to complete so the agent has family context
    if (pendingSeed.current) {
      await pendingSeed.current
      pendingSeed.current = null
    }
    setShowChips(false)
    sendMessage(content, attachments)
  }, [sendMessage])

  const handleChipSelect = useCallback((text: string) => {
    handleSend(text)
  }, [handleSend])

  const phase = (session?.phase || latestTrace?.phase_decision?.phase || "intake") as ConversationPhase

  return (
    <div className="flex h-[calc(100vh-10rem)]">
      <ChatSidebar
        isOpen={sidebarOpen}
        activeSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />

      <Card className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label={sidebarOpen ? "Close history" : "Open history"}
              className="text-muted-foreground"
            >
              <PanelLeft className="h-4 w-4" />
            </Button>
            <SessionProgress currentPhase={phase} />
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleNewChat}
            className="gap-2 text-xs text-muted-foreground"
          >
            <Plus className="h-3.5 w-3.5" />
            New Chat
          </Button>
        </div>

        {/* Chat area */}
        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          isStreaming={isStreaming}
          statusText={statusText}
          streamingContent={streamingContent}
          typewriterResetRef={typewriterResetRef}
          onStreamComplete={onStreamComplete}
          summaryText={summaryText}
        />

        {/* Quick reply chips */}
        <QuickReplyChips
          phase={phase}
          onSelect={handleChipSelect}
          visible={showChips && !isLoading && !isStreaming}
        />

        {/* Input */}
        <ChatInput onSend={handleSend} onStop={stopStreaming} isLoading={isLoading} isStreaming={isStreaming} sessionId={sessionId} />
      </Card>
    </div>
  )
}

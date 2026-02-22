import { useEffect, useState } from "react"
import { Plus, MessageCircle } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { api } from "@/lib/api"
import type { SessionListItem } from "@/types"

function formatSessionTime(dateStr: string): string {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    return "Today, " + date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
  }
  if (diffDays === 1) {
    return "Yesterday"
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

interface Props {
  isOpen: boolean
  activeSessionId: string
  onSelectSession: (sessionId: string) => void
  onNewChat: () => void
}

export function ChatSidebar({ isOpen, activeSessionId, onSelectSession, onNewChat }: Props) {
  const [sessions, setSessions] = useState<SessionListItem[]>([])

  useEffect(() => {
    if (isOpen) {
      api.listSessions()
        .then((data) => setSessions(data.sessions.filter((s) => s.turn_count > 0)))
        .catch(() => {})
    }
  }, [isOpen, activeSessionId])

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 256, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="shrink-0 overflow-hidden border-r border-border/50"
        >
          <div className="flex h-full w-64 flex-col">
            <div className="p-3">
              <Button
                variant="outline"
                size="sm"
                onClick={onNewChat}
                className="w-full gap-2"
              >
                <Plus className="h-3.5 w-3.5" />
                New Chat
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto px-2 pb-3">
              <p className="mb-2 px-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                History
              </p>
              {sessions.length === 0 ? (
                <p className="px-2 text-xs text-muted-foreground">No past sessions.</p>
              ) : (
                <div className="space-y-1">
                  {sessions.map((s) => {
                    const isActive = s.session_id === activeSessionId
                    return (
                      <button
                        key={s.session_id}
                        onClick={() => onSelectSession(s.session_id)}
                        aria-current={isActive ? "true" : undefined}
                        className={`w-full rounded-lg px-3 py-2.5 text-left transition-colors ${
                          isActive
                            ? "bg-primary/10 text-primary"
                            : "hover:bg-muted text-foreground"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <MessageCircle className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                          <span className="truncate text-xs font-medium">
                            {s.created_at ? formatSessionTime(s.created_at) : "Session"}
                          </span>
                        </div>
                        <div className="mt-1 pl-5.5">
                          <span className="text-[11px] text-muted-foreground">
                            {s.turn_count} messages
                          </span>
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

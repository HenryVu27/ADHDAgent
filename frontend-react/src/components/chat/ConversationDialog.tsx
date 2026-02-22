import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Sprout, User, ArrowRight } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import { setActiveSessionId } from "@/lib/auth"
import type { SessionListItem } from "@/types"

const phaseLabels: Record<string, string> = {
  intake: "Getting to Know You",
  strategy: "Exploring Strategies",
  progress: "Building a Plan",
  followup: "Following Up",
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
}

function formatMarkdown(text: string): string {
  const escaped = escapeHtml(text)
  return escaped
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/((?:^- .+\n?)+)/gm, (block) => {
      const items = block.trim().split("\n").map((l) => `<li>${l.slice(2)}</li>`).join("")
      return `<ul>${items}</ul>`
    })
    .split("\n\n")
    .map((p) => p.startsWith("<ul>") ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("")
}

interface Props {
  session: SessionListItem | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function ConversationDialog({ session, open, onOpenChange }: Props) {
  const navigate = useNavigate()
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (open && session) {
      let cancelled = false
      setLoading(true)
      api.getSessionMessages(session.session_id)
        .then((data) => {
          if (!cancelled) setMessages(data.messages.filter((m) => !m.blocked))
        })
        .catch(() => { if (!cancelled) setMessages([]) })
        .finally(() => { if (!cancelled) setLoading(false) })
      return () => { cancelled = true }
    } else {
      setMessages([])
    }
  }, [open, session])

  const handleResume = () => {
    if (!session) return
    setActiveSessionId(session.session_id)
    onOpenChange(false)
    navigate("/chat")
  }

  if (!session) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[80vh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Coaching Session
            {session.created_at && (
              <span className="text-sm font-normal text-muted-foreground">
                {new Date(session.created_at).toLocaleDateString(undefined, {
                  month: "long",
                  day: "numeric",
                  year: "numeric",
                })}
              </span>
            )}
          </DialogTitle>
          <DialogDescription asChild>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-xs">
                {phaseLabels[session.phase] || session.phase}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {session.turn_count} messages
              </span>
            </div>
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 py-4">
          {loading ? (
            <div className="flex h-32 items-center justify-center">
              <div className="flex items-center gap-1">
                <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:0ms]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:150ms]" />
                <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:300ms]" />
              </div>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
              No messages in this session.
            </div>
          ) : (
            messages.map((msg, i) => {
              const isUser = msg.role === "user"
              return (
                <div
                  key={i}
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
                      dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
                    />
                  </div>
                </div>
              )
            })
          )}
        </div>

        <DialogFooter>
          <Button onClick={handleResume} className="gap-2">
            Resume this chat
            <ArrowRight className="h-4 w-4" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

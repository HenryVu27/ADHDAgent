import { useRef, useEffect } from "react"
import { motion } from "framer-motion"
import { Sprout } from "lucide-react"
import { ChatBubble, StreamingContent } from "./ChatBubble"
import type { ChatMessage } from "@/types"

const SCROLL_THRESHOLD = 120 // px from bottom before auto-scroll disengages

interface ChatContainerProps {
  messages: ChatMessage[]
  isLoading: boolean
  isStreaming: boolean
  statusText: string
  summaryText: string
  streamingContent: string
}

export function ChatContainer({
  messages,
  isLoading,
  isStreaming,
  statusText,
  summaryText,
  streamingContent,
}: ChatContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const scrollToBottomIfNear = () => {
    const el = containerRef.current
    if (!el) return
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    if (distFromBottom < SCROLL_THRESHOLD) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  // Smart scroll during streaming
  useEffect(() => {
    scrollToBottomIfNear()
  }, [streamingContent])

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && !isLoading && !isStreaming && (
        <div className="flex h-full flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-coach/15">
            <Sprout className="h-8 w-8 text-coach" />
          </div>
          <h3 className="mb-1 text-lg font-semibold tracking-tight">
            Hi! I'm Ally, your ADHD parenting coach.
          </h3>
          <p className="max-w-sm text-sm text-muted-foreground leading-relaxed">
            Tell me what's going on with your family, and we'll figure it out together.
          </p>
        </div>
      )}

      {messages.map((msg) => (
        <ChatBubble key={msg.id} message={msg} />
      ))}

      {/* Loading state: summary + contextual status */}
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
            <p className="text-sm text-muted-foreground animate-pulse">
              {summaryText || statusText || "Thinking..."}
            </p>
          </div>
        </motion.div>
      )}

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
            <StreamingContent content={streamingContent} />
          </div>
        </motion.div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}

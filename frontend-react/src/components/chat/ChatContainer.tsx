import { useRef, useEffect } from "react"
import { motion } from "framer-motion"
import { Sprout } from "lucide-react"
import { ChatBubble, StreamingBubble } from "./ChatBubble"
import type { ChatMessage } from "@/types"

const SCROLL_THRESHOLD = 120 // px from bottom before auto-scroll disengages

interface Props {
  messages: ChatMessage[]
  isLoading: boolean
  isStreaming: boolean
  statusText: string
  streamingContent: string
  typewriterResetRef: React.MutableRefObject<((text: string) => void) | null>
}

export function ChatContainer({
  messages,
  isLoading,
  isStreaming,
  statusText,
  streamingContent,
  typewriterResetRef,
}: Props) {
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

      {/* Status line: shown while waiting for first token */}
      {isLoading && statusText && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="rounded-2xl bg-card px-4 py-3 shadow-sm border border-border/30">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-coach" />
              <span className="text-sm text-muted-foreground">{statusText}</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Fallback dots while loading with no status yet */}
      {isLoading && !statusText && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex gap-3"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-coach">
            <Sprout className="h-4 w-4 text-coach-foreground" />
          </div>
          <div className="rounded-2xl bg-card px-4 py-3 shadow-sm border border-border/30">
            <div className="flex items-center gap-1">
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:0ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:150ms]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-coach/40 [animation-delay:300ms]" />
            </div>
          </div>
        </motion.div>
      )}

      {/* Streaming bubble: typewriter animation */}
      {isStreaming && (
        <StreamingBubble
          content={streamingContent}
          typewriterResetRef={typewriterResetRef}
        />
      )}

      <div ref={bottomRef} />
    </div>
  )
}

import { motion } from "framer-motion"
import { Sprout, User } from "lucide-react"
import { useTypewriter } from "@/hooks/use-typewriter"
import type { ChatMessage } from "@/types"

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(.+)$/s, "<p>$1</p>")
}

interface Props {
  message: ChatMessage
}

interface StreamingBubbleProps {
  content: string
  typewriterResetRef: React.MutableRefObject<((text: string) => void) | null>
  onComplete?: () => void
}

/**
 * Content-only streaming component during SSE streaming.
 * Uses useTypewriter for smooth character-by-character animation.
 * onComplete fires when the typewriter catches up to all received content,
 * signaling the done handler that it's safe to finalize.
 * The avatar/wrapper is provided by ChatContainer.
 */
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

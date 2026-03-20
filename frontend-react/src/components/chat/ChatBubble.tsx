import { motion } from "framer-motion"
import { Sprout, User, FileText } from "lucide-react"
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

const markdownClasses = "text-sm leading-relaxed [&_ul]:ml-4 [&_ul]:list-disc [&_ul]:space-y-1 [&_p]:mb-2 [&_p:last-child]:mb-0 [&_strong]:font-semibold"

interface Props {
  message: ChatMessage
}

interface StreamingContentProps {
  content: string
}

/**
 * Renders streamed tokens directly as they arrive — no artificial animation.
 * Appends a blinking cursor to signal that more text is incoming.
 */
export function StreamingContent({ content }: StreamingContentProps) {
  return (
    <div className={markdownClasses}>
      <span dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }} />
      <span className="inline-block w-[2px] h-[1em] bg-current ml-0.5 align-text-bottom animate-[cursor-blink_1s_step-end_infinite]" />
    </div>
  )
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user"

  // Skip entrance animation for messages that were just streamed —
  // the content was already visible in the streaming bubble.
  const skipAnimation = message.streamed

  return (
    <motion.div
      initial={skipAnimation ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: skipAnimation ? 0 : 0.2 }}
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
        {message.attachments && message.attachments.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {message.attachments.map(att => (
              <div key={att.id} className="flex items-center gap-1.5 rounded border border-border/50 px-2 py-1 text-xs">
                {att.content_type.startsWith("image/") && att.thumbnail_url ? (
                  <a href={att.thumbnail_url} target="_blank" rel="noopener noreferrer">
                    <img
                      src={att.thumbnail_url}
                      alt={att.filename}
                      className="h-10 w-10 rounded object-cover cursor-pointer hover:opacity-80"
                    />
                  </a>
                ) : (
                  <FileText className="h-4 w-4" />
                )}
                <span className="max-w-[100px] truncate text-xs">{att.filename}</span>
              </div>
            ))}
          </div>
        )}
        <div
          className={markdownClasses}
          dangerouslySetInnerHTML={{ __html: formatMarkdown(message.content) }}
        />
      </div>
    </motion.div>
  )
}

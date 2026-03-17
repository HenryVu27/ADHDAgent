import { useState, useRef } from "react"
import { Send, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface Props {
  onSend: (message: string) => void
  onStop: () => void
  isLoading: boolean
  isStreaming: boolean
}

export function ChatInput({ onSend, onStop, isLoading, isStreaming }: Props) {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const busy = isLoading || isStreaming

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || busy) return
    onSend(trimmed)
    setValue("")
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex items-end gap-2 border-t border-border/50 bg-background p-4">
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Share what's going on with your family..."
        className="min-h-[44px] max-h-32 resize-none"
        rows={1}
        disabled={busy}
      />
      {isStreaming ? (
        <Button
          onClick={onStop}
          size="icon"
          variant="outline"
          className="shrink-0"
          title="Stop generating"
        >
          <Square className="h-4 w-4" />
        </Button>
      ) : (
        <Button
          onClick={handleSend}
          disabled={!value.trim() || isLoading}
          size="icon"
          className="shrink-0"
        >
          <Send className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}

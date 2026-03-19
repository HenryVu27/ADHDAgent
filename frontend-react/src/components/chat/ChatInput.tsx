import { useState, useRef, useCallback } from "react"
import { ArrowUp, Square, Paperclip, X, FileText, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { api } from "@/lib/api"
import type { Attachment, UploadResponse, UploadBlockedResponse } from "@/types"

const ALLOWED_TYPES = new Set([
  "image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf",
])
const MAX_FILES = 3

interface PendingFile {
  id: string // temp id until upload completes
  file: File
  previewUrl?: string
  uploading: boolean
  attachment?: Attachment // set after successful upload
  error?: string
}

interface Props {
  onSend: (message: string, attachments?: Attachment[]) => void
  onStop: () => void
  isLoading: boolean
  isStreaming: boolean
  sessionId: string
}

export function ChatInput({ onSend, onStop, isLoading, isStreaming, sessionId }: Props) {
  const [value, setValue] = useState("")
  const [pendingFiles, setPendingFiles] = useState<PendingFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const busy = isLoading || isStreaming
  const uploading = pendingFiles.some(f => f.uploading)

  const addFiles = useCallback(async (files: File[]) => {
    const available = MAX_FILES - pendingFiles.length
    const toAdd = files.slice(0, available).filter(f => ALLOWED_TYPES.has(f.type))
    if (toAdd.length === 0) return

    const newPending: PendingFile[] = toAdd.map(file => ({
      id: crypto.randomUUID(),
      file,
      previewUrl: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
      uploading: true,
    }))

    setPendingFiles(prev => [...prev, ...newPending])

    // Upload each file
    for (const pf of newPending) {
      try {
        const result = await api.uploadFile(pf.file, sessionId)
        if ("blocked" in result) {
          const blocked = result as UploadBlockedResponse
          setPendingFiles(prev =>
            prev.map(f => f.id === pf.id ? { ...f, uploading: false, error: blocked.response || "Blocked" } : f)
          )
        } else {
          const uploaded = result as UploadResponse
          const attachment: Attachment = {
            id: uploaded.id,
            filename: uploaded.filename,
            content_type: uploaded.content_type,
            thumbnail_url: uploaded.thumbnail_url,
          }
          setPendingFiles(prev =>
            prev.map(f => f.id === pf.id ? { ...f, uploading: false, attachment } : f)
          )
        }
      } catch (e) {
        setPendingFiles(prev =>
          prev.map(f => f.id === pf.id ? { ...f, uploading: false, error: String(e) } : f)
        )
      }
    }
  }, [pendingFiles.length, sessionId])

  const removeFile = (id: string) => {
    setPendingFiles(prev => {
      const file = prev.find(f => f.id === id)
      if (file?.previewUrl) URL.revokeObjectURL(file.previewUrl)
      return prev.filter(f => f.id !== id)
    })
  }

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || busy || uploading) return
    const attachments = pendingFiles
      .filter(f => f.attachment && !f.error)
      .map(f => f.attachment!)
    onSend(trimmed, attachments.length > 0 ? attachments : undefined)
    setValue("")
    setPendingFiles([])
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    addFiles(files)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => setIsDragOver(false)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
      e.target.value = "" // reset so same file can be re-selected
    }
  }

  return (
    <div
      className={`border-t border-border/50 bg-background p-4 ${isDragOver ? "ring-2 ring-primary ring-inset" : ""}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {/* Pending file chips */}
      {pendingFiles.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {pendingFiles.map(pf => (
            <div
              key={pf.id}
              className={`flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs ${
                pf.error ? "border-destructive bg-destructive/10" : "border-border bg-muted"
              }`}
            >
              {pf.uploading ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : pf.previewUrl ? (
                <img src={pf.previewUrl} alt="" className="h-6 w-6 rounded object-cover" />
              ) : (
                <FileText className="h-3 w-3" />
              )}
              <span className="max-w-[120px] truncate">{pf.file.name}</span>
              <button onClick={() => removeFile(pf.id)} className="ml-1 hover:text-destructive">
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
          multiple
          onChange={handleFileSelect}
        />
        <Button
          variant="ghost"
          size="icon"
          className="shrink-0"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy || pendingFiles.length >= MAX_FILES}
          title="Attach file"
        >
          <Paperclip className="h-4 w-4" />
        </Button>

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
            disabled={!value.trim() || isLoading || uploading}
            size="icon"
            className="shrink-0"
          >
            <ArrowUp className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  )
}

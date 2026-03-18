# Multimodal File Upload Design

**Date:** 2026-03-18
**Status:** Approved

## Overview

Add multimodal capability to the ADHD coaching chatbot: parents can drag-and-drop or attach images and PDFs (IEPs, report cards, behavior charts) to chat messages. The agent sees file content via Gemini's native multimodal support and can reference it in coaching responses. Files are scanned through the input gate before reaching the agent.

## Supported Files

- **Images:** JPEG, PNG, GIF, WebP
- **Documents:** PDF
- **Limits:** Max 10MB per file, up to 3 files per message

## Architecture

```
File selected/dropped in ChatInput
    |
    v
POST /api/upload (multipart/form-data)
    |
    v
Validate type + size
    |
    v
Upload to Gemini File API -> get file URI
    |
    v
Content extraction (Gemini Flash call)
    |
    v
InputGate.check() on extracted text
    |-- crisis -> return crisis resources, delete Gemini file
    |-- jailbreak -> return rejection, delete Gemini file
    |
    v
Generate thumbnail (Pillow, images only)
    |
    v
Store metadata in SQLite attachments table
    |
    v
Return { id, filename, content_type, thumbnail_url }
    |
    v
Frontend displays thumbnail chip in ChatInput
    |
    v
User sends message with attachment_ids
    |
    v
Orchestrator resolves IDs -> Gemini file URIs
    |
    v
HumanMessage(content=[text_part, file_part, ...])
    |
    v
Normal ReAct agent loop (agent sees file content natively)
```

## Storage

### SQLite: `attachments` table (SCHEMA_V6)

```sql
CREATE TABLE attachments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    gemini_file_name TEXT NOT NULL,
    gemini_file_uri TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    thumbnail_path TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_attachments_session ON attachments(session_id);
```

Note: `gemini_file_name` (e.g. `files/abc123`) is needed for deletion via `client.files.delete(name=...)`. `gemini_file_uri` is the URI used in content generation.

### Thumbnails

- Local directory: `data/thumbnails/{session_id}/`
- Generated via Pillow for images
- PDFs use a generic document icon (no generation needed)
- Served via `GET /api/attachments/{id}/thumbnail`

### Gemini File API

- Files uploaded to Gemini have a 48-hour TTL (sufficient for session-scoped usage)
- No manual cleanup needed for most cases
- On session deletion: best-effort delete via API

## API Changes

### New Endpoint: `POST /api/upload`

**Request:** `multipart/form-data`
- `file`: the file
- `session_id`: string (validated with same `[a-zA-Z0-9_-]{1,128}` pattern as ChatRequest to prevent path traversal in thumbnail storage)

**Rate limit:** 5/minute (file uploads are more expensive than text -- Gemini File API call + Flash extraction call).

**Response (success):**
```json
{
  "id": "att_abc123",
  "filename": "report_card.pdf",
  "content_type": "application/pdf",
  "thumbnail_url": "/api/attachments/att_abc123/thumbnail"
}
```

**Response (input gate blocked):**
```json
{
  "blocked": true,
  "reason": "crisis" | "jailbreak",
  "response": "..."
}
```

### New Endpoint: `GET /api/attachments/{id}/thumbnail`

Returns the thumbnail image for display in chat.

### Modified: `POST /api/chat/stream`

`ChatRequest` gains an optional field:
```python
attachment_ids: list[str] = []  # max 3
```

### Modified: `DELETE /api/session/{id}`

Extended to also:
- Delete attachment records from `attachments` table
- Delete local thumbnail files from disk
- Best-effort delete Gemini files

## Schema Changes

### Backend

```python
# New model
class Attachment(BaseModel):
    id: str
    filename: str
    content_type: str
    thumbnail_url: str | None

# ChatRequest update
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = "default"
    attachment_ids: list[str] = Field(default_factory=list, max_length=3)
```

### Frontend

```typescript
interface Attachment {
  id: string;
  filename: string;
  content_type: string;
  thumbnail_url: string | null;
}

// Message type gains:
interface Message {
  // ...existing fields
  attachments?: Attachment[];
}
```

## LangGraph Message Format

Currently: `HumanMessage(content="user text")`

With attachments, use the `media` content type supported by `langchain-google-genai`:
```python
HumanMessage(content=[
    {"type": "text", "text": "Look at my child's report card"},
    {"type": "media", "file_uri": "<gemini_file_uri>", "mime_type": "application/pdf"},
    {"type": "media", "file_uri": "<gemini_file_uri>", "mime_type": "image/jpeg"},
])
```

The `file_uri` comes from the Gemini File API upload response. The `mime_type` comes from the stored `content_type` in the attachments table.

## Input Gate File Scanning

1. File uploaded to Gemini File API
2. Flash call extracts text content from the file
   - Images: "extract any visible text from this image"
   - PDFs: "extract the text content of this document"
3. Extracted text passed through existing `InputGate.check()`
4. If flagged: reject upload, delete Gemini file
5. If clean (or no text extractable): proceed normally
6. **If extraction call fails** (timeout, API error): **fail closed** -- reject the upload. This is a safety-critical path; unlike the text input gate which fails open on errors, file content is opaque and must be scanned successfully before being allowed through.

**Cost:** One Flash call per uploaded file. Acceptable given upload frequency is much lower than text messages.

### Input gate on chat messages with attachments

The existing `input_gate_node` in `graph.py` passes `latest_human.content` to `InputGate.check()`, which expects a string. When `content` is a multipart list (text + media parts), the node must extract just the text portion before calling `check()`. Use the existing `_extract_text()` helper from `orchestrator.py` (which already handles list content by joining text parts and skipping non-text entries). The file content itself was already scanned at upload time, so only the text message needs input gate classification here.

## Context Assembly (hooks.py)

When reconstructing conversation history:
- Attachment metadata (filename, type) is included in stored messages
- File content is NOT re-sent every turn (avoids token bloat)
- Agent can reference files from earlier turns by description since it already saw them
- Within the 48-hour Gemini TTL, the agent could re-read a file if needed via the stored URI

### Multipart content handling

The state block injection in `hooks.py` currently does `original_content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)`. When `content` is a multipart list, `str()` destroys the structure. Instead:
- If `content` is a string: prepend the state block as before
- If `content` is a list: find the first `{"type": "text", ...}` part and prepend the state block to its `"text"` value. This preserves file/media parts intact.

The `_estimate_chars` function already handles list content correctly.

## Frontend UI

### ChatInput.tsx
- Paperclip/attach button next to the send button
- Drag-and-drop zone over the input area (visual highlight on drag-over)
- File picker filtered to images + PDF
- Attached files appear as removable thumbnail chips below the textarea
  - Images: small preview thumbnail
  - PDFs: document icon + filename
- Files upload immediately on attach (not on send) for early error handling
- Upload progress: simple spinner on each chip
- Send button disabled while uploads are in-flight
- Max 3 files enforced client-side

### ChatBubble.tsx
- User messages with attachments render thumbnail chips above the message text
- Images: small thumbnail, click to open full-size in modal or new tab
- PDFs: document icon + filename, click to download

## Session Lifecycle

- **Upload:** Files uploaded to Gemini, metadata stored in SQLite, thumbnails on disk
- **Chat:** Gemini file URIs included in messages, agent sees content natively
- **Re-reference:** Within 48h, agent can re-read files via stored URI. After 48h, metadata/thumbnails remain but content unavailable.
- **Deletion:** Session delete clears attachments table rows, local thumbnails, and best-effort Gemini file deletion

## Security

### File type validation

MIME types from `multipart/form-data` are client-provided and can be spoofed. The upload endpoint must perform server-side magic byte validation:
- JPEG: starts with `FF D8 FF`
- PNG: starts with `89 50 4E 47`
- GIF: starts with `47 49 46 38`
- WebP: starts with `52 49 46 46` + `57 45 42 50` at offset 8
- PDF: starts with `%PDF`

Reject files whose magic bytes don't match the declared MIME type. As an additional safety net, Gemini File API will also reject unsupported formats.

## Config (app/config.py)

```python
UPLOAD_MAX_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
UPLOAD_MAX_FILES_PER_MESSAGE: int = 3
UPLOAD_ALLOWED_TYPES: set = {"image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"}
UPLOAD_THUMBNAIL_DIR: str = "data/thumbnails"
UPLOAD_RATE_LIMIT: str = "5/minute"
```

## New Dependencies

- **Pillow** -- thumbnail generation for images

## Files Touched

| File | Change |
|------|--------|
| `app/api/routes.py` | New upload endpoint, thumbnail serving, session delete extension |
| `app/models/schemas.py` | ChatRequest attachment_ids, Attachment model |
| `app/db.py` | Attachments table migration |
| `app/agent/graph.py` | Input gate node: extract text from multipart content before checking |
| `app/agent/orchestrator.py` | Resolve attachment IDs to Gemini file parts, update `_extract_text` for media parts |
| `app/agent/hooks.py` | State block injection for multipart content, handle list content in history |
| `app/guardrails/validator.py` | File content scanning helper |
| `app/config.py` | File upload settings |
| `frontend-react/src/components/chat/ChatInput.tsx` | Drag-drop, attach button, thumbnail chips |
| `frontend-react/src/components/chat/ChatBubble.tsx` | Attachment rendering |
| `frontend-react/src/lib/api.ts` | uploadFile method, chatStream update |
| `frontend-react/src/types/index.ts` | Attachment type |

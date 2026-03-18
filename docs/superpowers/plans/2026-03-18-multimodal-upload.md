# Multimodal File Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable parents to attach images and PDFs to chat messages so the ADHD coaching agent can see and reference them natively via Gemini's multimodal support.

**Architecture:** Files upload to the Gemini File API at attach-time (not send-time), get scanned through the input gate via Flash text extraction, and are stored as metadata in SQLite. When the user sends a message with attachments, the orchestrator resolves attachment IDs to Gemini file URIs and builds multipart `HumanMessage` content. Existing input gate and hooks code is updated to handle list-type content.

**Tech Stack:** Gemini File API (google-genai SDK), Pillow (thumbnails), FastAPI (multipart upload endpoint), SQLite (attachments table), React 19 (drag-drop UI)

**Spec:** `docs/superpowers/specs/2026-03-18-multimodal-upload-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `app/config.py` | Modify | Add upload settings (max size, allowed types, thumbnail dir, rate limit) |
| `app/db.py` | Modify | Add SCHEMA_V6 with attachments table |
| `app/models/schemas.py` | Modify | Add `Attachment`, `UploadResponse`, `UploadBlockedResponse` models; extend `ChatRequest` with `attachment_ids` |
| `app/services/file_upload.py` | Create | File validation (magic bytes), Gemini upload, text extraction, thumbnail generation, input gate scanning |
| `app/api/routes.py` | Modify | Add `POST /api/upload`, `GET /api/attachments/{id}/thumbnail`; extend `DELETE /api/session/{id}` |
| `app/agent/orchestrator.py` | Modify | Resolve attachment IDs to multipart `HumanMessage` content in `process()` and `process_stream()` |
| `app/agent/graph.py` | Modify | Extract text from multipart content before `input_gate.check()` |
| `app/agent/hooks.py` | Modify | Handle list content in state block injection |
| `app/agent/sqlite_store.py` | Modify | Add attachments to `delete_session()` cascade |
| `app/agent/store_protocol.py` | Modify | Add `get_attachments()` and `save_attachment()` abstract methods |
| `frontend-react/src/types/index.ts` | Modify | Add `Attachment` interface, extend `ChatRequest` and `ChatMessage` |
| `frontend-react/src/lib/api.ts` | Modify | Add `uploadFile()` method, extend `chatStream` for `attachment_ids` |
| `frontend-react/src/components/chat/ChatInput.tsx` | Modify | Drag-drop zone, paperclip button, thumbnail chips, upload state management |
| `frontend-react/src/components/chat/ChatBubble.tsx` | Modify | Render attachment thumbnails on user messages |
| `requirements.txt` | Modify | Add `Pillow` |
| `tests/test_file_upload.py` | Create | Unit tests for file validation, upload service |
| `tests/test_multimodal_api.py` | Create | Unit tests for upload endpoint, thumbnail serving |
| `tests/test_multimodal_agent.py` | Create | Unit tests for orchestrator/graph/hooks multipart handling |

---

### Task 1: Config + Dependencies

**Files:**
- Modify: `app/config.py:78-84` (after SQLite persistence block)
- Modify: `requirements.txt`

- [ ] **Step 1: Add upload settings to config**

In `app/config.py`, add after line 84 (`EVENT_BUFFER_SIZE`), before the `@model_validator`:

```python
    # File upload
    UPLOAD_MAX_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_MAX_FILES_PER_MESSAGE: int = 3
    UPLOAD_ALLOWED_TYPES: set[str] = {"image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"}
    UPLOAD_THUMBNAIL_DIR: str = "data/thumbnails"
    UPLOAD_RATE_LIMIT: str = "5/minute"
```

- [ ] **Step 2: Add Pillow to requirements.txt**

Append to `requirements.txt`:

```
Pillow>=11.0.0
```

- [ ] **Step 3: Install Pillow**

Run: `./adhd312/Scripts/pip.exe install Pillow`

- [ ] **Step 4: Commit**

```bash
git add app/config.py requirements.txt
git commit -m "Add upload config settings and Pillow dependency"
```

---

### Task 2: Database Migration (SCHEMA_V6)

**Files:**
- Modify: `app/db.py:201-209` (after SCHEMA_V5, update MIGRATIONS dict)
- Test: `tests/test_db_migration.py` (new)

Note: SCHEMA_V5 already exists (creates `users` table and adds `user_id` to sessions). The attachments table must be SCHEMA_V6.

- [ ] **Step 1: Write failing test for V6 migration**

Create `tests/test_db_migration.py`:

```python
import sqlite3
import pytest
from app.db import init_db, _get_schema_version

def test_schema_v6_creates_attachments_table():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    # Check version
    assert _get_schema_version(conn) == 6
    # Check table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='attachments'"
    )
    assert cursor.fetchone() is not None
    # Check columns
    cursor = conn.execute("PRAGMA table_info(attachments)")
    columns = {row["name"] for row in cursor.fetchall()}
    assert columns == {
        "id", "session_id", "gemini_file_name", "gemini_file_uri",
        "filename", "content_type", "size_bytes", "thumbnail_path", "created_at",
    }
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_db_migration.py -v`
Expected: FAIL — schema version is 5, no attachments table

- [ ] **Step 3: Add SCHEMA_V6 and update MIGRATIONS**

In `app/db.py`, add after `SCHEMA_V5` (after line 201):

```python
SCHEMA_V6 = """
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    gemini_file_name TEXT NOT NULL,
    gemini_file_uri TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    thumbnail_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_attachments_session ON attachments(session_id);
"""
```

Update MIGRATIONS dict to include `6: SCHEMA_V6`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_db_migration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db_migration.py
git commit -m "Add SCHEMA_V6 with attachments table"
```

---

### Task 3: Pydantic Models

**Files:**
- Modify: `app/models/schemas.py:222-239` (ChatRequest), add new models after line 239
- Test: `tests/test_schemas_upload.py` (new)

- [ ] **Step 1: Write failing test for new models**

Create `tests/test_schemas_upload.py`:

```python
import pytest
from pydantic import ValidationError
from app.models.schemas import Attachment, UploadResponse, UploadBlockedResponse, ChatRequest

def test_attachment_model():
    att = Attachment(id="att_123", filename="report.pdf", content_type="application/pdf", thumbnail_url="/api/attachments/att_123/thumbnail")
    assert att.id == "att_123"
    assert att.thumbnail_url is not None

def test_attachment_optional_thumbnail():
    att = Attachment(id="att_123", filename="report.pdf", content_type="application/pdf")
    assert att.thumbnail_url is None

def test_upload_response():
    resp = UploadResponse(id="att_123", filename="report.pdf", content_type="application/pdf", thumbnail_url="/thumb")
    assert resp.id == "att_123"

def test_upload_blocked_response():
    resp = UploadBlockedResponse(blocked=True, reason="crisis", response="Call 988")
    assert resp.blocked is True

def test_chat_request_with_attachment_ids():
    req = ChatRequest(message="Look at this", session_id="test", attachment_ids=["att_1", "att_2"])
    assert req.attachment_ids == ["att_1", "att_2"]

def test_chat_request_default_empty_attachment_ids():
    req = ChatRequest(message="Hello", session_id="test")
    assert req.attachment_ids == []

def test_chat_request_max_3_attachments():
    with pytest.raises(ValidationError):
        ChatRequest(message="Hello", attachment_ids=["a", "b", "c", "d"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_schemas_upload.py -v`
Expected: FAIL — models don't exist

- [ ] **Step 3: Add models to schemas.py**

In `app/models/schemas.py`, add after `ChatRequest` class (after line 239):

```python
class Attachment(BaseModel):
    id: str
    filename: str
    content_type: str
    thumbnail_url: str | None = None


class UploadResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    thumbnail_url: str | None = None


class UploadBlockedResponse(BaseModel):
    blocked: bool = True
    reason: str  # "crisis" | "jailbreak"
    response: str
```

Modify `ChatRequest` to add `attachment_ids`:

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = "default"
    attachment_ids: list[str] = Field(default_factory=list, max_length=3)
    # ... existing validators unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_schemas_upload.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/schemas.py tests/test_schemas_upload.py
git commit -m "Add Attachment and upload response models, extend ChatRequest with attachment_ids"
```

---

### Task 4: File Upload Service

**Files:**
- Create: `app/services/file_upload.py`
- Test: `tests/test_file_upload.py` (new)

This is the core backend module: magic byte validation, Gemini File API upload, Flash text extraction for input gate scanning, and Pillow thumbnail generation.

- [ ] **Step 1: Write failing tests for magic byte validation**

Create `tests/test_file_upload.py`:

```python
import io
import os
import pytest
from app.services.file_upload import validate_magic_bytes

# Real magic bytes for each format
JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 20
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
GIF_HEADER = b"GIF89a" + b"\x00" * 20
WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20
PDF_HEADER = b"%PDF-1.7" + b"\x00" * 20

def test_valid_jpeg():
    assert validate_magic_bytes(JPEG_HEADER, "image/jpeg") is True

def test_valid_png():
    assert validate_magic_bytes(PNG_HEADER, "image/png") is True

def test_valid_gif():
    assert validate_magic_bytes(GIF_HEADER, "image/gif") is True

def test_valid_webp():
    assert validate_magic_bytes(WEBP_HEADER, "image/webp") is True

def test_valid_pdf():
    assert validate_magic_bytes(PDF_HEADER, "application/pdf") is True

def test_mismatched_type():
    """JPEG bytes but declared as PNG should fail."""
    assert validate_magic_bytes(JPEG_HEADER, "image/png") is False

def test_unknown_type():
    assert validate_magic_bytes(b"random bytes", "text/plain") is False

def test_empty_bytes():
    assert validate_magic_bytes(b"", "image/jpeg") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_file_upload.py -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Create file_upload.py with validate_magic_bytes**

Create `app/services/__init__.py` (empty) and `app/services/file_upload.py`:

```python
"""File upload service: validation, Gemini upload, scanning, thumbnails."""

import asyncio
import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Magic byte signatures for allowed file types
MAGIC_BYTES = {
    "image/jpeg": [(0, b"\xff\xd8\xff")],
    "image/png": [(0, b"\x89PNG")],
    "image/gif": [(0, b"GIF8")],
    "image/webp": [(0, b"RIFF"), (8, b"WEBP")],
    "application/pdf": [(0, b"%PDF")],
}


def validate_magic_bytes(data: bytes, content_type: str) -> bool:
    """Validate file content matches declared MIME type via magic bytes."""
    signatures = MAGIC_BYTES.get(content_type)
    if not signatures:
        return False
    for offset, expected in signatures:
        if len(data) < offset + len(expected):
            return False
        if data[offset : offset + len(expected)] != expected:
            return False
    return True


def generate_attachment_id() -> str:
    """Generate a unique attachment ID."""
    return f"att_{uuid.uuid4().hex[:12]}"


async def upload_to_gemini(data: bytes, filename: str, content_type: str) -> tuple[str, str]:
    """Upload file to Gemini File API. Returns (file_name, file_uri).

    Uses the google-genai SDK's synchronous file upload in a thread
    to avoid blocking the event loop.
    """
    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _upload():
        response = client.files.upload(
            file=io.BytesIO(data),
            config={"display_name": filename, "mime_type": content_type},
        )
        return response.name, response.uri

    return await asyncio.to_thread(_upload)


async def delete_gemini_file(gemini_file_name: str) -> None:
    """Best-effort delete a file from Gemini File API."""
    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        await asyncio.to_thread(client.files.delete, name=gemini_file_name)
        logger.info("Deleted Gemini file: %s", gemini_file_name)
    except Exception as e:
        logger.warning("Failed to delete Gemini file %s: %s", gemini_file_name, e)


async def extract_text_from_file(
    gemini_file_uri: str, content_type: str
) -> str:
    """Extract text from an uploaded file via Gemini Flash call.

    Returns extracted text, or empty string if no text found.
    Raises on API failure (caller should handle fail-closed).
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    if content_type.startswith("image/"):
        prompt = "Extract any visible text from this image. If there is no text, respond with EMPTY."
    else:
        prompt = "Extract the text content of this document. If there is no text, respond with EMPTY."

    def _extract():
        response = client.models.generate_content(
            model=settings.GEMINI_UTILITY_MODEL,
            contents=[
                {"file_data": {"file_uri": gemini_file_uri, "mime_type": content_type}},
                prompt,
            ],
            config=GenerateContentConfig(temperature=0.0, max_output_tokens=4096),
        )
        return response.text or ""

    text = await asyncio.wait_for(asyncio.to_thread(_extract), timeout=30.0)
    if text.strip().upper() == "EMPTY":
        return ""
    return text


def generate_thumbnail(data: bytes, content_type: str, session_id: str, attachment_id: str) -> str | None:
    """Generate a thumbnail for image files. Returns relative path or None for PDFs."""
    if not content_type.startswith("image/"):
        return None

    from PIL import Image

    thumb_dir = Path(settings.UPLOAD_THUMBNAIL_DIR) / session_id
    thumb_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(io.BytesIO(data))
    img.thumbnail((200, 200))

    thumb_path = thumb_dir / f"{attachment_id}.webp"
    img.save(thumb_path, "WEBP", quality=80)
    return str(thumb_path)
```

- [ ] **Step 4: Run tests to verify magic byte validation passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_file_upload.py -v`
Expected: PASS

- [ ] **Step 5: Write tests for thumbnail generation**

Add to `tests/test_file_upload.py`:

```python
import tempfile
from unittest.mock import patch
from app.services.file_upload import generate_thumbnail, generate_attachment_id

def test_generate_attachment_id_format():
    aid = generate_attachment_id()
    assert aid.startswith("att_")
    assert len(aid) == 16  # "att_" + 12 hex chars

def test_generate_thumbnail_image(tmp_path):
    """Generate a thumbnail from a small JPEG."""
    from PIL import Image
    img = Image.new("RGB", (400, 300), color="red")
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    data = buf.getvalue()

    with patch("app.services.file_upload.settings") as mock_settings:
        mock_settings.UPLOAD_THUMBNAIL_DIR = str(tmp_path)
        path = generate_thumbnail(data, "image/jpeg", "session1", "att_abc")
    assert path is not None
    assert path.endswith(".webp")
    assert os.path.exists(path)

def test_generate_thumbnail_pdf_returns_none():
    """PDFs should not generate thumbnails."""
    result = generate_thumbnail(b"%PDF-1.7 content", "application/pdf", "s1", "att_1")
    assert result is None
```

- [ ] **Step 6: Run tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_file_upload.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/services/__init__.py app/services/file_upload.py tests/test_file_upload.py
git commit -m "Add file upload service with validation, Gemini upload, and thumbnail generation"
```

---

### Task 5: Store Protocol + SQLite Store Extensions

**Files:**
- Modify: `app/agent/store_protocol.py` (add abstract methods)
- Modify: `app/agent/sqlite_store.py` (implement methods, extend delete_session)
- Test: `tests/test_sqlite_attachments.py` (new)

- [ ] **Step 1: Write failing tests for attachment store methods**

Create `tests/test_sqlite_attachments.py`:

```python
import sqlite3
import pytest
import pytest_asyncio
import aiosqlite
from app.agent.sqlite_store import SQLiteSessionStore
from app.db import init_db

@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    # Run sync init_db on the underlying connection
    sync_conn = sqlite3.connect(":memory:")
    sync_conn.row_factory = sqlite3.Row
    init_db(sync_conn)
    # Replay schema on async conn
    schema = sync_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' OR type='index'"
    ).fetchall()
    for row in schema:
        if row[0]:
            await conn.execute(row[0])
    # Insert schema version
    await conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (5)")
    await conn.commit()
    s = SQLiteSessionStore(conn)
    yield s
    await conn.close()

@pytest.mark.asyncio
async def test_save_and_get_attachment(store):
    # Ensure session exists
    await store.get("test_session")
    await store.save_attachment(
        session_id="test_session",
        attachment_id="att_abc",
        gemini_file_name="files/xyz",
        gemini_file_uri="https://generativelanguage.googleapis.com/files/xyz",
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=1024,
        thumbnail_path=None,
    )
    await store.commit()

    attachments = await store.get_attachments(["att_abc"])
    assert len(attachments) == 1
    assert attachments[0]["id"] == "att_abc"
    assert attachments[0]["gemini_file_uri"].endswith("files/xyz")

@pytest.mark.asyncio
async def test_get_attachments_missing_id(store):
    result = await store.get_attachments(["nonexistent"])
    assert len(result) == 0

@pytest.mark.asyncio
async def test_delete_session_removes_attachments(store):
    await store.get("test_session")
    await store.save_attachment(
        session_id="test_session",
        attachment_id="att_del",
        gemini_file_name="files/del",
        gemini_file_uri="uri://del",
        filename="img.jpg",
        content_type="image/jpeg",
        size_bytes=500,
    )
    await store.commit()

    await store.delete_session("test_session")
    result = await store.get_attachments(["att_del"])
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_sqlite_attachments.py -v`
Expected: FAIL — `save_attachment` and `get_attachments` don't exist

- [ ] **Step 3: Add abstract methods to store_protocol.py**

In `app/agent/store_protocol.py`, add after existing abstract methods:

```python
    @abstractmethod
    async def save_attachment(
        self,
        session_id: str,
        attachment_id: str,
        gemini_file_name: str,
        gemini_file_uri: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        thumbnail_path: str | None = None,
    ) -> None:
        """Save file attachment metadata."""

    @abstractmethod
    async def get_attachments(self, attachment_ids: list[str]) -> list[dict]:
        """Get attachment records by IDs. Returns list of dicts with all columns."""

    @abstractmethod
    async def get_attachments_by_session(self, session_id: str) -> list[dict]:
        """Get all attachment records for a session."""
```

- [ ] **Step 4: Implement in sqlite_store.py**

Add methods to `SQLiteSessionStore`:

```python
    async def save_attachment(
        self,
        session_id: str,
        attachment_id: str,
        gemini_file_name: str,
        gemini_file_uri: str,
        filename: str,
        content_type: str,
        size_bytes: int,
        thumbnail_path: str | None = None,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO attachments
               (id, session_id, gemini_file_name, gemini_file_uri, filename, content_type, size_bytes, thumbnail_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (attachment_id, session_id, gemini_file_name, gemini_file_uri,
             filename, content_type, size_bytes, thumbnail_path),
        )

    async def get_attachments(self, attachment_ids: list[str]) -> list[dict]:
        if not attachment_ids:
            return []
        placeholders = ",".join("?" for _ in attachment_ids)
        cursor = await self._conn.execute(
            f"SELECT * FROM attachments WHERE id IN ({placeholders})",
            attachment_ids,
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_attachments_by_session(self, session_id: str) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM attachments WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
```

Also extend `delete_session` to include `"attachments"` in the delete cascade. Add it to the tuple of tables (before `"sessions"`):

```python
    async def delete_session(self, session_id: str) -> None:
        for table in (
            "attachments", "tool_results", "turn_analyses", "traces", "episode_links", "episodes",
            "session_summaries", "active_strategies", "outcomes", "goals", "messages",
            "family_profiles", "profile_changelog", "sessions",
        ):
            await self._conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        await self._conn.commit()
```

- [ ] **Step 5: Run tests**

Note: `app/agent/session_store.py` is just a factory that creates a `SQLiteSessionStore` over an in-memory database. No stubs needed -- the methods added to `SQLiteSessionStore` are automatically available to both code paths.

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_sqlite_attachments.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/agent/store_protocol.py app/agent/sqlite_store.py tests/test_sqlite_attachments.py
git commit -m "Add attachment store methods and extend delete_session cascade"
```

---

### Task 6: Upload API Endpoint

**Files:**
- Modify: `app/api/routes.py` (add upload + thumbnail endpoints)
- Modify: `app/api/deps.py` (add input_gate dependency)
- Test: `tests/test_multimodal_api.py` (new)

- [ ] **Step 1: Write failing tests for upload endpoint**

Create `tests/test_multimodal_api.py`:

```python
import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

def _make_jpeg_bytes():
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()

@pytest.fixture
def mock_app():
    """Create a minimal app with mocked dependencies for upload testing."""
    from fastapi import FastAPI
    from app.api.routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")

    # Mock app state
    mock_store = AsyncMock()
    mock_store.session_exists = AsyncMock(return_value=True)
    mock_store.save_attachment = AsyncMock()
    mock_store.commit = AsyncMock()

    mock_orchestrator = MagicMock()
    mock_orchestrator.get_session_store = MagicMock(return_value=mock_store)

    mock_input_gate = AsyncMock()
    mock_input_gate.check = AsyncMock(return_value=MagicMock(is_allowed=True))

    app.state.orchestrator = mock_orchestrator
    app.state.session_store = mock_store
    app.state.input_gate = mock_input_gate
    app.state.limiter = MagicMock()

    return app, mock_store, mock_input_gate

def test_upload_rejects_oversized_file(mock_app):
    app, _, _ = mock_app
    client = TestClient(app)
    big_data = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)  # 11MB
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.UPLOAD_MAX_SIZE_BYTES = 10 * 1024 * 1024
        mock_settings.UPLOAD_ALLOWED_TYPES = {"image/jpeg"}
        mock_settings.UPLOAD_RATE_LIMIT = "100/minute"
        mock_settings.RATE_LIMIT_ENABLED = False
        response = client.post(
            "/api/upload",
            files={"file": ("big.jpg", io.BytesIO(big_data), "image/jpeg")},
            data={"session_id": "test"},
        )
    assert response.status_code == 413

def test_upload_rejects_unsupported_type(mock_app):
    app, _, _ = mock_app
    client = TestClient(app)
    with patch("app.api.routes.settings") as mock_settings:
        mock_settings.UPLOAD_ALLOWED_TYPES = {"image/jpeg"}
        mock_settings.UPLOAD_MAX_SIZE_BYTES = 10 * 1024 * 1024
        mock_settings.RATE_LIMIT_ENABLED = False
        response = client.post(
            "/api/upload",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"session_id": "test"},
        )
    assert response.status_code == 415
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_api.py -v`
Expected: FAIL — upload endpoint doesn't exist

- [ ] **Step 3: Add upload endpoint to routes.py**

Add imports and the upload endpoint to `app/api/routes.py`:

```python
import os
import re
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse

from app.models.schemas import (
    Attachment,
    ChatRequest,
    MessagesResponse,
    OutcomesResponse,
    SeedSessionRequest,
    SessionResponse,
    SessionsResponse,
    UploadBlockedResponse,
    UploadResponse,
)
from app.services.file_upload import (
    delete_gemini_file,
    extract_text_from_file,
    generate_attachment_id,
    generate_thumbnail,
    upload_to_gemini,
    validate_magic_bytes,
)
```

Then add the endpoint:

```python
@router.post("/upload")
@limiter.limit(lambda: settings.UPLOAD_RATE_LIMIT)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Upload a file for attachment to a chat message.

    Validates type + size, uploads to Gemini File API, extracts text for
    input gate scanning, generates thumbnail (images only), stores metadata.
    """
    # Validate session_id format (prevent path traversal in thumbnail dir)
    if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    # Validate content type
    if file.content_type not in settings.UPLOAD_ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    # Read file data
    data = await file.read()

    # Validate size
    if len(data) > settings.UPLOAD_MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # Validate magic bytes
    if not validate_magic_bytes(data, file.content_type):
        raise HTTPException(status_code=415, detail="File content does not match declared type")

    # Upload to Gemini File API
    try:
        gemini_file_name, gemini_file_uri = await upload_to_gemini(
            data, file.filename or "upload", file.content_type
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini upload failed: {e}")

    # Extract text and scan through input gate (fail-closed)
    input_gate = getattr(request.app.state, "input_gate", None)
    if input_gate:
        try:
            extracted_text = await extract_text_from_file(gemini_file_uri, file.content_type)
            if extracted_text.strip():
                check = await input_gate.check(extracted_text)
                if not check.is_allowed:
                    await delete_gemini_file(gemini_file_name)
                    return UploadBlockedResponse(
                        reason=check.blocked_reason or "content",
                        response=check.override_response or "",
                    )
        except Exception as e:
            # Fail closed: reject upload if extraction fails
            await delete_gemini_file(gemini_file_name)
            raise HTTPException(status_code=502, detail=f"File content scanning failed: {e}")

    # Generate thumbnail (images only)
    attachment_id = generate_attachment_id()
    thumbnail_path = generate_thumbnail(data, file.content_type, session_id, attachment_id)

    # Store metadata
    store = orchestrator.get_session_store()
    await store.save_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        gemini_file_name=gemini_file_name,
        gemini_file_uri=gemini_file_uri,
        filename=file.filename or "upload",
        content_type=file.content_type,
        size_bytes=len(data),
        thumbnail_path=thumbnail_path,
    )
    await store.commit()

    thumbnail_url = f"/api/attachments/{attachment_id}/thumbnail" if thumbnail_path else None

    return UploadResponse(
        id=attachment_id,
        filename=file.filename or "upload",
        content_type=file.content_type,
        thumbnail_url=thumbnail_url,
    )


@router.get("/attachments/{attachment_id}/thumbnail")
async def get_thumbnail(
    attachment_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Serve a thumbnail image for an attachment."""
    store = orchestrator.get_session_store()
    attachments = await store.get_attachments([attachment_id])
    if not attachments:
        raise HTTPException(status_code=404, detail="Attachment not found")

    thumb_path = attachments[0].get("thumbnail_path")
    if not thumb_path or not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path, media_type="image/webp")
```

Also extend the `delete_session` endpoint to clean up thumbnails and Gemini files:

```python
@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
):
    """Delete all data for a session (right-to-erasure)."""
    store = orchestrator.get_session_store()
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Clean up attachments: Gemini files + local thumbnails
    attachments = await store.get_attachments_by_session(session_id)
    for att in attachments:
        # Best-effort Gemini file deletion
        if att.get("gemini_file_name"):
            await delete_gemini_file(att["gemini_file_name"])
        # Delete local thumbnail
        if att.get("thumbnail_path") and os.path.exists(att["thumbnail_path"]):
            os.remove(att["thumbnail_path"])

    # Delete thumbnail directory if empty
    import shutil
    thumb_dir = os.path.join(settings.UPLOAD_THUMBNAIL_DIR, session_id)
    if os.path.isdir(thumb_dir):
        shutil.rmtree(thumb_dir, ignore_errors=True)

    await store.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
```

- [ ] **Step 4: Store input_gate on app.state in main.py**

In `app/main.py`, after creating the `input_gate` (around line 91), add:

```python
    app.state.input_gate = input_gate
```

- [ ] **Step 5: Run tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py app/main.py tests/test_multimodal_api.py
git commit -m "Add upload and thumbnail API endpoints with input gate scanning"
```

---

### Task 7: Input Gate Multipart Handling (graph.py)

**Files:**
- Modify: `app/agent/graph.py:78-94` (input_gate_node)
- Test: `tests/test_multimodal_agent.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/test_multimodal_agent.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
from app.agent.orchestrator import _extract_text

def test_extract_text_from_string():
    assert _extract_text("hello") == "hello"

def test_extract_text_from_multipart_list():
    content = [
        {"type": "text", "text": "Look at this"},
        {"type": "media", "file_uri": "uri://x", "mime_type": "image/jpeg"},
    ]
    assert _extract_text(content) == "Look at this"

def test_extract_text_from_list_no_text():
    content = [
        {"type": "media", "file_uri": "uri://x", "mime_type": "image/jpeg"},
    ]
    assert _extract_text(content) == ""

def test_extract_text_from_multipart_multiple_text():
    content = [
        {"type": "text", "text": "First "},
        {"type": "media", "file_uri": "uri://x", "mime_type": "image/jpeg"},
        {"type": "text", "text": "second"},
    ]
    assert _extract_text(content) == "First second"
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_agent.py -v`
Expected: PASS (existing `_extract_text` already handles lists correctly by joining text from dict parts)

- [ ] **Step 3: Update input_gate_node in graph.py**

In `app/agent/graph.py`, modify the `input_gate_node` function at line 94. Change:

```python
        check = await input_gate.check(latest_human.content)
```

To:

```python
        from app.agent.orchestrator import _extract_text
        user_text = _extract_text(latest_human.content)
        check = await input_gate.check(user_text)
```

This ensures that when `latest_human.content` is a multipart list (text + media parts), only the text portion is sent to the input gate classifier.

- [ ] **Step 4: Run existing tests to verify no regressions**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_guardrails.py tests/test_multimodal_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/test_multimodal_agent.py
git commit -m "Extract text from multipart content before input gate check"
```

---

### Task 8: Hooks Multipart Content Handling

**Files:**
- Modify: `app/agent/hooks.py:204-211` (state block injection)

- [ ] **Step 1: Write failing test for multipart state injection**

Add to `tests/test_multimodal_agent.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

def test_state_block_injection_with_list_content():
    """Verify state block is prepended to the text part, not str(list)."""
    from app.agent.hooks import _estimate_chars

    # Simulate multipart content
    content = [
        {"type": "text", "text": "Look at this report card"},
        {"type": "media", "file_uri": "uri://file", "mime_type": "application/pdf"},
    ]
    msg = HumanMessage(content=content)
    # _estimate_chars should count only the text parts
    chars = _estimate_chars([msg])
    assert chars == len("Look at this report card")
```

- [ ] **Step 2: Run test**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_agent.py::test_state_block_injection_with_list_content -v`
Expected: PASS (_estimate_chars already handles list content)

- [ ] **Step 3: Update state block injection in hooks.py**

In `app/agent/hooks.py`, replace lines 204-211:

```python
        if conversation_messages:
            last_msg = conversation_messages[-1]
            if isinstance(last_msg, HumanMessage):
                state_suffix = "\n\n" + state_block + budget_warning
                original_content = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                conversation_messages = conversation_messages[:-1] + [
                    HumanMessage(content=original_content + state_suffix)
                ]
```

With:

```python
        if conversation_messages:
            last_msg = conversation_messages[-1]
            if isinstance(last_msg, HumanMessage):
                state_suffix = "\n\n" + state_block + budget_warning
                if isinstance(last_msg.content, str):
                    new_content = last_msg.content + state_suffix
                elif isinstance(last_msg.content, list):
                    # Multipart content: prepend state block to first text part
                    new_content = list(last_msg.content)  # shallow copy
                    for i, part in enumerate(new_content):
                        if isinstance(part, dict) and part.get("type") == "text":
                            new_content[i] = {**part, "text": part["text"] + state_suffix}
                            break
                    else:
                        # No text part found — prepend one
                        new_content.insert(0, {"type": "text", "text": state_suffix})
                else:
                    new_content = str(last_msg.content) + state_suffix
                conversation_messages = conversation_messages[:-1] + [
                    HumanMessage(content=new_content)
                ]
```

- [ ] **Step 4: Run all hooks tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_hooks.py tests/test_multimodal_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/hooks.py
git commit -m "Handle multipart content in state block injection"
```

---

### Task 9: Orchestrator Attachment Resolution

**Files:**
- Modify: `app/agent/orchestrator.py:125-178` (process method), `392-444` (process_stream)

The orchestrator must resolve `attachment_ids` to Gemini file URIs and build multipart `HumanMessage` content when attachments are present.

- [ ] **Step 1: Write failing test**

Add to `tests/test_multimodal_agent.py`:

```python
def test_build_multipart_content():
    """Test building multipart HumanMessage content from attachments."""
    from app.agent.orchestrator import _build_multipart_content

    attachments = [
        {"gemini_file_uri": "uri://file1", "content_type": "image/jpeg"},
        {"gemini_file_uri": "uri://file2", "content_type": "application/pdf"},
    ]
    content = _build_multipart_content("Look at these files", attachments)
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Look at these files"}
    assert content[1] == {"type": "media", "file_uri": "uri://file1", "mime_type": "image/jpeg"}
    assert content[2] == {"type": "media", "file_uri": "uri://file2", "mime_type": "application/pdf"}

def test_build_multipart_content_no_attachments():
    """Without attachments, returns plain string."""
    from app.agent.orchestrator import _build_multipart_content
    content = _build_multipart_content("Hello", [])
    assert content == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_agent.py::test_build_multipart_content -v`
Expected: FAIL — function doesn't exist

- [ ] **Step 3: Add _build_multipart_content helper to orchestrator.py**

Add after `_extract_text` function (line 35):

```python
def _build_multipart_content(message: str, attachments: list[dict]):
    """Build HumanMessage content: plain string or multipart list with media parts.

    Args:
        message: The user's text message.
        attachments: List of attachment dicts with gemini_file_uri and content_type.

    Returns:
        str if no attachments, or list of content parts if attachments present.
    """
    if not attachments:
        return message
    parts = [{"type": "text", "text": message}]
    for att in attachments:
        parts.append({
            "type": "media",
            "file_uri": att["gemini_file_uri"],
            "mime_type": att["content_type"],
        })
    return parts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_multimodal_agent.py -v`
Expected: PASS

- [ ] **Step 5: Wire attachment resolution into process() and process_stream()**

In `process()` method signature, add `attachment_ids` parameter:

```python
    async def process(self, message: str, session_id: str, attachment_ids: list[str] | None = None) -> StreamDonePayload:
```

After building `history_messages` and before creating `input_data`, resolve attachments:

```python
        # Resolve attachments to multipart content
        attachments = []
        if attachment_ids:
            attachments = await self._session_store.get_attachments(attachment_ids)

        message_content = _build_multipart_content(message, attachments)
```

Then change the `HumanMessage` construction:

```python
        # Before: HumanMessage(content=message)
        # After:
        HumanMessage(content=message_content)
```

Apply the same pattern to `process_stream()`. Update its signature:

```python
    async def process_stream(self, message: str, session_id: str, attachment_ids: list[str] | None = None):
```

And add the same attachment resolution logic before `input_data` construction.

- [ ] **Step 6: Update routes.py to pass attachment_ids**

In `app/api/routes.py`, modify the `chat_stream` endpoint to pass `attachment_ids`:

```python
        async for event_type, data in orchestrator.process_stream(
            message=body.message,
            session_id=body.session_id,
            attachment_ids=body.attachment_ids,
        ):
```

- [ ] **Step 7: Run existing orchestrator tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_orchestrator.py tests/test_multimodal_agent.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/agent/orchestrator.py app/api/routes.py tests/test_multimodal_agent.py
git commit -m "Resolve attachment IDs to multipart HumanMessage content in orchestrator"
```

---

### Task 10: Frontend Types + API Client

**Files:**
- Modify: `frontend-react/src/types/index.ts`
- Modify: `frontend-react/src/lib/api.ts`

- [ ] **Step 1: Add Attachment type and extend ChatRequest/ChatMessage**

In `frontend-react/src/types/index.ts`, add after the `ChatRequest` interface (line 79):

```typescript
export interface Attachment {
  id: string
  filename: string
  content_type: string
  thumbnail_url: string | null
}

export interface UploadResponse {
  id: string
  filename: string
  content_type: string
  thumbnail_url: string | null
}

export interface UploadBlockedResponse {
  blocked: true
  reason: string
  response: string
}
```

Extend `ChatRequest`:

```typescript
export interface ChatRequest {
  message: string
  session_id: string
  attachment_ids?: string[]
}
```

Extend `ChatMessage`:

```typescript
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  agentUsed?: string
  pipelineTrace?: PipelineTrace
  summary?: string
  attachments?: Attachment[]
}
```

- [ ] **Step 2: Add uploadFile method to api.ts**

In `frontend-react/src/lib/api.ts`, add the upload types to imports and add the method:

```typescript
import type {
  // ...existing imports
  UploadResponse,
  UploadBlockedResponse,
} from "@/types"
```

Add to the `api` object:

```typescript
  async uploadFile(
    file: File,
    sessionId: string,
  ): Promise<UploadResponse | UploadBlockedResponse> {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("session_id", sessionId)

    const res = await fetch(`${SSE_BASE}/upload`, {
      method: "POST",
      body: formData,
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(error.detail || `Upload failed: ${res.status}`)
    }
    return res.json()
  },

  async deleteSession(sessionId: string): Promise<{ status: string }> {
    return request(`/session/${sessionId}`, { method: "DELETE" })
  },
```

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/types/index.ts frontend-react/src/lib/api.ts
git commit -m "Add Attachment types and uploadFile API method"
```

---

### Task 11: ChatInput Drag-Drop + Attach UI

**Files:**
- Modify: `frontend-react/src/components/chat/ChatInput.tsx`

- [ ] **Step 1: Update ChatInput props**

Change the Props interface:

```typescript
interface Props {
  onSend: (message: string, attachments?: Attachment[]) => void
  onStop: () => void
  isLoading: boolean
  isStreaming: boolean
  sessionId: string
}
```

- [ ] **Step 2: Implement full ChatInput with drag-drop and file attachment**

Replace the entire `ChatInput.tsx` with the upload-capable version:

```tsx
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
```

- [ ] **Step 3: Update parent component to pass sessionId and handle attachments**

Find the parent component that renders `ChatInput` and update its `onSend` handler to pass `attachment_ids` to `api.chatStream()`. The parent should:

1. Accept `(message: string, attachments?: Attachment[])` in the `onSend` callback
2. Include `attachment_ids: attachments?.map(a => a.id)` in the `ChatRequest` passed to `chatStream()`
3. Store attachments on the user's `ChatMessage` for display

Use grep to find: `<ChatInput` to locate the parent component.

- [ ] **Step 4: Commit**

```bash
git add frontend-react/src/components/chat/ChatInput.tsx
git commit -m "Add drag-drop file upload and attachment chips to ChatInput"
```

---

### Task 12: ChatBubble Attachment Rendering

**Files:**
- Modify: `frontend-react/src/components/chat/ChatBubble.tsx`

- [ ] **Step 1: Add attachment rendering to ChatBubble**

In `ChatBubble.tsx`, add attachment rendering inside the user message bubble, before the text content. Add between the outer div and the text div for user messages:

```tsx
import { motion } from "framer-motion"
import { Sprout, User, FileText } from "lucide-react"
import { useTypewriter } from "@/hooks/use-typewriter"
import type { ChatMessage } from "@/types"
```

Inside the `ChatBubble` component, before the text `<div>`, add:

```tsx
        {/* Attachment thumbnails */}
        {message.attachments && message.attachments.length > 0 && (
          <div className={`flex flex-wrap gap-2 ${isUser ? "mb-2" : "mb-2"}`}>
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
```

- [ ] **Step 2: Commit**

```bash
git add frontend-react/src/components/chat/ChatBubble.tsx
git commit -m "Render attachment thumbnails on user messages in ChatBubble"
```

---

### Task 13: Wire Parent Component + Integration Verification

**Files:**
- Depends on which component renders `ChatInput` (find via grep)

- [ ] **Step 1: Find and update the parent chat component**

Search for `<ChatInput` usage. Update:
1. Pass `sessionId` prop to `ChatInput`
2. Update `onSend` handler to accept optional `attachments` parameter
3. Include `attachment_ids` in the `ChatRequest` sent to `api.chatStream()`
4. Store `attachments` on the user `ChatMessage` object for `ChatBubble` to render

- [ ] **Step 2: Verify frontend build**

Run: `cd frontend-react && npm run build`
Expected: Build succeeds with no type errors

- [ ] **Step 3: Commit**

```bash
git add frontend-react/src/
git commit -m "Wire attachment flow through parent chat component"
```

---

### Task 14: Run All Tests + Final Verification

- [ ] **Step 1: Run all backend tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/ -v -m "not integration"`
Expected: All PASS

- [ ] **Step 2: Run frontend type check + build**

Run: `cd frontend-react && npx tsc --noEmit && npm run build`
Expected: No errors

- [ ] **Step 3: Final commit with any remaining fixes**

Fix any test failures or type errors, then commit.

"""File upload service: validation, Gemini upload, scanning, thumbnails."""

import asyncio
import io
import logging
import uuid
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

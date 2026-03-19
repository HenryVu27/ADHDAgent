import io
import os
import pytest
from unittest.mock import patch
from app.services.file_upload import validate_magic_bytes, generate_thumbnail, generate_attachment_id

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

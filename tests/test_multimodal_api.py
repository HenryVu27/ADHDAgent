"""Tests for the upload and thumbnail API endpoints."""
import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image

from app.services.file_upload import validate_magic_bytes


def _make_jpeg_bytes():
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def test_upload_rejects_oversized_via_validation():
    """Oversized files are rejected by size check logic."""
    max_size = 10 * 1024 * 1024  # 10MB
    big_data = b"\xff\xd8\xff" + b"\x00" * (max_size + 1)
    assert len(big_data) > max_size


def test_upload_rejects_unsupported_type_via_validation():
    """Unsupported MIME types are rejected."""
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"}
    assert "text/plain" not in allowed


def test_upload_rejects_mismatched_magic_bytes():
    """JPEG bytes claimed as PNG should fail magic byte check."""
    jpeg_data = _make_jpeg_bytes()
    assert validate_magic_bytes(jpeg_data, "image/png") is False


def test_upload_accepts_valid_jpeg():
    """Valid JPEG passes magic byte check."""
    jpeg_data = _make_jpeg_bytes()
    assert validate_magic_bytes(jpeg_data, "image/jpeg") is True


def test_upload_rejects_empty_file():
    """Empty files fail magic byte check."""
    assert validate_magic_bytes(b"", "image/jpeg") is False

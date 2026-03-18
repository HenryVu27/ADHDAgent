"""Tests for multipart content handling across agent components."""
import pytest
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


def test_estimate_chars_with_list_content():
    """_estimate_chars should count only text parts from multipart content."""
    from app.agent.hooks import _estimate_chars

    content = [
        {"type": "text", "text": "Look at this report card"},
        {"type": "media", "file_uri": "uri://file", "mime_type": "application/pdf"},
    ]
    msg = HumanMessage(content=content)
    chars = _estimate_chars([msg])
    assert chars == len("Look at this report card")

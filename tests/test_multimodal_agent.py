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

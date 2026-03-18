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
        ChatRequest(message="Hello", session_id="test", attachment_ids=["a", "b", "c", "d"])

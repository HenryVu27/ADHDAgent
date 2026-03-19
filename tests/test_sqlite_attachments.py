import sqlite3
import pytest
import pytest_asyncio
import aiosqlite
from app.agent.sqlite_store import SQLiteSessionStore
from app.db import init_db_async

@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db_async(conn)
    s = SQLiteSessionStore(conn)
    yield s
    await conn.close()

@pytest.mark.asyncio
async def test_save_and_get_attachment(store):
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
async def test_get_attachments_by_session(store):
    await store.get("test_session")
    await store.save_attachment(
        session_id="test_session",
        attachment_id="att_1",
        gemini_file_name="files/1",
        gemini_file_uri="uri://1",
        filename="img1.jpg",
        content_type="image/jpeg",
        size_bytes=500,
    )
    await store.save_attachment(
        session_id="test_session",
        attachment_id="att_2",
        gemini_file_name="files/2",
        gemini_file_uri="uri://2",
        filename="img2.png",
        content_type="image/png",
        size_bytes=600,
    )
    await store.commit()

    result = await store.get_attachments_by_session("test_session")
    assert len(result) == 2

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

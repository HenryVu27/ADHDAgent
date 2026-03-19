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

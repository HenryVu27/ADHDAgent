"""In-memory session store backed by SQLiteSessionStore(":memory:").

The old 350-line InMemorySessionStore class is replaced by a factory function
that creates a SQLiteSessionStore over an in-memory aiosqlite database.
This eliminates implementation drift between the two store backends.
"""

import sqlite3

import aiosqlite

from app.agent.sqlite_store import SQLiteSessionStore
from app.db import init_db_async


async def create_in_memory_store() -> SQLiteSessionStore:
    """Create a SQLiteSessionStore backed by an in-memory database."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db_async(conn)
    return SQLiteSessionStore(conn)


# Backward-compatibility alias
InMemorySessionStore = SQLiteSessionStore

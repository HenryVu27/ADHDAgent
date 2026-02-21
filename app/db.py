"""SQLite connection management, schema initialization, and migrations."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    turn_count INTEGER NOT NULL DEFAULT 0,
    phase TEXT NOT NULL DEFAULT 'intake',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS family_profiles (
    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
    child_name TEXT,
    child_age TEXT,
    diagnosis_status TEXT,
    challenge_areas TEXT NOT NULL DEFAULT '[]',
    attempted_strategies TEXT NOT NULL DEFAULT '[]',
    good_day_description TEXT,
    hardest_situations TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    blocked INTEGER NOT NULL DEFAULT 0,
    blocked_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session_turn ON messages(session_id, turn);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    description TEXT NOT NULL,
    strategy_id TEXT,
    created_turn INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id);

CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    goal_description TEXT NOT NULL,
    signal TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    turn INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_outcomes_session ON outcomes(session_id);

CREATE TABLE IF NOT EXISTS active_strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    strategy_name TEXT NOT NULL,
    UNIQUE(session_id, strategy_name)
);
"""

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    summary TEXT NOT NULL,
    covers_through_turn INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries(session_id);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '',
    strategies_involved TEXT NOT NULL DEFAULT '[]',
    emotional_context TEXT NOT NULL DEFAULT '',
    turn_range_start INTEGER NOT NULL DEFAULT 0,
    turn_range_end INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
"""

SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn INTEGER NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);

CREATE TABLE IF NOT EXISTS turn_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    turn INTEGER NOT NULL,
    analysis_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_turn_analyses_session ON turn_analyses(session_id);
"""

SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS observability_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    category TEXT NOT NULL,
    event_type TEXT NOT NULL,
    turn INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    detail_json TEXT NOT NULL DEFAULT '{}',
    level TEXT NOT NULL DEFAULT 'info',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_obs_events_session ON observability_events(session_id);
CREATE INDEX IF NOT EXISTS idx_obs_events_category ON observability_events(session_id, category);
"""

MIGRATIONS = {
    1: SCHEMA_V1,
    2: SCHEMA_V2,
    3: SCHEMA_V3,
    4: SCHEMA_V4,
}


def get_connection(db_path: str) -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode and foreign keys."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version, 0 if no schema_version table."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] or 0
    except sqlite3.OperationalError:
        return 0


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize or migrate the database schema."""
    current = _get_schema_version(conn)
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            logger.info("Applying migration v%d", version)
            conn.executescript(MIGRATIONS[version])
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (version,),
            )
            conn.commit()
    logger.info("Database schema at v%d", max(MIGRATIONS.keys()))


def run_migrations(conn: sqlite3.Connection) -> None:
    """Alias for init_db — applies any pending migrations."""
    init_db(conn)

"""Tiny SQLite store for user feedback. Uses parameterised queries (no SQL injection)."""
import sqlite3
import time
from pathlib import Path

_SCHEMA = """CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    reason TEXT,
    created_at REAL NOT NULL
)"""


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def add_feedback(path: str, request_id: str, rating: str, reason: str | None) -> None:
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO feedback (request_id, rating, reason, created_at) VALUES (?, ?, ?, ?)",
                (request_id, rating, reason, time.time()),
            )
    finally:
        conn.close()


def count_feedback(path: str) -> int:
    conn = _connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    finally:
        conn.close()

"""Append-only SQLite event records; one short-lived connection per operation."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from ppe.schemas import ViolationEvent


class EventStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, observed_at_utc TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS events_source_time ON events(source_id, observed_at_utc)")

    def add(self, event: ViolationEvent):
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
                               (event.event_id, event.source_id, event.observed_at_utc, event.model_dump_json()))

    def list(self, limit=100, offset=0, source_id=None):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid pagination")
        with closing(sqlite3.connect(self.path, timeout=5)) as connection:
            if source_id is None:
                rows = connection.execute("SELECT payload FROM events ORDER BY observed_at_utc DESC, event_id DESC LIMIT ? OFFSET ?", (limit, offset))
            else:
                rows = connection.execute("SELECT payload FROM events WHERE source_id=? ORDER BY observed_at_utc DESC, event_id DESC LIMIT ? OFFSET ?", (source_id, limit, offset))
            return [json.loads(row[0]) for row in rows]

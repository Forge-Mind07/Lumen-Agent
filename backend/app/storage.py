from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import (
    ArtifactRecord,
    DatasetProfile,
    DatasetRecord,
    MessageRecord,
    SessionRecord,
    new_id,
    utc_now,
)

CHAT_DATASET_ID = "__research_chat__"


class Storage:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    original_path TEXT NOT NULL,
                    duckdb_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    row_count INTEGER,
                    column_count INTEGER,
                    profile_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL REFERENCES datasets(id),
                    title TEXT NOT NULL,
                    working_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session_idx
                    ON messages(session_id, created_at);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_session_idx
                    ON artifacts(session_id, created_at);
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            if "mode" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'sql'")
            for table in ("ml_datasets", "ml_projects", "ml_runs", "ml_models"):
                db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, "
                           "session_id TEXT NOT NULL REFERENCES sessions(id), "
                           "payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
                db.execute(f"CREATE INDEX IF NOT EXISTS {table}_session ON {table}(session_id)")
            empty_profile = DatasetProfile(row_count=0, columns=[], sample_rows=[]).model_dump_json()
            db.execute(
                """INSERT OR IGNORE INTO datasets
                (id, filename, file_type, size_bytes, original_path, duckdb_path, status,
                 row_count, column_count, profile_json, error, created_at)
                VALUES (?, 'Research Chat', 'system', 0, '', '', 'ready', 0, 0, ?, NULL, ?)""",
                (CHAT_DATASET_ID, empty_profile, utc_now()),
            )

    def create_dataset(
        self,
        *,
        dataset_id: str,
        filename: str,
        file_type: str,
        size_bytes: int,
        original_path: str,
        duckdb_path: str,
    ) -> DatasetRecord:
        created_at = utc_now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO datasets
                (id, filename, file_type, size_bytes, original_path, duckdb_path, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'processing', ?)""",
                (
                    dataset_id,
                    filename,
                    file_type,
                    size_bytes,
                    original_path,
                    duckdb_path,
                    created_at,
                ),
            )
        return self.get_dataset(dataset_id)

    def finish_dataset(self, dataset_id: str, profile: DatasetProfile) -> DatasetRecord:
        with self.connect() as db:
            db.execute(
                """UPDATE datasets SET status = 'ready', row_count = ?, column_count = ?,
                profile_json = ?, error = NULL WHERE id = ?""",
                (
                    profile.row_count,
                    len(profile.columns),
                    profile.model_dump_json(),
                    dataset_id,
                ),
            )
        return self.get_dataset(dataset_id)

    def fail_dataset(self, dataset_id: str, error: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE datasets SET status = 'error', error = ? WHERE id = ?",
                (error[:2000], dataset_id),
            )

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self.connect() as db:
            row = db.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if not row:
            raise KeyError(f"Dataset {dataset_id} was not found")
        return self._dataset_from_row(row)

    def list_datasets(self) -> list[DatasetRecord]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM datasets WHERE id != ? ORDER BY created_at DESC", (CHAT_DATASET_ID,)).fetchall()
        return [self._dataset_from_row(row) for row in rows]

    def dataset_session_ids(self, dataset_id: str) -> list[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM sessions WHERE dataset_id = ?", (dataset_id,)).fetchall()]

    def delete_dataset(self, dataset_id: str) -> tuple[DatasetRecord, list[str]]:
        if dataset_id == CHAT_DATASET_ID:
            raise ValueError("The Research Chat workspace cannot be deleted")
        record = self.get_dataset(dataset_id)
        cleanup_paths = [record.original_path, record.duckdb_path]
        with self.connect() as db:
            session_ids = [row[0] for row in db.execute("SELECT id FROM sessions WHERE dataset_id = ?", (dataset_id,)).fetchall()]
            for session_id in session_ids:
                for table in ("ml_datasets", "ml_projects", "ml_runs", "ml_models"):
                    rows = db.execute(f"SELECT payload_json FROM {table} WHERE session_id = ?", (session_id,)).fetchall()
                    for row in rows:
                        payload = json.loads(row[0])
                        for key in ("path", "duckdb_path", "artifact_path"):
                            value = payload.get(key)
                            if isinstance(value, str) and value:
                                cleanup_paths.append(value)
                    db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                db.execute("DELETE FROM artifacts WHERE session_id = ?", (session_id,))
                db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM sessions WHERE dataset_id = ?", (dataset_id,))
            db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        return record, cleanup_paths

    def create_session(self, dataset_id: str, title: str, mode: str = "sql") -> SessionRecord:
        session_id = new_id("ses")
        now = utc_now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO sessions (id, dataset_id, title, working_summary, created_at, updated_at, mode) VALUES (?, ?, ?, '', ?, ?, ?)",
                (session_id, dataset_id, title[:100], now, now, mode),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> SessionRecord:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise KeyError(f"Session {session_id} was not found")
        return SessionRecord(**dict(row))

    def list_sessions(self, limit: int = 30) -> list[SessionRecord]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [SessionRecord(**dict(row)) for row in rows]

    def add_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        message = MessageRecord(
            id=new_id("msg"),
            session_id=session_id,
            role=role,
            content=content,
            created_at=utc_now(),
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (message.id, message.session_id, message.role, message.content, message.created_at),
            )
            db.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (message.created_at, session_id),
            )
        return message

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [MessageRecord(**dict(row)) for row in rows]

    def update_summary(self, session_id: str, summary: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE sessions SET working_summary = ?, updated_at = ? WHERE id = ?",
                (summary, utc_now(), session_id),
            )

    def add_artifact(
        self, session_id: str, kind: str, payload: dict[str, Any]
    ) -> ArtifactRecord:
        artifact = ArtifactRecord(
            id=new_id("res" if kind == "query_result" else "chart"),
            session_id=session_id,
            kind=kind,
            payload=payload,
            created_at=utc_now(),
        )
        with self.connect() as db:
            db.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                (
                    artifact.id,
                    artifact.session_id,
                    artifact.kind,
                    json.dumps(artifact.payload, default=str),
                    artifact.created_at,
                ),
            )
        return artifact

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self.connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if not row:
            raise KeyError(f"Artifact {artifact_id} was not found")
        return ArtifactRecord(
            id=row["id"],
            session_id=row["session_id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
        )

    def list_artifacts(self, session_id: str) -> list[ArtifactRecord]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [
            ArtifactRecord(
                id=row["id"],
                session_id=row["session_id"],
                kind=row["kind"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _dataset_from_row(row: sqlite3.Row) -> DatasetRecord:
        values = dict(row)
        profile_json = values.pop("profile_json")
        values["profile"] = DatasetProfile.model_validate_json(profile_json) if profile_json else None
        return DatasetRecord(**values)

    def put_ml(self, kind: str, record: dict[str, Any]) -> dict[str, Any]:
        table = self._ml_table(kind)
        with self.connect() as db:
            db.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                       "payload_json = excluded.payload_json, updated_at = excluded.updated_at",
                       (record['id'], record['session_id'], json.dumps(record, allow_nan=False, default=str), utc_now()))
        return record

    def recover_ml_runs(self) -> None:
        """Single-process startup recovery; never claim an interrupted fit finished."""
        with self.connect() as db:
            for table, statuses in (("ml_runs", {"queued", "running"}), ("ml_projects", {"training"})):
                for row in db.execute(f"SELECT id, payload_json FROM {table}").fetchall():
                    record = json.loads(row["payload_json"])
                    if record.get("status") in statuses:
                        record.update(status="interrupted", error="The API restarted during this operation. Run it again.")
                        db.execute(f"UPDATE {table} SET payload_json = ?, updated_at = ? WHERE id = ?",
                                   (json.dumps(record), utc_now(), row["id"]))

    def get_ml(self, kind: str, record_id: str, session_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(f"SELECT payload_json FROM {self._ml_table(kind)} WHERE id = ? AND session_id = ?",
                             (record_id, session_id)).fetchone()
        if row is None:
            raise ValueError(f"{kind.capitalize()} reference was not found in this conversation")
        return json.loads(row[0])

    def list_ml(self, kind: str, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(f"SELECT payload_json FROM {self._ml_table(kind)} WHERE session_id = ? ORDER BY updated_at DESC",
                              (session_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    @staticmethod
    def _ml_table(kind: str) -> str:
        if kind not in {"datasets", "projects", "runs", "models"}:
            raise ValueError("Unknown ML record type")
        return "ml_" + kind

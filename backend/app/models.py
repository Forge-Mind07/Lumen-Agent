from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ColumnProfile(BaseModel):
    name: str
    data_type: str
    kind: Literal["dimension", "measure", "date", "other"]
    null_percent: float | None = None
    approximate_distinct: int | None = None
    minimum: Any | None = None
    maximum: Any | None = None


class DatasetProfile(BaseModel):
    table_name: str = "data"
    row_count: int
    columns: list[ColumnProfile]
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    profile_is_partial: bool = False


class DatasetRecord(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    original_path: str
    duckdb_path: str
    status: Literal["processing", "ready", "error"]
    row_count: int | None = None
    column_count: int | None = None
    profile: DatasetProfile | None = None
    error: str | None = None
    created_at: str


class SessionRecord(BaseModel):
    id: str
    dataset_id: str
    title: str
    working_summary: str = ""
    created_at: str
    updated_at: str
    mode: Literal["sql", "ml", "chat"] = "sql"


class MessageRecord(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ArtifactRecord(BaseModel):
    id: str
    session_id: str
    kind: Literal["query_result", "chart", "ml_result"]
    payload: dict[str, Any]
    created_at: str


class ChatRequest(BaseModel):
    dataset_id: str | None = None
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    mode: Literal["sql", "ml", "chat"] = "sql"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class AssistantTurn(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str | None = None


class AgentEvent(BaseModel):
    version: int = 1
    id: str = Field(default_factory=lambda: new_id("evt"))
    type: Literal[
        "session_started",
        "analysis_started",
        "tool_started",
        "tool_finished",
        "query_result",
        "chart_created",
        "ml_progress",
        "ml_result",
        "message_delta",
        "analysis_complete",
        "error",
    ]
    timestamp: str = Field(default_factory=utc_now)
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)

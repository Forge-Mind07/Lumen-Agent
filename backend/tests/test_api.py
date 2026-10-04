import json
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.models import AssistantTurn, ToolCall


class ApiScriptedProvider:
    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: Literal["primary", "secondary"] = "primary",
    ) -> AssistantTurn:
        self.calls += 1
        if self.calls == 1:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="call_api",
                        name="run_sql",
                        arguments={
                            "query": "SELECT sum(revenue) AS total_revenue FROM data",
                            "purpose": "Calculating total revenue",
                        },
                    )
                ]
            )
        return AssistantTurn(content="Total revenue is 6,450. [source: result]")


class ChatProvider:
    def __init__(self):
        self.tools: list[dict[str, Any]] = []

    async def complete(self, messages, tools, role="primary") -> AssistantTurn:
        self.tools = tools
        return AssistantTurn(content="Hello from plain Qwen chat.")


@pytest.mark.asyncio
async def test_upload_to_streaming_answer_vertical_path(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    app = create_app(settings=settings, provider=ApiScriptedProvider())
    transport = httpx.ASGITransport(app=app)
    fixture = Path(__file__).parent / "fixtures" / "sales.csv"

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with fixture.open("rb") as source:
                upload = await client.post(
                    "/api/datasets",
                    files={"file": ("sales.csv", source, "text/csv")},
                )
            assert upload.status_code == 200
            dataset = upload.json()
            assert dataset["row_count"] == 6

            response = await client.post(
                "/api/chat/stream",
                json={"dataset_id": dataset["id"], "message": "What is total revenue?"},
            )

    assert response.status_code == 200
    envelopes = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    event_types = [item["type"] for item in envelopes]
    assert event_types[0] == "session_started"
    assert "query_result" in event_types
    assert event_types[-1] == "analysis_complete"


@pytest.mark.asyncio
async def test_research_chat_requires_no_dataset_and_has_only_research_tools(tmp_path: Path) -> None:
    provider = ChatProvider()
    app = create_app(settings=Settings(data_dir=tmp_path / "data"), provider=provider)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat/stream", json={"mode": "chat", "message": "Hello"})
            datasets = await client.get("/api/datasets")
            sessions = await client.get("/api/sessions")
    assert response.status_code == 200
    assert datasets.json() == []
    assert sessions.json()[0]["mode"] == "chat"
    assert {tool["function"]["name"] for tool in provider.tools} == {"web_search", "read_web_page"}


@pytest.mark.asyncio
async def test_delete_dataset_removes_registry_and_files(tmp_path: Path) -> None:
    app = create_app(settings=Settings(data_dir=tmp_path / "data"), provider=ApiScriptedProvider())
    fixture = Path(__file__).parent / "fixtures" / "sales.csv"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with fixture.open("rb") as source:
                uploaded = await client.post("/api/datasets", files={"file": ("sales.csv", source, "text/csv")})
            dataset = uploaded.json()
            directory = Path(dataset["original_path"]).parent
            response = await client.delete(f"/api/datasets/{dataset['id']}")
            missing = await client.get(f"/api/datasets/{dataset['id']}")
    assert response.status_code == 204
    assert missing.status_code == 404
    assert not directory.exists()

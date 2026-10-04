from pathlib import Path
from typing import Any, Literal

import pytest

from app.agent import run_analysis
from app.config import Settings
from app.datasets import register_dataset
from app.models import AssistantTurn, ToolCall
from app.storage import Storage


class ScriptedProvider:
    def __init__(self, turns: list[AssistantTurn]):
        self.turns = turns
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        role: Literal["primary", "secondary"] = "primary",
    ) -> AssistantTurn:
        self.requests.append(messages.copy())
        return self.turns.pop(0)


@pytest.mark.asyncio
async def test_agent_executes_sql_then_answers(settings: Settings, storage: Storage) -> None:
    fixture = Path(__file__).parent / "fixtures" / "sales.csv"
    with fixture.open("rb") as source:
        dataset = register_dataset(
            filename="sales.csv", source=source, storage=storage, settings=settings
        )
    session = storage.create_session(dataset.id, "Compare months")
    storage.add_message(session.id, "user", "How did August compare with July?")
    provider = ScriptedProvider(
        [
            AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="run_sql",
                        arguments={
                            "purpose": "Comparing monthly revenue",
                            "query": "SELECT date_trunc('month', order_date) AS month_start, "
                            "sum(revenue) AS revenue FROM data GROUP BY 1 ORDER BY 1",
                        },
                    )
                ]
            ),
            AssistantTurn(content="August revenue was lower. [source: res_test]"),
        ]
    )

    events = [
        item
        async for item in run_analysis(
            session_id=session.id,
            storage=storage,
            provider=provider,
            settings=settings,
        )
    ]

    event_types = [item.type for item in events]
    assert "query_result" in event_types
    assert event_types[-1] == "analysis_complete"
    assert len(storage.list_artifacts(session.id)) == 1
    tool_message = provider.requests[1][-1]
    assert tool_message["role"] == "tool"
    assert '"returned_row_count":2' in tool_message["content"]

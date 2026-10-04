import json

import httpx
import pytest

from app.config import Settings
from app.provider import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_provider_parses_standard_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    response_payload = {
        "model": "local-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_7",
                            "type": "function",
                            "function": {
                                "name": "run_sql",
                                "arguments": json.dumps(
                                    {"query": "SELECT count(*) FROM data", "purpose": "Counting rows"}
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    async def fake_post(self, url, **kwargs):  # type: ignore[no-untyped-def]
        return httpx.Response(200, json=response_payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    provider = OpenAICompatibleProvider(Settings())
    turn = await provider.complete([{"role": "user", "content": "Count"}], [])

    assert turn.model == "local-model"
    assert turn.tool_calls[0].name == "run_sql"
    assert turn.tool_calls[0].arguments["query"] == "SELECT count(*) FROM data"
    assert turn.usage["prompt_tokens"] == 100

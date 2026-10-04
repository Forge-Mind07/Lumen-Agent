from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.research import execute_research_tool


class FakeResponse:
    def __init__(self, body): self.body = body
    def raise_for_status(self): return None
    def json(self): return self.body


class FakeClient:
    def __init__(self, **_): self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *_): return None
    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/search"):
            return FakeResponse({"results": [{"title": "Official source", "url": "https://example.com/report", "content": "Measured result", "published_date": "2026-09-01"}]})
        return FakeResponse({"results": [{"title": "Official source", "raw_content": "Full source text"}]})


@pytest.mark.asyncio
async def test_research_tools_return_bounded_structured_sources(settings, monkeypatch) -> None:
    settings.tavily_api_key = SecretStr("test-key")
    monkeypatch.setattr("app.research.httpx.AsyncClient", FakeClient)
    result = await execute_research_tool("web_search", {"query": "latest result", "max_results": 3}, settings)
    assert result["source_count"] == 1
    assert result["sources"][0]["url"] == "https://example.com/report"
    page = await execute_research_tool("read_web_page", {"url": "https://example.com/report"}, settings)
    assert page["content"] == "Full source text"


@pytest.mark.asyncio
async def test_research_tools_fail_clearly_without_configuration(settings) -> None:
    settings.tavily_api_key = None
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        await execute_research_tool("web_search", {"query": "anything"}, settings)

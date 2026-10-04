from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Settings


RESEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current or source-sensitive information. Returns compact source metadata and snippets for citation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "maxLength": 500},
                    "topic": {"type": "string", "enum": ["general", "news"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                    "purpose": {"type": "string", "maxLength": 100},
                },
                "required": ["query", "purpose"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_web_page",
            "description": "Read the main text of one public http(s) page returned by web_search when its snippet is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 8, "maxLength": 2000},
                    "purpose": {"type": "string", "maxLength": 100},
                },
                "required": ["url", "purpose"],
                "additionalProperties": False,
            },
        },
    },
]


async def execute_research_tool(name: str, arguments: dict[str, Any], settings: Settings) -> dict[str, Any]:
    key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    if not key:
        raise ValueError("Web research is not configured. Add TAVILY_API_KEY to backend/.env and restart the API.")
    timeout = httpx.Timeout(settings.web_search_timeout_seconds)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            if len(query) < 2:
                raise ValueError("web_search requires a useful query")
            limit = min(max(int(arguments.get("max_results", settings.web_search_max_results)), 1), 8)
            response = await client.post(
                "https://api.tavily.com/search",
                headers=headers,
                json={
                    "query": query,
                    "topic": arguments.get("topic", "general"),
                    "search_depth": "advanced",
                    "max_results": limit,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
            response.raise_for_status()
            body = response.json()
            sources = [
                {
                    "title": str(item.get("title") or "Untitled")[:240],
                    "url": str(item.get("url") or "")[:2000],
                    "snippet": " ".join(str(item.get("content") or "").split())[:1600],
                    **({"published_at": item["published_date"]} if item.get("published_date") else {}),
                }
                for item in body.get("results", [])[:limit]
                if item.get("url")
            ]
            return {"query": query, "sources": sources, "source_count": len(sources)}
        if name == "read_web_page":
            url = str(arguments.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("read_web_page accepts only public http(s) URLs")
            response = await client.post(
                "https://api.tavily.com/extract",
                headers=headers,
                json={"urls": [url], "extract_depth": "basic", "include_images": False},
            )
            response.raise_for_status()
            body = response.json()
            results = body.get("results", [])
            if not results:
                raise ValueError("The page could not be extracted")
            result = results[0]
            content = " ".join(str(result.get("raw_content") or "").split())
            return {"url": url, "title": str(result.get("title") or parsed.netloc)[:240], "content": content[:14_000], "truncated": len(content) > 14_000}
    raise ValueError("Unknown research tool")

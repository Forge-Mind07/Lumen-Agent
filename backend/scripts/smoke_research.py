from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from app.config import Settings
from app.research import execute_research_tool


async def main() -> None:
    result = await execute_research_tool(
        "web_search",
        {"query": "Apache ECharts official features", "max_results": 3, "topic": "general"},
        Settings(),
    )
    print(f"sources={result['source_count']}")
    for source in result["sources"]:
        print(f"- {source['title']} ({urlparse(source['url']).netloc})")


if __name__ == "__main__":
    asyncio.run(main())

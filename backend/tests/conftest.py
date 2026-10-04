from pathlib import Path

import pytest

from app.config import Settings
from app.storage import Storage


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        query_timeout_seconds=5,
        query_row_limit=25,
        max_tool_calls=5,
        max_analysis_iterations=4,
    )


@pytest.fixture
def storage(settings: Settings) -> Storage:
    instance = Storage(settings.sqlite_path)
    instance.initialize()
    return instance

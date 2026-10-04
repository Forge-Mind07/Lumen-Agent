from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class ModelEndpoint(BaseModel):
    base_url: str
    model: str
    api_key: str = "local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    data_dir: Path = Path("../data")
    primary_model_base_url: str = "http://127.0.0.1:8001/v1"
    primary_model_name: str = "qwen-local"
    primary_model_api_key: str = "local"
    secondary_model_base_url: str | None = None
    secondary_model_name: str | None = None
    secondary_model_api_key: str = "local"
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=256)
    context_limit_tokens: int = Field(default=65_536, ge=4096)
    model_timeout_seconds: float = Field(default=180, gt=0)
    stream_idle_timeout_seconds: float = Field(default=120, gt=0)
    query_timeout_seconds: float = Field(default=30, gt=0)
    query_row_limit: int = Field(default=200, ge=1, le=5000)
    max_tool_calls: int = Field(default=10, ge=1, le=30)
    max_analysis_iterations: int = Field(default=8, ge=1, le=20)
    recent_message_limit: int = Field(default=8, ge=2, le=30)
    compaction_threshold_chars: int = Field(default=24_000, ge=2000)
    max_summary_chars: int = Field(default=6000, ge=500)
    max_upload_bytes: int = Field(default=2_000_000_000, ge=1_000_000)
    max_profile_columns: int = Field(default=60, ge=1, le=500)
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    debug: bool = False
    ml_max_rows: int = Field(default=20_000, ge=100, le=100_000)
    ml_max_columns: int = Field(default=80, ge=2, le=200)
    ml_timeout_seconds: float = Field(default=180, ge=1, le=900)
    ml_max_candidates: int = Field(default=3, ge=1, le=4)
    ml_max_snapshot_bytes: int = Field(default=50_000_000, ge=1000)
    tavily_api_key: SecretStr | None = None
    web_search_timeout_seconds: float = Field(default=25, gt=0, le=120)
    web_search_max_results: int = Field(default=6, ge=1, le=10)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"

    def model_endpoint(self, role: Literal["primary", "secondary"]) -> ModelEndpoint:
        if role == "secondary" and self.secondary_model_base_url and self.secondary_model_name:
            return ModelEndpoint(
                base_url=self.secondary_model_base_url,
                model=self.secondary_model_name,
                api_key=self.secondary_model_api_key,
            )
        return ModelEndpoint(
            base_url=self.primary_model_base_url,
            model=self.primary_model_name,
            api_key=self.primary_model_api_key,
        )

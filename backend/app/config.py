from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./market_intelligence.db"
    cors_origins: list[str] | str = ["http://localhost:8000"]
    model_dir: Path = Path("models")
    cache_dir: Path = Path("data/raw/cache")
    market_data_provider: str = "yahoo_research"
    quote_provider: str = "historical"
    quote_delay_minutes: int = Field(15, ge=0)
    twelve_data_api_key: str | None = None
    newsdata_api_key: str | None = None
    news_provider: str = "newsdata"
    finbert_enabled: bool = False
    default_history_years: int = Field(8, ge=2, le=20)
    transaction_cost_bps: float = Field(10, ge=0)
    slippage_bps: float = Field(5, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

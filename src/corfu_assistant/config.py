from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CORFU_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "Corfu AI Tourist Assistant"
    environment: Literal["development", "test", "production"] = "development"
    dataset_path: Path = _PROJECT_ROOT / "corfu_dataset.json"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: Path | None = None
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="GOOGLE_API_KEY",
        repr=False,
    )
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    weather_fixture_path: Path = _PROJECT_ROOT / "corfu_synthetic_rain.json"
    weather_fixture_enabled: bool = False
    weather_latitude: float = Field(default=39.6231266, ge=-90, le=90)
    weather_longitude: float = Field(default=19.9285412, ge=-180, le=180)
    weather_timezone: str = "Europe/Athens"
    weather_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    weather_retry_budget: int = Field(default=1, ge=0, le=2)
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    @model_validator(mode="after")
    def validate_weather_fixture_environment(self) -> Settings:
        if self.weather_fixture_enabled and self.environment != "development":
            raise ValueError(
                "CORFU_WEATHER_FIXTURE_ENABLED is allowed only when "
                "CORFU_ENVIRONMENT=development"
            )
        return self

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for origin in value:
            candidate = origin.strip().rstrip("/")
            parsed = urlsplit(candidate)
            if (
                not candidate
                or candidate == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "CORS origins must be explicit HTTP(S) origins without paths or wildcards"
                )
            if candidate not in normalized:
                normalized.append(candidate)
        return normalized


def get_settings() -> Settings:
    return Settings()

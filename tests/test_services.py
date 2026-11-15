from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from corfu_assistant.config import Settings
from corfu_assistant.data import FixtureLoadError, load_dataset
from corfu_assistant.repository import POIRepository
from corfu_assistant.services import ServiceOverrides, build_application_services
from corfu_assistant.weather import (
    FixtureWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherProviderError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedEmbeddings:
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def make_repository() -> POIRepository:
    dataset = load_dataset(PROJECT_ROOT / "corfu_dataset.json")
    return POIRepository(dataset, embedding_model=FixedEmbeddings())


def make_settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        dataset_path=PROJECT_ROOT / "corfu_dataset.json",
        google_api_key=None,
        **overrides,
    )


def test_application_defaults_to_live_open_meteo_provider() -> None:
    services = build_application_services(make_settings(), make_repository())

    assert isinstance(services.weather_provider, OpenMeteoWeatherProvider)


def test_development_fixture_switch_selects_synthetic_provider_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORFU_ENVIRONMENT", "development")
    monkeypatch.setenv("CORFU_WEATHER_FIXTURE_ENABLED", "true")
    monkeypatch.setenv(
        "CORFU_WEATHER_FIXTURE_PATH",
        str(PROJECT_ROOT / "corfu_synthetic_rain.json"),
    )
    settings = Settings(
        _env_file=None,
        dataset_path=PROJECT_ROOT / "corfu_dataset.json",
        google_api_key=None,
    )
    services = build_application_services(settings, make_repository())

    assert isinstance(services.weather_provider, FixtureWeatherProvider)
    forecast = services.weather_provider.get_forecast(
        datetime(2026, 10, 7, 13, tzinfo=ZoneInfo("Europe/Athens")),
        datetime(2026, 10, 7, 18, tzinfo=ZoneInfo("Europe/Athens")),
    )
    assert forecast.source == "synthetic_fixture"
    with pytest.raises(WeatherProviderError, match="cannot be presented as current live"):
        services.weather_provider.get_current()


def test_fixture_switch_is_rejected_outside_development() -> None:
    with pytest.raises(ValidationError, match="allowed only.*development"):
        make_settings(environment="production", weather_fixture_enabled=True)

    settings = make_settings(environment="production")
    fixture = FixtureWeatherProvider(PROJECT_ROOT / "corfu_synthetic_rain.json")
    with pytest.raises(ValueError, match="allowed only.*development"):
        build_application_services(
            settings,
            make_repository(),
            overrides=ServiceOverrides(weather_provider=fixture),
        )


@pytest.mark.parametrize("fixture_contents", [None, "not valid json", "{}"])
def test_fixture_mode_fails_clearly_for_missing_or_invalid_fixture(
    tmp_path: Path,
    fixture_contents: str | None,
) -> None:
    fixture_path = tmp_path / "weather.json"
    if fixture_contents is not None:
        fixture_path.write_text(fixture_contents, encoding="utf-8")
    settings = make_settings(
        weather_fixture_enabled=True,
        weather_fixture_path=fixture_path,
    )

    with pytest.raises(FixtureLoadError):
        build_application_services(settings, make_repository())

import json
from pathlib import Path

import pytest

from corfu_assistant.data import (
    FixtureLoadError,
    load_dataset,
    load_weather_fixture,
    load_weather_forecast,
)
from corfu_assistant.models import WeatherFixture, WeatherForecast


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loads_supplied_dataset_and_preserves_null_semantics() -> None:
    dataset = load_dataset(PROJECT_ROOT / "corfu_dataset.json")

    assert len(dataset.pois) == 50
    assert dataset.timezone == "Europe/Athens"
    assert dataset.fields_note.opening_hours_fixture.startswith("Weekday schedule")
    assert dataset.pois[1].opening_hours_fixture is not None
    assert dataset.pois[1].opening_hours_fixture.tue is None
    assert dataset.pois[3].opening_hours_fixture is None
    assert dataset.pois[0].family_suitability_fixture is None


def test_geographically_verified_area_corrections_keep_stable_poi_ids() -> None:
    dataset = load_dataset(PROJECT_ROOT / "corfu_dataset.json")
    by_id = {poi.id: poi for poi in dataset.pois}

    assert by_id["CF-021"].area == "Gastouri"
    assert by_id["CF-031"].area == "Garitsa / Corfu Town"
    assert by_id["CF-026"].area == "Old Town of Corfu"


def test_opening_windows_include_full_day_and_overnight_ranges() -> None:
    dataset = load_dataset(PROJECT_ROOT / "corfu_dataset.json")
    windows = [
        window
        for poi in dataset.pois
        if poi.opening_hours_fixture is not None
        for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        if (window := getattr(poi.opening_hours_fixture, day)) is not None
    ]

    assert any(window.start_minute == 0 and window.end_minute == 1440 for window in windows)
    overnight = [window for window in windows if window.crosses_midnight]
    assert overnight
    assert overnight[0].start_minute == 20 * 60
    assert overnight[0].end_minute == 26 * 60


def test_rejects_missing_weekday_in_schedule(tmp_path: Path) -> None:
    dataset_data = json.loads((PROJECT_ROOT / "corfu_dataset.json").read_text())
    del dataset_data["pois"][0]["opening_hours_fixture"]["mon"]
    fixture_path = tmp_path / "invalid-dataset.json"
    fixture_path.write_text(json.dumps(dataset_data))

    with pytest.raises(FixtureLoadError, match="schema validation"):
        load_dataset(fixture_path)


def test_rejects_duplicate_poi_ids(tmp_path: Path) -> None:
    dataset_data = json.loads((PROJECT_ROOT / "corfu_dataset.json").read_text())
    dataset_data["pois"][1]["id"] = dataset_data["pois"][0]["id"]
    fixture_path = tmp_path / "duplicate-id.json"
    fixture_path.write_text(json.dumps(dataset_data))

    with pytest.raises(FixtureLoadError, match="POI ids must be unique"):
        load_dataset(fixture_path)


def test_loads_and_validates_original_synthetic_weather_fixture() -> None:
    fixture = load_weather_fixture(PROJECT_ROOT / "corfu_synthetic_rain.json")

    assert isinstance(fixture, WeatherFixture)
    assert fixture.is_live is False
    assert fixture.timezone == "Europe/Athens"
    assert len(fixture.hourly) == 5
    assert fixture.hourly[0].local_time.hour == 13
    assert [entry.precipitation_mm for entry in fixture.hourly[2:]] == [3.2, 4.1, 1.8]
    assert fixture.hourly[0].local_time.tzinfo is None


def test_weather_fixture_converts_to_provider_independent_forecast() -> None:
    fixture = load_weather_fixture(PROJECT_ROOT / "corfu_synthetic_rain.json")
    forecast = fixture.to_forecast()

    assert isinstance(forecast, WeatherForecast)
    assert not isinstance(forecast, WeatherFixture)
    assert forecast.source == "synthetic_fixture"
    assert forecast.place == fixture.place
    assert forecast.timezone == fixture.timezone
    assert forecast.hourly[0].at.hour == 13
    assert forecast.hourly[0].at.tzinfo is not None
    assert [entry.precipitation_mm for entry in forecast.hourly[2:]] == [3.2, 4.1, 1.8]


def test_weather_forecast_loader_returns_normalized_forecast() -> None:
    forecast = load_weather_forecast(PROJECT_ROOT / "corfu_synthetic_rain.json")

    assert isinstance(forecast, WeatherForecast)
    assert forecast.source == "synthetic_fixture"
    assert forecast.hourly[0].at.tzinfo is not None


def test_reports_invalid_json_with_fixture_path(tmp_path: Path) -> None:
    fixture_path = tmp_path / "broken.json"
    fixture_path.write_text("{", encoding="utf-8")

    with pytest.raises(FixtureLoadError, match="not valid JSON"):
        load_dataset(fixture_path)

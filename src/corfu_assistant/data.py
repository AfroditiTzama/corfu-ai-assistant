import json
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from corfu_assistant.models import CorfuDataset, WeatherFixture, WeatherForecast


class FixtureLoadError(ValueError):
    """Raised when a supplied JSON fixture cannot be read or validated."""


FixtureModel = TypeVar("FixtureModel", CorfuDataset, WeatherFixture)


def _load_fixture(path: Path, model: type[FixtureModel]) -> FixtureModel:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FixtureLoadError(f"Unable to read fixture at {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise FixtureLoadError(
            f"Fixture at {path} is not valid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})"
        ) from error

    try:
        return model.model_validate(raw_data)
    except ValidationError as error:
        raise FixtureLoadError(f"Fixture at {path} failed schema validation:\n{error}") from error


def load_dataset(path: Path) -> CorfuDataset:
    return _load_fixture(path, CorfuDataset)


def load_weather_fixture(path: Path) -> WeatherFixture:
    return _load_fixture(path, WeatherFixture)


def load_weather_forecast(path: Path) -> WeatherForecast:
    """Load fixture data as the provider-independent forecast contract."""
    return load_weather_fixture(path).to_forecast()

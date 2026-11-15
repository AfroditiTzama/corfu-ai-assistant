import os
from datetime import datetime, timezone

import pytest

from corfu_assistant.config import get_settings
from corfu_assistant.weather import OpenMeteoWeatherProvider


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("CORFU_RUN_OPEN_METEO_INTEGRATION") != "1",
    reason="Set CORFU_RUN_OPEN_METEO_INTEGRATION=1 to call the live Open-Meteo API",
)
def test_live_open_meteo_next_24_hours() -> None:
    settings = get_settings()
    provider = OpenMeteoWeatherProvider(
        latitude=settings.weather_latitude,
        longitude=settings.weather_longitude,
        timezone_name=settings.weather_timezone,
        timeout_seconds=settings.weather_timeout_seconds,
        retry_budget=settings.weather_retry_budget,
    )

    forecast = provider.next_24_hours()

    assert forecast.source == "live"
    assert forecast.timezone == "Europe/Athens"
    assert forecast.retrieved_at is not None
    assert forecast.retrieved_at.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    assert forecast.hourly

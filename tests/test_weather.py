import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from corfu_assistant.config import Settings
from corfu_assistant.data import FixtureLoadError
from corfu_assistant.models import ForecastHour, WeatherForecast
from corfu_assistant.weather import (
    DEFAULT_RAIN_THRESHOLD_MM,
    FixtureWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherProviderError,
    WeatherResponseError,
    WeatherTransportError,
    assess_rain,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "corfu_synthetic_rain.json"
ATHENS = ZoneInfo("Europe/Athens")


def hourly_response(
    start: datetime,
    count: int = 26,
    *,
    precipitation: list[float | None] | None = None,
) -> dict[str, object]:
    hours = [start + timedelta(hours=index) for index in range(count)]
    return {
        "timezone": "Europe/Athens",
        "hourly_units": {
            "precipitation": "mm",
            "precipitation_probability": "%",
            "temperature_2m": "°C",
            "weather_code": "wmo code",
        },
        "hourly": {
            "time": [hour.strftime("%Y-%m-%dT%H:%M") for hour in hours],
            "precipitation": precipitation or [0.0] * count,
            "precipitation_probability": [0] * count,
            "temperature_2m": [21.5] * count,
            "weather_code": [1] * count,
        },
    }


def make_live_provider(
    *,
    now: datetime | None = None,
    retry_budget: int = 1,
    sleep=lambda _delay: None,
    jitter=lambda _low, _high: 0.0,
) -> OpenMeteoWeatherProvider:
    return OpenMeteoWeatherProvider(
        latitude=39.6231266,
        longitude=19.9285412,
        timeout_seconds=4,
        retry_budget=retry_budget,
        clock=lambda: now or datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS),
        sleep=sleep,
        jitter=jitter,
    )


def test_settings_include_dataset_verified_corfu_weather_defaults() -> None:
    settings = Settings()

    assert settings.weather_latitude == 39.6231266
    assert settings.weather_longitude == 19.9285412
    assert settings.weather_timezone == "Europe/Athens"
    assert settings.weather_fixture_path.name == "corfu_synthetic_rain.json"


def test_open_meteo_parses_live_hourly_values_and_provenance(monkeypatch) -> None:
    now = datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS)
    payload = hourly_response(datetime(2026, 11, 11, 0, tzinfo=ATHENS), count=48)
    calls: list[tuple[str, dict[str, object], float]] = []

    def fake_get(url: str, *, params: dict[str, object], timeout: float) -> httpx.Response:
        calls.append((url, params, timeout))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr("corfu_assistant.weather.httpx.get", fake_get)
    forecast = make_live_provider(now=now).next_24_hours()

    assert forecast.source == "live"
    assert forecast.timezone == "Europe/Athens"
    assert forecast.retrieved_at == now
    assert forecast.hourly[0].at.tzinfo is not None
    assert forecast.hourly[0].temperature_c == 21.5
    assert forecast.hourly[0].precipitation_probability_percent == 0
    assert forecast.hourly[0].precipitation_mm == 0
    assert forecast.hourly[0].weather_code == 1
    assert calls[0][0] == "https://api.open-meteo.com/v1/forecast"
    assert calls[0][1]["timezone"] == "Europe/Athens"
    assert calls[0][1]["forecast_days"] == 2
    assert calls[0][2] == 4
    assert len(forecast.hourly) == 25


def test_open_meteo_current_weather_uses_common_forecast_contract(monkeypatch) -> None:
    now = datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS)
    payload = {
        "timezone": "Europe/Athens",
        "current_units": {
            "temperature_2m": "°C",
            "precipitation": "mm",
            "weather_code": "wmo code",
        },
        "current": {
            "time": "2026-11-11T12:30",
            "temperature_2m": 22.0,
            "precipitation": 0.0,
            "weather_code": 2,
        },
    }
    monkeypatch.setattr(
        "corfu_assistant.weather.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("GET", url)
        ),
    )

    forecast = make_live_provider(now=now).get_current()

    assert forecast.source == "live"
    assert len(forecast.hourly) == 1
    assert forecast.hourly[0].at.isoformat() == "2026-11-11T12:30:00+02:00"
    assert forecast.hourly[0].temperature_c == 22
    assert forecast.hourly[0].precipitation_probability_percent is None


def test_open_meteo_orders_repeated_local_hour_across_dst_fallback(monkeypatch) -> None:
    start = datetime(2026, 10, 25, 0, tzinfo=ATHENS)
    payload = {
        "timezone": "Europe/Athens",
        "hourly_units": {
            "precipitation": "mm",
            "precipitation_probability": "%",
            "temperature_2m": "°C",
            "weather_code": "wmo code",
        },
        "hourly": {
            "time": [
                "2026-10-25T00:00",
                "2026-10-25T01:00",
                "2026-10-25T02:00",
                "2026-10-25T03:00",
                "2026-10-25T03:00",
                "2026-10-25T04:00",
                "2026-10-25T05:00",
            ],
            "precipitation": [0] * 7,
            "precipitation_probability": [0] * 7,
            "temperature_2m": [20] * 7,
            "weather_code": [1] * 7,
        },
    }
    monkeypatch.setattr(
        "corfu_assistant.weather.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("GET", url)
        ),
    )
    provider = make_live_provider(now=start)
    forecast = provider.get_forecast(start, datetime(2026, 10, 25, 5, tzinfo=ATHENS))
    repeated = [hour.at for hour in forecast.hourly if hour.at.hour == 3]

    assert len(repeated) == 2
    assert repeated[0].utcoffset() == timedelta(hours=3)
    assert repeated[1].utcoffset() == timedelta(hours=2)
    assert repeated[1].astimezone(timezone.utc) - repeated[0].astimezone(
        timezone.utc
    ) == timedelta(hours=1)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"timezone": "UTC", "hourly": {}},
        {
            **hourly_response(datetime(2026, 11, 11, 0, tzinfo=ATHENS)),
            "hourly_units": {"precipitation": "inches"},
        },
        {
            "timezone": "Europe/Athens",
            "hourly_units": {"precipitation": "mm"},
            "hourly": {"time": ["2026-11-11T00:00"], "precipitation": [1, 2]},
        },
        {
            "timezone": "Europe/Athens",
            "hourly_units": {
                "precipitation": "mm",
                "precipitation_probability": "%",
                "temperature_2m": "°C",
                "weather_code": "wmo code",
            },
            "hourly": {
                "time": ["2026-11-11T14:00"],
                "precipitation": [0],
                "precipitation_probability": [0],
                "temperature_2m": [21],
                "weather_code": [120],
            },
        },
    ],
)
def test_invalid_open_meteo_responses_fail_explicitly(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        "corfu_assistant.weather.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("GET", url)
        ),
    )

    with pytest.raises(WeatherResponseError):
        make_live_provider().next_24_hours()


def test_open_meteo_timeout_retries_once_and_preserves_cause(monkeypatch) -> None:
    sleeps: list[float] = []
    requests = 0
    root_error = httpx.ReadTimeout("request timed out")
    payload = hourly_response(datetime(2026, 11, 11, 0, tzinfo=ATHENS), count=48)

    def fake_get(url: str, **kwargs) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise root_error
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr("corfu_assistant.weather.httpx.get", fake_get)
    forecast = make_live_provider(sleep=sleeps.append, jitter=lambda low, high: high).next_24_hours()

    assert forecast.source == "live"
    assert requests == 2
    assert sleeps == [0.25]


def test_open_meteo_transient_server_error_retries_within_budget(monkeypatch) -> None:
    requests = 0
    sleeps: list[float] = []
    payload = hourly_response(datetime(2026, 11, 11, 0, tzinfo=ATHENS), count=48)

    def fake_get(url: str, **kwargs) -> httpx.Response:
        nonlocal requests
        requests += 1
        status = 503 if requests == 1 else 200
        body = {"unavailable": True} if status == 503 else payload
        return httpx.Response(status, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr("corfu_assistant.weather.httpx.get", fake_get)
    forecast = make_live_provider(
        sleep=sleeps.append,
        jitter=lambda low, high: high,
    ).next_24_hours()

    assert forecast.source == "live"
    assert requests == 2
    assert sleeps == [0.25]


def test_open_meteo_nontransient_http_failure_does_not_retry(monkeypatch) -> None:
    requests = 0

    def fake_get(url: str, **kwargs) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(403, request=httpx.Request("GET", url))

    monkeypatch.setattr("corfu_assistant.weather.httpx.get", fake_get)
    with pytest.raises(WeatherProviderError, match="HTTP 403") as raised:
        make_live_provider().next_24_hours()

    assert requests == 1
    assert isinstance(raised.value.__cause__, httpx.HTTPStatusError)


def test_open_meteo_exhausted_timeout_reports_transport_error_with_cause(monkeypatch) -> None:
    root_error = httpx.ReadTimeout("offline")
    sleeps: list[float] = []
    requests = 0

    def fake_get(url: str, **kwargs) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise root_error

    monkeypatch.setattr("corfu_assistant.weather.httpx.get", fake_get)
    with pytest.raises(WeatherTransportError) as raised:
        make_live_provider(sleep=sleeps.append).next_24_hours()

    assert requests == 2
    assert sleeps == [0]
    assert raised.value.__cause__ is root_error


def test_live_provider_rejects_naive_clock_and_historical_range(monkeypatch) -> None:
    provider = make_live_provider(now=datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS))
    now = datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS)
    with pytest.raises(WeatherProviderError, match="historical"):
        provider.get_forecast(now - timedelta(hours=1), now)
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_forecast(datetime(2026, 11, 11), datetime(2026, 11, 12))


def test_missing_remote_values_remain_none_and_rain_is_unknown(monkeypatch) -> None:
    now = datetime(2026, 11, 11, 12, 30, tzinfo=ATHENS)
    payload = {
        "timezone": "Europe/Athens",
        "hourly_units": {},
        "hourly": {
            "time": ["2026-11-11T13:00"],
        },
    }
    monkeypatch.setattr(
        "corfu_assistant.weather.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("GET", url)
        ),
    )

    forecast = make_live_provider(now=now).get_forecast(
        now, now + timedelta(hours=1)
    )
    assessment = assess_rain(forecast, now, now + timedelta(hours=1))

    assert forecast.hourly[0].precipitation_mm is None
    assert forecast.hourly[0].temperature_c is None
    assert forecast.hourly[0].weather_code is None
    assert assessment.rain_forecast is None
    assert assessment.information_missing


def test_fixture_provider_loads_offline_and_reports_rain_from_15_through_17() -> None:
    retrieved_at = datetime(2026, 10, 7, 12, tzinfo=ATHENS)
    provider = FixtureWeatherProvider(FIXTURE_PATH, clock=lambda: retrieved_at)
    start = datetime(2026, 10, 7, 13, tzinfo=ATHENS)
    end = datetime(2026, 10, 7, 18, tzinfo=ATHENS)

    forecast = provider.get_forecast(start, end)
    assessment = assess_rain(forecast, start, end)

    assert forecast.source == "synthetic_fixture"
    assert forecast.retrieved_at == retrieved_at
    assert [hour.at.hour for hour in forecast.hourly] == [13, 14, 15, 16, 17]
    assert assessment.threshold_mm == DEFAULT_RAIN_THRESHOLD_MM
    assert assessment.rain_forecast is True
    assert [interval.start.hour for interval in assessment.affected_intervals] == [15, 16, 17]
    assert assessment.first_affected_interval.start == datetime(
        2026, 10, 7, 15, tzinfo=ATHENS
    )
    assert assessment.exposure_overlaps is True
    assert not assessment.information_missing


def test_fixture_rain_begins_at_exactly_1500_and_dry_window_is_false() -> None:
    provider = FixtureWeatherProvider(
        FIXTURE_PATH,
        clock=lambda: datetime(2026, 10, 7, 12, tzinfo=ATHENS),
    )
    start = datetime(2026, 10, 7, 13, tzinfo=ATHENS)
    forecast = provider.get_forecast(start, datetime(2026, 10, 7, 18, tzinfo=ATHENS))
    dry_until_15 = assess_rain(
        forecast,
        start,
        datetime(2026, 10, 7, 15, tzinfo=ATHENS),
    )
    exposure = assess_rain(
        forecast,
        start,
        datetime(2026, 10, 7, 18, tzinfo=ATHENS),
        exposure_start=datetime(2026, 10, 7, 14, tzinfo=ATHENS),
        exposure_end=datetime(2026, 10, 7, 15, tzinfo=ATHENS),
    )

    assert dry_until_15.rain_forecast is False
    assert dry_until_15.affected_intervals == []
    assert exposure.exposure_overlaps is False


def test_fixture_does_not_claim_rain_absent_when_window_has_missing_data() -> None:
    forecast = WeatherForecast(
        place="Corfu Town",
        timezone="Europe/Athens",
        source="live",
        hourly=[
            ForecastHour(
                at=datetime(2026, 11, 11, 12, tzinfo=ATHENS),
                precipitation_mm=0,
            ),
            ForecastHour(
                at=datetime(2026, 11, 11, 14, tzinfo=ATHENS),
                precipitation_mm=0,
            ),
        ],
    )
    start = datetime(2026, 11, 11, 12, tzinfo=ATHENS)
    result = assess_rain(forecast, start, start + timedelta(hours=3))

    assert result.rain_forecast is None
    assert result.information_missing


def test_unknown_precipitation_is_not_reported_as_no_rain() -> None:
    forecast = WeatherForecast(
        place="Corfu Town",
        timezone="Europe/Athens",
        source="live",
        hourly=[
            ForecastHour(
                at=datetime(2026, 11, 11, 12, tzinfo=ATHENS),
                precipitation_mm=None,
            )
        ],
    )
    start = datetime(2026, 11, 11, 12, tzinfo=ATHENS)
    result = assess_rain(forecast, start, start + timedelta(hours=1))

    assert result.rain_forecast is None
    assert result.exposure_overlaps is None
    assert result.information_missing


def test_fixture_results_are_deterministic_with_fixed_clock() -> None:
    retrieved_at = datetime(2026, 10, 7, 12, tzinfo=ATHENS)
    provider = FixtureWeatherProvider(FIXTURE_PATH, clock=lambda: retrieved_at)
    start = datetime(2026, 10, 7, 13, tzinfo=ATHENS)
    end = datetime(2026, 10, 7, 18, tzinfo=ATHENS)

    first = provider.get_forecast(start, end)
    second = provider.get_forecast(start, end)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert assess_rain(first, start, end) == assess_rain(second, start, end)


def test_fixture_provider_rejects_unsupported_dates_without_live_fallback() -> None:
    provider = FixtureWeatherProvider(FIXTURE_PATH)
    start = datetime(2026, 10, 8, 13, tzinfo=ATHENS)

    with pytest.raises(WeatherProviderError, match="no live fallback"):
        provider.get_forecast(start, start + timedelta(hours=1))

    with pytest.raises(WeatherProviderError, match="cannot be presented as current"):
        provider.get_current()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timezone", "UTC"),
        ("hourly.0.local_time", "2026-10-08T13:00:00"),
        ("hourly.0.precipitation_mm", -1),
    ],
)
def test_fixture_provider_rejects_invalid_fixture_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if field.startswith("hourly."):
        _, index, key = field.split(".")
        fixture["hourly"][int(index)][key] = value
    else:
        fixture[field] = value
    path = tmp_path / "invalid-weather.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(FixtureLoadError):
        FixtureWeatherProvider(path)


def test_open_meteo_precipitation_uses_api_preceding_hour_interval(monkeypatch) -> None:
    now = datetime(2026, 11, 11, 13, tzinfo=ATHENS)
    precipitation = [0.0] * 26
    precipitation[14] = 2.5
    payload = hourly_response(
        datetime(2026, 11, 11, 0, tzinfo=ATHENS),
        precipitation=precipitation,
    )
    monkeypatch.setattr(
        "corfu_assistant.weather.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("GET", url)
        ),
    )
    forecast = make_live_provider(now=now).get_forecast(
        now, now + timedelta(hours=1)
    )
    assessment = assess_rain(forecast, now, now + timedelta(hours=1))

    assert assessment.rain_forecast is True
    assert assessment.first_affected_interval.start == now
    assert assessment.first_affected_interval.end == now + timedelta(hours=1)


def test_rain_assessment_validates_threshold_and_window() -> None:
    forecast = WeatherForecast(
        place="Corfu Town",
        timezone="Europe/Athens",
        source="live",
        hourly=[
            ForecastHour(
                at=datetime(2026, 11, 11, 12, tzinfo=ATHENS),
                precipitation_mm=0,
            )
        ],
    )
    start = datetime(2026, 11, 11, 12, tzinfo=ATHENS)

    with pytest.raises(ValueError, match="threshold_mm"):
        assess_rain(forecast, start, start + timedelta(hours=1), threshold_mm=0)
    with pytest.raises(ValueError, match="timezone-aware"):
        assess_rain(forecast, datetime(2026, 11, 11, 12), start + timedelta(hours=1))

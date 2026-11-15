from __future__ import annotations

import math
import logging
import random
import time
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
from typing import Callable, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import Field, ValidationError, model_validator

from corfu_assistant.data import load_weather_fixture
from corfu_assistant.models import DomainModel, ForecastHour, WeatherForecast


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CORFU_TIMEZONE = "Europe/Athens"
DEFAULT_RAIN_THRESHOLD_MM = 0.1
_HOURLY_FIELDS = (
    "precipitation",
    "precipitation_probability",
    "temperature_2m",
    "weather_code",
)
_EXPECTED_UNITS = {
    "precipitation": "mm",
    "precipitation_probability": "%",
    "temperature_2m": "°C",
    "weather_code": "wmo code",
}
_LOGGER = logging.getLogger(__name__)


class WeatherProviderError(RuntimeError):
    """Base error for weather provider failures."""


class WeatherTransportError(WeatherProviderError):
    """Raised when the weather service cannot be reached."""


class WeatherResponseError(WeatherProviderError):
    """Raised when a provider response is invalid or incomplete."""


class WeatherProvider(Protocol):
    def next_24_hours(self) -> WeatherForecast:
        """Return the rolling 24 hours anchored to the provider's current clock."""

    def get_forecast(self, start_at: datetime, end_at: datetime) -> WeatherForecast:
        """Return provider data overlapping the requested local-time interval."""

    def get_current(self) -> WeatherForecast:
        """Return current conditions in the common forecast representation."""


class OpenMeteoWeatherProvider:
    """Synchronous adapter for the public Open-Meteo Forecast API."""

    def __init__(
        self,
        *,
        latitude: float,
        longitude: float,
        timezone_name: str = CORFU_TIMEZONE,
        timeout_seconds: float = 10.0,
        retry_budget: int = 1,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if not math.isfinite(latitude) or not -90 <= latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not math.isfinite(longitude) or not -180 <= longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")
        try:
            self._zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown IANA timezone: {timezone_name}") from error
        if not 0 < timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 0 and 120")
        if isinstance(retry_budget, bool) or not isinstance(retry_budget, int) or not 0 <= retry_budget <= 2:
            raise ValueError("retry_budget must be an integer between 0 and 2")
        self._latitude = latitude
        self._longitude = longitude
        self._timezone_name = timezone_name
        self._timeout_seconds = timeout_seconds
        self._retry_budget = retry_budget
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep
        self._jitter = jitter

    def get_forecast(self, start_at: datetime, end_at: datetime) -> WeatherForecast:
        start, end = self._normalize_window(start_at, end_at)
        now = self._now()
        if start.astimezone(timezone.utc) < now.astimezone(timezone.utc):
            raise WeatherProviderError("Live forecasts cannot include a historical start time")
        if (
            end.astimezone(timezone.utc) - start.astimezone(timezone.utc)
            > timedelta(hours=24)
        ):
            raise WeatherProviderError("Open-Meteo requests are limited to a 24-hour window")

        return self._fetch_forecast(start, end)

    def _fetch_forecast(self, start: datetime, end: datetime) -> WeatherForecast:
        payload = self._get_json(
            {
                "latitude": self._latitude,
                "longitude": self._longitude,
                "timezone": self._timezone_name,
                "forecast_days": 2,
                "hourly": ",".join(_HOURLY_FIELDS),
            }
        )
        return self._parse_hourly_forecast(payload, start, end)

    def next_24_hours(self) -> WeatherForecast:
        start = self._now()
        end = (start.astimezone(timezone.utc) + timedelta(hours=24)).astimezone(
            self._zone
        )
        return self._fetch_forecast(start, end)

    def get_current(self) -> WeatherForecast:
        payload = self._get_json(
            {
                "latitude": self._latitude,
                "longitude": self._longitude,
                "timezone": self._timezone_name,
                "current": "temperature_2m,precipitation,weather_code",
            }
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("current"), dict):
            raise WeatherResponseError("Open-Meteo response is missing current conditions")
        response_timezone = payload.get("timezone")
        if response_timezone != self._timezone_name:
            raise WeatherResponseError(
                f"Open-Meteo timezone mismatch: expected {self._timezone_name}"
            )

        current = payload["current"]
        units = payload.get("current_units")
        if not isinstance(units, dict):
            raise WeatherResponseError("Open-Meteo response is missing current units")
        values = self._weather_values(current, units)
        raw_offset = payload.get("utc_offset_seconds")
        if raw_offset is not None and (
            isinstance(raw_offset, bool) or not isinstance(raw_offset, int)
        ):
            raise WeatherResponseError("Open-Meteo utc_offset_seconds must be an integer")
        at = self._parse_timestamp(current.get("time"), offset_seconds=raw_offset)
        period_start = (
            at.astimezone(timezone.utc) - timedelta(hours=1)
        ).astimezone(self._zone)
        try:
            return WeatherForecast(
                place="Corfu Town",
                timezone=self._timezone_name,
                source="live",
                hourly=[
                    ForecastHour(
                        at=at,
                        precipitation_interval_start=period_start,
                        precipitation_interval_end=at,
                        **values,
                    )
                ],
                retrieved_at=self._now(),
            )
        except ValidationError as error:
            raise WeatherResponseError(
                "Open-Meteo current conditions failed weather schema validation"
            ) from error

    def _parse_hourly_forecast(
        self,
        payload: object,
        start: datetime,
        end: datetime,
    ) -> WeatherForecast:
        if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
            raise WeatherResponseError("Open-Meteo response is missing hourly forecast data")
        if payload.get("timezone") != self._timezone_name:
            raise WeatherResponseError(
                f"Open-Meteo timezone mismatch: expected {self._timezone_name}"
            )

        hourly = payload["hourly"]
        units = payload.get("hourly_units")
        if not isinstance(units, dict):
            raise WeatherResponseError("Open-Meteo response is missing hourly units")
        raw_times = hourly.get("time")
        if not isinstance(raw_times, list) or not raw_times:
            raise WeatherResponseError("Open-Meteo response contains no hourly timestamps")
        times = self._parse_timestamp_sequence(raw_times)
        arrays = {
            field: self._optional_array(hourly, field, units, len(raw_times))
            for field in _HOURLY_FIELDS
        }

        selected: list[ForecastHour] = []
        for index, at in enumerate(times):
            period_start = (
                at.astimezone(timezone.utc) - timedelta(hours=1)
            ).astimezone(self._zone)
            if (
                self._overlaps(period_start, start, end)
                or start.astimezone(timezone.utc)
                <= at.astimezone(timezone.utc)
                < end.astimezone(timezone.utc)
            ):
                values = {
                    "precipitation_mm": self._optional_number(arrays["precipitation"][index]),
                    "precipitation_probability_percent": self._optional_number(
                        arrays["precipitation_probability"][index]
                    ),
                    "temperature_c": self._optional_number(arrays["temperature_2m"][index]),
                    "weather_code": self._optional_code(arrays["weather_code"][index]),
                }
                try:
                    selected.append(
                        ForecastHour(
                            at=at,
                            precipitation_interval_start=period_start,
                            precipitation_interval_end=at,
                            **values,
                        )
                    )
                except ValidationError as error:
                    raise WeatherResponseError(
                        "Open-Meteo hourly values failed weather schema validation"
                    ) from error

        if not selected:
            raise WeatherResponseError("Open-Meteo response has no hours in the requested interval")
        try:
            return WeatherForecast(
                place="Corfu Town",
                timezone=self._timezone_name,
                source="live",
                hourly=selected,
                retrieved_at=self._now(),
            )
        except ValidationError as error:
            raise WeatherResponseError(
                "Open-Meteo forecast failed weather schema validation"
            ) from error

    def _optional_array(
        self,
        hourly: dict[str, object],
        field: str,
        units: dict[str, object],
        count: int,
    ) -> list[object]:
        values = hourly.get(field)
        if values is None:
            return [None] * count
        if not isinstance(values, list) or len(values) != count:
            raise WeatherResponseError(f"Open-Meteo hourly field {field} has invalid shape")
        unit = units.get(field)
        if unit != _EXPECTED_UNITS[field]:
            raise WeatherResponseError(
                f"Open-Meteo unit mismatch for {field}: expected {_EXPECTED_UNITS[field]}"
            )
        return values

    @staticmethod
    def _weather_values(
        current: dict[str, object],
        units: dict[str, object],
    ) -> dict[str, object]:
        values: dict[str, object] = {
            "precipitation_mm": None,
            "precipitation_probability_percent": None,
            "temperature_c": None,
            "weather_code": None,
        }
        field_mapping = {
            "precipitation": ("precipitation_mm", "mm"),
            "precipitation_probability": ("precipitation_probability_percent", "%"),
            "temperature_2m": ("temperature_c", "°C"),
            "weather_code": ("weather_code", "wmo code"),
        }
        for api_field, (model_field, expected_unit) in field_mapping.items():
            if api_field not in current:
                continue
            if units.get(api_field) != expected_unit:
                raise WeatherResponseError(
                    f"Open-Meteo unit mismatch for {api_field}: expected {expected_unit}"
                )
            value = current[api_field]
            values[model_field] = (
                OpenMeteoWeatherProvider._optional_code(value)
                if model_field == "weather_code"
                else OpenMeteoWeatherProvider._optional_number(value)
            )
        return values

    def _get_json(self, params: dict[str, object]) -> object:
        started = time.perf_counter()
        try:
            payload = self._get_json_with_retries(params)
        except Exception as error:
            _LOGGER.info(
                "Weather API stage completed",
                extra={
                    "stage": "weather_api",
                    "outcome": "error",
                    "error_type": type(error).__name__,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        _LOGGER.info(
            "Weather API stage completed",
            extra={
                "stage": "weather_api",
                "outcome": "success",
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return payload

    def _get_json_with_retries(self, params: dict[str, object]) -> object:
        for attempt in range(self._retry_budget + 1):
            try:
                response = httpx.get(
                    OPEN_METEO_FORECAST_URL,
                    params=params,
                    timeout=self._timeout_seconds,
                )
            except httpx.TimeoutException as error:
                if attempt < self._retry_budget:
                    self._wait_before_retry(attempt)
                    continue
                raise WeatherTransportError("Open-Meteo request timed out") from error
            except httpx.RequestError as error:
                if attempt < self._retry_budget:
                    self._wait_before_retry(attempt)
                    continue
                raise WeatherTransportError("Unable to reach Open-Meteo") from error

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code in {500, 502, 503, 504} and attempt < self._retry_budget:
                    self._wait_before_retry(attempt)
                    continue
                raise WeatherProviderError(
                    f"Open-Meteo returned HTTP {error.response.status_code}"
                ) from error

            try:
                return response.json()
            except ValueError as error:
                raise WeatherResponseError("Open-Meteo returned invalid JSON") from error

        raise AssertionError("bounded Open-Meteo retry loop exited unexpectedly")

    def _wait_before_retry(self, attempt: int) -> None:
        delay_cap = min(2.0, 0.25 * (2**attempt))
        self._sleep(self._jitter(0.0, delay_cap))

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Weather provider clock must return a timezone-aware datetime")
        return now.astimezone(self._zone)

    def _parse_timestamp(
        self,
        value: object,
        *,
        offset_seconds: int | None = None,
    ) -> datetime:
        if not isinstance(value, str):
            raise WeatherResponseError("Open-Meteo timestamp must be a string")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise WeatherResponseError("Open-Meteo returned an invalid timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            if offset_seconds is not None:
                fixed_offset = timezone(timedelta(seconds=offset_seconds))
                return parsed.replace(tzinfo=fixed_offset).astimezone(self._zone)
            return parsed.replace(tzinfo=self._zone)
        return parsed.astimezone(self._zone)

    def _parse_timestamp_sequence(self, values: list[object]) -> list[datetime]:
        parsed_times: list[datetime] = []
        previous: datetime | None = None
        for raw_value in values:
            if not isinstance(raw_value, str):
                raise WeatherResponseError("Open-Meteo timestamp must be a string")
            try:
                local_time = datetime.fromisoformat(raw_value)
            except ValueError as error:
                raise WeatherResponseError(
                    "Open-Meteo returned an invalid timestamp"
                ) from error
            if local_time.tzinfo is not None and local_time.utcoffset() is not None:
                candidate = local_time.astimezone(self._zone)
            else:
                candidate = self._localize_in_sequence(local_time, previous)
            if previous is not None and candidate.astimezone(
                timezone.utc
            ) <= previous.astimezone(timezone.utc):
                raise WeatherResponseError(
                    "Open-Meteo timestamps are not strictly chronological"
                )
            parsed_times.append(candidate)
            previous = candidate
        return parsed_times

    def _localize_in_sequence(
        self,
        value: datetime,
        previous: datetime | None,
    ) -> datetime:
        candidates: list[datetime] = []
        seen: set[datetime] = set()
        for fold in (0, 1):
            candidate = value.replace(tzinfo=self._zone, fold=fold)
            round_trip = candidate.astimezone(timezone.utc).astimezone(self._zone)
            if round_trip.replace(tzinfo=None) != value:
                continue
            instant = candidate.astimezone(timezone.utc)
            if instant not in seen:
                candidates.append(candidate)
                seen.add(instant)
        if not candidates:
            raise WeatherResponseError(
                "Open-Meteo returned a nonexistent local timestamp"
            )
        if previous is None:
            return candidates[0]
        previous_utc = previous.astimezone(timezone.utc)
        later = [
            candidate
            for candidate in candidates
            if candidate.astimezone(timezone.utc) > previous_utc
        ]
        if not later:
            raise WeatherResponseError(
                "Open-Meteo local timestamps cannot be ordered unambiguously"
            )
        return min(later, key=lambda candidate: candidate.astimezone(timezone.utc))

    @staticmethod
    def _optional_number(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WeatherResponseError("Open-Meteo weather values must be numeric or null")
        number = float(value)
        if not math.isfinite(number):
            raise WeatherResponseError("Open-Meteo weather values must be finite")
        return number

    @staticmethod
    def _optional_code(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise WeatherResponseError("Open-Meteo weather codes must be integers or null")
        return value

    def _normalize_window(
        self,
        start_at: datetime,
        end_at: datetime,
    ) -> tuple[datetime, datetime]:
        return _aware_window(start_at, end_at, self._zone)

    @staticmethod
    def _overlaps(at: datetime, start: datetime, end: datetime) -> bool:
        hour_end = at.astimezone(timezone.utc) + timedelta(hours=1)
        return (
            at.astimezone(timezone.utc) < end.astimezone(timezone.utc)
            and hour_end > start.astimezone(timezone.utc)
        )


class FixtureWeatherProvider:
    """Offline adapter for the supplied synthetic rain exercise fixture."""

    def __init__(
        self,
        fixture_path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._fixture = load_weather_fixture(Path(fixture_path))
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def get_forecast(self, start_at: datetime, end_at: datetime) -> WeatherForecast:
        if self._fixture.timezone != CORFU_TIMEZONE:
            raise WeatherProviderError("Synthetic fixture timezone must be Europe/Athens")
        zone = ZoneInfo(self._fixture.timezone)
        start, end = _aware_window(start_at, end_at, zone)
        forecast = self._fixture.to_forecast(retrieved_at=self._now())
        selected = [
            hour
            for hour in forecast.hourly
            if _hour_overlaps(hour.at, start, end)
        ]
        if not selected:
            raise WeatherProviderError(
                "Synthetic fixture has no data for the requested interval; no live fallback was attempted"
            )
        return forecast.model_copy(update={"hourly": selected})

    def next_24_hours(self) -> WeatherForecast:
        start = self._now()
        end = (start.astimezone(timezone.utc) + timedelta(hours=24)).astimezone(
            ZoneInfo(self._fixture.timezone)
        )
        return self.get_forecast(start, end)

    def get_current(self) -> WeatherForecast:
        raise WeatherProviderError(
            "Synthetic fixture data cannot be presented as current live conditions"
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Weather provider clock must return a timezone-aware datetime")
        return now.astimezone(ZoneInfo(self._fixture.timezone))


class WeatherInterval(DomainModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_interval(self) -> WeatherInterval:
        if (
            self.start.tzinfo is None
            or self.start.utcoffset() is None
            or self.end.tzinfo is None
            or self.end.utcoffset() is None
        ):
            raise ValueError("Weather intervals must be timezone-aware")
        if self.end.astimezone(timezone.utc) <= self.start.astimezone(timezone.utc):
            raise ValueError("Weather interval end must be later than start")
        return self


class RainAssessment(DomainModel):
    threshold_mm: float = Field(gt=0)
    rain_forecast: bool | None
    affected_intervals: list[WeatherInterval]
    first_affected_interval: WeatherInterval | None
    exposure_overlaps: bool | None
    information_missing: bool


def assess_rain(
    forecast: WeatherForecast,
    start_at: datetime,
    end_at: datetime,
    *,
    threshold_mm: float = DEFAULT_RAIN_THRESHOLD_MM,
    exposure_start: datetime | None = None,
    exposure_end: datetime | None = None,
) -> RainAssessment:
    """Assess hourly precipitation without treating missing data as dry weather."""
    if not math.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("threshold_mm must be a finite value greater than zero")
    zone = ZoneInfo(forecast.timezone)
    start, end = _aware_window(start_at, end_at, zone)
    exposure_start = exposure_start or start
    exposure_end = exposure_end or end
    exposure_start, exposure_end = _aware_window(exposure_start, exposure_end, zone)

    records = sorted(forecast.hourly, key=lambda hour: hour.at.astimezone(timezone.utc))
    affected: list[WeatherInterval] = []
    missing = False
    cursor = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    for hour in records:
        hour_start, hour_end = _precipitation_window(hour)
        overlap_start = max(start.astimezone(timezone.utc), hour_start)
        overlap_end = min(end_utc, hour_end)
        if overlap_start >= overlap_end:
            continue
        if overlap_start > cursor:
            missing = True
        cursor = max(cursor, overlap_end)
        if hour.precipitation_mm is None:
            missing = True
        elif hour.precipitation_mm >= threshold_mm:
            precipitation_start, precipitation_end = _precipitation_window(hour)
            affected.append(
                WeatherInterval(
                    start=precipitation_start.astimezone(zone),
                    end=precipitation_end.astimezone(zone),
                )
            )
    if cursor < end_utc:
        missing = True

    rain_forecast: bool | None
    if affected:
        rain_forecast = True
    elif missing:
        rain_forecast = None
    else:
        rain_forecast = False

    exposure_start_utc = exposure_start.astimezone(timezone.utc)
    exposure_end_utc = exposure_end.astimezone(timezone.utc)
    exposure_has_rain = any(
        interval.start.astimezone(timezone.utc) < exposure_end_utc
        and interval.end.astimezone(timezone.utc) > exposure_start_utc
        for interval in affected
    )
    if exposure_has_rain:
        exposure_overlaps: bool | None = True
    elif _period_has_missing_data(records, exposure_start_utc, exposure_end_utc):
        exposure_overlaps = None
    else:
        exposure_overlaps = False

    return RainAssessment(
        threshold_mm=threshold_mm,
        rain_forecast=rain_forecast,
        affected_intervals=affected,
        first_affected_interval=affected[0] if affected else None,
        exposure_overlaps=exposure_overlaps,
        information_missing=missing,
    )


def _aware_window(
    start: datetime,
    end: datetime,
    zone: ZoneInfo,
) -> tuple[datetime, datetime]:
    if (
        start.tzinfo is None
        or start.utcoffset() is None
        or end.tzinfo is None
        or end.utcoffset() is None
    ):
        raise ValueError("Weather assessment timestamps must be timezone-aware")
    start_local = start.astimezone(zone)
    end_local = end.astimezone(zone)
    if end_local.astimezone(timezone.utc) <= start_local.astimezone(timezone.utc):
        raise ValueError("Weather assessment end must be later than start")
    return start_local, end_local


def _hour_overlaps(at: datetime, start: datetime, end: datetime) -> bool:
    hour_start = at.astimezone(timezone.utc)
    hour_end = hour_start + timedelta(hours=1)
    return (
        hour_start < end.astimezone(timezone.utc)
        and hour_end > start.astimezone(timezone.utc)
    )


def _period_has_missing_data(
    records: list[ForecastHour],
    start: datetime,
    end: datetime,
) -> bool:
    cursor = start
    for hour in records:
        hour_start, hour_end = _precipitation_window(hour)
        overlap_start = max(start, hour_start)
        overlap_end = min(end, hour_end)
        if overlap_start >= overlap_end:
            continue
        if overlap_start > cursor or hour.precipitation_mm is None:
            return True
        cursor = max(cursor, overlap_end)
    return cursor < end


def _precipitation_window(hour: ForecastHour) -> tuple[datetime, datetime]:
    interval_start = hour.precipitation_interval_start
    interval_end = hour.precipitation_interval_end
    if interval_start is not None and interval_end is not None:
        return (
            interval_start.astimezone(timezone.utc),
            interval_end.astimezone(timezone.utc),
        )
    hour_start = hour.at.astimezone(timezone.utc)
    return hour_start, hour_start + timedelta(hours=1)

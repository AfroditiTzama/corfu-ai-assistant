import { CloudRain, CloudSun, Info } from "lucide-react";
import type { RainAssessment, WeatherForecast } from "../../lib/types";
import { Badge } from "../ui/badge";
import { Card } from "../ui/card";

interface WeatherPanelProps {
  forecast: WeatherForecast | null;
  assessment: RainAssessment | null;
  error: string | null;
  wasRequested: boolean;
}

function localTime(value: string, timeZone: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone,
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function valueOrUnknown(value: number | null, unit: string): string {
  return value === null ? "Unavailable" : `${value}${unit}`;
}

export function WeatherPanel({ forecast, assessment, error, wasRequested }: WeatherPanelProps) {
  return (
    <Card role="region" aria-label="Weather information" className="p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-stone-400">
            <CloudSun className="h-4 w-4 text-sea" /> WEATHER IN CORFU
          </div>
          <h3 className="mt-2 font-display text-lg font-bold text-ink">
            {forecast
              ? "Weather outlook"
              : error
                ? "Weather is unavailable"
                : wasRequested
                  ? "No forecast details available"
                  : "Plan around the weather"}
          </h3>
        </div>
        {forecast && (
          <Badge tone={forecast.source === "live" ? "green" : "sand"}>
            {forecast.source === "live" ? "Live forecast" : "Demonstration data"}
          </Badge>
        )}
      </div>

      {!forecast ? (
        <div className="mt-5 flex items-center gap-4 rounded-2xl bg-[#F5F8F6] p-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-sea shadow-sm">
            {error ? <Info className="h-6 w-6 text-amber-700" /> : <CloudSun className="h-6 w-6" />}
          </div>
          <p className={`text-sm leading-6 ${error ? "text-amber-900" : "text-stone-500"}`}>
            {error
              ? "We couldn't retrieve weather just now. Please try again later."
              : wasRequested
                ? "No weather details were available for this request."
                : "Ask for the current weather to see the latest available forecast."}
          </p>
        </div>
      ) : (
        <>
          <div className={`mt-4 rounded-2xl p-4 text-sm leading-6 ${forecast.source === "synthetic_fixture" ? "bg-amber-50 text-amber-900" : "bg-teal-50 text-teal-900"}`}>
            {forecast.source === "synthetic_fixture"
              ? "Demonstration scenario with synthetic weather data — not a live forecast."
              : "Live forecast from Open-Meteo."}
            <p className="mt-1 text-xs opacity-75">{forecast.place}</p>
            <p className="mt-1 text-xs opacity-75">
              Updated: {forecast.retrieved_at ? localTime(forecast.retrieved_at, forecast.timezone) : "Time unavailable"}
            </p>
          </div>

          <div className="mt-4 max-h-64 space-y-2 overflow-y-auto" aria-label="Hourly weather forecast">
            {forecast.hourly.length > 0 ? forecast.hourly.map((hour) => (
              <div key={hour.at} className="grid grid-cols-2 gap-2 rounded-xl border border-stone-100 bg-white p-3 text-xs sm:grid-cols-[1.4fr_1fr_1fr_1fr]">
                <span className="col-span-2 font-semibold text-ink sm:col-span-1">{localTime(hour.at, forecast.timezone)}</span>
                <span className="text-stone-600">Temperature · {valueOrUnknown(hour.temperature_c, "°C")}</span>
                <span className="text-stone-600">Rain · {valueOrUnknown(hour.precipitation_mm, " mm")}</span>
                <span className="text-stone-600">Rain chance · {valueOrUnknown(hour.precipitation_probability_percent, "%")}</span>
              </div>
            )) : (
              <p className="rounded-xl border border-stone-100 bg-white p-3 text-xs text-stone-500">
                No hourly details were provided.
              </p>
            )}
          </div>

          {assessment && (
            <div className="mt-4 border-t border-stone-100 pt-4">
              <div className="flex items-center gap-2">
                <CloudRain className="h-4 w-4 text-sea" />
                <p className="text-sm font-semibold text-ink">Rain during your plans</p>
                <Badge tone={assessment.rain_forecast === true ? "sand" : "neutral"}>
                  {assessment.rain_forecast === null ? "Unclear" : assessment.rain_forecast ? "Rain expected" : "No rain indicated"}
                </Badge>
              </div>
              {assessment.information_missing && (
                <p className="mt-2 text-xs leading-5 text-stone-500">Some forecast details are unavailable.</p>
              )}
              {assessment.exposure_overlaps !== null && (
                <p className="mt-2 text-xs text-stone-500">
                  {assessment.exposure_overlaps
                    ? "Rain overlaps planned outdoor stops."
                    : "Rain does not overlap planned outdoor stops."}
                </p>
              )}
              {assessment.affected_intervals.length > 0 ? (
                <ul className="mt-2 space-y-1">
                  {assessment.affected_intervals.map((interval) => (
                    <li key={`${interval.start}-${interval.end}`} className="text-xs text-stone-600">
                      {localTime(interval.start, forecast.timezone)} – {localTime(interval.end, forecast.timezone)}
                    </li>
                  ))}
                </ul>
              ) : assessment.rain_forecast === null ? (
                <p className="mt-2 flex gap-1.5 text-xs text-stone-500"><Info className="h-3.5 w-3.5 shrink-0" /> Rain conditions could not be determined for the full period.</p>
              ) : assessment.rain_forecast === false ? (
                <p className="mt-2 text-xs text-stone-500">Rain is not indicated during the forecast period.</p>
              ) : null}
            </div>
          )}
        </>
      )}
    </Card>
  );
}

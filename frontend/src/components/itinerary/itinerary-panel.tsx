import { useMemo } from "react";
import { AlertTriangle, ArrowRight, Clock3, Compass, MapPin, Route } from "lucide-react";
import { pois } from "../../lib/dataset";
import type { PlanningResult, WeatherReplanning } from "../../lib/types";
import { formatDuration } from "../../lib/utils";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card } from "../ui/card";

interface ItineraryPanelProps {
  planningResult: PlanningResult | null;
  weatherReplanning: WeatherReplanning | null;
}

function formatTime(value: string, timeZone: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatDateTime(value: string, timeZone: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone,
    day: "numeric",
    month: "short",
  }).format(new Date(value));
}

function statusLabel(status: PlanningResult["status"]): string {
  if (status === "feasible") return "Ready to explore";
  if (status === "uncertain") return "Hours to confirm";
  return "Needs adjustment";
}

function friendlyText(value: string, poiById: Map<string, (typeof pois)[number]>): string {
  return value
    .replace(/\bCF-\d{3}\b/gi, (id) => poiById.get(id.toUpperCase())?.name ?? "a place")
    .replace(/opening_hours_fixture/gi, "opening hours")
    .replace(/exercise fixture/gi, "planning data")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function messageKey(value: string): string {
  return value.toLocaleLowerCase().replace(/\s+/g, " ").trim();
}

function uniqueMessages(values: string[], seen: Set<string>): string[] {
  return values.filter((value) => {
    const key = messageKey(value);
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function visibleStopReason(value: string | null | undefined): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  return trimmed === "Selected from eligible candidates." ? null : trimmed;
}

function poiName(poiId: string, poiById: Map<string, (typeof pois)[number]>, fallback: string): string {
  return poiById.get(poiId)?.name ?? fallback;
}

function displayLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

export function ItineraryPanel({ planningResult, weatherReplanning }: ItineraryPanelProps) {
  const itinerary = planningResult?.itinerary ?? null;
  const poiById = useMemo(() => new Map(pois.map((poi) => [poi.id, poi])), []);
  const transfersByPair = useMemo(
    () => new Map((planningResult?.transfers ?? []).map((transfer) => [
      `${transfer.from_poi_id}:${transfer.to_poi_id}`,
      transfer,
    ])),
    [planningResult],
  );
  const waitingByPoiId = useMemo(
    () => new Map((planningResult?.waiting_periods ?? []).map((wait) => [wait.poi_id, wait])),
    [planningResult],
  );
  const seenValidationMessages = new Set<string>();
  const warningMessages = uniqueMessages(
    (planningResult?.validation.warnings ?? []).map((warning) => friendlyText(warning, poiById)),
    seenValidationMessages,
  );
  const issueMessages = uniqueMessages(
    (planningResult?.validation.issues ?? []).map((issue) => friendlyText(issue.message, poiById)),
    seenValidationMessages,
  );
  const explanationMessages = uniqueMessages(
    (planningResult?.explanations ?? []).map((explanation) => friendlyText(explanation, poiById)),
    seenValidationMessages,
  );

  return (
    <Card role="region" aria-label="Itinerary plan" className="overflow-hidden lg:min-h-[470px]">
      <div className="border-b border-stone-100 p-5 sm:p-6">
        <div className="mb-4 flex items-center justify-between">
          {planningResult ? (
            <Badge tone={planningResult.status === "feasible" ? "green" : planningResult.status === "uncertain" ? "sand" : "neutral"}>
              <Route className="h-3 w-3" /> {statusLabel(planningResult.status)}
            </Badge>
          ) : (
            <span className="inline-flex items-center gap-1.5 text-xs font-medium text-stone-500">
              <Route className="h-3.5 w-3.5 text-sea" /> Plan your day
            </span>
          )}
          {itinerary && <Badge><MapPin className="h-3 w-3" /> {itinerary.stops.length} stops</Badge>}
        </div>
        <h2 className="font-display text-xl font-bold tracking-tight text-ink sm:text-2xl">Your itinerary</h2>
        <p className="mt-1.5 text-sm text-stone-500">
          {itinerary
            ? `${formatDateTime(itinerary.starts_at, itinerary.timezone)} · ${formatTime(itinerary.starts_at, itinerary.timezone)}–${formatTime(itinerary.ends_at, itinerary.timezone)} local time`
            : "A clear route for your time in Corfu."}
        </p>
        {planningResult?.status === "uncertain" && (
          <p className="mt-3 rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-900">
            Some opening hours need confirmation. Please check before you go.
          </p>
        )}
        {planningResult?.status === "infeasible" && (
          <p className="mt-3 rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-900">
            This route cannot fit the requested constraints as planned. Here is what needs to change.
          </p>
        )}
      </div>

      {itinerary && itinerary.stops.length > 0 ? (
        <div className="max-h-[420px] overflow-y-auto px-5 py-3 sm:px-6">
          <div className="relative">
            <div className="absolute bottom-7 left-[11px] top-5 w-px bg-stone-200" />
            <div className="space-y-1">
              {itinerary.stops.map((stop, index) => {
                const poi = poiById.get(stop.poi_id);
                const previous = itinerary.stops[index - 1];
                const transfer = previous && transfersByPair.get(`${previous.poi_id}:${stop.poi_id}`);
                const waitingPeriod = waitingByPoiId.get(stop.poi_id);
                const minutes = Math.max(0, Math.round(
                  (new Date(stop.ends_at).getTime() - new Date(stop.starts_at).getTime()) / 60_000,
                ));
                const stopReason = visibleStopReason(stop.reason);
                return (
                  <div key={`${stop.poi_id}-${stop.starts_at}`}>
                    {transfer && (
                      <div className="ml-9 flex items-center gap-2 py-2 text-[11px] text-stone-500">
                        <ArrowRight className="h-3 w-3 text-sea" />
                        Walk · {formatDuration(transfer.estimate.minutes)} · {transfer.estimate.estimated_walking_distance_km.toFixed(1)} km estimate
                      </div>
                    )}
                    {waitingPeriod && (
                      <div className="ml-9 flex items-center gap-2 py-2 text-[11px] text-stone-500">
                        <Clock3 className="h-3 w-3 text-sea" />
                        Wait for opening window · {formatDuration(Math.round(
                          (new Date(waitingPeriod.ends_at).getTime() - new Date(waitingPeriod.starts_at).getTime()) / 60_000,
                        ))}
                      </div>
                    )}
                    <div className="relative flex gap-3 py-3">
                      <div className="z-10 mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 border-white bg-teal-100 text-[10px] font-bold text-teal-800 ring-1 ring-teal-100">
                        {index + 1}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                          <h3 className="truncate text-sm font-semibold text-ink">{poi?.name ?? "Place details unavailable"}</h3>
                          <span className="font-mono text-[11px] font-semibold text-sea">
                            {formatTime(stop.starts_at, itinerary.timezone)} – {formatTime(stop.ends_at, itinerary.timezone)}
                          </span>
                        </div>
                        {poi?.area && <p className="mt-1 text-xs text-stone-500">{poi.area}</p>}
                        {stopReason && <p className="mt-2 text-xs leading-5 text-stone-500">{friendlyText(stopReason, poiById)}</p>}
                        <div className="mt-2 flex flex-wrap items-center gap-1.5">
                          {poi && <Badge tone="teal">{displayLabel(poi.category)}</Badge>}
                          {poi && <Badge>{poi.setting === "mixed" ? "Indoor & outdoor" : poi.setting}</Badge>}
                          <span className="ml-auto inline-flex items-center gap-1 text-[11px] text-stone-500">
                            <Clock3 className="h-3 w-3" /> {formatDuration(minutes)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        <div className="mx-5 my-5 flex items-center gap-3 rounded-2xl bg-[#F5F8F6] p-4 sm:mx-6">
          <Compass className="h-5 w-5 shrink-0 text-sea" />
          <p className="text-sm leading-6 text-stone-500">
            {planningResult?.status === "infeasible"
              ? "No stops fit the request as written. Review the details below."
              : planningResult
                ? "No stops were returned for this plan."
                : "Ask for a route to see your stops and estimated walking times."}
          </p>
        </div>
      )}

      {planningResult && (
        <div className="space-y-3 border-t border-stone-100 p-5 sm:p-6">
          {warningMessages.map((warning) => (
            <p key={`warning-${messageKey(warning)}`} className="flex gap-2 text-xs leading-5 text-amber-800">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {warning}
            </p>
          ))}
          {issueMessages.map((issue) => (
            <p key={`issue-${messageKey(issue)}`} className="flex gap-2 text-xs leading-5 text-rose-700">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" /> {issue}
            </p>
          ))}
          {explanationMessages.map((explanation) => (
            <p key={`explanation-${messageKey(explanation)}`} className="text-xs leading-5 text-stone-500">{explanation}</p>
          ))}
        </div>
      )}

      {weatherReplanning && (
        <div className="border-t border-stone-100 bg-teal-50/50 p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-teal-800">
            Weather update · {statusLabel(weatherReplanning.status)}
          </p>
          {weatherReplanning.changed_stops.map((change) => (
            <p key={`${change.stop_index}-${change.previous_poi_id}`} className="mt-2 text-xs text-stone-600">
              Stop {change.stop_index + 1}: {poiName(change.previous_poi_id, poiById, "previous place")} → {poiName(change.revised_poi_id, poiById, "updated place")}
            </p>
          ))}
          {weatherReplanning.removed_poi_ids
            .filter((id) => !weatherReplanning.changed_stops.some((change) => change.previous_poi_id === id))
            .map((id) => (
            <p key={id} className="mt-2 text-xs text-stone-600">Removed stop: {poiName(id, poiById, "Place details unavailable")}</p>
            ))}
          {weatherReplanning.reordered && (
            <p className="mt-2 text-xs text-stone-600">The existing stops were reordered.</p>
          )}
          {weatherReplanning.conditional_exposure_poi_ids.length > 0 && (
            <p className="mt-2 text-xs leading-5 text-amber-800">
              Outdoor conditions still need consideration at: {weatherReplanning.conditional_exposure_poi_ids.map((id) => poiName(id, poiById, "a place")).join(", ")}.
            </p>
          )}
          {weatherReplanning.explanations.map((explanation, index) => (
            <p key={`weather-explanation-${index}`} className="mt-2 text-xs leading-5 text-stone-600">{friendlyText(explanation, poiById)}</p>
          ))}
        </div>
      )}

      <div className="border-t border-stone-100 px-5 py-4 sm:px-6">
        <Button variant="soft" size="sm" onClick={() => document.getElementById("explore")?.scrollIntoView({ behavior: "smooth" })}>
          Explore places <ArrowRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </Card>
  );
}

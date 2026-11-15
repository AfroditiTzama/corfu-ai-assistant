import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { pois } from "./lib/dataset";
import { getPoiImage } from "./lib/poi-image-manifest";
import type { OrchestrationResult } from "./lib/types";

const conversationId = "f6757de4-f389-42e5-ac24-9e86f6b3ee77";
const secondConversationId = "585cdb1a-30ad-41d7-8fd2-25577ed5d726";

function result(
  overrides: Partial<OrchestrationResult> = {},
): OrchestrationResult {
  return {
    conversation_id: conversationId,
    intent: {
      intent: "tourism_qa",
      field_actions: {
        requested_date: "preserve",
        start_time: "preserve",
        end_time: "preserve",
        timezone: "preserve",
        semantic_interests: "preserve",
        filter_category: "preserve",
        filter_tags: "preserve",
        filter_setting: "preserve",
        filter_family_suitability: "preserve",
        travel_mode: "preserve",
        family_required: "preserve",
        child_ages: "preserve",
        itinerary_modification: "preserve",
        question: "set",
        clarification_question: "preserve",
        weather_request: "preserve",
      },
    },
    status: "completed",
    response: "The available description identifies Old Fortress as a fortified landmark in Corfu Town. [CF-001]",
    relevant_poi_ids: ["CF-001"],
    evidence: [{
      poi_id: "CF-001",
      name: "Old Fortress",
      description: "A fortified landmark in Corfu Town.",
      source_urls: ["https://example.org/fortress"],
      description_provenance: "Curated dataset description",
      method: "semantic",
      relevance_score: 0.83,
    }],
    planning_result: null,
    modification_applied: null,
    weather_forecast: null,
    rain_assessment: null,
    weather_replanning: null,
    weather_error: null,
    conversation_state: {
      conversation_id: conversationId,
      timezone: "Europe/Athens",
      preferences: {
        interests: [],
        excluded_categories: [],
        excluded_tags: [],
        preferred_setting: null,
        family_required: null,
        child_ages: [],
      },
      hard_filters: {
        category: null,
        tags: null,
        setting: null,
        family_suitability_fixture: null,
      },
      travel_mode: null,
      active_request: null,
      itinerary: null,
      pending_modification: null,
      replacement_family_requirements: null,
      last_modification_outcome: null,
      updated_at: "2026-10-06T08:00:00+03:00",
    },
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  } as Response;
}

function setupFetch(fetchMock: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("fetch", fetchMock);
}

async function submitMessage(text: string) {
  fireEvent.change(screen.getByRole("textbox", { name: "Message your Corfu guide" }), {
    target: { value: text },
  });
  await waitFor(() => expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

describe("Corfu assistant API integration", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses a compact Corfu identity and functional introduction above the workspace", () => {
    render(<App />);

    expect(screen.getByRole("link", { name: "Corfu AI Travel Assistant home" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Explore Corfu", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Discover attractions, plan itineraries and check the weather.")).toBeInTheDocument();
    expect(screen.queryByText("Your Corfu story")).not.toBeInTheDocument();
    expect(screen.queryByText("An island worth getting lost in")).not.toBeInTheDocument();
    expect(screen.queryByText("50 curated places")).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Itinerary plan" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Weather information" })).toBeInTheDocument();
    expect(screen.queryByText(/Yassas/i)).not.toBeInTheDocument();
    expect(screen.getByText(/Hello and welcome to Corfu!/)).toBeInTheDocument();
    expect(screen.getByText(/I'm your AI travel guide/)).toBeInTheDocument();
    expect(screen.getByText(/What would you like to explore today\?/)).toBeInTheDocument();
    expect(screen.getByText("A few ideas")).toBeInTheDocument();
  });

  it("sends real chat messages and reuses the conversation ID across turns", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(result()))
      .mockResolvedValueOnce(jsonResponse(result({
        response: "The route can be adapted for a child.",
      })));
    setupFetch(fetchMock);
    render(<App />);

    await submitMessage("Tell me about Old Fortress");
    expect(screen.queryByText(/Hello and welcome to Corfu!/)).not.toBeInTheDocument();
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
    await submitMessage("Make it suitable for a child");
    await screen.findByText("The route can be adapted for a child.");

    const first = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    const second = JSON.parse(String(fetchMock.mock.calls[1][1].body));
    expect(first.message).toBe("Tell me about Old Fortress");
    expect(first.message).not.toContain("Hello and welcome to Corfu!");
    expect(first.request_id).toBeTruthy();
    expect(first.conversation_id).toBeUndefined();
    expect(second.conversation_id).toBe(conversationId);
    expect(second.request_id).not.toBe(first.request_id);
  });

  it("renders API evidence and accessible source attribution", async () => {
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result())));
    render(<App />);
    await submitMessage("Tell me about Old Fortress");

    expect(await screen.findByText(
      "The available description identifies Old Fortress as a fortified landmark in Corfu Town.",
    )).toBeInTheDocument();
    expect(screen.queryByText("A fortified landmark in Corfu Town.")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain("CF-001");
    expect(screen.getByRole("link", { name: /Background information about Old Fortress from example.org/ })).toHaveAttribute(
      "href",
      "https://example.org/fortress",
    );
    fireEvent.click(within(screen.getByRole("region", { name: "Sources and places mentioned" }))
      .getByRole("button", { name: "Open Old Fortress details" }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });

  it("renders grounded recommendations as photo cards without technical labels or IDs", async () => {
    const recommendation = result({
      intent: { ...result().intent, intent: "poi_recommendation" },
      response: "The available description identifies Old Fortress as a fortified landmark in Corfu Town.",
    });
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(recommendation)));
    render(<App />);
    await submitMessage("Recommend a historic place");

    const recommendations = await screen.findByRole("region", { name: "Recommended for you" });
    expect(screen.getByText(
      "The available description identifies Old Fortress as a fortified landmark in Corfu Town.",
    )).toBeInTheDocument();
    expect(within(recommendations).getByRole("heading", { name: "Old Fortress" })).toBeInTheDocument();
    expect(within(recommendations).queryByText("A fortified landmark in Corfu Town.")).not.toBeInTheDocument();
    expect(within(recommendations).getByRole("img", { name: /Old Fortress rising/ })).toHaveAttribute(
      "src",
      "/images/poi/cf-001-old-fortress.jpg",
    );
    expect(within(recommendations).getByRole("link", { name: /Photo attribution for Old Fortress/ })).toHaveAttribute(
      "href",
      "https://commons.wikimedia.org/wiki/File:The_Old_Fortress_and_the_Old_Town_of_Corfu_-_September_2017.jpg",
    );
    expect(within(recommendations).getByRole("link", { name: /CC BY-SA 4.0 license for Old Fortress photo/ })).toHaveAttribute(
      "href",
      "https://creativecommons.org/licenses/by-sa/4.0/",
    );
    expect(document.body.textContent).not.toMatch(/Dataset matches|semantic match|CF-001|poi_recommendation|completed/);
    expect(screen.queryByText("API mode")).not.toBeInTheDocument();
    expect(screen.queryByText(/FastAPI assistant|grounded answers/i)).not.toBeInTheDocument();
  });

  it("orders selected recommendations and separates remaining retrieved places without duplicates", async () => {
    const evidence = ["CF-001", "CF-002", "CF-003"].map((poiId) => {
      const poi = pois.find((item) => item.id === poiId)!;
      return {
        poi_id: poi.id,
        name: poi.name,
        description: poi.description,
        source_urls: poi.source_urls,
        description_provenance: poi.description_provenance,
        method: "semantic" as const,
        relevance_score: 0.8,
      };
    });
    const recommendation = result({
      intent: { ...result().intent, intent: "poi_recommendation" },
      evidence,
      recommended_poi_ids: ["CF-002", "CF-001", "CF-002"],
    });
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(recommendation)));
    render(<App />);
    await submitMessage("Recommend two historic places");

    const selected = await screen.findByRole("region", { name: "Recommended for you" });
    const more = screen.getByRole("region", { name: "More places to explore" });
    expect(within(selected).getAllByRole("heading", { level: 3 }).map((node) => node.textContent))
      .toEqual(["New Fortress", "Old Fortress"]);
    expect(within(more).getAllByRole("heading", { level: 3 }).map((node) => node.textContent))
      .toEqual([pois.find((poi) => poi.id === "CF-003")!.name]);
    const selectedNames = within(selected).getAllByRole("heading", { level: 3 })
      .map((node) => node.textContent);
    const moreNames = within(more).getAllByRole("heading", { level: 3 })
      .map((node) => node.textContent);
    expect(new Set(selectedNames).size).toBe(selectedNames.length);
    expect(selectedNames.some((name) => moreNames.includes(name))).toBe(false);
  });

  it("uses category artwork for unphotographed places and falls back when an image fails", async () => {
    const unphotographed = pois.find((poi) => getPoiImage(poi.id) === undefined);
    expect(unphotographed).toBeDefined();
    render(<App />);
    const explore = within(screen.getByRole("region", { name: "Explore places in Corfu" }));
    const search = explore.getByRole("textbox", { name: "Search Corfu places" });

    fireEvent.change(search, { target: { value: "Old Fortress" } });
    const oldFortressImage = await explore.findByRole("img", { name: /Old Fortress rising/ });
    fireEvent.error(oldFortressImage);
    expect(await explore.findByRole("img", { name: "Old Fortress illustration" })).toBeInTheDocument();

    fireEvent.change(search, { target: { value: unphotographed!.name } });
    expect(await explore.findByRole("img", { name: `${unphotographed!.name} illustration` })).toBeInTheDocument();
  });

  it("preserves the original request ID when retrying a failed logical message", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(jsonResponse(result()));
    setupFetch(fetchMock);
    render(<App />);
    await submitMessage("Tell me about Old Fortress");
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");

    const first = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    const retry = JSON.parse(String(fetchMock.mock.calls[1][1].body));
    expect(retry).toEqual(first);
  });

  it("prevents duplicate submissions while a request is pending", async () => {
    let resolveRequest!: (response: Response) => void;
    const fetchMock = vi.fn(() => new Promise<Response>((resolve) => {
      resolveRequest = resolve;
    }));
    setupFetch(fetchMock);
    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Message your Corfu guide" }), {
      target: { value: "Find a museum" },
    });
    const form = screen.getByRole("button", { name: "Send message" }).closest("form");
    fireEvent.submit(form!);
    fireEvent.submit(form!);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => resolveRequest(jsonResponse(result())));
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");
  });

  it("starts a fresh conversation with a new request ID", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(result()))
      .mockResolvedValueOnce(jsonResponse(result({ conversation_id: secondConversationId })));
    setupFetch(fetchMock);
    render(<App />);
    await submitMessage("First question");
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");
    fireEvent.click(screen.getByRole("button", { name: "Start a new conversation" }));
    await submitMessage("A new question");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    const first = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    const second = JSON.parse(String(fetchMock.mock.calls[1][1].body));
    expect(second.conversation_id).toBeUndefined();
    expect(second.request_id).not.toBe(first.request_id);
  });

  it("deletes the current backend conversation", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(result()))
      .mockResolvedValueOnce(jsonResponse(null, 204));
    setupFetch(fetchMock);
    render(<App />);
    await submitMessage("Start a conversation");
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");
    fireEvent.click(screen.getByRole("button", { name: "Delete conversation" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(String(fetchMock.mock.calls[1][0])).toBe(
      `http://127.0.0.1:8000/conversations/${conversationId}`,
    );
    expect(fetchMock.mock.calls[1][1].method).toBe("DELETE");
    await waitFor(() => expect(screen.queryByRole("button", { name: "Delete conversation" })).not.toBeInTheDocument());
  });

  it("explains an expired conversation and offers a fresh chat", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(result()))
      .mockResolvedValueOnce(jsonResponse({ detail: "Conversation not found" }, 404));
    setupFetch(fetchMock);
    render(<App />);
    await submitMessage("Start a conversation");
    await screen.findByText("The available description identifies Old Fortress as a fortified landmark in Corfu Town.");
    await submitMessage("Continue the conversation");

    expect(await screen.findByText("This stored conversation has expired or was lost after a backend restart.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start a new chat" })).toBeInTheDocument();
  });

  it("renders a returned uncertain itinerary without claiming feasibility", async () => {
    const planning = {
      status: "uncertain" as const,
      itinerary: {
        conversation_id: conversationId,
        timezone: "Europe/Athens",
        starts_at: "2026-10-06T11:00:00+03:00",
        ends_at: "2026-10-06T15:00:00+03:00",
        stops: [{
          poi_id: "CF-001",
          starts_at: "2026-10-06T11:00:00+03:00",
          ends_at: "2026-10-06T12:00:00+03:00",
          reason: "Historical interest",
        }, {
          poi_id: "CF-002",
          starts_at: "2026-10-06T12:20:00+03:00",
          ends_at: "2026-10-06T13:10:00+03:00",
          reason: null,
        }],
      },
      transfers: [{
        from_poi_id: "CF-001",
        to_poi_id: "CF-002",
        starts_at: "2026-10-06T12:00:00+03:00",
        ends_at: "2026-10-06T12:10:00+03:00",
        estimate: {
          straight_line_distance_km: 0.5,
          estimated_walking_distance_km: 0.6,
          minutes: 10,
        },
      }],
      waiting_periods: [{
        poi_id: "CF-002",
        starts_at: "2026-10-06T12:10:00+03:00",
        ends_at: "2026-10-06T12:20:00+03:00",
        reason: "opening_hours" as const,
      }],
      validation: { is_valid: true, issues: [], warnings: ["Hours are unknown."] },
      explanations: [
        "This plan is conditional.",
        "Scheduled visits finish at 13:10, leaving 110 minutes unallocated in the requested time window.",
      ],
      selected_poi_ids: ["CF-001"],
    };
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result({
      intent: {
        intent: "itinerary_creation",
        field_actions: {
          requested_date: "set", start_time: "set", end_time: "set", timezone: "preserve",
          semantic_interests: "set", filter_category: "preserve", filter_tags: "preserve",
          filter_setting: "preserve", filter_family_suitability: "preserve",
          travel_mode: "set", family_required: "preserve", child_ages: "preserve",
          itinerary_modification: "preserve", question: "preserve",
          clarification_question: "preserve", weather_request: "preserve",
        },
      },
      status: "uncertain",
      response: "This plan is conditional.",
      planning_result: planning,
    }))));
    render(<App />);
    await submitMessage("Plan a historic walk");

    expect(await screen.findByText(/Some opening hours need confirmation/)).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Itinerary plan" })).getByRole("heading", { name: "Old Fortress" })).toBeInTheDocument();
    expect(screen.getByText("Hours are unknown.")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Itinerary plan" })).getByText("This plan is conditional.")).toBeInTheDocument();
    expect(screen.getByText("Scheduled visits finish at 13:10, leaving 110 minutes unallocated in the requested time window.")).toBeInTheDocument();
    expect(screen.getByText(/Walk · 10 min/)).toBeInTheDocument();
    expect(screen.getByText("Wait for opening window · 10 min")).toBeInTheDocument();
  });

  it("shows an infeasible planning explanation and does not invent stops", async () => {
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result({
      status: "infeasible",
      response: "Five stops cannot fit in the requested window.",
      planning_result: {
        status: "infeasible",
        itinerary: null,
        transfers: [],
        waiting_periods: [],
        validation: {
          is_valid: false,
          issues: [{ code: "time_window", message: "The requested end time cannot be met.", poi_id: null }],
          warnings: [],
        },
        explanations: ["Reduce the number of requested stops."],
        selected_poi_ids: [],
      },
    }))));
    render(<App />);
    await submitMessage("Plan five stops in 90 minutes");
    expect(await screen.findByText("The itinerary cannot meet all requested constraints. Review the validation details in the plan.")).toBeInTheDocument();
    expect(screen.getByText("The requested end time cannot be met.")).toBeInTheDocument();
    expect(screen.getByText("Reduce the number of requested stops.")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Itinerary plan" })).queryByRole("heading", { name: "Old Fortress" })).not.toBeInTheDocument();
  });

  it("renders duplicated validation explanations once while preserving distinct guidance", async () => {
    const durationIssue = "Required visit durations total 355 minutes, exceeding the 90-minute window before travel.";
    const closureIssue = "Required POI New Fortress is closed on the requested date.";
    const planning = {
      status: "infeasible" as const,
      itinerary: null,
      transfers: [],
      waiting_periods: [],
      validation: {
        is_valid: false,
        issues: [
          { code: "insufficient_overall_time", message: durationIssue, poi_id: null },
          { code: "poi_closed", message: closureIssue, poi_id: "CF-002" },
        ],
        warnings: [],
      },
      explanations: [durationIssue, closureIssue, "Try a shorter route or a different day."],
      selected_poi_ids: [],
    };
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result({
      status: "infeasible",
      response: "This route cannot be completed as requested.",
      planning_result: planning,
    }))));
    render(<App />);
    await submitMessage("Plan five stops in 90 minutes");
    await screen.findByText("The itinerary cannot meet all requested constraints. Review the validation details in the plan.");

    const panel = within(screen.getByRole("region", { name: "Itinerary plan" }));
    expect(screen.getAllByText(durationIssue)).toHaveLength(1);
    expect(screen.getAllByText(closureIssue)).toHaveLength(1);
    expect(panel.getByText("Try a shorter route or a different day.")).toBeInTheDocument();
    expect(planning.validation.issues).toHaveLength(2);
  });

  it("distinguishes a synthetic fixture from live weather and keeps null values unknown", async () => {
    const weatherResult = result({
      intent: { intent: "live_weather_query", field_actions: result().intent.field_actions },
      status: "completed",
      response: "Forecast returned.",
      evidence: [],
      relevant_poi_ids: [],
      weather_forecast: {
        place: "Corfu",
        timezone: "Europe/Athens",
        source: "synthetic_fixture",
        retrieved_at: "2026-10-07T12:00:00+03:00",
        hourly: [{
          at: "2026-10-07T15:00:00+03:00",
          temperature_c: null,
          precipitation_mm: 3.2,
          precipitation_probability_percent: null,
          weather_code: null,
          precipitation_interval_start: "2026-10-07T15:00:00+03:00",
          precipitation_interval_end: "2026-10-07T16:00:00+03:00",
        }],
      },
      rain_assessment: {
        threshold_mm: 0.1,
        rain_forecast: true,
        affected_intervals: [{
          start: "2026-10-07T15:00:00+03:00",
          end: "2026-10-07T16:00:00+03:00",
        }],
        first_affected_interval: {
          start: "2026-10-07T15:00:00+03:00",
          end: "2026-10-07T16:00:00+03:00",
        },
        exposure_overlaps: true,
        information_missing: false,
      },
    });
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(weatherResult)));
    render(<App />);
    await submitMessage("Tell me about the forecast");

    expect(await screen.findByText("Demonstration data")).toBeInTheDocument();
    expect(screen.getByText(/not a live forecast/i)).toBeInTheDocument();
    expect(screen.getByText("Temperature · Unavailable")).toBeInTheDocument();
    expect(screen.getByText("Rain · 3.2 mm")).toBeInTheDocument();
    expect(screen.getByText("Rain expected")).toBeInTheDocument();
  });

  it("labels live weather as live and shows values supplied by the backend", async () => {
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result({
      intent: { intent: "live_weather_query", field_actions: result().intent.field_actions },
      response: "Live forecast returned.",
      evidence: [],
      relevant_poi_ids: [],
      weather_forecast: {
        place: "Corfu",
        timezone: "Europe/Athens",
        source: "live",
        retrieved_at: "2026-10-08T10:00:00+03:00",
        hourly: [
          {
            at: "2026-10-08T11:00:00+03:00",
            temperature_c: 22,
            precipitation_mm: 0,
            precipitation_probability_percent: 5,
            weather_code: 1,
            precipitation_interval_start: null,
            precipitation_interval_end: null,
          },
          {
            at: "2026-10-08T12:00:00+03:00",
            temperature_c: 23,
            precipitation_mm: 1.7,
            precipitation_probability_percent: 60,
            weather_code: 61,
            precipitation_interval_start: null,
            precipitation_interval_end: null,
          },
        ],
      },
    }))));
    render(<App />);
    await submitMessage("What is the weather now?");

    expect(await screen.findByText("Live forecast")).toBeInTheDocument();
    expect(screen.getByText("Live forecast from Open-Meteo.")).toBeInTheDocument();
    expect(screen.getByText("Temperature · 22°C")).toBeInTheDocument();
    expect(screen.getByText("Temperature · 23°C")).toBeInTheDocument();
    expect(screen.getByText("Rain · 1.7 mm")).toBeInTheDocument();
    expect(screen.queryByText("Code 1")).not.toBeInTheDocument();
    expect(screen.queryByText(/not a live observation/)).not.toBeInTheDocument();
  });

  it("shows a typed weather-unavailable result without substituting fixture values", async () => {
    setupFetch(vi.fn().mockResolvedValue(jsonResponse(result({
      intent: { intent: "live_weather_query", field_actions: result().intent.field_actions },
      status: "weather_unavailable",
      response: "Live weather is currently unavailable.",
      evidence: [],
      relevant_poi_ids: [],
      weather_error: "The live weather provider is unavailable.",
    }))));
    render(<App />);
    await submitMessage("What is the current weather?");

    const weather = within(screen.getByRole("region", { name: "Weather information" }));
    expect(await weather.findByText("Weather is unavailable")).toBeInTheDocument();
    expect(weather.getByText(/We couldn't retrieve weather just now/)).toBeInTheDocument();
    expect(screen.queryByText("Demonstration data")).not.toBeInTheDocument();
  });

  it("keeps the supplied Explore dataset and POI detail dialog", async () => {
    render(<App />);
    const explore = within(screen.getByRole("region", { name: "Explore places in Corfu" }));
    const search = explore.getByRole("textbox", { name: "Search Corfu places" });
    fireEvent.change(search, { target: { value: "Asian Art" } });
    const detailsButton = await explore.findByRole("button", { name: "Open Museum of Asian Art details" });
    fireEvent.click(detailsButton);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText(/Planning schedule only; confirm current hours before visiting/i)).toBeInTheDocument();
    expect(screen.getByText(/Marked suitable in the planning data; not a safety guarantee/i)).toBeInTheDocument();
  });
});

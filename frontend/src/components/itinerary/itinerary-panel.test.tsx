import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ItineraryPanel } from "./itinerary-panel";
import type { PlanningResult, WeatherReplanning } from "../../lib/types";

describe("ItineraryPanel weather change summary", () => {
  it("hides generic selection text and renders each venue-specific hours warning once", () => {
    const churchWarning = "Opening hours for Saint Spyridon Church are unknown in the exercise fixture; this stop is conditional, not confirmed open.";
    const galleryWarning = "Opening hours for Lotos / Alcia Art Space are unknown in the exercise fixture; this stop is conditional, not confirmed open.";
    const planning: PlanningResult = {
      status: "uncertain",
      itinerary: {
        timezone: "Europe/Athens",
        starts_at: "2026-10-05T13:00:00+03:00",
        ends_at: "2026-10-05T15:05:00+03:00",
        stops: [
          {
            poi_id: "CF-004",
            starts_at: "2026-10-05T13:00:00+03:00",
            ends_at: "2026-10-05T13:45:00+03:00",
            reason: "Selected from eligible candidates.",
          },
          {
            poi_id: "CF-045",
            starts_at: "2026-10-05T14:05:00+03:00",
            ends_at: "2026-10-05T15:05:00+03:00",
            reason: "Strong match for the requested art theme.",
          },
        ],
      },
      transfers: [],
      waiting_periods: [],
      validation: {
        is_valid: false,
        issues: [
          { code: "unknown_opening_hours", message: churchWarning, poi_id: "CF-004" },
          { code: "unknown_opening_hours", message: galleryWarning, poi_id: "CF-045" },
        ],
        warnings: [churchWarning, churchWarning, galleryWarning],
      },
      explanations: [churchWarning, galleryWarning, "Confirm access directly before visiting."],
      selected_poi_ids: ["CF-004", "CF-045"],
    };

    render(<ItineraryPanel planningResult={planning} weatherReplanning={null} />);

    expect(screen.queryByText("Selected from eligible candidates.")).not.toBeInTheDocument();
    expect(screen.getByText("Strong match for the requested art theme.")).toBeInTheDocument();
    expect(screen.getAllByText(/Opening hours for Saint Spyridon Church are unknown/)).toHaveLength(1);
    expect(screen.getAllByText(/Opening hours for Lotos \/ Alcia Art Space are unknown/)).toHaveLength(1);
    expect(screen.getByText("Confirm access directly before visiting.")).toBeInTheDocument();
  });

  it("numbers replacements by revised route position and avoids duplicate removal labels", () => {
    const planning: PlanningResult = {
      status: "feasible",
      itinerary: {
        timezone: "Europe/Athens",
        starts_at: "2026-10-07T13:00:00+03:00",
        ends_at: "2026-10-07T17:00:00+03:00",
        stops: [
          { poi_id: "CF-032", starts_at: "2026-10-07T13:00:00+03:00", ends_at: "2026-10-07T13:25:00+03:00" },
          { poi_id: "CF-016", starts_at: "2026-10-07T13:28:00+03:00", ends_at: "2026-10-07T13:48:00+03:00" },
          { poi_id: "CF-009", starts_at: "2026-10-07T13:49:00+03:00", ends_at: "2026-10-07T14:09:00+03:00" },
          { poi_id: "CF-003", starts_at: "2026-10-07T14:15:00+03:00", ends_at: "2026-10-07T14:35:00+03:00" },
          { poi_id: "CF-014", starts_at: "2026-10-07T14:41:00+03:00", ends_at: "2026-10-07T15:36:00+03:00" },
          { poi_id: "CF-007", starts_at: "2026-10-07T15:41:00+03:00", ends_at: "2026-10-07T16:36:00+03:00" },
        ],
      },
      transfers: [],
      waiting_periods: [],
      validation: { is_valid: true, issues: [], warnings: [] },
      explanations: [],
      selected_poi_ids: ["CF-032", "CF-016", "CF-009", "CF-003", "CF-014", "CF-007"],
    };
    const weather: WeatherReplanning = {
      status: "feasible",
      planning_result: planning,
      original_poi_ids: ["CF-032", "CF-016", "CF-003", "CF-005", "CF-006", "CF-009"],
      revised_poi_ids: ["CF-032", "CF-016", "CF-003", "CF-005", "CF-014", "CF-007"],
      changed_stops: [
        { stop_index: 4, previous_poi_id: "CF-006", revised_poi_id: "CF-014" },
        { stop_index: 5, previous_poi_id: "CF-009", revised_poi_id: "CF-007" },
      ],
      removed_poi_ids: ["CF-006", "CF-009"],
      reordered: false,
      conditional_exposure_poi_ids: [],
      explanations: [
        "Stop 5 was replaced with an indoor-setting POI because its visit overlapped a rain interval.",
        "Stop 6 was replaced with an indoor-setting POI because its visit overlapped a rain interval.",
      ],
    };

    render(<ItineraryPanel planningResult={planning} weatherReplanning={weather} />);

    expect(screen.getByText("Stop 5: Maitland Rotunda → Municipal Art Gallery of Corfu")).toBeInTheDocument();
    expect(screen.getByText("Stop 6: Old Town Hall (Loggia dei Nobili) → Casa Parlante Museum")).toBeInTheDocument();
    expect(screen.queryByText("The existing stops were reordered.")).not.toBeInTheDocument();
    expect(screen.queryByText("Removed stop: Maitland Rotunda")).not.toBeInTheDocument();
    expect(screen.queryByText("Removed stop: Old Town Hall (Loggia dei Nobili)")).not.toBeInTheDocument();
  });
});

export type PoiSetting = "indoor" | "outdoor" | "mixed";

export interface POI {
  id: string;
  name: string;
  area: string;
  category: string;
  tags: string[];
  location: {
    latitude: number;
    longitude: number;
  };
  description: string;
  setting: PoiSetting;
  suggested_visit_minutes: number;
  opening_hours_fixture: unknown | null;
  source_urls: string[];
  family_suitability_fixture: boolean | null;
  description_provenance: string;
}

type PatchAction = "set" | "clear" | "preserve";

export interface Intent {
  intent:
    | "tourism_qa"
    | "poi_recommendation"
    | "itinerary_creation"
    | "itinerary_modification"
    | "live_weather_query"
    | "weather_based_replanning"
    | "clarification";
  field_actions: {
    requested_date: PatchAction;
    start_time: PatchAction;
    end_time: PatchAction;
    timezone: PatchAction;
    semantic_interests: PatchAction;
    filter_category: PatchAction;
    filter_tags: PatchAction;
    filter_setting: PatchAction;
    filter_family_suitability: PatchAction;
    travel_mode: PatchAction;
    family_required: PatchAction;
    child_ages: PatchAction;
    itinerary_modification: PatchAction;
    question: PatchAction;
    clarification_question: PatchAction;
    weather_request: PatchAction;
  };
  question?: string | null;
  recommendation_count?: number | null;
  requested_date?: string | null;
  time_window?: { start_time: string | null; end_time: string | null } | null;
  timezone?: string | null;
  semantic_preferences?: { interests: string[] | null } | null;
  hard_filters?: {
    category: string | null;
    tags: string[] | null;
    setting: PoiSetting | null;
    family_suitability_fixture: boolean | "unknown" | null;
  } | null;
  travel_mode?: "walking" | "driving" | "cycling" | "public_transit" | null;
  family_requirements?: { required: boolean | null; child_ages: number[] | null } | null;
  itinerary_modification?: {
    action: "replace_stop" | "remove_stop" | "add_stop" | "change_order" | "change_constraints";
    stop_index: number | null;
    requested_replacement: string | null;
    preserve_other_stops: boolean | null;
  } | null;
  clarification_question?: string | null;
  weather_request?: {
    relative_horizon: "next_24_hours" | "tomorrow" | "today" | "tonight" | null;
    duration_hours: number | null;
    forecast_date: string | null;
    aspects: Array<"precipitation" | "temperature" | "wind" | "general_conditions">;
  } | null;
}

export interface Evidence {
  poi_id: string;
  name: string;
  description: string;
  source_urls: string[];
  description_provenance: string;
  method: "exact_name" | "semantic";
  relevance_score?: number | null;
}

export interface ItineraryStop {
  poi_id: string;
  starts_at: string;
  ends_at: string;
  reason?: string | null;
}

export interface Itinerary {
  conversation_id?: string | null;
  timezone: string;
  starts_at: string;
  ends_at: string;
  stops: ItineraryStop[];
}

export interface ValidationIssue {
  code: string;
  message: string;
  poi_id?: string | null;
}

export interface ItineraryValidationResult {
  is_valid: boolean;
  issues: ValidationIssue[];
  warnings: string[];
}

export interface TravelEstimate {
  straight_line_distance_km: number;
  estimated_walking_distance_km: number;
  minutes: number;
}

export interface WalkingTransfer {
  from_poi_id: string;
  to_poi_id: string;
  starts_at: string;
  ends_at: string;
  estimate: TravelEstimate;
}

export interface WaitingPeriod {
  poi_id: string;
  starts_at: string;
  ends_at: string;
  reason: "opening_hours";
}

export type PlanningStatus = "feasible" | "uncertain" | "infeasible";

export interface PlanningResult {
  status: PlanningStatus;
  itinerary: Itinerary | null;
  transfers: WalkingTransfer[];
  waiting_periods: WaitingPeriod[];
  validation: ItineraryValidationResult;
  explanations: string[];
  selected_poi_ids: string[];
}

export interface ForecastHour {
  at: string;
  temperature_c: number | null;
  precipitation_mm: number | null;
  precipitation_probability_percent: number | null;
  weather_code: number | null;
  precipitation_interval_start: string | null;
  precipitation_interval_end: string | null;
}

export interface WeatherForecast {
  place: string;
  timezone: string;
  source: "live" | "synthetic_fixture";
  hourly: ForecastHour[];
  retrieved_at: string | null;
}

export interface WeatherInterval {
  start: string;
  end: string;
}

export interface RainAssessment {
  threshold_mm: number;
  rain_forecast: boolean | null;
  affected_intervals: WeatherInterval[];
  first_affected_interval: WeatherInterval | null;
  exposure_overlaps: boolean | null;
  information_missing: boolean;
}

export interface WeatherReplanning {
  status: PlanningStatus;
  planning_result: PlanningResult | null;
  original_poi_ids: string[];
  revised_poi_ids: string[];
  changed_stops: Array<{
    /** Zero-based position of the replacement in the revised itinerary. */
    stop_index: number;
    previous_poi_id: string;
    revised_poi_id: string;
  }>;
  removed_poi_ids: string[];
  reordered: boolean;
  conditional_exposure_poi_ids: string[];
  explanations: string[];
}

export interface ConversationState {
  conversation_id: string;
  timezone: string;
  preferences: {
    interests: string[];
    excluded_categories: string[];
    excluded_tags: string[];
    preferred_setting: PoiSetting | null;
    family_required: boolean | null;
    child_ages: number[];
  };
  hard_filters: {
    category: string | null;
    tags: string[] | null;
    setting: PoiSetting | null;
    family_suitability_fixture: boolean | "unknown" | null;
  };
  travel_mode: "walking" | "driving" | "cycling" | "public_transit" | null;
  active_request: {
    message: string;
    requested_date: string | null;
    start_time: string | null;
    end_time: string | null;
    timezone: string;
    interests: string[];
    setting: PoiSetting | null;
    child_ages: number[];
  } | null;
  itinerary: Itinerary | null;
  pending_modification: Intent["itinerary_modification"] | null;
  replacement_family_requirements: {
    required: boolean | null;
    child_ages: number[] | null;
  } | null;
  last_modification_outcome: "applied" | "rejected" | null;
  updated_at: string;
}

export type OrchestrationStatus =
  | "completed"
  | "no_evidence"
  | "clarification_required"
  | "not_implemented"
  | "feasible"
  | "uncertain"
  | "infeasible"
  | "weather_unavailable"
  | "unsupported_action"
  | (string & {});

export interface OrchestrationResult {
  conversation_id: string;
  intent: Intent;
  status: OrchestrationStatus;
  response: string;
  relevant_poi_ids: string[];
  recommended_poi_ids?: string[];
  evidence: Evidence[];
  planning_result: PlanningResult | null;
  modification_applied: boolean | null;
  weather_forecast: WeatherForecast | null;
  rain_assessment: RainAssessment | null;
  weather_replanning: WeatherReplanning | null;
  weather_error: string | null;
  conversation_state: ConversationState;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  request_id: string;
  top_k?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  result?: OrchestrationResult;
  request?: ChatRequest;
  error?: {
    message: string;
    conversationMissing?: boolean;
  };
}

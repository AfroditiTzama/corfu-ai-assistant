# Corfu AI Assistant — Design Note

## Scope and execution model

The application is a local prototype over the supplied 50-POI Corfu dataset.
A React client calls a typed FastAPI API through Nginx. A Python orchestrator
routes validated intent to retrieval, recommendations, deterministic planning,
conversation updates, or weather. It is not a multi-agent system: Gemini
interprets language and can phrase grounded answers, while Python owns
eligibility, dates, state transitions, schedule arithmetic, and validation.

```text
React -> Nginx -> FastAPI -> orchestrator
                          -> intent policy / state
                          -> POI repository / hybrid retrieval
                          -> planner / independent validator
                          -> Open-Meteo or synthetic fixture
                          -> grounded answer generator
```

## RAG, filtering, and relevance

Each validated POI record is one retrieval unit. Its semantic document contains
the name, category, area, tags, and supplied description. The repository lazily
loads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, embeds all
50 records in memory, and uses cosine similarity. Hybrid scoring combines 70%
normalized semantic similarity with 30% lexical relevance; lexical scoring
weights matches in names, metadata, areas, and descriptions. Retrieval returns
POI IDs, source URLs, provenance, and scores.

Explicit category, tag, setting, exclusion, and family-fixture requirements are
applied before ranking. This prevents top-k semantic retrieval from hiding an
eligible hard-filter match. Natural themes remain soft interests used for
ranking and candidate focus. Intent policy checks current-message provenance so
an unsupported model-generated filter does not silently become mandatory.
Named claims and citations are checked against repository evidence. A malformed
or unsupported generated answer falls back to deterministic grounded text.
Retrieved text is treated as data, never instructions.

## Intent extraction and normalization

`GeminiIntentExtractor` uses `ChatGoogleGenerativeAI` from
`langchain-google-genai` with `gemini-3.5-flash-lite`, temperature zero, and a
JSON-schema structured-output contract. Pydantic enforces `set`, `clear`, and
`preserve` action/value invariants. Provider retries and one malformed-schema
recovery attempt are bounded. Europe/Athens is resolved by the application
rather than invented by the model.

The normalization layer distinguishes soft interests, explicit inclusion and
exclusion constraints, literal metadata-tag requests, and inherited state.
Constraint evidence is checked against the actual user message and dataset
vocabulary. Clear wording can proceed even if a model evidence annotation is
poor; real contradiction or essential ambiguity can request clarification.
Safe diagnostics record stages, exception classes, validation paths, and
sanitized trace frames without prompts, model bodies, credentials, or local
variables.

## Planning and validation

The bounded deterministic planner uses the requested local date/window,
suggested visit durations, fixture opening hours, hard constraints, required
POIs, and approximate Haversine walking times. It checks full visit intervals,
including cross-midnight opening windows. Unknown hours yield conditional
results; they never mean confirmed access. The requested window is a maximum,
so a relevant route may finish early rather than add unrelated filler.

A separate validator rechecks timezone and exact window boundaries, POI
identity, chronological order, visit duration, transfers, opening intervals,
hard constraints, required stops, and duplicates. Invalid plans are not
committed. Walking estimates are explainable approximations rather than route
directions or accessibility guarantees.

## Conversation state and concurrency

`ConversationStateManager` stores typed, session-scoped state in memory. Intent
updates are patches: relevant date/window, preferences, filters, child context,
and unaffected itinerary stops are preserved unless explicitly changed or
cleared. Failed workflows roll back only when their state snapshot is still
current. Planned itineraries use optimistic commits that compare the complete
expected state and pending modification; a genuine concurrent update is
reported instead of overwritten. Request IDs provide process-local idempotency.
One Uvicorn worker is intentional because state is not shared or durable.

## Weather

Production/default configuration constructs `OpenMeteoWeatherProvider` for the
configured Corfu coordinates and Europe/Athens timezone. It validates response
shape, bounds forecast windows, and uses bounded transport retries. Live results
retain source and retrieval time.

For the dated rain demonstration, a development-only Compose override selects
the existing `FixtureWeatherProvider`. Synthetic results retain
`synthetic_fixture`; the adapter has no live fallback and refuses to represent
fixture data as current weather. Replanning assesses scheduled visits and
walking transfers, preserves unaffected stops where feasible, recomputes the
schedule, and independently validates before an optimistic state commit.

## Reliability, trade-offs, and extensions

The final non-demo evaluation reports 7 PASS, 2 PARTIAL, and 1 FAIL. Observed
limitations include intermittent Gemini structured-intent failures, a natural
feasibility request routed as QA, and inconsistent thematic matching for the
soft term `heritage` during indoor weather replacement. In-memory sessions are
lost on restart; embedding cold start can be slow; the dataset fixtures are not
live visitor guarantees; and approximate walking estimates omit real street,
terrain, and accessibility conditions.

Safe extensions would preserve the same boundaries: add persistent versioned
state before multiple workers, introduce a routing provider behind the travel
estimator interface, improve intent/relevance evaluation before changing
weights, and add richer weather/location adapters behind existing protocols.
Every extension should retain strict schemas, hard-filter-first eligibility,
independent itinerary validation, source labels, bounded provider behavior, and
privacy-safe diagnostics.

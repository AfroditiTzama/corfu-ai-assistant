# Corfu AI Tourist Assistant

A local prototype for grounded Corfu place discovery, conversational itinerary planning, deterministic itinerary validation, and weather-aware replanning. The application consists of a FastAPI backend and a React frontend. It is not a publicly hosted or production-hardened travel service.

## Capabilities

- Dataset-grounded tourism questions and POI recommendations with source links
- Soft thematic ranking alongside explicit metadata requirements and exclusions
- Deterministic walking itineraries with fixture hours, visit durations, approximate transfers, and independent validation
- Session-scoped follow-ups that preserve applicable constraints and unaffected stops
- Live Open-Meteo forecasts plus an explicitly enabled synthetic-rain adapter for reproducible replanning
- Grounded Gemini answer generation with citation validation and deterministic fallbacks

## Architecture sketch

```text
React UI -> Nginx /api proxy -> FastAPI -> deterministic orchestrator
                                         |-> Gemini structured intent adapter
                                         |-> intent policy + in-memory state
                                         |-> metadata filters + hybrid retrieval
                                         |-> planner + independent validator
                                         |-> Open-Meteo or opt-in fixture provider
                                         `-> grounded answer adapter + fallback
```

The backend is deliberately a deterministic orchestrator rather than an agent system. Each component has a narrow responsibility:

| **Component**         | **Responsibility**                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FastAPI API           | Typed `/chat`, conversation, health, and readiness endpoints; maps provider failures to stable public errors.                                                             |
| Gemini adapters       | Extract a strict `UserIntent` and optionally phrase a grounded answer. Both use `gemini-3.5-flash-lite` through `langchain-google-genai` structured output.               |
| Intent policy         | Validates field actions and provenance, separates explicit metadata requirements from soft interests, and normalizes current-turn updates.                                |
| POI repository        | Validates the 50-record dataset, applies hard filters first, then combines Sentence Transformers cosine similarity (70%) with lexical relevance (30%).                    |
| Planner and validator | Builds bounded walking schedules from coordinates, visit durations, fixture hours, and constraints; a separate validator rechecks the complete result.                    |
| State manager         | Stores typed conversation state in memory and applies patch semantics. Snapshot comparison protects itinerary commits from concurrent updates.                            |
| Weather adapters      | Use Open-Meteo by default or the explicitly enabled synthetic fixture in development. Synthetic data retains its source label and cannot answer current-weather requests. |
| React frontend        | Presents chat, POI cards, itinerary timing, warnings, and weather details through an Nginx `/api` proxy.                                                                  |

LangChain is used only as the Google Gemini SDK adapter for structured model calls. Retrieval, orchestration, state, planning, validation, and weather logic remain ordinary typed Python services.

The 70% semantic / 30% lexical retrieval blend is a fixed heuristic for this small dataset, not a universally optimal weighting. Explicit metadata requirements are applied as hard filters before ranking, so high semantic similarity cannot override them.

See [docs/DESIGN_NOTE.md](docs/DESIGN_NOTE.md) for the concise design note and [docs/EVALUATION_CASES.md](docs/EVALUATION_CASES.md) for ten representative non-demo evaluation cases.

## Demo Video

A recorded demonstration of the Corfu AI Assistant covering all five official DOTSOFT evaluation scenarios (A–E).

**Duration:** Approximately 3 minutes

[Watch the Demo Video (Google Drive)](https://drive.google.com/file/d/1d4_4-FN3V8Of2Kzah54SEWbLMJ3TtAg_/view?usp=sharing)

## Run with Docker Compose

### Prerequisites

- Docker Desktop or Docker Engine with Docker Compose v2.24 or newer
- Internet access for image/dependency downloads and, on first semantic search, the embedding model
- A Google Gemini API key to use chat features that require intent extraction

### Configure and start

Create `.env` from the example only if you do not already have a local `.env`. This preserves existing local configuration:

```bash
test -f .env || cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` for Gemini-backed chat. The key is passed to the backend at container runtime; it is not included in either image or the browser bundle. The app can start without the key, and its health checks remain available, but chat requests requiring Gemini will return a clear service configuration error.

Start both services with live Open-Meteo as the default weather provider:

```bash
docker compose up --build -d
```

- Frontend: [http://localhost:8080](http://localhost:8080/)
- Backend API for local debugging: [http://localhost:8000](http://localhost:8000/)
- Liveness: [http://localhost:8000/health](http://localhost:8000/health)
- Readiness (dataset loaded): [http://localhost:8000/ready](http://localhost:8000/ready)

The browser sends API requests to the same origin at `/api/*`; Nginx removes the `/api` prefix and forwards requests to the `backend` Compose service. For example, `POST /api/chat` is forwarded to `POST /chat`. Static Vite assets are served by Nginx with long-lived caching for hashed `/assets/` files, while the app shell is revalidated so browser refresh and client-side routes continue to work.

Check service health and follow logs:

```bash
docker compose ps
docker compose logs -f backend frontend
```

Stop the containers:

```bash
docker compose down
```

The named `huggingface-model-cache` volume survives container rebuilds and holds model files. The embedding model is not downloaded during image build or application startup. Its first semantic search may download and load `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which can take time and disk space. The container uses CPU-only PyTorch; no GPU or CUDA setup is required. Metadata-based and deterministic behavior does not require the embedding model.

The synthetic rain fixture is included only for controlled application scenarios. It is not substituted for live weather. Live weather requests use the existing Open-Meteo provider and therefore need outbound network access.

For the dated rain scenario, recreate the backend with the development-only Compose override:

```bash
docker compose -f compose.yaml -f compose.weather-demo.yaml \
  up --build -d --force-recreate backend
```

This sets `CORFU_ENVIRONMENT=development` and `CORFU_WEATHER_FIXTURE_ENABLED=true`; the fixture path remains `/app/corfu_synthetic_rain.json`. Do not run the live-weather scenario in this mode. Restore the normal Open-Meteo backend with:

```bash
docker compose -f compose.yaml up -d --force-recreate backend
```

## API example

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Recommend a few Corfu museums",
    "request_id": "8ba6e65d-b715-4f15-9c95-351cc66d12cb"
  }'
```

Through the frontend reverse proxy, the same request is available at `http://localhost:8080/api/chat`. Keep the returned `conversation_id` for follow-up messages and generate a new `request_id` for each new request; reuse that request ID if retrying the same logical message.

## Retrieval evaluation

The evaluation commands in this section and the next require a local Python environment with the project dependencies installed; see [Local development without Docker](#local-development-without-docker). Their JSON outputs are written under `evaluation/results/`, which is excluded from version control.

The labeled benchmark in `evaluation/retrieval_benchmark.json` contains 68 human-authored queries with dataset-grounded relevance grades, hard constraints, and explicit multi-turn examples. Labels were authored from the supplied POI records, not copied from the assistant's search results. The runner reports pure retrieval ranking metrics separately from recommendation-count simulation; it does not call Gemini or measure final generated-answer quality.

Run the evaluation on CPU with:

```bash
python3 evaluation/run_retrieval_benchmark.py \
  --output evaluation/results/current.json
```

The first query includes model initialization/download if needed; subsequent queries report warm retrieval latency. The command generates `evaluation/results/current.json` locally; previously generated reports are not included in the repository. These scores are from a small manually labeled exercise benchmark, not a guarantee of production retrieval quality.

## End-to-end reliability and geography

Run the deterministic, fault-injected end-to-end evaluation and its regression test with:

```bash
python3 evaluation/run_e2e_evaluation.py --output evaluation/results/e2e-current.json
python3 -m pytest -q tests/test_e2e_evaluation.py
```

The optional Gemini run is limited to one QA turn and two provider calls:

```bash
python3 evaluation/run_e2e_evaluation.py --real-gemini --max-provider-requests 2
```

It requires `GOOGLE_API_KEY`; no live Gemini result is included unless that command is actually run. For the 50-record coordinate and area audit, run:

```bash
python3 evaluation/run_geographic_audit.py --output evaluation/results/geographic-audit.json
```

The commands above generate `evaluation/results/e2e-current.json` and `evaluation/results/geographic-audit.json` locally. Generated reports are excluded from Git; the documented expected and observed outcomes of the ten non-demo cases are available in [docs/EVALUATION_CASES.md](docs/EVALUATION_CASES.md). The geographic audit separates dataset-coordinate findings from limitations of straight-line walking estimation.

## Local development without Docker

Use Python 3.10 or newer and Node.js 20.19+ or 22.12+. From the repository root, create and activate a Python virtual environment:

```bash
test -f .env || cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,llm,retrieval]"
python -m uvicorn corfu_assistant.main:app --app-dir src --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
test -f .env || cp .env.example .env
npm ci
npm run dev
```

The local Vite configuration continues to use `VITE_API_BASE_URL` from `frontend/.env` (default: `http://127.0.0.1:8000`); the backend allows the default Vite origins `localhost:5173` and `127.0.0.1:5173`. Docker builds use the same-origin `/api` prefix instead. Never put `GOOGLE_API_KEY` in a `VITE_*` variable.

Run the complete automated checks from the repository root in a separate terminal (the Python test command explicitly uses the virtual environment):

```bash
.venv/bin/python -m pytest -q
cd frontend
npm test -- --run
npm run build
```

## Official demo A–E

The executable client in `scripts/run_demo.py` sends the assignment's exact prompts through `/chat`, starts a new conversation for A–E, and preserves the conversation ID for the B and D follow-ups. It does not bypass intent extraction, retrieval, planning, validation, or weather routing.

With the normal live-weather Compose stack on port 8000, run A, B, and C:

```bash
python3 scripts/run_demo.py --base-url http://127.0.0.1:8000 --scenarios A B C
```

Recreate the backend in fixture mode and run D:

```bash
docker compose -f compose.yaml -f compose.weather-demo.yaml \
  up -d --force-recreate backend
python3 scripts/run_demo.py --base-url http://127.0.0.1:8000 --scenarios D
```

The script rejects a D response not labeled `synthetic_fixture`. Restore the default backend and run E against actual Open-Meteo data:

```bash
docker compose -f compose.yaml up -d --force-recreate backend
python3 scripts/run_demo.py --base-url http://127.0.0.1:8000 --scenarios E
```

The script rejects a synthetic E response. Starting a new main scenario creates a new conversation; B and D retain their conversation ID for their follow-up.

The exact prompt sequence is:

- **A:** art and museums, Monday 5 October 2026, 13:00–17:00.
- **B:** historical walking plan, Tuesday 6 October 2026, 11:00–15:00; then replace stop two for a 10-year-old while preserving the rest.
- **C:** five named attractions on foot in 90 minutes; explain infeasibility rather than fabricate a route.
- **D:** outdoor historical plan, Wednesday 7 October 2026, 13:00–17:00; then replan with the injected rain fixture while preserving the window.
- **E:** real next-24-hour Corfu forecast and rain question.

For a UI recording, use `http://localhost:8080`. The existing Nginx proxy continues to target the recreated `backend` service in either weather mode.

## Architecture and limitations

- The backend loads and validates the supplied dataset at startup. The embedding model/index is initialized lazily by the existing repository.
- Docker uses one Uvicorn worker because conversation state and idempotency records are in memory. A backend restart loses conversations; the frontend handles an expired conversation by offering a fresh one. There is no persistent database or cross-process state.
- `/health` is liveness; `/ready` confirms the startup-loaded POI repository. Neither endpoint waits for the embedding model or Gemini credentials.
- The service binds host ports to loopback by default. It has no authentication, and is intended for local review only; do not expose it directly to the public internet.
- Opening hours, family-suitability flags, and visit durations are exercise assumptions, not verified visitor information. Walking times are approximate, and historical opening-hour accuracy is not guaranteed.
- Unknown fixture opening hours produce a conditional plan; they never confirm that indoor access is available.
- Live weather requires the Open-Meteo service. Conversation and model caches are process-local and volume-backed respectively; the cache volume does not preserve conversation state.
- Intent extraction and answer wording depend on Gemini availability and may vary semantically despite strict structured validation and bounded recovery. The final non-demo evaluation observed intermittent structured-intent HTTP 502 responses on a multi-turn preference update.
- The final evaluation recorded **7 PASS, 2 PARTIAL, and 1 FAIL**. A natural five-stop feasibility question was routed as QA instead of planning, and a broad `heritage` interest did not always identify thematic indoor weather replacements. See [docs/EVALUATION_CASES.md](docs/EVALUATION_CASES.md).
- The in-memory vector index and session store are appropriate for this 50-record prototype, not for horizontally scaled deployment.

## Troubleshooting

- **Cannot connect to Docker:** Start Docker Desktop / the Docker Engine, then retry `docker compose up --build`.
- **Chat reports missing Gemini configuration:** Add a valid server-side `GOOGLE_API_KEY` to `.env` and recreate the backend with `docker compose up -d --force-recreate backend`.
- **Frontend reports an unavailable service:** Check `docker compose ps` and `docker compose logs backend frontend`; wait for the backend `/ready` health check to pass.
- **First semantic request is slow or fails to download the model:** Confirm outbound network access and available disk space. Subsequent containers reuse the named model-cache volume. If operating offline, pre-populate that volume with the model before issuing semantic requests.
- **Conversation ID returns 404 after a restart:** Process-local state is lost when the backend restarts. Start a new conversation.

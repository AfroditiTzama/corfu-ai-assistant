# Corfu AI Travel Assistant — Frontend

The React + TypeScript frontend connects to the FastAPI chat API and continues
to use the supplied `corfu_dataset.json` for the Explore browser and POI detail
dialogs. Chat answers, evidence, itinerary plans, and weather data come from
the backend; the browser does not generate substitute AI responses or plans.

## Requirements and startup

- Node.js 20.19+ or 22.12+
- npm
- The FastAPI backend running at `http://127.0.0.1:8000`

From the backend project root, install the backend extras and start FastAPI:

```bash
python3 -m pip install -e ".[llm,retrieval]"
cp .env.example .env
python3 -m uvicorn corfu_assistant.main:app --app-dir src --host 127.0.0.1 --port 8000
```

Set `GOOGLE_API_KEY` in the backend `.env` if using Gemini. Then, in this
directory:

```bash
npm install
cp .env.example .env
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The default API base URL
is `http://127.0.0.1:8000`; override it with `VITE_API_BASE_URL` when needed.
The backend must allow the frontend origin through its explicit CORS settings.
For local Vite development, the backend defaults include
`http://localhost:5173` and `http://127.0.0.1:5173`.
If Vite uses another port, add that exact origin to the backend's
`CORFU_CORS_ORIGINS` JSON list (for example
`["http://localhost:5173","http://localhost:5174"]`); do not use wildcard
origins.

## API usage

The client uses the existing backend contract without introducing a second
response schema.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Recommend a few Corfu museums",
    "request_id": "8ba6e65d-b715-4f15-9c95-351cc66d12cb"
  }'
```

The response contains the backend's `conversation_id`, `intent`, `status`,
`response`, `evidence`, `planning_result`, weather fields, and
`conversation_state`. Subsequent messages include the returned conversation
ID and a new request ID:

```json
{
  "conversation_id": "returned-uuid",
  "intent": { "intent": "tourism_qa" },
  "status": "completed",
  "response": "Grounded answer with POI references.",
  "relevant_poi_ids": ["CF-001"],
  "evidence": [],
  "planning_result": null,
  "modification_applied": null,
  "weather_forecast": null,
  "rain_assessment": null,
  "weather_replanning": null,
  "weather_error": null,
  "conversation_state": { "conversation_id": "returned-uuid" }
}
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Make that route suitable for a 10-year-old",
    "conversation_id": "CONVERSATION_UUID_FROM_PREVIOUS_RESPONSE",
    "request_id": "f40c6972-5f4b-4db4-bf51-bfb71dc22037"
  }'
```

Read or delete an existing process-local conversation:

```bash
curl http://127.0.0.1:8000/conversations/CONVERSATION_UUID
curl -X DELETE http://127.0.0.1:8000/conversations/CONVERSATION_UUID
```

## Frontend architecture

- `src/lib/api.ts` provides typed chat, health/readiness, and conversation
  requests with HTTP, JSON, network, and cancellation handling.
- `src/lib/types.ts` mirrors the Pydantic response fields used by the UI.
- `src/App.tsx` owns the process-local conversation ID, messages, retries, and
  latest returned planning/weather data.
- Chat evidence and source URLs are rendered from API results;
  `src/components/itinerary` and `src/components/weather` only render backend
  data and do not infer feasibility or weather.
- `src/lib/dataset.ts` continues to import the supplied dataset for browsing.

`request_id` is retained with a failed user message and reused when retrying
that same request. Conversation state is held in the backend process and can
be lost on restart; a 404 is explained and the user can start a fresh chat.
`GOOGLE_API_KEY` belongs only in the backend environment and must not be added
to Vite variables or browser code.

## Limitations

- The backend has no authentication or persistent conversation database.
- Opening-hour, family-suitability, and suggested-duration values are supplied
  exercise assumptions, not verified live visitor information. The API does
  not guarantee historical opening-hour accuracy beyond its supplied fixtures.
- Walking estimates and weather availability depend on the backend's
  documented providers and assumptions. Synthetic fixture weather is labeled
  and is never represented as live weather.
- There is no streaming response support.

## Build and tests

```bash
npm run build
npm test
```

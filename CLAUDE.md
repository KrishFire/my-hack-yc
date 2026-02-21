# CLAUDE.md

## Purpose
This repository is an MCP App for synthetic focus-group research.

It has two runtimes:
- A TypeScript MCP server (`index.ts`) that exposes tools and widget metadata to MCP hosts (Claude/ChatGPT/Inspector).
- A Python FastAPI backend (`backend/main.py`) that performs persona simulation and follow-up Q&A.

The frontend widget (`resources/focus-group-results.tsx`) renders simulation output and supports interactive reruns and follow-ups.

## High-Level Architecture
1. User (Claude/ChatGPT/Inspector) calls MCP tool `run_synthetic_focus_group`.
2. MCP server forwards request to backend `POST /synthetic/simulate`.
3. Backend generates/normalizes personas, aggregates manager summary, and returns structured JSON.
4. MCP server normalizes response and returns `widget(...)` result with props for `focus-group-results`.
5. Widget renders aggregation + roster and allows:
   - Rerunning simulation with edited stimulus/audience/dimensions.
   - Asking follow-ups to one, many, or segmented personas via `ask_persona_followup`.

## Core Files
- `index.ts`: MCP server setup, tool definitions, backend HTTP wrapper, response normalization, widget linkage.
- `backend/main.py`: FastAPI app, OpenAI calls, persona generation, retrieval memory, aggregation.
- `resources/focus-group-results.tsx`: Main focus-group widget UI and interaction logic.
- `resources/product-search-result/*`: Starter demo widget (kept for reference/demo).

## MCP Tools
### `run_synthetic_focus_group`
Input:
- `target_audience?: string`
- `stimulus_description: string`
- `image_url?: string`
- `analysis_dimensions?: string[]`

Behavior:
- POSTs to backend `/synthetic/simulate`.
- Normalizes backend response shape.
- Returns a widget-linked result for `focus-group-results`.

### `ask_persona_followup`
Input:
- `agent_id: string`
- `question: string`

Behavior:
- POSTs to backend `/synthetic/followup`.
- Returns normalized structured object (`answer`, `memory_snippets`, etc.).

## Backend API
### `GET /health`
Returns service status, active model, fallback models, embedding model, and in-memory store size.

### `POST /synthetic/simulate`
Request:
- `target_audience?: string`
- `stimulus_description: string`
- `image_url?: string`
- `persona_count?: number`
- `analysis_dimensions?: string[]`

Flow:
1. Resolve target audience (use provided or auto-generate).
2. Resolve analysis dimensions (merge user-defined + auto-generated + fallback).
3. Build diverse persona blueprints with sentiment balancing.
4. Generate personas in parallel (`asyncio.gather` + semaphore).
5. Normalize and sanitize persona output.
6. Persist personas in in-memory store.
7. Build manager summary (dynamic heatmap, sentiment, demographics).

### `POST /synthetic/followup`
Request:
- `agent_id: string`
- `question: string`

Flow:
1. Retrieve persona from in-memory store.
2. Retrieve top memory snippets (embedding/cosine fallback to lexical scoring).
3. Generate grounded first-person response.
4. Append Q/A back into memory stream.

## Data Model Notes
- Persona state is stored in memory only (reset on backend restart).
- Each persona has profile fields + memory chunks.
- `cognitive_load` is dynamic by resolved `analysis_dimensions` keys (not hardcoded dashboard keys).
- Manager summary includes:
  - `cognitive_load_heatmap`
  - `sentiment`
  - `demographics`
  - `analysis_dimensions`

## Realism and Quality Controls
- Sentiment targets are distributed across positive/neutral/negative.
- Blueprint generation enforces diversity of names/occupations/contexts.
- Output normalization guards against malformed model responses.
- Claim sanitizer prevents obvious implausible claims for certain stimuli (e.g. fast-food "health benefits").
- Model fallback chain is used automatically if primary model fails.

## Widget Behavior (`focus-group-results`)
- Renders:
  - Aggregation metrics with dynamic dimension labels.
  - Sentiment and demographic sections.
  - Persona roster cards.
- Supports:
  - Selecting an active persona.
  - Batch targeting follow-ups (`selected`, `all`, `segment`).
  - Editing and rerunning simulation:
    - target audience
    - stimulus description
    - image URL
    - analysis dimensions (comma/newline list)

Implementation note:
- Form fields are hydrated only when a new simulation run signature arrives, so user typing is not overwritten by re-renders.

## Environment Variables
### MCP server (TypeScript)
- `FOCUS_GROUP_BACKEND_URL` (default: `http://127.0.0.1:8000`)
- `MCP_URL` (optional server base URL override)

### Backend (Python)
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default: `gpt-5.2-mini`)
- `OPENAI_MODEL_FALLBACKS` (default: `gpt-5-mini,gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `FG_PERSONA_COUNT`, `FG_MAX_PERSONA_COUNT`, `FG_MAX_PARALLEL_TASKS`, `FG_TOP_K_MEMORIES`, `FG_MAX_PERSONA_STORE`

## Local Run
### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your-key>"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### MCP app
```bash
npm install
npm run dev
```

Inspector:
- `http://localhost:3000/inspector`

Tunnel mode:
```bash
npm run dev -- --tunnel
```

## Build/Test Commands
```bash
python3 -m py_compile backend/main.py
npm run build
```

## Known Operational Constraints
- Persona/memory store is in-memory; no persistence across backend restarts.
- Tunnel provider may rate-limit rapid reconnects.
- Bundle-size warnings from Vite are currently non-blocking.

## Security
- Never commit API keys.
- If a key is exposed in logs/chat/history, rotate it immediately and replace in local env.

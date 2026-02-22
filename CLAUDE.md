# CLAUDE.md

## Purpose
This repository is a Manufact hackathon MCP app for synthetic focus-group research.

It runs as:
- A TypeScript MCP server in `index.ts` (tool definitions + widget wiring).
- A Python FastAPI simulation backend in `backend/main.py`.
- A React widget in `resources/focus-group-results.tsx`.

## Current System Flow
1. Host (Claude/ChatGPT/Inspector) calls `run_synthetic_focus_group`.
2. MCP server POSTs to backend `/synthetic/simulate`.
3. Backend classifies the task, builds personas/memory, runs simulation in parallel, and returns summary + roster.
4. MCP server normalizes response and returns a widget result (`focus-group-results`).
5. Widget supports rerun edits and targeted follow-up questions.

## MCP Tools
### `run_synthetic_focus_group`
Input:
- `target_audience?: string`
- `stimulus_description: string`
- `image_url?: string`
- `persona_count?: number` (up to backend max)

Notes:
- `analysis_dimensions` is intentionally **not** an input anymore.
- Response keeps the same schema expected by the widget:
  - `manager_summary` (`cognitive_load_heatmap`, `sentiment`, `demographics`, `analysis_dimensions`)
  - `personas`
  - run metadata fields

### `ask_persona_followup`
Input:
- `agent_id: string`
- `question: string`
- `persona_profile?: Record<string, unknown>`

Notes:
- `persona_profile` is used as fallback bootstrapping if backend in-memory store was reset.

## Backend Behavior (`backend/main.py`)
### Key upgrades
- Sentiment split is no longer hardcoded; sentiment emerges from persona + memory + stimulus.
- Pre-simulation classifier sets:
  - `Objective UI Task` for factual UI/UX checks.
  - `Subjective Concept` for opinion/concept/ad/product review tasks.
- For `Objective UI Task`, sentiment is marked `not_applicable` in manager summary.
- Persona grounding uses Hugging Face persona data:
  - default: `SynthlabsAI/PERSONA`
  - configurable fallbacks supported
- Additional real-world memory snippets come from a general-population dataset (default `yahoo_answers_topics`).
- Persona generation executes in parallel with bounded concurrency.

### Scale settings
- Backend allows larger runs (default max persona count is 1000 via env).
- In-memory persona store capacity is configurable.

### Follow-up stability
- `/synthetic/followup` can rebuild transient persona context from `persona_profile` when `agent_id` is missing from store.
- Prevents common 404 failures after backend restarts while widget still has old roster state.

## Widget Behavior (`focus-group-results`)
### Main UX
- Header with audience/stimulus context and simulation metadata.
- In-widget **Refine Simulation** form:
  - edit audience
  - edit stimulus text
  - edit image URL
  - rerun without leaving widget
- **Interactive Agent Roster** with preference bar (derived from sentiment score).
- **Persona Follow-up** supports:
  - selected persona
  - all personas
  - segment targeting (sentiment, gender, ethnicity, occupation, region, persona type)
- Follow-up sends:
  - direct tool calls (`ask_persona_followup`) for immediate widget responses
  - host follow-up prompt message for continued LLM-thread interaction

### Summary presentation
- Background summary is intentionally de-emphasized/faded to keep focus on roster + follow-up.
- Demographics are shown in a collapsed `<details>` section by default.

## Files
- `index.ts`: MCP tools + normalization + widget metadata.
- `backend/main.py`: Simulation engine, dataset integration, follow-up memory retrieval.
- `resources/focus-group-results.tsx`: Widget UI/state/actions.
- `backend/requirements.txt`: Python dependencies (includes `datasets` for Hugging Face loading).
- `resources/product-search-result/*`: legacy starter widget code kept in repo.

## Environment Variables
### MCP server
- `FOCUS_GROUP_BACKEND_URL` (default `http://127.0.0.1:8000`)

### Backend
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default `gpt-5.2-mini`)
- `OPENAI_MODEL_FALLBACKS` (default `gpt-5-mini,gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `FG_PERSONA_COUNT`
- `FG_MAX_PERSONA_COUNT` (default `1000`)
- `FG_MAX_PARALLEL_TASKS`
- `FG_TOP_K_MEMORIES`
- `FG_MAX_PERSONA_STORE`
- `FG_HF_PERSONA_DATASET` (default `SynthlabsAI/PERSONA`)
- `FG_HF_PERSONA_DATASET_FALLBACKS`
- `FG_HF_MEMORY_DATASET` (default `yahoo_answers_topics`)

## Run Locally
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

## Build / Checks
```bash
python3 -m py_compile backend/main.py
npm run build
```

## Operational Notes
- Persona state is in-memory only; restarting backend clears store.
- Tunnel setup may rate-limit repeated rapid tunnel requests.
- Vite bundle-size warnings are non-blocking for hackathon demo use.

## Security
- Do not commit API keys.
- If a key was exposed anywhere, rotate it immediately and update local env.

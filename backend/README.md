# Synthetic Focus Group Backend (FastAPI)

This service powers the MCP tools:
- `POST /synthetic/simulate`
- `POST /synthetic/followup`

It is designed to run locally at `http://127.0.0.1:8000`.

## 1) Create and activate a Python environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Set environment variables

```bash
export OPENAI_API_KEY="<your-key>"
# Optional overrides:
# export OPENAI_MODEL="gpt-4o-mini"
# export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
# export FG_PERSONA_COUNT="6"
# export FG_MAX_PARALLEL_TASKS="8"
```

## 4) Run server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## 5) Quick checks

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/synthetic/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "target_audience": "middle-aged suburban women with kids",
    "stimulus_description": "analyze car dashboard for ease of use during stressed times"
  }'
```

## Notes

- Persona simulation is parallelized using `asyncio.gather` to improve speed.
- Follow-up uses embedding-based memory retrieval for persona grounding.
- This store is in-memory; restarting the backend clears prior personas/memory.

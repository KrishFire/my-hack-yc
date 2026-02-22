# Synthetic Focus Group MCP App - Pneuma

An interactive MCP app for running fast, grounded synthetic focus groups on ads, product concepts, and UI experiences.

Built for the Manufact MCP Apps Hackathon at YC.

## Why This Is Useful
- Product teams can test messaging before launch.
- Designers can pressure-test ideas with realistic audience segments.
- Researchers can ask follow-up questions to specific personas or entire subgroups, directly inside the widget.

## What We Built
- Two MCP tools:
  - `run_synthetic_focus_group`
  - `ask_persona_followup`
- A custom React widget (`focus-group-results`) that renders:
  - simulation summary
  - persona roster
  - in-widget rerun controls
  - single or batch follow-up chat
- A FastAPI backend that:
  - classifies task type (`Objective UI Task` vs `Subjective Concept`)
  - grounds personas with `SynthlabsAI/PERSONA` from Hugging Face
  - applies sentiment/preference calibration based on actual persona language
  - supports image grounding for visible brand/price cues

## How It Works
1. User asks host (Claude/ChatGPT/Inspector) to run a focus group.
2. MCP tool calls backend `POST /synthetic/simulate`.
3. Backend returns manager summary + personas.
4. MCP server returns widget-linked result.
5. Widget lets the user:
   - edit target audience / stimulus / image and rerun
   - ask follow-ups to selected personas, all personas, or segments

## MCP Tools

| Tool | Input | Output |
|---|---|---|
| `run_synthetic_focus_group` | `target_audience?`, `stimulus_description`, `image_url?`, `persona_count?` | Widget props (`manager_summary`, `personas`, metadata) |
| `ask_persona_followup` | `agent_id`, `question`, `persona_profile?` | Structured answer + supporting memory snippets |

## Quickstart (Local)

### 1) Start backend (Terminal A)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="<your-key>"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2) Start MCP server + widget (Terminal B)
```bash
npm install
npm run dev
```

Inspector:
- `http://localhost:3000/inspector`

Health check:
```bash
curl http://127.0.0.1:8000/health
```

## Run in Claude / ChatGPT

Start with tunnel:
```bash
npm run dev -- --tunnel
```

Use the printed URL (example):
- `https://<subdomain>.tunnel.mcp-use.com/mcp`

Then add that MCP URL:
- Claude: Settings → Integrations → Add integration
- ChatGPT: Settings → Connectors → Add MCP server
// End of Selection
```

import asyncio
import json
import math
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from openai import AsyncOpenAI
except Exception as import_error:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None

APP_NAME = "synthetic-focus-group-backend"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_PERSONA_COUNT = int(os.getenv("FG_PERSONA_COUNT", "6"))
MAX_PERSONA_COUNT = int(os.getenv("FG_MAX_PERSONA_COUNT", "12"))
MAX_PARALLEL_TASKS = int(os.getenv("FG_MAX_PARALLEL_TASKS", "8"))
TOP_K_MEMORIES = int(os.getenv("FG_TOP_K_MEMORIES", "5"))
MAX_PERSONA_STORE = int(os.getenv("FG_MAX_PERSONA_STORE", "200"))


class SimulateRequest(BaseModel):
    target_audience: str = Field(min_length=3)
    stimulus_description: str = Field(min_length=3)
    image_url: str | None = None
    persona_count: int = Field(default=DEFAULT_PERSONA_COUNT, ge=3, le=MAX_PERSONA_COUNT)


class FollowupRequest(BaseModel):
    agent_id: str = Field(min_length=3)
    question: str = Field(min_length=2)


@dataclass
class MemoryChunk:
    text: str
    kind: str
    embedding: list[float] | None = None
    created_at: float = field(default_factory=lambda: time.time())


@dataclass
class PersonaState:
    agent_id: str
    profile: dict[str, Any]
    memories: list[MemoryChunk]
    created_at: float = field(default_factory=lambda: time.time())


class PersonaStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: dict[str, PersonaState] = {}

    async def upsert_many(self, personas: list[PersonaState]) -> None:
        async with self._lock:
            for persona in personas:
                self._store[persona.agent_id] = persona

            if len(self._store) > MAX_PERSONA_STORE:
                sorted_items = sorted(
                    self._store.items(), key=lambda item: item[1].created_at, reverse=True
                )
                keep = dict(sorted_items[:MAX_PERSONA_STORE])
                self._store = keep

    async def get(self, agent_id: str) -> PersonaState | None:
        async with self._lock:
            return self._store.get(agent_id)

    async def append_memories(self, agent_id: str, chunks: list[MemoryChunk]) -> None:
        async with self._lock:
            persona = self._store.get(agent_id)
            if not persona:
                return
            persona.memories.extend(chunks)

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)


app = FastAPI(title=APP_NAME, version="1.0.0")
store = PersonaStore()
_client: AsyncOpenAI | None = None


def ensure_openai_client() -> AsyncOpenAI:
    global _client

    if _IMPORT_ERROR is not None or AsyncOpenAI is None:
        raise HTTPException(
            status_code=500,
            detail=f"openai package is not installed: {_IMPORT_ERROR}",
        )

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY is missing. Set it before starting the backend.",
            )
        _client = AsyncOpenAI(api_key=api_key)

    return _client


def _safe_int(value: Any, default: int, minimum: int = 0, maximum: int = 100) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw_text": raw_text}

    return {"raw_text": raw_text}


async def _chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.8) -> dict[str, Any]:
    client = ensure_openai_client()

    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    return _extract_json_object(content)


async def _embed_text(text: str) -> list[float] | None:
    client = ensure_openai_client()

    try:
        response = await client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _lexical_score(question: str, text: str) -> float:
    question_tokens = {token for token in question.lower().split() if len(token) > 2}
    text_tokens = {token for token in text.lower().split() if len(token) > 2}
    if not question_tokens or not text_tokens:
        return 0.0
    overlap = len(question_tokens.intersection(text_tokens))
    return overlap / max(1, len(question_tokens))


def _default_persona(index: int) -> dict[str, Any]:
    persona_number = index + 1
    return {
        "name": f"Persona {persona_number}",
        "age_range": "35-49",
        "occupation": "Parent",
        "household": "Suburban household with kids",
        "region": "United States",
        "stress_level": 70,
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "cognitive_load": {
            "visual_clarity": 50,
            "control_discoverability": 50,
            "glanceability_under_stress": 50,
            "error_recovery": 50,
        },
        "reaction_summary": "I need clear, simple controls during stressful moments.",
        "quote": "When my kids are loud, I need the dashboard to be obvious at a glance.",
        "reasons": [
            "Simple visual hierarchy reduces mental effort",
            "Quick access controls matter during time pressure",
        ],
        "concerns": ["Too many nested menus", "Small touch targets while driving"],
        "delights": ["Large, clearly labeled controls"],
    }


def _normalize_persona(payload: dict[str, Any], index: int) -> dict[str, Any]:
    default_persona = _default_persona(index)

    cognitive_load_raw = payload.get("cognitive_load")
    if not isinstance(cognitive_load_raw, dict):
        cognitive_load_raw = {}

    cognitive_load = {
        "visual_clarity": _safe_int(cognitive_load_raw.get("visual_clarity"), 50),
        "control_discoverability": _safe_int(
            cognitive_load_raw.get("control_discoverability"), 50
        ),
        "glanceability_under_stress": _safe_int(
            cognitive_load_raw.get("glanceability_under_stress"), 50
        ),
        "error_recovery": _safe_int(cognitive_load_raw.get("error_recovery"), 50),
    }

    sentiment_score = _safe_float(payload.get("sentiment_score"), 0.0)
    sentiment_label = _safe_str(payload.get("sentiment_label"), "neutral").lower()
    if sentiment_label not in {"positive", "neutral", "negative"}:
        sentiment_label = (
            "positive"
            if sentiment_score > 0.2
            else "negative"
            if sentiment_score < -0.2
            else "neutral"
        )

    return {
        "name": _safe_str(payload.get("name"), default_persona["name"]),
        "age_range": _safe_str(payload.get("age_range"), default_persona["age_range"]),
        "occupation": _safe_str(payload.get("occupation"), default_persona["occupation"]),
        "household": _safe_str(payload.get("household"), default_persona["household"]),
        "region": _safe_str(payload.get("region"), default_persona["region"]),
        "stress_level": _safe_int(payload.get("stress_level"), 70),
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "cognitive_load": cognitive_load,
        "reaction_summary": _safe_str(
            payload.get("reaction_summary"), default_persona["reaction_summary"]
        ),
        "quote": _safe_str(payload.get("quote"), default_persona["quote"]),
        "reasons": [
            _safe_str(item, "")
            for item in payload.get("reasons", default_persona["reasons"])
            if _safe_str(item, "")
        ],
        "concerns": [
            _safe_str(item, "")
            for item in payload.get("concerns", default_persona["concerns"])
            if _safe_str(item, "")
        ],
        "delights": [
            _safe_str(item, "")
            for item in payload.get("delights", default_persona["delights"])
            if _safe_str(item, "")
        ],
    }


def _persona_seed_prompt(
    target_audience: str,
    stimulus_description: str,
    image_url: str | None,
    index: int,
    total: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are generating one synthetic focus-group participant. "
        "Return valid JSON only. Keep responses realistic and concise."
    )

    user_prompt = f"""
Generate participant {index + 1} of {total}.

Target audience: {target_audience}
Stimulus: {stimulus_description}
Stimulus image URL (optional context): {image_url or "none"}

Return exactly one JSON object with keys:
- name (string)
- age_range (string)
- occupation (string)
- household (string)
- region (string)
- stress_level (integer 0-100)
- sentiment_score (number from -1 to 1)
- sentiment_label ("positive" | "neutral" | "negative")
- cognitive_load (object with integers 0-100):
  - visual_clarity
  - control_discoverability
  - glanceability_under_stress
  - error_recovery
- reaction_summary (string)
- quote (string, first-person)
- reasons (array of strings, 2-4)
- concerns (array of strings, 2-4)
- delights (array of strings, 1-3)
"""
    return system_prompt, user_prompt


async def _generate_one_persona(
    request: SimulateRequest,
    index: int,
    semaphore: asyncio.Semaphore,
) -> PersonaState:
    async with semaphore:
        system_prompt, user_prompt = _persona_seed_prompt(
            request.target_audience,
            request.stimulus_description,
            request.image_url,
            index,
            request.persona_count,
        )

        raw = await _chat_json(system_prompt, user_prompt)
        normalized = _normalize_persona(raw, index)

        agent_id = f"agent_{uuid.uuid4().hex[:10]}"

        memory_texts = [
            normalized["reaction_summary"],
            normalized["quote"],
            "Reasons: " + "; ".join(normalized["reasons"]),
            "Concerns: " + "; ".join(normalized["concerns"]),
            "Delights: " + "; ".join(normalized["delights"]),
        ]

        memory_chunks = [MemoryChunk(text=text, kind="seed") for text in memory_texts if text]

        embeddings = await asyncio.gather(*[_embed_text(chunk.text) for chunk in memory_chunks])
        for chunk, embedding in zip(memory_chunks, embeddings):
            chunk.embedding = embedding

        profile = {
            "agent_id": agent_id,
            **normalized,
        }

        return PersonaState(agent_id=agent_id, profile=profile, memories=memory_chunks)


def _build_manager_summary(personas: list[PersonaState]) -> dict[str, Any]:
    if not personas:
        return {
            "cognitive_load_heatmap": {},
            "sentiment": {"overall": "neutral", "breakdown": {}, "raw": {}},
            "demographics": [],
        }

    dimensions = [
        "visual_clarity",
        "control_discoverability",
        "glanceability_under_stress",
        "error_recovery",
    ]

    dimension_totals: defaultdict[str, int] = defaultdict(int)
    for persona in personas:
        for dim in dimensions:
            dimension_totals[dim] += _safe_int(
                persona.profile.get("cognitive_load", {}).get(dim),
                50,
            )

    cognitive_load_heatmap = {
        dim: round(dimension_totals[dim] / len(personas), 1) for dim in dimensions
    }

    sentiment_counter = Counter(
        _safe_str(persona.profile.get("sentiment_label"), "neutral") for persona in personas
    )
    avg_sentiment = round(
        sum(_safe_float(persona.profile.get("sentiment_score"), 0.0) for persona in personas)
        / max(1, len(personas)),
        3,
    )

    if avg_sentiment >= 0.2:
        overall_sentiment = "positive"
    elif avg_sentiment <= -0.2:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    demographics = [
        {
            "segment": "age_range",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("age_range"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "household",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("household"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "occupation",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("occupation"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "region",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("region"), "unknown") for p in personas)
            ),
        },
    ]

    return {
        "cognitive_load_heatmap": cognitive_load_heatmap,
        "sentiment": {
            "overall": overall_sentiment,
            "breakdown": {
                "positive": sentiment_counter.get("positive", 0),
                "neutral": sentiment_counter.get("neutral", 0),
                "negative": sentiment_counter.get("negative", 0),
                "average_score": avg_sentiment,
            },
            "raw": {
                "average_score": avg_sentiment,
                "counts": dict(sentiment_counter),
            },
        },
        "demographics": demographics,
    }


async def _retrieve_relevant_memories(
    persona: PersonaState,
    question: str,
    top_k: int = TOP_K_MEMORIES,
) -> list[dict[str, Any]]:
    query_embedding = await _embed_text(question)

    scored: list[tuple[float, MemoryChunk]] = []
    for chunk in persona.memories:
        if query_embedding is not None and chunk.embedding is not None:
            score = _cosine_similarity(query_embedding, chunk.embedding)
        else:
            score = _lexical_score(question, chunk.text)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(1, top_k)]

    return [
        {
            "text": chunk.text,
            "score": round(score, 4),
            "kind": chunk.kind,
            "created_at": chunk.created_at,
        }
        for score, chunk in top
    ]


def _followup_prompt(persona: PersonaState, question: str, memory_snippets: list[dict[str, Any]]) -> tuple[str, str]:
    system_prompt = f"""
You are role-playing this persona in first person:
- Name: {_safe_str(persona.profile.get("name"), "Unknown")}
- Age range: {_safe_str(persona.profile.get("age_range"), "Unknown")}
- Occupation: {_safe_str(persona.profile.get("occupation"), "Unknown")}
- Household: {_safe_str(persona.profile.get("household"), "Unknown")}
- Region: {_safe_str(persona.profile.get("region"), "Unknown")}
- Reaction summary: {_safe_str(persona.profile.get("reaction_summary"), "")}
- Quote: {_safe_str(persona.profile.get("quote"), "")}

Answer naturally as this persona would.
Return valid JSON only with keys:
- answer (string)
- confidence (number 0-1)
- rationale (string)
"""

    snippet_text = "\n".join(
        [f"{idx + 1}. {item.get('text', '')}" for idx, item in enumerate(memory_snippets)]
    )

    user_prompt = f"""
Question from researcher: {question}

Relevant memory snippets:
{snippet_text or "(none)"}

Provide a grounded first-person response as the persona.
"""

    return system_prompt, user_prompt


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "model": OPENAI_MODEL,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "persona_store_size": await store.size(),
    }


@app.post("/synthetic/simulate")
async def simulate_focus_group(request: SimulateRequest) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)

    tasks = [
        _generate_one_persona(request=request, index=index, semaphore=semaphore)
        for index in range(request.persona_count)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    personas: list[PersonaState] = []
    failures: list[str] = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            failures.append(f"persona_{idx + 1}: {result}")
            fallback_profile = _default_persona(idx)
            fallback_agent_id = f"agent_{uuid.uuid4().hex[:10]}"
            fallback_state = PersonaState(
                agent_id=fallback_agent_id,
                profile={
                    "agent_id": fallback_agent_id,
                    **fallback_profile,
                    "generation_error": str(result),
                },
                memories=[
                    MemoryChunk(text=fallback_profile["reaction_summary"], kind="seed"),
                    MemoryChunk(text=fallback_profile["quote"], kind="seed"),
                ],
            )
            personas.append(fallback_state)
            continue

        personas.append(result)

    await store.upsert_many(personas)

    manager_summary = _build_manager_summary(personas)

    persona_payloads = []
    for persona in personas:
        persona_payloads.append(
            {
                **persona.profile,
                "memory_count": len(persona.memories),
            }
        )

    return {
        "run_id": f"run_{uuid.uuid4().hex[:10]}",
        "target_audience": request.target_audience,
        "stimulus_description": request.stimulus_description,
        "image_url": request.image_url,
        "manager_summary": manager_summary,
        "personas": persona_payloads,
        "generation_failures": failures,
    }


@app.post("/synthetic/followup")
async def ask_persona_followup(request: FollowupRequest) -> dict[str, Any]:
    persona = await store.get(request.agent_id)
    if persona is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown agent_id '{request.agent_id}'. Run /synthetic/simulate first and use a returned agent_id."
            ),
        )

    snippets = await _retrieve_relevant_memories(persona, request.question, TOP_K_MEMORIES)
    system_prompt, user_prompt = _followup_prompt(persona, request.question, snippets)

    followup_raw = await _chat_json(system_prompt, user_prompt, temperature=0.7)
    answer = _safe_str(
        followup_raw.get("answer"),
        "I need more context, but based on my prior reaction I would simplify the high-stress controls.",
    )

    new_chunks = [
        MemoryChunk(text=f"Researcher asked: {request.question}", kind="question"),
        MemoryChunk(text=f"Persona answered: {answer}", kind="answer"),
    ]

    embeddings = await asyncio.gather(*[_embed_text(chunk.text) for chunk in new_chunks])
    for chunk, embedding in zip(new_chunks, embeddings):
        chunk.embedding = embedding

    await store.append_memories(request.agent_id, new_chunks)

    return {
        "agent_id": request.agent_id,
        "question": request.question,
        "answer": answer,
        "memory_snippets": snippets,
        "raw": {
            "model_response": followup_raw,
            "persona_profile": persona.profile,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

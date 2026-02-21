import asyncio
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import pstdev
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

try:
    from datasets import get_dataset_split_names, load_dataset
except Exception as datasets_import_error:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]
    get_dataset_split_names = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = datasets_import_error
else:
    _DATASETS_IMPORT_ERROR = None

APP_NAME = "synthetic-focus-group-backend"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL_FALLBACKS = [
    model.strip()
    for model in os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-5-mini,gpt-4o-mini").split(",")
    if model.strip()
]
DEFAULT_PERSONA_COUNT = int(os.getenv("FG_PERSONA_COUNT", "6"))
MAX_PERSONA_COUNT = int(os.getenv("FG_MAX_PERSONA_COUNT", "1000"))
MAX_PARALLEL_TASKS = int(os.getenv("FG_MAX_PARALLEL_TASKS", "8"))
TOP_K_MEMORIES = int(os.getenv("FG_TOP_K_MEMORIES", "5"))
MAX_PERSONA_STORE = int(os.getenv("FG_MAX_PERSONA_STORE", "5000"))
HF_MEMORY_DATASET = os.getenv("FG_HF_MEMORY_DATASET", "yahoo_answers_topics")
HF_MEMORY_POOL_MIN = int(os.getenv("FG_HF_MEMORY_POOL_MIN", "10"))
HF_MEMORY_POOL_MAX = int(os.getenv("FG_HF_MEMORY_POOL_MAX", "15"))
HF_MEMORY_PER_PERSONA_MIN = int(os.getenv("FG_HF_MEMORY_PER_PERSONA_MIN", "2"))
HF_MEMORY_PER_PERSONA_MAX = int(os.getenv("FG_HF_MEMORY_PER_PERSONA_MAX", "3"))
HF_MEMORY_CACHE_TTL_SEC = int(os.getenv("FG_HF_MEMORY_CACHE_TTL_SEC", "1800"))
HF_PERSONA_DATASET = os.getenv("FG_HF_PERSONA_DATASET", "SynthlabsAI/PERSONA")
HF_PERSONA_DATASET_FALLBACKS = [
    value.strip()
    for value in os.getenv(
        "FG_HF_PERSONA_DATASET_FALLBACKS",
        "SYNTTHLABSAI/Persona,SYNTHLABSAI/Persona",
    ).split(",")
    if value.strip()
]
HF_PERSONA_POOL_MIN = int(os.getenv("FG_HF_PERSONA_POOL_MIN", "30"))
HF_PERSONA_POOL_MAX = int(os.getenv("FG_HF_PERSONA_POOL_MAX", "160"))
HF_PERSONA_CACHE_TTL_SEC = int(os.getenv("FG_HF_PERSONA_CACHE_TTL_SEC", "1800"))

FALLBACK_FIRST_NAMES = [
    "Maya",
    "Nora",
    "Elena",
    "Priya",
    "Danielle",
    "Monica",
    "Renee",
    "Jasmine",
    "Alicia",
    "Sofia",
    "Brenda",
    "Carla",
]

FALLBACK_OCCUPATIONS = [
    "Nurse",
    "School Administrator",
    "Insurance Specialist",
    "Restaurant Manager",
    "Office Coordinator",
    "Small Business Owner",
    "Dental Hygienist",
    "Paralegal",
    "Retail Operations Lead",
    "Teacher",
    "HR Generalist",
    "Project Coordinator",
]

FALLBACK_REGIONS = [
    "Midwest suburb",
    "Southwest suburb",
    "Southeast suburb",
    "Northeast suburb",
    "Pacific Northwest suburb",
    "Mountain West suburb",
]

FALLBACK_DAILY_CONTEXTS = [
    "weekday routine with limited attention and time",
    "back-to-back tasks between work and family obligations",
    "weekend planning with competing priorities",
    "quick decisions while managing interruptions",
    "comparison-shopping mindset with budget constraints",
]

FALLBACK_PERSONA_TYPES = [
    "time-pressed multitasker",
    "safety-first planner",
    "tech-cautious pragmatist",
    "efficiency-focused optimizer",
    "value-conscious decision maker",
    "confidence-seeking adopter",
]

FALLBACK_GENDER_IDENTITIES = [
    "woman",
    "man",
    "non-binary",
]

FALLBACK_ETHNICITIES = [
    "Black",
    "White",
    "Latina",
    "South Asian",
    "East Asian",
    "Middle Eastern",
    "Mixed",
]

FAST_FOOD_HINTS = [
    "kfc",
    "mcdonald",
    "burger king",
    "wendy",
    "popeyes",
    "chick-fil-a",
    "fast food",
    "fried chicken",
    "fries",
    "soda",
    "combo meal",
]

HEALTH_CLAIM_PATTERN = re.compile(
    r"\b(health(y|ier)?|nutriti(ous|on|onal)|wellness|good for (my|your|our) health|health benefits?)\b",
    re.IGNORECASE,
)
OBJECTIVE_UI_HINTS = [
    "does this image have text",
    "find the",
    "button",
    "click",
    "tap",
    "locate",
    "where is",
    "checkout",
    "sign in",
    "navbar",
    "menu",
    "screen",
    "ui",
    "ux",
    "form field",
    "error message",
]
FALLBACK_REAL_WORLD_MEMORIES = [
    "I bought a product once because the reviews sounded convincing, but it underdelivered in daily use.",
    "I usually ignore flashy ads and look for practical details, especially return policy and long-term value.",
    "A checkout flow confused me recently because key actions were hidden below too much content.",
    "I changed my mind on a service after hearing mixed feedback from friends with different priorities.",
    "I prefer clear, direct language over hype because I often compare options quickly between errands.",
    "I once chose a cheaper option, then regretted it when reliability problems showed up within a month.",
    "I trust products more when I can quickly understand what problem they solve for someone like me.",
    "I am skeptical of broad claims unless I can find concrete examples or third-party validation.",
    "I have clicked away from pages that felt cluttered or demanded too much effort to complete one task.",
    "I am more likely to try something new if the first interaction feels simple and low-risk.",
    "I value honest tradeoff explanations more than polished branding language.",
    "I often read both highest and lowest reviews to find recurring patterns before deciding.",
]


class SimulateRequest(BaseModel):
    target_audience: str | None = Field(
        default=None,
        description="Optional audience description. If omitted, backend auto-generates one.",
    )
    stimulus_description: str = Field(min_length=3)
    image_url: str | None = None
    persona_count: int = Field(default=DEFAULT_PERSONA_COUNT, ge=3, le=MAX_PERSONA_COUNT)
    analysis_dimensions: list[str] | None = Field(
        default=None,
        description=(
            "Optional custom evaluation dimensions (e.g. ['Trust', 'Clarity']). "
            "Backend can mix these with auto-generated dimensions."
        ),
    )


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
                self._store = dict(sorted_items[:MAX_PERSONA_STORE])

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


app = FastAPI(title=APP_NAME, version="1.1.0")
store = PersonaStore()
_client: AsyncOpenAI | None = None
_real_world_memory_cache: list[str] = []
_real_world_memory_cache_at: float = 0.0
_persona_seed_cache: list[dict[str, Any]] = []
_persona_seed_cache_at: float = 0.0


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


def _clean_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _dimension_key(label: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in label.strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized or f"dimension_{uuid.uuid4().hex[:6]}"


def _make_dimension_spec(label: str, description: str = "") -> dict[str, str]:
    cleaned_label = label.strip() if label.strip() else "General Clarity"
    return {
        "key": _dimension_key(cleaned_label),
        "label": cleaned_label,
        "description": description.strip(),
    }


def _fallback_dimension_specs() -> list[dict[str, str]]:
    return [
        _make_dimension_spec(
            "Message Clarity",
            "How clear and understandable the core message is.",
        ),
        _make_dimension_spec(
            "Perceived Value",
            "How strongly the stimulus communicates useful or worthwhile value.",
        ),
        _make_dimension_spec(
            "Trust & Credibility",
            "How believable and trustworthy the claims and presentation feel.",
        ),
        _make_dimension_spec(
            "Actionability",
            "How easy it is to decide what action to take next after viewing.",
        ),
    ]


def _dedupe_dimension_specs(specs: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()

    for spec in specs:
        key = _dimension_key(_safe_str(spec.get("key"), spec.get("label", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            _make_dimension_spec(
                _safe_str(spec.get("label"), key.replace("_", " ").title()),
                _safe_str(spec.get("description"), ""),
            )
        )

    return deduped


async def _auto_generate_dimension_specs(
    target_audience: str,
    stimulus_description: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a UX research methodologist. Return strict JSON listing evaluation dimensions "
        "specific to the provided stimulus."
    )

    user_prompt = f"""
Target audience:
{target_audience}

Stimulus:
{stimulus_description}

Return JSON with key dimensions: array of 3-6 objects, each object with:
- label (short phrase)
- description (one sentence)

Rules:
- Avoid automotive-specific assumptions unless explicitly in stimulus.
- Use dimensions that can apply to this exact stimulus category.
"""

    try:
        generated = await _chat_json(system_prompt, user_prompt, temperature=0.6)
        raw_dimensions = generated.get("dimensions", [])
    except Exception:
        raw_dimensions = []

    specs: list[dict[str, str]] = []
    if isinstance(raw_dimensions, list):
        for raw in raw_dimensions:
            if isinstance(raw, dict):
                specs.append(
                    _make_dimension_spec(
                        _safe_str(raw.get("label"), "General Clarity"),
                        _safe_str(raw.get("description"), ""),
                    )
                )
            elif isinstance(raw, str):
                specs.append(_make_dimension_spec(raw))

    return _dedupe_dimension_specs(specs)


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
    model_candidates: list[str] = []
    for candidate in [OPENAI_MODEL, *OPENAI_MODEL_FALLBACKS]:
        cleaned = _clean_optional_str(candidate)
        if cleaned and cleaned not in model_candidates:
            model_candidates.append(cleaned)

    if not model_candidates:
        raise RuntimeError("No chat model configured.")

    last_error: Exception | None = None
    for model_name in model_candidates:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content or "{}"
            parsed = _extract_json_object(content)
            parsed.setdefault("_model", model_name)
            return parsed
        except Exception as error:  # pragma: no cover - network/model runtime dependent
            last_error = error

    raise RuntimeError(
        f"All configured models failed ({', '.join(model_candidates)}): {last_error}"
    )


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


def _heuristic_task_type(stimulus_description: str) -> str:
    normalized = stimulus_description.lower()
    if any(hint in normalized for hint in OBJECTIVE_UI_HINTS):
        return "Objective UI Task"
    return "Subjective Concept"


async def _classify_task_type(stimulus_description: str) -> dict[str, str]:
    heuristic = _heuristic_task_type(stimulus_description)
    client = ensure_openai_client()

    system_prompt = (
        "Classify the user request into one of exactly two labels and return strict JSON: "
        "'Objective UI Task' or 'Subjective Concept'."
    )
    user_prompt = f"""
Stimulus/request:
{stimulus_description}

Rules:
- Use "Objective UI Task" for factual UI/UX tasks (locating elements, checking text, validating states).
- Use "Subjective Concept" for ads, concepts, product opinions, or perception-based critique.

Return JSON:
- task_type
- rationale (<= 25 words)
"""

    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json_object(content)

        task_type = _safe_str(parsed.get("task_type"), heuristic)
        if task_type not in {"Objective UI Task", "Subjective Concept"}:
            task_type = heuristic

        return {
            "task_type": task_type,
            "task_type_rationale": _safe_str(parsed.get("rationale"), "Classified by primary model"),
        }
    except Exception as error:
        return {
            "task_type": heuristic,
            "task_type_rationale": f"Heuristic fallback used: {error}",
        }


def _normalize_memory_seed_text(value: Any, max_chars: int = 320) -> str:
    text = _safe_str(value, "")
    if not text:
        return ""
    compact = " ".join(text.split())
    return compact[:max_chars]


def _dataset_candidates(primary: str, fallbacks: list[str]) -> list[str]:
    ordered: list[str] = []
    for value in [primary, *fallbacks]:
        cleaned = _clean_optional_str(value)
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered


def _row_first_text(row: dict[str, Any], keys: list[str], max_chars: int = 120) -> str:
    for key in keys:
        cleaned = _normalize_memory_seed_text(row.get(key), max_chars=max_chars)
        if cleaned:
            return cleaned
    return ""


def _row_text_list(row: dict[str, Any], keys: list[str], max_items: int = 3) -> list[str]:
    collected: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            for entry in value:
                cleaned = _normalize_memory_seed_text(entry, max_chars=220)
                if cleaned:
                    collected.append(cleaned)
                    if len(collected) >= max_items:
                        return collected
        else:
            cleaned = _normalize_memory_seed_text(value, max_chars=220)
            if cleaned:
                collected.append(cleaned)
                if len(collected) >= max_items:
                    return collected
    return collected


def _normalize_budget_sensitivity(raw: str) -> str:
    lowered = raw.lower()
    if any(token in lowered for token in ["high", "very", "strict", "sensitive", "tight"]):
        return "high"
    if any(token in lowered for token in ["low", "not sensitive", "flexible", "premium"]):
        return "low"
    return "medium"


def _normalize_hf_persona_seed_row(row: dict[str, Any]) -> dict[str, Any]:
    name = _row_first_text(
        row,
        ["name", "full_name", "persona_name", "username", "first_name"],
        max_chars=80,
    )
    age_range = _row_first_text(row, ["age_range", "age_bracket", "age"], max_chars=24)
    occupation = _row_first_text(
        row,
        ["occupation", "job_title", "profession", "role", "work"],
        max_chars=80,
    )
    household = _row_first_text(
        row,
        ["household", "family_status", "family", "living_situation"],
        max_chars=120,
    )
    region = _row_first_text(row, ["region", "location", "country", "state"], max_chars=80)
    persona_type = _row_first_text(
        row,
        ["persona_type", "archetype", "personality_type", "segment"],
        max_chars=80,
    )
    gender_identity = _row_first_text(
        row,
        ["gender_identity", "gender", "sex"],
        max_chars=32,
    )
    ethnicity = _row_first_text(row, ["ethnicity", "race"], max_chars=48)
    tech_comfort = _row_first_text(
        row,
        ["tech_comfort", "technology_adoption", "digital_literacy", "tech_savviness"],
        max_chars=32,
    ).lower()
    daily_context = _row_first_text(
        row,
        ["daily_context", "context", "lifestyle", "habits", "routine"],
        max_chars=180,
    )
    budget_raw = _row_first_text(
        row,
        ["budget_sensitivity", "price_sensitivity", "budget", "spending_style"],
        max_chars=48,
    )
    budget_sensitivity = _normalize_budget_sensitivity(budget_raw) if budget_raw else ""

    memory_stream = _row_text_list(
        row,
        [
            "past_experiences",
            "core_opinions",
            "preferences",
            "interests",
            "bio",
            "background",
            "description",
            "profile",
            "persona",
            "opinion",
            "experience",
        ],
        max_items=3,
    )

    seed: dict[str, Any] = {
        "name": name,
        "age_range": age_range,
        "occupation": occupation,
        "household": household,
        "region": region,
        "persona_type": persona_type,
        "gender_identity": gender_identity,
        "ethnicity": ethnicity,
        "tech_comfort": tech_comfort,
        "daily_context": daily_context,
        "budget_sensitivity": budget_sensitivity,
        "memory_stream": memory_stream,
    }

    return {key: value for key, value in seed.items() if value}


def _extract_memory_text_from_row(row: dict[str, Any]) -> str:
    candidates: list[str] = []

    for key in [
        "question_title",
        "question_content",
        "best_answer",
        "summary",
        "text",
        "content",
        "review_text",
        "body",
        "title",
        "question",
        "answer",
        "bio",
        "background",
        "description",
        "profile",
        "persona",
        "opinion",
        "experience",
    ]:
        cleaned = _normalize_memory_seed_text(row.get(key))
        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return ""

    combined = " | ".join(candidates[:3])
    return _normalize_memory_seed_text(combined, max_chars=420)


def _load_hf_memory_pool_sync(sample_size: int) -> list[str]:
    if load_dataset is None:
        return []

    min_pool = max(1, HF_MEMORY_POOL_MIN)
    max_pool = max(min_pool, HF_MEMORY_POOL_MAX)
    requested = max(min_pool, min(max_pool, sample_size))

    try:
        start_pct = random.randint(0, 97)
        end_pct = min(100, start_pct + 2)
        split = f"train[{start_pct}%:{end_pct}%]"
        dataset = load_dataset(HF_MEMORY_DATASET, split=split)
    except Exception:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    try:
        for row in dataset:
            if not isinstance(row, dict):
                continue
            text = _extract_memory_text_from_row(row)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(text)
            if len(candidates) >= requested * 5:
                break
    except Exception:
        return []

    random.shuffle(candidates)
    return candidates[:requested]


def _sample_from_pool(pool: list[str], sample_count: int) -> list[str]:
    clean_pool = [_normalize_memory_seed_text(item) for item in pool if _normalize_memory_seed_text(item)]
    if not clean_pool:
        return []
    if len(clean_pool) >= sample_count:
        return random.sample(clean_pool, sample_count)
    return [random.choice(clean_pool) for _ in range(sample_count)]


async def _get_real_world_memory_pool() -> list[str]:
    global _real_world_memory_cache
    global _real_world_memory_cache_at

    now = time.time()
    if (
        _real_world_memory_cache
        and (now - _real_world_memory_cache_at) < HF_MEMORY_CACHE_TTL_SEC
    ):
        return _real_world_memory_cache.copy()

    target_size = random.randint(
        max(1, HF_MEMORY_POOL_MIN),
        max(max(1, HF_MEMORY_POOL_MIN), HF_MEMORY_POOL_MAX),
    )
    loaded = await asyncio.to_thread(_load_hf_memory_pool_sync, target_size)

    if loaded:
        _real_world_memory_cache = loaded
        _real_world_memory_cache_at = now
        return loaded.copy()

    _real_world_memory_cache = FALLBACK_REAL_WORLD_MEMORIES.copy()
    _real_world_memory_cache_at = now
    return _real_world_memory_cache.copy()


def _normalize_tech_comfort(raw: str) -> str:
    lowered = raw.lower()
    if any(token in lowered for token in ["high", "expert", "advanced", "power"]):
        return "high"
    if any(token in lowered for token in ["low", "beginner", "novice", "minimal"]):
        return "low"
    return "medium"


def _persona_seed_fingerprint(seed: dict[str, Any]) -> str:
    return "|".join(
        [
            _safe_str(seed.get("name"), "").lower(),
            _safe_str(seed.get("occupation"), "").lower(),
            _safe_str(seed.get("persona_type"), "").lower(),
            _safe_str(seed.get("region"), "").lower(),
            ";".join(_normalize_string_list(seed.get("memory_stream"), [])),
        ]
    )


def _safe_dataset_split_names(dataset_name: str) -> list[str]:
    split_names: list[str] = []
    if get_dataset_split_names is not None:
        try:
            names = get_dataset_split_names(dataset_name)
            if isinstance(names, list):
                split_names = [str(name) for name in names if str(name).strip()]
        except Exception:
            split_names = []

    ordered: list[str] = []
    for value in [*split_names, "train", "validation", "test"]:
        cleaned = _clean_optional_str(value)
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)

    return ordered or ["train"]


def _load_hf_persona_pool_sync(sample_size: int) -> list[dict[str, Any]]:
    if load_dataset is None:
        return []

    dataset_candidates = _dataset_candidates(
        HF_PERSONA_DATASET,
        HF_PERSONA_DATASET_FALLBACKS,
    )
    if not dataset_candidates:
        return []

    min_pool = max(1, HF_PERSONA_POOL_MIN)
    max_pool = max(min_pool, HF_PERSONA_POOL_MAX)
    requested = max(min_pool, min(max_pool, sample_size))

    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for dataset_name in dataset_candidates:
        split_names = _safe_dataset_split_names(dataset_name)

        for split_name in split_names:
            start_pct = random.randint(0, 97)
            end_pct = min(100, start_pct + 2)
            split_slice = f"{split_name}[{start_pct}%:{end_pct}%]"

            datasets_to_scan = []
            try:
                datasets_to_scan.append(load_dataset(dataset_name, split=split_slice))
            except Exception:
                pass

            if not datasets_to_scan:
                try:
                    stream_ds = load_dataset(dataset_name, split=split_name, streaming=True)
                    datasets_to_scan.append(stream_ds)
                except Exception:
                    pass

            for dataset_obj in datasets_to_scan:
                try:
                    skip = random.randint(0, 150)
                    idx = 0
                    for row in dataset_obj:
                        if idx < skip:
                            idx += 1
                            continue
                        idx += 1

                        if not isinstance(row, dict):
                            continue
                        seed = _normalize_hf_persona_seed_row(row)
                        if not seed:
                            continue

                        fingerprint = _persona_seed_fingerprint(seed)
                        if fingerprint in seen:
                            continue
                        seen.add(fingerprint)
                        collected.append(seed)
                        if len(collected) >= requested * 4:
                            break
                except Exception:
                    continue

            if len(collected) >= requested:
                break

        if len(collected) >= requested:
            break

    random.shuffle(collected)
    return collected[:requested]


def _query_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _persona_seed_corpus(seed: dict[str, Any]) -> str:
    parts = [
        _safe_str(seed.get("name"), ""),
        _safe_str(seed.get("age_range"), ""),
        _safe_str(seed.get("occupation"), ""),
        _safe_str(seed.get("household"), ""),
        _safe_str(seed.get("region"), ""),
        _safe_str(seed.get("persona_type"), ""),
        _safe_str(seed.get("gender_identity"), ""),
        _safe_str(seed.get("ethnicity"), ""),
        _safe_str(seed.get("daily_context"), ""),
        _safe_str(seed.get("budget_sensitivity"), ""),
        " ".join(_normalize_string_list(seed.get("memory_stream"), [])),
    ]
    return " ".join([part for part in parts if part]).strip()


def _token_overlap_score(query: set[str], doc: set[str]) -> float:
    if not query or not doc:
        return 0.0
    overlap = len(query.intersection(doc))
    return overlap / max(1, len(query))


def _select_persona_seeds_for_audience(
    seed_pool: list[dict[str, Any]],
    target_audience: str,
    stimulus_description: str,
    desired_count: int,
) -> list[dict[str, Any]]:
    if not seed_pool:
        return []

    audience_tokens = _query_tokens(target_audience)
    stimulus_tokens = _query_tokens(stimulus_description)

    scored: list[dict[str, Any]] = []
    for seed in seed_pool:
        corpus_tokens = _query_tokens(_persona_seed_corpus(seed))
        if not corpus_tokens:
            continue
        audience_score = _token_overlap_score(audience_tokens, corpus_tokens)
        stimulus_score = _token_overlap_score(stimulus_tokens, corpus_tokens)
        base_score = (0.7 * audience_score) + (0.3 * stimulus_score) + random.random() * 0.01
        scored.append({"score": base_score, "seed": seed})

    if not scored:
        return seed_pool[: max(1, min(desired_count, len(seed_pool)))]

    selected: list[dict[str, Any]] = []
    remaining = scored[:]
    occupation_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()

    target_count = max(1, min(desired_count, len(seed_pool)))
    while remaining and len(selected) < target_count:
        best_idx = 0
        best_score = float("-inf")

        for idx, entry in enumerate(remaining):
            seed = entry["seed"]
            occupation = _safe_str(seed.get("occupation"), "").lower()
            region = _safe_str(seed.get("region"), "").lower()
            penalty = 0.08 * occupation_counts.get(occupation, 0) + 0.04 * region_counts.get(region, 0)
            adjusted = float(entry["score"]) - penalty
            if adjusted > best_score:
                best_idx = idx
                best_score = adjusted

        best_entry = remaining.pop(best_idx)
        best_seed = best_entry["seed"]
        selected.append(best_seed)

        occupation = _safe_str(best_seed.get("occupation"), "").lower()
        region = _safe_str(best_seed.get("region"), "").lower()
        if occupation:
            occupation_counts[occupation] += 1
        if region:
            region_counts[region] += 1

    return selected


async def _get_persona_seed_pool(
    target_audience: str,
    stimulus_description: str,
    persona_count: int,
) -> list[dict[str, Any]]:
    global _persona_seed_cache
    global _persona_seed_cache_at

    now = time.time()
    pool_is_fresh = _persona_seed_cache and (now - _persona_seed_cache_at) < HF_PERSONA_CACHE_TTL_SEC

    if not pool_is_fresh:
        desired_pool_size = max(
            HF_PERSONA_POOL_MIN,
            min(HF_PERSONA_POOL_MAX, max(persona_count * 3, HF_PERSONA_POOL_MIN)),
        )
        loaded = await asyncio.to_thread(_load_hf_persona_pool_sync, desired_pool_size)
        if loaded:
            _persona_seed_cache = loaded
            _persona_seed_cache_at = now
        else:
            _persona_seed_cache = []
            _persona_seed_cache_at = now

    base_pool = [dict(entry) for entry in _persona_seed_cache]
    if not base_pool:
        return []

    selected = _select_persona_seeds_for_audience(
        base_pool,
        target_audience=target_audience,
        stimulus_description=stimulus_description,
        desired_count=max(1, min(len(base_pool), max(persona_count, HF_PERSONA_POOL_MIN))),
    )
    return [dict(entry) for entry in selected]


def _merge_blueprint_with_persona_seed(
    blueprint: dict[str, Any],
    persona_seed: dict[str, Any],
) -> dict[str, Any]:
    if not persona_seed:
        return blueprint

    merged = dict(blueprint)
    for key in [
        "name",
        "age_range",
        "occupation",
        "household",
        "region",
        "persona_type",
        "gender_identity",
        "ethnicity",
        "daily_context",
    ]:
        value = _safe_str(persona_seed.get(key), "")
        if value:
            merged[key] = value

    budget = _safe_str(persona_seed.get("budget_sensitivity"), "")
    if budget:
        merged["budget_sensitivity"] = _normalize_budget_sensitivity(budget)

    tech = _safe_str(persona_seed.get("tech_comfort"), "")
    if tech:
        merged["tech_comfort"] = _normalize_tech_comfort(tech)

    seed_memories = _normalize_string_list(persona_seed.get("memory_stream"), [])
    existing_memories = _normalize_string_list(merged.get("memory_stream"), [])
    combined = [*existing_memories]
    for memory in seed_memories:
        if memory not in combined:
            combined.append(memory)
    merged["memory_stream"] = combined

    return merged


def _default_target_audience(stimulus_description: str) -> str:
    stimulus_fragment = stimulus_description.strip().lower()[:80]
    return (
        "U.S. adults likely to encounter this type of product/message in everyday decision-making, "
        f"evaluating: {stimulus_fragment}"
    )


async def _resolve_target_audience(
    requested_target: str | None,
    stimulus_description: str,
) -> dict[str, Any]:
    normalized_request = _clean_optional_str(requested_target)
    if normalized_request:
        return {
            "target_audience": normalized_request,
            "target_audience_generated": False,
            "target_audience_generation_notes": "Provided by user or host model",
        }

    system_prompt = (
        "You are a market research planner. Return strict JSON with a concrete, specific "
        "target audience statement suited for synthetic focus groups."
    )

    user_prompt = f"""
Stimulus to analyze:
{stimulus_description}

Return JSON with keys:
- target_audience (string, 1 sentence, very specific)
- rationale (string, <= 25 words)
"""

    try:
        generated = await _chat_json(system_prompt, user_prompt, temperature=0.7)
        target_audience = _safe_str(
            generated.get("target_audience"),
            _default_target_audience(stimulus_description),
        )
        rationale = _safe_str(generated.get("rationale"), "Auto-generated from stimulus")
        return {
            "target_audience": target_audience,
            "target_audience_generated": True,
            "target_audience_generation_notes": rationale,
        }
    except Exception as error:
        return {
            "target_audience": _default_target_audience(stimulus_description),
            "target_audience_generated": True,
            "target_audience_generation_notes": f"Fallback audience used: {error}",
        }


async def _resolve_analysis_dimensions(
    requested_dimensions: list[str] | None,
    target_audience: str,
    stimulus_description: str,
) -> list[dict[str, str]]:
    requested_specs: list[dict[str, str]] = []
    if requested_dimensions:
        for label in requested_dimensions:
            cleaned = _clean_optional_str(label)
            if cleaned:
                requested_specs.append(_make_dimension_spec(cleaned))

    requested_specs = _dedupe_dimension_specs(requested_specs)

    auto_specs = await _auto_generate_dimension_specs(
        target_audience=target_audience,
        stimulus_description=stimulus_description,
    )

    fallback_specs = _fallback_dimension_specs()

    merged = _dedupe_dimension_specs([*requested_specs, *auto_specs, *fallback_specs])

    # Keep a focused set so scoring remains interpretable.
    return merged[:6]


def _dimension_keys(dimensions: list[dict[str, str]]) -> list[str]:
    keys: list[str] = []
    for idx, spec in enumerate(dimensions):
        key = _safe_str(spec.get("key"), f"dimension_{idx + 1}")
        if key not in keys:
            keys.append(key)
    return keys


def _dimension_score_default(
    *,
    index: int,
    key_index: int,
) -> int:
    base = 48 + ((index * 9 + key_index * 7) % 20)
    return _safe_int(base, 50)


def _resolve_dimension_score(
    cognitive_load_raw: dict[str, Any],
    default_cognitive_load: dict[str, Any],
    spec: dict[str, str],
    key_index: int,
) -> int:
    key = _safe_str(spec.get("key"), f"dimension_{key_index + 1}")
    label = _safe_str(spec.get("label"), key.replace("_", " ").title())
    alias_keys = [
        key,
        _dimension_key(label),
        label,
        label.lower(),
        label.replace(" ", "_").lower(),
    ]

    for alias in alias_keys:
        if alias in cognitive_load_raw:
            return _safe_int(
                cognitive_load_raw.get(alias),
                _safe_int(default_cognitive_load.get(key), 50),
            )

    return _safe_int(
        default_cognitive_load.get(key),
        50,
    )


def _stimulus_flags(stimulus_description: str) -> dict[str, bool]:
    normalized = stimulus_description.lower()
    is_fast_food = any(hint in normalized for hint in FAST_FOOD_HINTS)
    return {
        "is_fast_food": is_fast_food,
    }


def _sanitize_claim_text(text: str, flags: dict[str, bool]) -> str:
    cleaned = _safe_str(text, "")
    if not cleaned:
        return ""

    if flags.get("is_fast_food") and HEALTH_CLAIM_PATTERN.search(cleaned):
        return "I like the taste and convenience, but I do not see this as a health-focused option."

    return cleaned


def _sanitize_persona_claims(persona: dict[str, Any], stimulus_description: str) -> dict[str, Any]:
    flags = _stimulus_flags(stimulus_description)
    if not flags.get("is_fast_food"):
        return persona

    sanitized = dict(persona)
    sanitized["reaction_summary"] = _sanitize_claim_text(
        _safe_str(persona.get("reaction_summary"), ""),
        flags,
    )
    sanitized["quote"] = _sanitize_claim_text(_safe_str(persona.get("quote"), ""), flags)

    for list_key in ["reasons", "concerns", "delights"]:
        values = persona.get(list_key, [])
        if not isinstance(values, list):
            values = []
        sanitized[list_key] = [
            _sanitize_claim_text(_safe_str(item, ""), flags) for item in values if _safe_str(item, "")
        ]

    return sanitized


def _fallback_blueprints(persona_count: int) -> list[dict[str, Any]]:
    random_seed = int(time.time()) % 10_000
    rng = random.Random(random_seed)

    first_names = FALLBACK_FIRST_NAMES.copy()
    occupations = FALLBACK_OCCUPATIONS.copy()
    regions = FALLBACK_REGIONS.copy()
    contexts = FALLBACK_DAILY_CONTEXTS.copy()
    persona_types = FALLBACK_PERSONA_TYPES.copy()
    gender_identities = FALLBACK_GENDER_IDENTITIES.copy()
    ethnicities = FALLBACK_ETHNICITIES.copy()

    rng.shuffle(first_names)
    rng.shuffle(occupations)
    rng.shuffle(regions)
    rng.shuffle(contexts)
    rng.shuffle(persona_types)
    rng.shuffle(gender_identities)
    rng.shuffle(ethnicities)

    blueprints: list[dict[str, Any]] = []
    for idx in range(persona_count):
        stance = ["champion", "fence-sitter", "skeptic"][idx % 3]

        blueprints.append(
            {
                "name": first_names[idx % len(first_names)],
                "age_range": ["30-39", "35-44", "40-49", "45-54"][idx % 4],
                "occupation": occupations[idx % len(occupations)],
                "household": "Suburban household with kids",
                "region": regions[idx % len(regions)],
                "persona_type": persona_types[idx % len(persona_types)],
                "gender_identity": gender_identities[idx % len(gender_identities)],
                "ethnicity": ethnicities[idx % len(ethnicities)],
                "tech_comfort": ["low", "medium", "high"][idx % 3],
                "daily_context": contexts[idx % len(contexts)],
                "budget_sensitivity": ["high", "medium", "low"][idx % 3],
                "stance": stance,
                "must_include": [
                    "one concrete usability concern",
                    "one realistic tradeoff",
                    "one contextual stress trigger",
                ],
                "memory_stream": [],
            }
        )

    return blueprints


def _normalize_blueprint_entry(payload: dict[str, Any], index: int) -> dict[str, Any]:
    stance = _safe_str(payload.get("stance"), "fence-sitter").lower()
    if stance not in {"champion", "fence-sitter", "skeptic"}:
        stance = "fence-sitter"

    daily_context = _safe_str(
        payload.get("daily_context"),
        _safe_str(payload.get("driving_context"), "weekday routine with competing demands"),
    )

    memory_stream_raw = payload.get("memory_stream", [])
    memory_stream = []
    if isinstance(memory_stream_raw, list):
        memory_stream = [
            _normalize_memory_seed_text(item)
            for item in memory_stream_raw
            if _normalize_memory_seed_text(item)
        ]

    return {
        "name": _safe_str(payload.get("name"), f"Persona {index + 1}"),
        "age_range": _safe_str(payload.get("age_range"), "35-49"),
        "occupation": _safe_str(payload.get("occupation"), "Working parent"),
        "household": _safe_str(payload.get("household"), "Suburban household with kids"),
        "region": _safe_str(payload.get("region"), "United States suburb"),
        "persona_type": _safe_str(payload.get("persona_type"), "time-pressed multitasker"),
        "gender_identity": _safe_str(payload.get("gender_identity"), "woman"),
        "ethnicity": _safe_str(payload.get("ethnicity"), "Black"),
        "tech_comfort": _safe_str(payload.get("tech_comfort"), "medium").lower(),
        "daily_context": daily_context,
        "driving_context": daily_context,
        "budget_sensitivity": _safe_str(payload.get("budget_sensitivity"), "medium").lower(),
        "stance": stance,
        "memory_stream": memory_stream,
        "must_include": [
            _safe_str(item, "")
            for item in payload.get("must_include", [])
            if _safe_str(item, "")
        ],
    }


async def _build_diversity_blueprints(
    target_audience: str,
    stimulus_description: str,
    task_type: str,
    persona_count: int,
) -> list[dict[str, Any]]:
    system_prompt = (
        "You are designing a synthetic focus group panel. Create highly diverse personas with "
        "distinct names, occupations, life constraints, and opinions. Return strict JSON."
    )

    user_prompt = f"""
Target audience:
{target_audience}

Stimulus:
{stimulus_description}

Task type:
{task_type}

Create exactly {persona_count} panel_blueprints with maximum diversity.
Avoid repeating occupations or near-identical names.

Return JSON with key panel_blueprints: array of objects, each with keys:
- name
- age_range
- occupation
- household
- region
- persona_type
- gender_identity
- ethnicity
- tech_comfort (low|medium|high)
- daily_context
- budget_sensitivity (low|medium|high)
- stance (champion|fence-sitter|skeptic)
- must_include (array of 2-4 short points)

Rules:
- Persona details must feel plausible for the stimulus category.
- Avoid fabricated objective claims (health, legal, safety, scientific) unless directly supported.
- Do NOT force any sentiment distribution. Let sentiment emerge organically from persona context.
"""

    raw_panel: list[dict[str, Any]] = []
    try:
        generated = await _chat_json(system_prompt, user_prompt, temperature=1.0)
        candidate = generated.get("panel_blueprints") or generated.get("panel") or generated.get("personas")
        if isinstance(candidate, list):
            raw_panel = [entry for entry in candidate if isinstance(entry, dict)]
    except Exception:
        raw_panel = []

    if len(raw_panel) < persona_count:
        fallback = _fallback_blueprints(persona_count)
        if raw_panel:
            raw_panel.extend(fallback[len(raw_panel) :])
        else:
            raw_panel = fallback

    normalized: list[dict[str, Any]] = []
    used_names: set[str] = set()
    persona_seed_pool = await _get_persona_seed_pool(
        target_audience=target_audience,
        stimulus_description=stimulus_description,
        persona_count=persona_count,
    )
    if persona_seed_pool:
        random.shuffle(persona_seed_pool)

    for idx in range(persona_count):
        source = raw_panel[idx] if idx < len(raw_panel) else {}
        entry = _normalize_blueprint_entry(source, idx)
        if persona_seed_pool:
            entry = _merge_blueprint_with_persona_seed(
                entry,
                persona_seed_pool[idx % len(persona_seed_pool)],
            )

        base_name = entry["name"]
        unique_name = base_name
        counter = 2
        while unique_name.lower() in used_names:
            unique_name = f"{base_name} {counter}"
            counter += 1
        used_names.add(unique_name.lower())
        entry["name"] = unique_name

        normalized.append(entry)

    memory_pool = await _get_real_world_memory_pool()
    min_mem = max(1, HF_MEMORY_PER_PERSONA_MIN)
    max_mem = max(min_mem, HF_MEMORY_PER_PERSONA_MAX)

    for entry in normalized:
        per_persona_count = random.randint(min_mem, max_mem)
        assigned = _sample_from_pool(memory_pool, per_persona_count)
        if assigned:
            entry["memory_stream"] = assigned
        elif not entry.get("memory_stream"):
            entry["memory_stream"] = _sample_from_pool(
                FALLBACK_REAL_WORLD_MEMORIES,
                per_persona_count,
            )

    return normalized


def _default_persona(
    index: int,
    blueprint: dict[str, Any],
    analysis_dimensions: list[dict[str, str]],
) -> dict[str, Any]:
    dimension_specs = analysis_dimensions or _fallback_dimension_specs()
    cognitive_load: dict[str, int] = {}
    for key_index, spec in enumerate(dimension_specs):
        key = _safe_str(spec.get("key"), f"dimension_{key_index + 1}")
        cognitive_load[key] = _dimension_score_default(
            index=index,
            key_index=key_index,
        )

    daily_context = _safe_str(
        blueprint.get("daily_context"),
        _safe_str(blueprint.get("driving_context"), "weekday routine with competing demands"),
    )

    return {
        "name": _safe_str(blueprint.get("name"), f"Persona {index + 1}"),
        "age_range": _safe_str(blueprint.get("age_range"), "35-49"),
        "occupation": _safe_str(blueprint.get("occupation"), "Working parent"),
        "household": _safe_str(
            blueprint.get("household"), "Suburban household with kids"
        ),
        "region": _safe_str(blueprint.get("region"), "United States"),
        "persona_type": _safe_str(
            blueprint.get("persona_type"), "time-pressed multitasker"
        ),
        "gender_identity": _safe_str(blueprint.get("gender_identity"), "woman"),
        "ethnicity": _safe_str(blueprint.get("ethnicity"), "Black"),
        "tech_comfort": _safe_str(blueprint.get("tech_comfort"), "medium"),
        "daily_context": daily_context,
        "driving_context": daily_context,
        "budget_sensitivity": _safe_str(blueprint.get("budget_sensitivity"), "medium"),
        "stance": _safe_str(blueprint.get("stance"), "fence-sitter"),
        "sentiment_target": "organic",
        "stress_level": 65,
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "cognitive_load": cognitive_load,
        "reaction_summary": "I evaluate this based on real-world fit, trust, and practical tradeoffs.",
        "quote": "If this helps in my actual routine, I can get behind it.",
        "reasons": [
            "It needs to match how I make decisions in day-to-day life",
            "I look for concrete value, not just surface appeal",
        ],
        "concerns": ["Some claims feel too broad without proof", "I need clearer tradeoff details"],
        "delights": ["Clear message hierarchy", "Feels relevant to my real context"],
        "memory_stream": _normalize_string_list(blueprint.get("memory_stream"), []),
    }


def _normalize_string_list(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [_safe_str(item, "") for item in value if _safe_str(item, "")]
        if cleaned:
            return cleaned
    return fallback


def _normalize_persona(
    payload: dict[str, Any],
    index: int,
    blueprint: dict[str, Any],
    analysis_dimensions: list[dict[str, str]],
    stimulus_description: str,
) -> dict[str, Any]:
    default_persona = _default_persona(index, blueprint, analysis_dimensions)
    dimension_specs = analysis_dimensions or _fallback_dimension_specs()

    cognitive_load_raw = payload.get("cognitive_load")
    if not isinstance(cognitive_load_raw, dict):
        cognitive_load_raw = {}

    default_cognitive_load = default_persona.get("cognitive_load", {})
    if not isinstance(default_cognitive_load, dict):
        default_cognitive_load = {}

    cognitive_load: dict[str, int] = {}
    for key_index, spec in enumerate(dimension_specs):
        key = _safe_str(spec.get("key"), f"dimension_{key_index + 1}")
        cognitive_load[key] = _resolve_dimension_score(
            cognitive_load_raw=cognitive_load_raw,
            default_cognitive_load=default_cognitive_load,
            spec=spec,
            key_index=key_index,
        )

    sentiment_score = _safe_float(
        payload.get("sentiment_score"),
        _safe_float(default_persona.get("sentiment_score"), 0.0),
    )

    sentiment_label = _safe_str(
        payload.get("sentiment_label"),
        _safe_str(default_persona.get("sentiment_label"), "neutral"),
    ).lower()
    if sentiment_label not in {"positive", "neutral", "negative"}:
        if sentiment_score > 0.15:
            sentiment_label = "positive"
        elif sentiment_score < -0.15:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

    daily_context = _safe_str(
        payload.get("daily_context"),
        _safe_str(
            payload.get("driving_context"),
            _safe_str(default_persona.get("daily_context"), "weekday routine"),
        ),
    )

    normalized = {
        "name": _safe_str(
            payload.get("name"), _safe_str(default_persona["name"], f"Persona {index + 1}")
        ),
        "age_range": _safe_str(
            payload.get("age_range"), _safe_str(default_persona["age_range"], "35-49")
        ),
        "occupation": _safe_str(
            payload.get("occupation"),
            _safe_str(default_persona["occupation"], "Working parent"),
        ),
        "household": _safe_str(
            payload.get("household"),
            _safe_str(default_persona["household"], "Suburban household with kids"),
        ),
        "region": _safe_str(
            payload.get("region"), _safe_str(default_persona["region"], "United States")
        ),
        "persona_type": _safe_str(
            payload.get("persona_type"),
            _safe_str(default_persona["persona_type"], "time-pressed multitasker"),
        ),
        "gender_identity": _safe_str(
            payload.get("gender_identity"), _safe_str(default_persona["gender_identity"], "woman")
        ),
        "ethnicity": _safe_str(
            payload.get("ethnicity"), _safe_str(default_persona["ethnicity"], "Black")
        ),
        "tech_comfort": _safe_str(
            payload.get("tech_comfort"),
            _safe_str(default_persona["tech_comfort"], "medium"),
        ).lower(),
        "daily_context": daily_context,
        "driving_context": daily_context,
        "budget_sensitivity": _safe_str(
            payload.get("budget_sensitivity"),
            _safe_str(default_persona["budget_sensitivity"], "medium"),
        ).lower(),
        "stance": _safe_str(
            payload.get("stance"), _safe_str(default_persona["stance"], "fence-sitter")
        ),
        "sentiment_target": _safe_str(
            payload.get("sentiment_target"),
            _safe_str(default_persona.get("sentiment_target"), "organic"),
        ),
        "stress_level": _safe_int(
            payload.get("stress_level"), _safe_int(default_persona["stress_level"], 65)
        ),
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "cognitive_load": cognitive_load,
        "reaction_summary": _safe_str(
            payload.get("reaction_summary"),
            _safe_str(default_persona["reaction_summary"], ""),
        ),
        "quote": _safe_str(payload.get("quote"), _safe_str(default_persona["quote"], "")),
        "reasons": _normalize_string_list(
            payload.get("reasons"),
            _normalize_string_list(default_persona.get("reasons"), []),
        ),
        "concerns": _normalize_string_list(
            payload.get("concerns"),
            _normalize_string_list(default_persona.get("concerns"), []),
        ),
        "delights": _normalize_string_list(
            payload.get("delights"),
            _normalize_string_list(default_persona.get("delights"), []),
        ),
        "memory_stream": _normalize_string_list(
            payload.get("memory_stream"),
            _normalize_string_list(default_persona.get("memory_stream"), []),
        ),
    }

    return _sanitize_persona_claims(normalized, stimulus_description)


def _persona_seed_prompt(
    target_audience: str,
    stimulus_description: str,
    image_url: str | None,
    task_type: str,
    blueprint: dict[str, Any],
    analysis_dimensions: list[dict[str, str]],
    index: int,
    total: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are simulating one specific focus group participant. "
        "Return valid JSON only. Keep details concrete and realistic."
    )

    must_include = blueprint.get("must_include", [])
    must_include_text = "\n".join([f"- {item}" for item in must_include]) or "- one specific context"
    dimension_specs = analysis_dimensions or _fallback_dimension_specs()
    dimension_details = "\n".join(
        [
            f"- {spec.get('key')}: {spec.get('label')}"
            + (
                f" ({_safe_str(spec.get('description'), '')})"
                if _safe_str(spec.get("description"), "")
                else ""
            )
            for spec in dimension_specs
        ]
    )
    dimension_keys = "\n".join([f"  - {spec.get('key')}" for spec in dimension_specs])
    daily_context = _safe_str(
        blueprint.get("daily_context"),
        _safe_str(blueprint.get("driving_context"), "weekday routine with competing demands"),
    )
    memory_stream = _normalize_string_list(blueprint.get("memory_stream"), [])
    memory_stream_text = (
        "\n".join([f"- {item}" for item in memory_stream]) if memory_stream else "- none"
    )
    task_type_guidance = (
        "Focus on factual UI observations. Be concrete and verifiable. Avoid broad brand-opinion language."
        if task_type == "Objective UI Task"
        else "Focus on authentic preference, trust, and tradeoff judgments rooted in lived experience."
    )

    user_prompt = f"""
Generate participant {index + 1} of {total}.

Target audience: {target_audience}
Stimulus: {stimulus_description}
Stimulus image URL (optional context): {image_url or "none"}
Task type: {task_type}

Persona blueprint (must follow):
- name: {blueprint.get("name")}
- age_range: {blueprint.get("age_range")}
- occupation: {blueprint.get("occupation")}
- household: {blueprint.get("household")}
- region: {blueprint.get("region")}
- persona_type: {blueprint.get("persona_type")}
- gender_identity: {blueprint.get("gender_identity")}
- ethnicity: {blueprint.get("ethnicity")}
- tech_comfort: {blueprint.get("tech_comfort")}
- daily_context: {daily_context}
- budget_sensitivity: {blueprint.get("budget_sensitivity")}
- stance: {blueprint.get("stance")}

Mandatory content points:
{must_include_text}

Real-world memory stream (past experiences/core opinions). Draw on these when reacting:
{memory_stream_text}

Evaluation dimensions (use exactly these keys in cognitive_load):
{dimension_details}

Return exactly one JSON object with keys:
- name (string)
- age_range (string)
- occupation (string)
- household (string)
- region (string)
- persona_type (string)
- gender_identity (string)
- ethnicity (string)
- tech_comfort ("low"|"medium"|"high")
- daily_context (string)
- budget_sensitivity ("low"|"medium"|"high")
- stance ("champion"|"fence-sitter"|"skeptic")
- stress_level (integer 0-100)
- sentiment_score (number from -1 to 1)
- sentiment_label ("positive"|"neutral"|"negative")
- cognitive_load (object with integers 0-100):
{dimension_keys}
- reaction_summary (string)
- quote (string, first-person)
- reasons (array of strings, 2-4)
- concerns (array of strings, 2-4)
- delights (array of strings, 1-3)

Rules:
- Keep this persona distinct from generic answers.
- Sentiment must emerge organically from memory stream + stimulus, do not force distribution.
- Mention at least one concrete concern and one potential upside.
- Do not invent objective claims (health/safety/legal/scientific) unless clearly supported by the stimulus.
- If the stimulus is fast food or indulgent food, do not call it healthy.
- {task_type_guidance}
"""
    return system_prompt, user_prompt


async def _generate_one_persona(
    target_audience: str,
    stimulus_description: str,
    image_url: str | None,
    task_type: str,
    persona_count: int,
    blueprint: dict[str, Any],
    analysis_dimensions: list[dict[str, str]],
    index: int,
    semaphore: asyncio.Semaphore,
) -> PersonaState:
    async with semaphore:
        system_prompt, user_prompt = _persona_seed_prompt(
            target_audience=target_audience,
            stimulus_description=stimulus_description,
            image_url=image_url,
            task_type=task_type,
            blueprint=blueprint,
            analysis_dimensions=analysis_dimensions,
            index=index,
            total=persona_count,
        )

        raw = await _chat_json(system_prompt, user_prompt, temperature=0.9)
        normalized = _normalize_persona(
            raw,
            index,
            blueprint,
            analysis_dimensions,
            stimulus_description,
        )

        agent_id = f"agent_{uuid.uuid4().hex[:10]}"
        profile = {
            "agent_id": agent_id,
            "stimulus_description": stimulus_description,
            "task_type": task_type,
            **normalized,
        }

        memory_texts = [
            f"Persona type: {profile.get('persona_type', '')}",
            f"Daily context: {profile.get('daily_context', profile.get('driving_context', ''))}",
            "Memory stream: " + " | ".join(_normalize_string_list(profile.get("memory_stream"), [])),
            profile.get("reaction_summary", ""),
            profile.get("quote", ""),
            "Reasons: " + "; ".join(profile.get("reasons", [])),
            "Concerns: " + "; ".join(profile.get("concerns", [])),
            "Delights: " + "; ".join(profile.get("delights", [])),
        ]

        memory_chunks = [MemoryChunk(text=text, kind="seed") for text in memory_texts if text]

        embeddings = await asyncio.gather(*[_embed_text(chunk.text) for chunk in memory_chunks])
        for chunk, embedding in zip(memory_chunks, embeddings):
            chunk.embedding = embedding

        return PersonaState(agent_id=agent_id, profile=profile, memories=memory_chunks)


def _build_manager_summary(
    personas: list[PersonaState],
    analysis_dimensions: list[dict[str, str]],
) -> dict[str, Any]:
    dimension_specs = analysis_dimensions or _fallback_dimension_specs()
    dimension_keys = _dimension_keys(dimension_specs)

    if not personas:
        return {
            "cognitive_load_heatmap": {},
            "sentiment": {"overall": "neutral", "breakdown": {}, "raw": {}},
            "demographics": [],
            "analysis_dimensions": dimension_specs,
        }

    dimension_totals: defaultdict[str, int] = defaultdict(int)
    for persona in personas:
        persona_cognitive_load = persona.profile.get("cognitive_load", {})
        if not isinstance(persona_cognitive_load, dict):
            persona_cognitive_load = {}

        for dim in dimension_keys:
            dimension_totals[dim] += _safe_int(
                persona_cognitive_load.get(dim),
                50,
            )

    cognitive_load_heatmap = {
        dim: round(dimension_totals[dim] / len(personas), 1) for dim in dimension_keys
    }

    sentiment_scores = [
        _safe_float(persona.profile.get("sentiment_score"), 0.0) for persona in personas
    ]
    sentiment_counter = Counter(
        _safe_str(persona.profile.get("sentiment_label"), "neutral") for persona in personas
    )

    avg_sentiment = round(sum(sentiment_scores) / max(1, len(sentiment_scores)), 3)
    score_spread = round(pstdev(sentiment_scores), 3) if len(sentiment_scores) > 1 else 0.0

    if avg_sentiment >= 0.2:
        overall_sentiment = "positive"
    elif avg_sentiment <= -0.2:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    stance_counter = Counter(
        _safe_str(persona.profile.get("stance"), "fence-sitter") for persona in personas
    )

    demographics = [
        {
            "segment": "age_range",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("age_range"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "occupation",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("occupation"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "gender_identity",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("gender_identity"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "ethnicity",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("ethnicity"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "region",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("region"), "unknown") for p in personas)
            ),
        },
        {
            "segment": "tech_comfort",
            "distribution": dict(
                Counter(_safe_str(p.profile.get("tech_comfort"), "unknown") for p in personas)
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
                "score_spread": score_spread,
            },
            "raw": {
                "average_score": avg_sentiment,
                "score_spread": score_spread,
                "counts": dict(sentiment_counter),
                "stance_mix": dict(stance_counter),
            },
        },
        "demographics": demographics,
        "analysis_dimensions": dimension_specs,
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


def _followup_prompt(
    persona: PersonaState,
    question: str,
    memory_snippets: list[dict[str, Any]],
) -> tuple[str, str]:
    system_prompt = f"""
You are role-playing this persona in first person:
- Name: {_safe_str(persona.profile.get("name"), "Unknown")}
- Age range: {_safe_str(persona.profile.get("age_range"), "Unknown")}
- Occupation: {_safe_str(persona.profile.get("occupation"), "Unknown")}
- Gender identity: {_safe_str(persona.profile.get("gender_identity"), "Unknown")}
- Ethnicity: {_safe_str(persona.profile.get("ethnicity"), "Unknown")}
- Persona type: {_safe_str(persona.profile.get("persona_type"), "Unknown")}
- Stance: {_safe_str(persona.profile.get("stance"), "fence-sitter")}
- Daily context: {_safe_str(persona.profile.get("daily_context"), _safe_str(persona.profile.get("driving_context"), ""))}
- Reaction summary: {_safe_str(persona.profile.get("reaction_summary"), "")}

Answer naturally as this persona would.
Do not invent objective claims (health/safety/legal/scientific) without support in memory snippets.
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
        "model_fallbacks": OPENAI_MODEL_FALLBACKS,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "persona_dataset": HF_PERSONA_DATASET,
        "persona_store_size": await store.size(),
    }


@app.post("/synthetic/simulate")
async def simulate_focus_group(request: SimulateRequest) -> dict[str, Any]:
    audience_info = await _resolve_target_audience(
        requested_target=request.target_audience,
        stimulus_description=request.stimulus_description,
    )
    task_type_info = await _classify_task_type(request.stimulus_description)
    task_type = _safe_str(task_type_info.get("task_type"), "Subjective Concept")

    resolved_target_audience = audience_info["target_audience"]
    analysis_dimensions = await _resolve_analysis_dimensions(
        requested_dimensions=request.analysis_dimensions,
        target_audience=resolved_target_audience,
        stimulus_description=request.stimulus_description,
    )

    blueprints = await _build_diversity_blueprints(
        target_audience=resolved_target_audience,
        stimulus_description=request.stimulus_description,
        task_type=task_type,
        persona_count=request.persona_count,
    )

    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    tasks = [
        _generate_one_persona(
            target_audience=resolved_target_audience,
            stimulus_description=request.stimulus_description,
            image_url=request.image_url,
            task_type=task_type,
            persona_count=request.persona_count,
            blueprint=blueprints[index],
            analysis_dimensions=analysis_dimensions,
            index=index,
            semaphore=semaphore,
        )
        for index in range(request.persona_count)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    personas: list[PersonaState] = []
    failures: list[str] = []

    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            failures.append(f"persona_{idx + 1}: {result}")
            fallback_blueprint = blueprints[idx] if idx < len(blueprints) else {}
            fallback_profile = _default_persona(idx, fallback_blueprint, analysis_dimensions)
            fallback_agent_id = f"agent_{uuid.uuid4().hex[:10]}"

            fallback_state = PersonaState(
                agent_id=fallback_agent_id,
                profile={
                    "agent_id": fallback_agent_id,
                    "stimulus_description": request.stimulus_description,
                    "task_type": task_type,
                    **fallback_profile,
                    "generation_error": str(result),
                },
                memories=[
                    MemoryChunk(text=_safe_str(fallback_profile.get("reaction_summary"), ""), kind="seed"),
                    MemoryChunk(text=_safe_str(fallback_profile.get("quote"), ""), kind="seed"),
                ],
            )
            personas.append(fallback_state)
            continue

        personas.append(result)

    await store.upsert_many(personas)

    manager_summary = _build_manager_summary(personas, analysis_dimensions)

    persona_payloads = [
        {
            **persona.profile,
            "memory_count": len(persona.memories),
        }
        for persona in personas
    ]

    return {
        "run_id": f"run_{uuid.uuid4().hex[:10]}",
        "target_audience": resolved_target_audience,
        "target_audience_generated": bool(audience_info.get("target_audience_generated")),
        "target_audience_generation_notes": _safe_str(
            audience_info.get("target_audience_generation_notes"),
            "",
        ),
        "stimulus_description": request.stimulus_description,
        "image_url": request.image_url,
        "task_type": task_type,
        "task_type_rationale": _safe_str(task_type_info.get("task_type_rationale"), ""),
        "analysis_dimensions": analysis_dimensions,
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
        "I need more context, but based on my prior reaction I would focus on practical fit and trust.",
    )
    answer = _sanitize_claim_text(
        answer,
        _stimulus_flags(_safe_str(persona.profile.get("stimulus_description"), "")),
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

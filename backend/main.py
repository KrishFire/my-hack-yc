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

APP_NAME = "synthetic-focus-group-backend"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL_FALLBACKS = [
    model.strip()
    for model in os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-5-mini,gpt-4o-mini").split(",")
    if model.strip()
]
DEFAULT_PERSONA_COUNT = int(os.getenv("FG_PERSONA_COUNT", "6"))
MAX_PERSONA_COUNT = int(os.getenv("FG_MAX_PERSONA_COUNT", "12"))
MAX_PARALLEL_TASKS = int(os.getenv("FG_MAX_PARALLEL_TASKS", "8"))
TOP_K_MEMORIES = int(os.getenv("FG_TOP_K_MEMORIES", "5"))
MAX_PERSONA_STORE = int(os.getenv("FG_MAX_PERSONA_STORE", "200"))

SENTIMENT_CYCLE = ("positive", "neutral", "negative")

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


def _sentiment_targets(count: int) -> list[str]:
    targets: list[str] = []
    while len(targets) < count:
        targets.extend(SENTIMENT_CYCLE)
    return targets[:count]


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
    sentiment_target: str,
) -> int:
    base = 48 + ((index * 9 + key_index * 7) % 20)
    if sentiment_target == "positive":
        base += 7
    elif sentiment_target == "negative":
        base -= 7
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
    targets = _sentiment_targets(persona_count)
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
        sentiment_target = targets[idx]
        stance = (
            "champion"
            if sentiment_target == "positive"
            else "skeptic"
            if sentiment_target == "negative"
            else "fence-sitter"
        )

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
                "sentiment_target": sentiment_target,
                "stance": stance,
                "must_include": [
                    "one concrete usability concern",
                    "one realistic tradeoff",
                    "one contextual stress trigger",
                ],
            }
        )

    return blueprints


def _normalize_blueprint_entry(
    payload: dict[str, Any],
    index: int,
    sentiment_target: str,
) -> dict[str, Any]:
    stance = _safe_str(payload.get("stance"), "fence-sitter").lower()
    if stance not in {"champion", "fence-sitter", "skeptic"}:
        stance = (
            "champion"
            if sentiment_target == "positive"
            else "skeptic"
            if sentiment_target == "negative"
            else "fence-sitter"
        )

    daily_context = _safe_str(
        payload.get("daily_context"),
        _safe_str(payload.get("driving_context"), "weekday routine with competing demands"),
    )

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
        "sentiment_target": sentiment_target,
        "stance": stance,
        "must_include": [
            _safe_str(item, "")
            for item in payload.get("must_include", [])
            if _safe_str(item, "")
        ],
    }


async def _build_diversity_blueprints(
    target_audience: str,
    stimulus_description: str,
    persona_count: int,
) -> list[dict[str, Any]]:
    sentiment_targets = _sentiment_targets(persona_count)

    system_prompt = (
        "You are designing a synthetic focus group panel. Create highly diverse personas with "
        "distinct names, occupations, life constraints, and opinions. Return strict JSON."
    )

    user_prompt = f"""
Target audience:
{target_audience}

Stimulus:
{stimulus_description}

Create exactly {persona_count} panel_blueprints with maximum diversity.
You MUST include sentiment mix across positive, neutral, negative.
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

    for idx in range(persona_count):
        source = raw_panel[idx] if idx < len(raw_panel) else {}
        entry = _normalize_blueprint_entry(source, idx, sentiment_targets[idx])

        base_name = entry["name"]
        unique_name = base_name
        counter = 2
        while unique_name.lower() in used_names:
            unique_name = f"{base_name} {counter}"
            counter += 1
        used_names.add(unique_name.lower())
        entry["name"] = unique_name

        normalized.append(entry)

    return normalized


def _default_persona(
    index: int,
    blueprint: dict[str, Any],
    analysis_dimensions: list[dict[str, str]],
) -> dict[str, Any]:
    sentiment_target = _safe_str(blueprint.get("sentiment_target"), "neutral")
    target_score = (
        0.45
        if sentiment_target == "positive"
        else -0.45
        if sentiment_target == "negative"
        else 0.0
    )

    dimension_specs = analysis_dimensions or _fallback_dimension_specs()
    cognitive_load: dict[str, int] = {}
    for key_index, spec in enumerate(dimension_specs):
        key = _safe_str(spec.get("key"), f"dimension_{key_index + 1}")
        cognitive_load[key] = _dimension_score_default(
            index=index,
            key_index=key_index,
            sentiment_target=sentiment_target,
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
        "sentiment_target": sentiment_target,
        "stress_level": 65,
        "sentiment_score": target_score,
        "sentiment_label": sentiment_target,
        "cognitive_load": cognitive_load,
        "reaction_summary": "I evaluate this based on real-world fit, trust, and practical tradeoffs.",
        "quote": "If this helps in my actual routine, I can get behind it.",
        "reasons": [
            "It needs to match how I make decisions in day-to-day life",
            "I look for concrete value, not just surface appeal",
        ],
        "concerns": ["Some claims feel too broad without proof", "I need clearer tradeoff details"],
        "delights": ["Clear message hierarchy", "Feels relevant to my real context"],
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

    sentiment_target = _safe_str(blueprint.get("sentiment_target"), "neutral")
    sentiment_score = _safe_float(
        payload.get("sentiment_score"),
        _safe_float(default_persona.get("sentiment_score"), 0.0),
    )

    sentiment_label = _safe_str(payload.get("sentiment_label"), sentiment_target).lower()
    if sentiment_label not in {"positive", "neutral", "negative"}:
        sentiment_label = sentiment_target

    if sentiment_target == "positive" and sentiment_score < 0.15:
        sentiment_score = max(0.15, abs(sentiment_score))
    if sentiment_target == "negative" and sentiment_score > -0.15:
        sentiment_score = -max(0.15, abs(sentiment_score))
    if sentiment_target == "neutral":
        sentiment_score = max(-0.25, min(0.25, sentiment_score))

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
        "sentiment_target": sentiment_target,
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
    }

    return _sanitize_persona_claims(normalized, stimulus_description)


def _persona_seed_prompt(
    target_audience: str,
    stimulus_description: str,
    image_url: str | None,
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

    user_prompt = f"""
Generate participant {index + 1} of {total}.

Target audience: {target_audience}
Stimulus: {stimulus_description}
Stimulus image URL (optional context): {image_url or "none"}

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
- required sentiment target: {blueprint.get("sentiment_target")}

Mandatory content points:
{must_include_text}

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
- Match sentiment_label and sentiment_score direction to required sentiment target.
- Even positive personas should mention at least one concern.
- Even skeptical personas should mention at least one potential upside.
- Do not invent objective claims (health/safety/legal/scientific) unless clearly supported by the stimulus.
- If the stimulus is fast food or indulgent food, do not call it healthy.
"""
    return system_prompt, user_prompt


async def _generate_one_persona(
    target_audience: str,
    stimulus_description: str,
    image_url: str | None,
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
            **normalized,
        }

        memory_texts = [
            f"Persona type: {profile.get('persona_type', '')}",
            f"Daily context: {profile.get('daily_context', profile.get('driving_context', ''))}",
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
        "persona_store_size": await store.size(),
    }


@app.post("/synthetic/simulate")
async def simulate_focus_group(request: SimulateRequest) -> dict[str, Any]:
    audience_info = await _resolve_target_audience(
        requested_target=request.target_audience,
        stimulus_description=request.stimulus_description,
    )

    resolved_target_audience = audience_info["target_audience"]
    analysis_dimensions = await _resolve_analysis_dimensions(
        requested_dimensions=request.analysis_dimensions,
        target_audience=resolved_target_audience,
        stimulus_description=request.stimulus_description,
    )

    blueprints = await _build_diversity_blueprints(
        target_audience=resolved_target_audience,
        stimulus_description=request.stimulus_description,
        persona_count=request.persona_count,
    )

    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    tasks = [
        _generate_one_persona(
            target_audience=resolved_target_audience,
            stimulus_description=request.stimulus_description,
            image_url=request.image_url,
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

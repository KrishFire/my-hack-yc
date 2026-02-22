import { MCPServer, object, text, widget } from "mcp-use/server";
import { z } from "zod";

const BACKEND_URL =
  process.env.FOCUS_GROUP_BACKEND_URL ?? "http://127.0.0.1:8000";

type UnknownRecord = Record<string, unknown>;

type NormalizedSentiment = {
  overall: string | number | null;
  breakdown: Record<string, unknown>;
  raw: unknown;
};

type AnalysisDimension = {
  key: string;
  label: string;
  description?: string;
};

type NormalizedManagerSummary = {
  cognitive_load_heatmap: Record<string, unknown>;
  sentiment: NormalizedSentiment;
  demographics: UnknownRecord[];
  analysis_dimensions: AnalysisDimension[];
};

type NormalizedPersona = UnknownRecord & {
  agent_id: string;
  name: string;
};

type FocusGroupWidgetProps = {
  target_audience: string;
  target_audience_generated: boolean;
  target_audience_generation_notes: string | null;
  stimulus_description: string;
  image_url: string | null;
  analysis_dimensions: AnalysisDimension[];
  manager_summary: NormalizedManagerSummary;
  personas: NormalizedPersona[];
  raw: unknown;
  error?: string;
};

type FollowupResult = {
  agent_id: string;
  question: string;
  answer: string;
  memory_snippets: unknown[];
  raw: unknown;
  error?: string;
};

class BackendRequestError extends Error {
  status: number;
  payload: unknown;

  constructor(status: number, message: string, payload: unknown) {
    super(message);
    this.name = "BackendRequestError";
    this.status = status;
    this.payload = payload;
  }
}

const server = new MCPServer({
  name: "hack-yc",
  title: "hack-yc",
  version: "1.0.0",
  description: "MCP server with MCP Apps integration",
  baseUrl: process.env.MCP_URL || "http://localhost:3000",
  favicon: "favicon.ico",
  websiteUrl: "https://mcp-use.com",
  icons: [
    {
      src: "icon.svg",
      mimeType: "image/svg+xml",
      sizes: ["512x512"],
    },
  ],
});

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function pickFirst(record: UnknownRecord | null, keys: string[]): unknown {
  if (!record) {
    return undefined;
  }

  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }

  return undefined;
}

function pickFromSources(sources: UnknownRecord[], keys: string[]): unknown {
  for (const source of sources) {
    const value = pickFirst(source, keys);
    if (value !== undefined) {
      return value;
    }
  }

  return undefined;
}

function asString(value: unknown): string | undefined {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "number") {
    return value !== 0;
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true" || normalized === "yes" || normalized === "1") {
      return true;
    }
    if (normalized === "false" || normalized === "no" || normalized === "0") {
      return false;
    }
  }

  return undefined;
}

function toDimensionKey(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || `dimension_${Math.random().toString(36).slice(2, 8)}`;
}

function formatDimensionLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
}

function getPayloadSources(payload: unknown): UnknownRecord[] {
  const sources: UnknownRecord[] = [];

  if (!isRecord(payload)) {
    return sources;
  }

  sources.push(payload);

  const data = payload.data;
  if (isRecord(data)) {
    sources.push(data);
  }

  const result = payload.result;
  if (isRecord(result)) {
    sources.push(result);
  }

  return sources;
}

function normalizeHeatmap(value: unknown): Record<string, unknown> {
  if (isRecord(value)) {
    return value;
  }

  if (!Array.isArray(value)) {
    return {};
  }

  const entries: Array<[string, unknown]> = [];

  value.forEach((item, index) => {
    if (isRecord(item)) {
      const label =
        asString(
          pickFirst(item, ["label", "name", "dimension", "metric", "key"])
        ) ?? `metric_${index + 1}`;
      const score =
        pickFirst(item, ["value", "score", "load", "intensity"]) ?? item;
      entries.push([label, score]);
      return;
    }

    entries.push([`metric_${index + 1}`, item]);
  });

  return Object.fromEntries(entries);
}

function normalizeSentiment(value: unknown): NormalizedSentiment {
  if (isRecord(value)) {
    const overallCandidate = pickFirst(value, [
      "overall",
      "score",
      "sentiment",
      "label",
    ]);

    const overall =
      typeof overallCandidate === "string" || typeof overallCandidate === "number"
        ? overallCandidate
        : null;

    const breakdownCandidate = pickFirst(value, [
      "breakdown",
      "distribution",
      "scores",
    ]);

    const breakdown = isRecord(breakdownCandidate)
      ? breakdownCandidate
      : Object.fromEntries(
          Object.entries(value).filter(
            ([key, entryValue]) =>
              key !== "overall" &&
              (typeof entryValue === "string" ||
                typeof entryValue === "number" ||
                typeof entryValue === "boolean")
          )
        );

    return {
      overall,
      breakdown,
      raw: value,
    };
  }

  if (typeof value === "string" || typeof value === "number") {
    return {
      overall: value,
      breakdown: {},
      raw: value,
    };
  }

  return {
    overall: null,
    breakdown: {},
    raw: value ?? null,
  };
}

function normalizeDemographics(value: unknown): UnknownRecord[] {
  if (Array.isArray(value)) {
    return value.map((entry, index) => {
      if (isRecord(entry)) {
        return entry;
      }

      return {
        segment: `segment_${index + 1}`,
        value: entry,
      };
    });
  }

  if (isRecord(value)) {
    const entries = Object.entries(value);
    const isFlatMap = entries.every(([, entryValue]) => {
      return (
        entryValue === null ||
        typeof entryValue === "string" ||
        typeof entryValue === "number" ||
        typeof entryValue === "boolean"
      );
    });

    if (isFlatMap) {
      return entries.map(([segment, count]) => ({
        segment,
        count,
      }));
    }

    return [value];
  }

  return [];
}

function normalizeAnalysisDimensions(value: unknown): AnalysisDimension[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const normalized: AnalysisDimension[] = [];
  const seen = new Set<string>();

  value.forEach((entry, index) => {
    if (typeof entry === "string") {
      const label = entry.trim();
      if (!label) {
        return;
      }

      const key = toDimensionKey(label);
      if (seen.has(key)) {
        return;
      }

      seen.add(key);
      normalized.push({
        key,
        label,
      });
      return;
    }

    if (!isRecord(entry)) {
      return;
    }

    const label =
      asString(pickFirst(entry, ["label", "name", "dimension", "metric"]))?.trim() ??
      "";
    const keyCandidate =
      asString(pickFirst(entry, ["key", "id", "slug"]))?.trim() ?? "";
    const key = keyCandidate ? toDimensionKey(keyCandidate) : toDimensionKey(label || `dimension_${index + 1}`);
    if (seen.has(key)) {
      return;
    }

    seen.add(key);
    normalized.push({
      key,
      label: label || formatDimensionLabel(key),
      description: asString(pickFirst(entry, ["description", "details", "hint"])),
    });
  });

  return normalized;
}

function normalizeManagerSummary(payload: unknown): NormalizedManagerSummary {
  const sources = getPayloadSources(payload);

  const managerSummary =
    (isRecord(
      pickFromSources(sources, [
        "manager_summary",
        "managerSummary",
        "summary",
        "aggregated_summary",
        "aggregate_summary",
      ])
    )
      ? (pickFromSources(sources, [
          "manager_summary",
          "managerSummary",
          "summary",
          "aggregated_summary",
          "aggregate_summary",
        ]) as UnknownRecord)
      : null) ?? {};

  const cognitiveHeatmap = normalizeHeatmap(
    pickFirst(managerSummary, [
      "cognitive_load_heatmap",
      "cognitiveLoadHeatmap",
      "cognitive_load",
      "cognitiveLoad",
      "heatmap",
    ])
  );

  let analysisDimensions = normalizeAnalysisDimensions(
    pickFirst(managerSummary, [
      "analysis_dimensions",
      "analysisDimensions",
      "dimensions",
    ])
  );

  if (analysisDimensions.length === 0) {
    analysisDimensions = normalizeAnalysisDimensions(
      pickFromSources(sources, [
        "analysis_dimensions",
        "analysisDimensions",
        "dimensions",
      ])
    );
  }

  if (analysisDimensions.length === 0) {
    analysisDimensions = Object.keys(cognitiveHeatmap).map((key) => {
      const normalizedKey = key.trim() || toDimensionKey(`dimension_${Math.random().toString(36).slice(2, 8)}`);
      return {
        key: normalizedKey,
        label: formatDimensionLabel(normalizedKey),
      };
    });
  }

  return {
    cognitive_load_heatmap: cognitiveHeatmap,
    sentiment: normalizeSentiment(
      pickFirst(managerSummary, [
        "sentiment",
        "sentiments",
        "overall_sentiment",
        "overallSentiment",
      ])
    ),
    demographics: normalizeDemographics(
      pickFirst(managerSummary, [
        "demographics",
        "demographic_breakdown",
        "demographicBreakdown",
        "segments",
      ])
    ),
    analysis_dimensions: analysisDimensions,
  };
}

function normalizePersonas(payload: unknown): NormalizedPersona[] {
  const sources = getPayloadSources(payload);
  const rawCandidates = pickFromSources(sources, [
    "personas",
    "agents",
    "participants",
    "synthetic_personas",
    "syntheticPersonas",
  ]);

  const normalizedArray = Array.isArray(rawCandidates)
    ? rawCandidates
    : isRecord(rawCandidates)
      ? Object.entries(rawCandidates).map(([agentId, value]) => {
          if (isRecord(value)) {
            return {
              ...value,
              agent_id: asString(value.agent_id) ?? agentId,
            };
          }

          return {
            agent_id: agentId,
            value,
          };
        })
      : [];

  return normalizedArray.map((entry, index) => {
    if (!isRecord(entry)) {
      return {
        agent_id: `agent_${index + 1}`,
        name: `Persona ${index + 1}`,
        response: entry,
      };
    }

    const agentId =
      asString(
        pickFirst(entry, [
          "agent_id",
          "agentId",
          "id",
          "persona_id",
          "personaId",
        ])
      ) ?? `agent_${index + 1}`;

    const name =
      asString(
        pickFirst(entry, [
          "name",
          "persona_name",
          "personaName",
          "display_name",
          "displayName",
          "agent_name",
          "agentName",
        ])
      ) ?? `Persona ${index + 1}`;

    const sanitized: NormalizedPersona = {
      agent_id: agentId,
      name,
    };

    const allowedKeys = [
      "age_range",
      "occupation",
      "persona_type",
      "gender_identity",
      "ethnicity",
      "region",
      "sentiment_label",
      "sentiment_score",
      "stress_level",
      "quote",
      "reaction_summary",
      "tech_comfort",
      "budget_sensitivity",
      "stance",
    ];

    for (const key of allowedKeys) {
      if (key in entry) {
        sanitized[key] = entry[key];
      }
    }

    const cognitiveLoadCandidate = pickFirst(entry, [
      "cognitive_load",
      "cognitiveLoad",
    ]);
    if (isRecord(cognitiveLoadCandidate)) {
      sanitized.cognitive_load = cognitiveLoadCandidate;
    }

    return sanitized;
  });
}

function summarizeSimulationRaw(
  payload: unknown,
  personaCount: number
): UnknownRecord {
  const sources = getPayloadSources(payload);

  const runId = asString(pickFromSources(sources, ["run_id", "runId"])) ?? null;
  const generationFailures = pickFromSources(sources, [
    "generation_failures",
    "generationFailures",
    "failures",
  ]);

  const failureList = Array.isArray(generationFailures)
    ? generationFailures.slice(0, 20)
    : [];

  return {
    run_id: runId,
    persona_count: personaCount,
    generation_failures: failureList,
    backend_summary_included: true,
  };
}

function normalizeTargetAudienceDetails(
  payload: unknown,
  fallbackTargetAudience?: string
): {
  targetAudience: string;
  targetAudienceGenerated: boolean;
  targetAudienceGenerationNotes: string | null;
} {
  const sources = getPayloadSources(payload);

  const targetAudience =
    asString(
      pickFromSources(sources, [
        "target_audience",
        "targetAudience",
        "resolved_target_audience",
        "resolvedTargetAudience",
      ])
    ) ??
    (fallbackTargetAudience?.trim() ? fallbackTargetAudience : "Audience unavailable");

  const targetAudienceGenerated =
    asBoolean(
      pickFromSources(sources, [
        "target_audience_generated",
        "targetAudienceGenerated",
        "audience_generated",
        "audienceGenerated",
      ])
    ) ?? false;

  const targetAudienceGenerationNotes =
    asString(
      pickFromSources(sources, [
        "target_audience_generation_notes",
        "targetAudienceGenerationNotes",
        "audience_generation_notes",
        "audienceGenerationNotes",
      ])
    ) ?? null;

  return {
    targetAudience,
    targetAudienceGenerated,
    targetAudienceGenerationNotes,
  };
}

function normalizeAnswer(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    const joined = value
      .map((entry) => {
        if (typeof entry === "string") {
          return entry;
        }

        if (isRecord(entry)) {
          return asString(pickFirst(entry, ["text", "content", "answer"])) ?? "";
        }

        return "";
      })
      .filter(Boolean)
      .join("\n");

    return joined;
  }

  if (isRecord(value)) {
    return (
      asString(pickFirst(value, ["text", "content", "answer", "response"])) ?? ""
    );
  }

  return "";
}

function normalizeMemorySnippets(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value;
  }

  if (value === null || value === undefined) {
    return [];
  }

  if (isRecord(value)) {
    return Object.entries(value).map(([key, entry]) => ({ key, entry }));
  }

  return [value];
}

function normalizeFollowupResponse(
  payload: unknown,
  requestedAgentId: string,
  requestedQuestion: string
): FollowupResult {
  const sources = getPayloadSources(payload);

  const answer = normalizeAnswer(
    pickFromSources(sources, [
      "answer",
      "response",
      "reply",
      "message",
      "content",
      "output",
    ])
  );

  const memorySnippets = normalizeMemorySnippets(
    pickFromSources(sources, [
      "memory_snippets",
      "memorySnippets",
      "memory",
      "memory_stream",
      "memoryStream",
      "context",
      "retrieved_memories",
      "retrievedMemories",
    ])
  );

  const agentId =
    asString(
      pickFromSources(sources, [
        "agent_id",
        "agentId",
        "id",
        "persona_id",
        "personaId",
      ])
    ) ?? requestedAgentId;

  const question =
    asString(pickFromSources(sources, ["question", "query", "prompt"])) ??
    requestedQuestion;

  const errorMessage = asString(
    pickFromSources(sources, ["error", "detail", "message"])
  );

  return {
    agent_id: agentId,
    question,
    answer,
    memory_snippets: memorySnippets,
    raw: payload,
    ...(errorMessage ? { error: errorMessage } : {}),
  };
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === "string") {
    return error;
  }

  try {
    return JSON.stringify(error);
  } catch {
    return "Unexpected error";
  }
}

function extractHttpErrorMessage(body: unknown, fallback: string): string {
  if (typeof body === "string" && body.trim().length > 0) {
    return body;
  }

  if (isRecord(body)) {
    const detail = asString(pickFirst(body, ["detail", "error", "message"]));
    if (detail) {
      return detail;
    }

    try {
      return JSON.stringify(body);
    } catch {
      return fallback;
    }
  }

  return fallback;
}

async function postJson(path: string, payload: Record<string, unknown>) {
  const url = new URL(path, BACKEND_URL).toString();

  let response: Response;

  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    throw new Error(`Unable to reach backend at ${url}: ${toErrorMessage(error)}`);
  }

  const responseText = await response.text();
  let parsedBody: unknown = null;

  if (responseText) {
    try {
      parsedBody = JSON.parse(responseText);
    } catch {
      parsedBody = responseText;
    }
  }

  if (!response.ok) {
    const detail = extractHttpErrorMessage(
      parsedBody,
      response.statusText || "Request failed"
    );

    throw new BackendRequestError(
      response.status,
      `POST ${path} failed (${response.status}): ${detail}`,
      parsedBody
    );
  }

  return parsedBody;
}

function makeFallbackSimulationProps(
  targetAudience: string | undefined,
  stimulusDescription: string,
  imageUrl: string | undefined,
  errorMessage: string,
  raw: unknown
): FocusGroupWidgetProps {
  const normalizedDimensions: AnalysisDimension[] = [];
  return {
    target_audience: targetAudience?.trim() ? targetAudience : "Audience unavailable",
    target_audience_generated: false,
    target_audience_generation_notes: null,
    stimulus_description: stimulusDescription,
    image_url: imageUrl ?? null,
    analysis_dimensions: normalizedDimensions,
    manager_summary: {
      cognitive_load_heatmap: {},
      sentiment: {
        overall: null,
        breakdown: {},
        raw: null,
      },
      demographics: [],
      analysis_dimensions: normalizedDimensions,
    },
    personas: [],
    raw,
    error: errorMessage,
  };
}

server.tool(
  {
    name: "run_synthetic_focus_group",
    description:
      "Run a synthetic focus group simulation and render aggregated results in a widget",
    schema: z.object({
      target_audience: z
        .string()
        .optional()
        .describe(
          "Optional target audience. If omitted, the backend auto-generates one from the stimulus."
        ),
      stimulus_description: z
        .string()
        .describe("Description of the concept, prompt, or stimulus being tested"),
      image_url: z
        .string()
        .optional()
        .describe("Optional image URL used as simulation stimulus"),
      persona_count: z
        .number()
        .int()
        .min(3)
        .max(1000)
        .optional()
        .describe("Optional number of personas to simulate (3-1000)"),
    }),
    widget: {
      name: "focus-group-results",
      invoking: "Running synthetic focus group...",
      invoked: "Focus group results ready",
      widgetAccessible: true,
      resultCanProduceWidget: true,
    },
  },
  async ({
    target_audience,
    stimulus_description,
    image_url,
    persona_count,
  }) => {
    const requestPayload: Record<string, unknown> = {
      stimulus_description,
    };

    if (target_audience?.trim()) {
      requestPayload.target_audience = target_audience;
    }

    if (image_url) {
      requestPayload.image_url = image_url;
    }

    if (typeof persona_count === "number") {
      requestPayload.persona_count = persona_count;
    }

    try {
      const simulationResponse = await postJson(
        "/synthetic/simulate",
        requestPayload
      );

      const audienceDetails = normalizeTargetAudienceDetails(
        simulationResponse,
        target_audience
      );
      const managerSummary = normalizeManagerSummary(simulationResponse);
      const personas = normalizePersonas(simulationResponse);

      const props: FocusGroupWidgetProps = {
        target_audience: audienceDetails.targetAudience,
        target_audience_generated: audienceDetails.targetAudienceGenerated,
        target_audience_generation_notes:
          audienceDetails.targetAudienceGenerationNotes,
        stimulus_description,
        image_url: image_url ?? null,
        analysis_dimensions: managerSummary.analysis_dimensions,
        manager_summary: managerSummary,
        personas,
        raw: summarizeSimulationRaw(simulationResponse, personas.length),
      };

      return widget({
        props,
        output: text(
          `Synthetic focus group completed for "${props.target_audience}" with ${props.personas.length} personas.`
        ),
      });
    } catch (error) {
      const message = toErrorMessage(error);
      const raw =
        error instanceof BackendRequestError ? error.payload ?? null : null;

      return widget({
        props: makeFallbackSimulationProps(
          target_audience,
          stimulus_description,
          image_url,
          message,
          raw
        ),
        output: text(`Synthetic focus group failed: ${message}`),
      });
    }
  }
);

server.tool(
  {
    name: "ask_persona_followup",
    description:
      "Ask a follow-up question to a specific synthetic persona and retrieve memory-grounded response",
    schema: z.object({
      agent_id: z.string().describe("ID of the synthetic persona to query"),
      question: z.string().describe("Follow-up question for the selected persona"),
      persona_profile: z
        .record(z.string(), z.unknown())
        .optional()
        .describe("Optional persona snapshot used if backend store was reset"),
    }),
    outputSchema: z.object({
      agent_id: z.string(),
      question: z.string(),
      answer: z.string(),
      memory_snippets: z.array(z.unknown()),
      raw: z.unknown(),
      error: z.string().optional(),
    }),
  },
  async ({ agent_id, question, persona_profile }) => {
    try {
      const payload: Record<string, unknown> = {
        agent_id,
        question,
      };
      if (persona_profile) {
        payload.persona_profile = persona_profile;
      }
      const followupResponse = await postJson("/synthetic/followup", payload);

      return object(normalizeFollowupResponse(followupResponse, agent_id, question));
    } catch (error) {
      const message = toErrorMessage(error);
      const raw =
        error instanceof BackendRequestError ? error.payload ?? null : null;

      return object({
        agent_id,
        question,
        answer: "",
        memory_snippets: [],
        raw,
        error: message,
      });
    }
  }
);

server.listen().then(() => {
  console.log("Server running");
});

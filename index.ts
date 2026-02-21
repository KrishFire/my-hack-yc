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

type NormalizedManagerSummary = {
  cognitive_load_heatmap: Record<string, unknown>;
  sentiment: NormalizedSentiment;
  demographics: UnknownRecord[];
};

type NormalizedPersona = UnknownRecord & {
  agent_id: string;
  name: string;
};

type FocusGroupWidgetProps = {
  target_audience: string;
  stimulus_description: string;
  image_url: string | null;
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

  return {
    cognitive_load_heatmap: normalizeHeatmap(
      pickFirst(managerSummary, [
        "cognitive_load_heatmap",
        "cognitiveLoadHeatmap",
        "cognitive_load",
        "cognitiveLoad",
        "heatmap",
      ])
    ),
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

    return {
      ...entry,
      agent_id: agentId,
      name,
    };
  });
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
  targetAudience: string,
  stimulusDescription: string,
  imageUrl: string | undefined,
  errorMessage: string,
  raw: unknown
): FocusGroupWidgetProps {
  return {
    target_audience: targetAudience,
    stimulus_description: stimulusDescription,
    image_url: imageUrl ?? null,
    manager_summary: {
      cognitive_load_heatmap: {},
      sentiment: {
        overall: null,
        breakdown: {},
        raw: null,
      },
      demographics: [],
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
        .describe("Description of the target audience for the simulation"),
      stimulus_description: z
        .string()
        .describe("Description of the concept, prompt, or stimulus being tested"),
      image_url: z
        .string()
        .optional()
        .describe("Optional image URL used as simulation stimulus"),
    }),
    widget: {
      name: "focus-group-results",
      invoking: "Running synthetic focus group...",
      invoked: "Focus group results ready",
      widgetAccessible: true,
      resultCanProduceWidget: true,
    },
  },
  async ({ target_audience, stimulus_description, image_url }) => {
    const requestPayload: Record<string, unknown> = {
      target_audience,
      stimulus_description,
    };

    if (image_url) {
      requestPayload.image_url = image_url;
    }

    try {
      const simulationResponse = await postJson(
        "/synthetic/simulate",
        requestPayload
      );

      const props: FocusGroupWidgetProps = {
        target_audience,
        stimulus_description,
        image_url: image_url ?? null,
        manager_summary: normalizeManagerSummary(simulationResponse),
        personas: normalizePersonas(simulationResponse),
        raw: simulationResponse,
      };

      return widget({
        props,
        output: text(
          `Synthetic focus group completed for "${target_audience}" with ${props.personas.length} personas.`
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
  async ({ agent_id, question }) => {
    try {
      const followupResponse = await postJson("/synthetic/followup", {
        agent_id,
        question,
      });

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

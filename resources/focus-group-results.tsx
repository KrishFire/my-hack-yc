import { McpUseProvider, useCallTool, useWidget, type WidgetMetadata } from "mcp-use/react";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { z } from "zod";
import "./styles.css";

const personaSchema = z
  .object({
    agent_id: z.string(),
    name: z.string(),
  })
  .passthrough();

const sentimentSchema = z.object({
  overall: z.union([z.string(), z.number(), z.null()]),
  breakdown: z.record(z.string(), z.unknown()),
  raw: z.unknown(),
});

const analysisDimensionSchema = z.object({
  key: z.string(),
  label: z.string(),
  description: z.string().optional(),
});

const managerSummarySchema = z.object({
  cognitive_load_heatmap: z.record(z.string(), z.unknown()),
  sentiment: sentimentSchema,
  demographics: z.array(z.record(z.string(), z.unknown())),
  analysis_dimensions: z.array(analysisDimensionSchema).optional(),
});

export const propSchema = z.object({
  target_audience: z.string(),
  target_audience_generated: z.boolean().optional(),
  target_audience_generation_notes: z.string().nullable().optional(),
  stimulus_description: z.string(),
  image_url: z.string().nullable(),
  analysis_dimensions: z.array(analysisDimensionSchema).optional(),
  manager_summary: managerSummarySchema,
  personas: z.array(personaSchema),
  raw: z.unknown(),
  error: z.string().optional(),
});

export type FocusGroupProps = z.infer<typeof propSchema>;

type FocusGroupState = {
  activeAgentId: string | null;
};

type FollowupStructuredContent = {
  agent_id: string;
  question: string;
  answer: string;
  memory_snippets: unknown[];
  raw: unknown;
  error?: string;
};

type RunToolInput = {
  target_audience?: string;
  stimulus_description: string;
  image_url?: string;
};

type FollowupBatchItem = {
  agent_id: string;
  persona_name: string;
  answer: string;
  memory_snippets: unknown[];
  error?: string;
};

type SentimentStats = {
  positive: number;
  neutral: number;
  negative: number;
};

type DemographicRow = {
  label: string;
  value: number;
};

type DemographicChart = {
  segment: string;
  rows: DemographicRow[];
};

type TargetMode = "selected" | "all" | "segment";

type SegmentKey =
  | "sentiment_label"
  | "gender_identity"
  | "ethnicity"
  | "occupation"
  | "region"
  | "persona_type";

const SEGMENT_LABELS: Record<SegmentKey, string> = {
  sentiment_label: "Sentiment",
  gender_identity: "Gender",
  ethnicity: "Ethnicity",
  occupation: "Occupation",
  region: "Region",
  persona_type: "Persona Type",
};

const DEFAULT_PROPS: FocusGroupProps = {
  target_audience: "",
  target_audience_generated: false,
  target_audience_generation_notes: null,
  stimulus_description: "",
  image_url: null,
  analysis_dimensions: [],
  manager_summary: {
    cognitive_load_heatmap: {},
    sentiment: {
      overall: null,
      breakdown: {},
      raw: null,
    },
    demographics: [],
    analysis_dimensions: [],
  },
  personas: [],
  raw: null,
};

export const widgetMetadata: WidgetMetadata = {
  description:
    "Displays synthetic focus group manager summary and supports persona follow-up questions",
  props: propSchema,
  exposeAsTool: false,
  metadata: {
    invoking: "Running synthetic focus group...",
    invoked: "Focus group results ready",
    prefersBorder: true,
  },
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return fallback;
}

function pickFirst(record: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }

  return undefined;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "N/A";
  }

  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.map((entry) => formatValue(entry)).join(", ");
  }

  if (isRecord(value)) {
    try {
      return JSON.stringify(value);
    } catch {
      return "[object]";
    }
  }

  return "N/A";
}

function formatDemographicSegment(segment: string): string {
  return segment
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function errorToMessage(error: unknown): string {
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

function appendError(previous: string | null, next: string): string {
  return previous ? `${previous} | ${next}` : next;
}

function memorySnippetToText(snippet: unknown): string {
  if (typeof snippet === "string") {
    return snippet;
  }

  if (typeof snippet === "number" || typeof snippet === "boolean") {
    return String(snippet);
  }

  if (isRecord(snippet)) {
    const directText = asString(
      pickFirst(snippet, ["text", "content", "memory", "summary", "entry"])
    );
    if (directText) {
      return directText;
    }

    try {
      return JSON.stringify(snippet);
    } catch {
      return "[memory item]";
    }
  }

  return "[memory item]";
}

function getPersonaSentiment(persona: Record<string, unknown>): "positive" | "neutral" | "negative" {
  const raw = asString(
    pickFirst(persona, ["sentiment_label", "sentimentLabel", "sentiment"])
  );
  const normalized = (raw ?? "neutral").toLowerCase();

  if (normalized === "positive" || normalized === "negative") {
    return normalized;
  }

  return "neutral";
}

function getPersonaRole(persona: Record<string, unknown>): string | null {
  return (
    asString(
      pickFirst(persona, ["occupation", "persona_type", "personaType", "role", "archetype"])
    ) ?? null
  );
}

function getPersonaPreference(persona: Record<string, unknown>): number {
  const score = toNumber(pickFirst(persona, ["sentiment_score", "sentimentScore"]), 0);
  return clampPercent(((score + 1) / 2) * 100);
}

function getPersonaSegmentValue(
  persona: Record<string, unknown>,
  segmentKey: SegmentKey
): string | null {
  if (segmentKey === "sentiment_label") {
    return getPersonaSentiment(persona);
  }

  const direct = asString(pickFirst(persona, [segmentKey]));
  if (direct) {
    return direct;
  }

  if (segmentKey === "persona_type") {
    return asString(pickFirst(persona, ["personaType", "role", "archetype"])) ?? null;
  }

  return null;
}

function normalizeSentimentStats(breakdown: Record<string, unknown>): SentimentStats {
  return {
    positive: Math.max(0, toNumber(breakdown.positive, 0)),
    neutral: Math.max(0, toNumber(breakdown.neutral, 0)),
    negative: Math.max(0, toNumber(breakdown.negative, 0)),
  };
}

function normalizeDemographicCharts(
  demographics: Array<Record<string, unknown>>
): DemographicChart[] {
  return demographics
    .map((segment, index) => {
      const segmentName =
        asString(pickFirst(segment, ["segment", "label", "name", "group"])) ??
        `Segment ${index + 1}`;

      const distributionCandidate = pickFirst(segment, ["distribution", "counts", "breakdown"]);
      const distribution = isRecord(distributionCandidate)
        ? distributionCandidate
        : Object.fromEntries(
            Object.entries(segment).filter(
              ([key, value]) =>
                key !== "segment" &&
                key !== "label" &&
                key !== "name" &&
                key !== "group" &&
                typeof value !== "object"
            )
          );

      const rows = Object.entries(distribution)
        .map(([label, value]) => ({
          label,
          value: Math.max(0, toNumber(value, 0)),
        }))
        .filter((row) => row.value > 0)
        .sort((a, b) => b.value - a.value)
        .slice(0, 6);

      return {
        segment: segmentName,
        rows,
      };
    })
    .filter((chart) => chart.rows.length > 0);
}

function sentimentToneClasses(sentiment: "positive" | "neutral" | "negative"): string {
  if (sentiment === "positive") {
    return "border-emerald-300/60 bg-emerald-500/10";
  }

  if (sentiment === "negative") {
    return "border-rose-300/60 bg-rose-500/10";
  }

  return "border-amber-300/60 bg-amber-500/10";
}

const FocusGroupResults: React.FC = () => {
  const {
    props,
    isPending,
    state,
    setState,
    sendFollowUpMessage,
  } = useWidget<FocusGroupProps, FocusGroupState>(DEFAULT_PROPS);

  const {
    callToolAsync: runSyntheticFocusGroup,
    isPending: isRerunPending,
  } = useCallTool<
    RunToolInput,
    {
      structuredContent?: FocusGroupProps;
    }
  >("run_synthetic_focus_group");

  const {
    callToolAsync: callAskPersonaFollowup,
    isPending: isFollowupPending,
  } = useCallTool<
    {
      agent_id: string;
      question: string;
      persona_profile?: Record<string, unknown>;
    },
    { structuredContent?: FollowupStructuredContent }
  >("ask_persona_followup");

  const [currentRunProps, setCurrentRunProps] = useState<FocusGroupProps | null>(null);
  const [questionInput, setQuestionInput] = useState("");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [targetMode, setTargetMode] = useState<TargetMode>("selected");
  const [segmentKey, setSegmentKey] = useState<SegmentKey>("sentiment_label");
  const [segmentValue, setSegmentValue] = useState("");

  const [editableTargetAudience, setEditableTargetAudience] = useState("");
  const [editableStimulus, setEditableStimulus] = useState("");
  const [editableImageUrl, setEditableImageUrl] = useState("");
  const lastHydratedRunSignature = useRef<string | null>(null);

  const [latestFollowups, setLatestFollowups] = useState<FollowupBatchItem[]>([]);

  useEffect(() => {
    if (!isPending && currentRunProps === null) {
      setCurrentRunProps(props);
    }
  }, [currentRunProps, isPending, props]);

  const displayProps = currentRunProps ?? props;
  const runHydrationSignature = useMemo(() => {
    const runId =
      isRecord(displayProps.raw) && asString(displayProps.raw.run_id)
        ? asString(displayProps.raw.run_id)
        : "";
    return [
      runId ?? "",
      displayProps.target_audience ?? "",
      displayProps.stimulus_description ?? "",
      displayProps.image_url ?? "",
    ].join("::");
  }, [
    displayProps.image_url,
    displayProps.raw,
    displayProps.stimulus_description,
    displayProps.target_audience,
  ]);

  useEffect(() => {
    if (lastHydratedRunSignature.current === runHydrationSignature) {
      return;
    }

    setEditableTargetAudience(displayProps.target_audience ?? "");
    setEditableStimulus(displayProps.stimulus_description ?? "");
    setEditableImageUrl(displayProps.image_url ?? "");
    lastHydratedRunSignature.current = runHydrationSignature;
  }, [displayProps, runHydrationSignature]);

  const personas = useMemo(() => {
    return Array.isArray(displayProps.personas) ? displayProps.personas : [];
  }, [displayProps.personas]);

  useEffect(() => {
    if (personas.length === 0) {
      return;
    }

    if (!state?.activeAgentId) {
      void setState({ activeAgentId: personas[0].agent_id });
      return;
    }

    const stillExists = personas.some((persona) => persona.agent_id === state.activeAgentId);
    if (!stillExists) {
      void setState({ activeAgentId: personas[0].agent_id });
    }
  }, [personas, setState, state?.activeAgentId]);

  const segmentValues = useMemo(() => {
    const values = new Set<string>();

    personas.forEach((persona) => {
      const value = getPersonaSegmentValue(persona, segmentKey);
      if (value) {
        values.add(value);
      }
    });

    return Array.from(values).sort((a, b) => a.localeCompare(b));
  }, [personas, segmentKey]);

  useEffect(() => {
    if (targetMode !== "segment") {
      return;
    }

    if (!segmentValues.length) {
      setSegmentValue("");
      return;
    }

    if (!segmentValues.includes(segmentValue)) {
      setSegmentValue(segmentValues[0]);
    }
  }, [segmentValue, segmentValues, targetMode]);

  const activeAgentId = state?.activeAgentId ?? null;

  const activePersona =
    personas.find((persona) => persona.agent_id === activeAgentId) ?? null;

  const targetedPersonas = useMemo(() => {
    if (targetMode === "all") {
      return personas;
    }

    if (targetMode === "selected") {
      return activePersona ? [activePersona] : [];
    }

    if (!segmentValue) {
      return [];
    }

    return personas.filter((persona) => {
      const value = getPersonaSegmentValue(persona, segmentKey);
      return value?.toLowerCase() === segmentValue.toLowerCase();
    });
  }, [activePersona, personas, segmentKey, segmentValue, targetMode]);

  const sentimentStats = normalizeSentimentStats(
    displayProps.manager_summary?.sentiment?.breakdown ?? {}
  );

  const totalSentimentVotes = Math.max(
    1,
    sentimentStats.positive + sentimentStats.neutral + sentimentStats.negative
  );

  const demographics = normalizeDemographicCharts(
    Array.isArray(displayProps.manager_summary?.demographics)
      ? displayProps.manager_summary.demographics
      : []
  );

  const averagePreference = personas.length
    ? Math.round(
        personas.reduce((sum, persona) => sum + getPersonaPreference(persona), 0) /
          personas.length
      )
    : 0;

  const sentimentOverallRaw = displayProps.manager_summary?.sentiment?.overall;
  const sentimentApplicable =
    String(sentimentOverallRaw ?? "").toLowerCase() !== "not_applicable";

  const simulationError = displayProps.error;

  const handleSelectPersona = useCallback(
    (agentId: string) => {
      void setState((previous) => ({
        ...(previous ?? { activeAgentId: null }),
        activeAgentId: agentId,
      }));
    },
    [setState]
  );

  const handleRerunSimulation = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();

      const stimulus = editableStimulus.trim();
      if (!stimulus) {
        setRerunError("Stimulus description is required to rerun simulation.");
        return;
      }

      setRerunError(null);

      const payload: RunToolInput = {
        stimulus_description: stimulus,
      };

      const targetAudience = editableTargetAudience.trim();
      if (targetAudience) {
        payload.target_audience = targetAudience;
      }

      const imageUrl = editableImageUrl.trim();
      if (imageUrl) {
        payload.image_url = imageUrl;
      }

      try {
        const result = await runSyntheticFocusGroup(payload);
        const structured = result.structuredContent;

        if (!structured) {
          throw new Error("Rerun did not return structured content.");
        }

        const parsed = propSchema.safeParse(structured);
        if (!parsed.success) {
          throw new Error("Rerun returned an unexpected schema.");
        }

        setCurrentRunProps(parsed.data);
        setLatestFollowups([]);
        setSubmitError(null);

        if (parsed.data.personas.length > 0) {
          await setState({ activeAgentId: parsed.data.personas[0].agent_id });
        }
      } catch (error) {
        setRerunError(errorToMessage(error));
      }
    },
    [
      editableImageUrl,
      editableStimulus,
      editableTargetAudience,
      runSyntheticFocusGroup,
      setState,
    ]
  );

  const handleSubmitFollowup = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();

      const trimmedQuestion = questionInput.trim();

      if (trimmedQuestion.length === 0) {
        return;
      }

      if (targetedPersonas.length === 0) {
        setSubmitError("No personas match this selection.");
        return;
      }

      setIsSubmitting(true);
      setSubmitError(null);

      const calls = targetedPersonas.map(async (persona) => {
        try {
          const result = await callAskPersonaFollowup({
            agent_id: persona.agent_id,
            question: trimmedQuestion,
            persona_profile: persona,
          });

          const structured = result.structuredContent;
          return {
            agent_id: persona.agent_id,
            persona_name: persona.name,
            answer: structured?.answer ?? "No answer returned.",
            memory_snippets: structured?.memory_snippets ?? [],
            error: structured?.error,
          } as FollowupBatchItem;
        } catch (error) {
          return {
            agent_id: persona.agent_id,
            persona_name: persona.name,
            answer: "",
            memory_snippets: [],
            error: errorToMessage(error),
          } as FollowupBatchItem;
        }
      });

      const batchResults = await Promise.all(calls);
      setLatestFollowups(batchResults);

      const errors = batchResults
        .filter((item) => item.error)
        .map((item) => `${item.persona_name}: ${item.error}`);

      if (errors.length > 0) {
        setSubmitError(errors.join(" | "));
      }

      const hostPrompt =
        targetedPersonas.length === 1
          ? `Ask persona ${targetedPersonas[0].agent_id} this follow-up question: ${trimmedQuestion}`
          : `Ask personas [${targetedPersonas
              .map((persona) => persona.agent_id)
              .join(", ")}] this follow-up question: ${trimmedQuestion}`;

      try {
        await sendFollowUpMessage(hostPrompt);
      } catch (error) {
        setSubmitError((previous) =>
          appendError(previous, `Host follow-up failed: ${errorToMessage(error)}`)
        );
      }

      setQuestionInput("");
      setIsSubmitting(false);
    },
    [
      callAskPersonaFollowup,
      questionInput,
      sendFollowUpMessage,
      targetedPersonas,
    ]
  );

  if (isPending) {
    return (
      <McpUseProvider>
        <div className="rounded-3xl border border-default bg-surface-elevated p-6 space-y-4">
          <div className="h-7 w-72 rounded bg-default/15 animate-pulse" />
          <div className="h-4 w-full rounded bg-default/10 animate-pulse" />
          <div className="h-4 w-4/5 rounded bg-default/10 animate-pulse" />
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
            <div className="h-24 rounded-xl bg-default/10 animate-pulse" />
            <div className="h-24 rounded-xl bg-default/10 animate-pulse" />
            <div className="h-24 rounded-xl bg-default/10 animate-pulse" />
          </div>
        </div>
      </McpUseProvider>
    );
  }

  return (
    <McpUseProvider>
      <div
        className="relative overflow-hidden rounded-[30px] border border-default/70 p-5 sm:p-6 space-y-6"
        style={{
          fontFamily: '"Avenir Next", "Trebuchet MS", "Gill Sans", sans-serif',
          background:
            "radial-gradient(circle at 8% 8%, rgba(50,130,184,0.18), transparent 38%), radial-gradient(circle at 90% 0%, rgba(237,101,42,0.16), transparent 36%), linear-gradient(160deg, rgba(255,248,240,0.96), rgba(247,252,255,0.96))",
        }}
      >
        <div className="pointer-events-none absolute -top-10 -right-8 h-36 w-36 rounded-full bg-info/15 blur-2xl" />
        <div className="pointer-events-none absolute -bottom-12 -left-10 h-40 w-40 rounded-full bg-warning/20 blur-2xl" />

        <header className="relative space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-2xl sm:text-3xl font-semibold text-default tracking-tight">
                Synthetic Focus Group Results
              </h2>
              <p className="text-sm text-secondary mt-1">
                Aggregated insights with persona-level follow-up in one view.
              </p>
            </div>

            <div className="rounded-xl border border-default bg-white/65 px-3 py-2 text-xs text-secondary">
              <p>
                Audience Source: {displayProps.target_audience_generated ? "Auto-generated" : "Provided"}
              </p>
              {displayProps.target_audience_generation_notes && (
                <p className="mt-1 text-[11px] text-secondary/80">
                  {displayProps.target_audience_generation_notes}
                </p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.45fr_1fr] gap-3">
            <div className="rounded-2xl border border-default bg-white/70 p-4 space-y-2 shadow-sm">
              <p className="text-[11px] uppercase tracking-[0.18em] text-secondary">
                Target Audience
              </p>
              <p className="text-sm sm:text-base text-default leading-relaxed">
                {displayProps.target_audience || "Not provided"}
              </p>
              <div className="mt-3 rounded-xl border border-default/70 bg-surface px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.16em] text-secondary mb-1">
                  Stimulus
                </p>
                <p className="text-sm text-default">
                  {displayProps.stimulus_description || "Not provided"}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="rounded-2xl border border-default bg-white/70 p-3 text-center shadow-sm">
                <p className="text-[11px] uppercase tracking-wide text-secondary">Personas</p>
                <p className="mt-1 text-xl font-semibold text-default">{personas.length}</p>
              </div>
              <div className="rounded-2xl border border-default bg-white/70 p-3 text-center shadow-sm">
                <p className="text-[11px] uppercase tracking-wide text-secondary">Avg Preference</p>
                <p className="mt-1 text-xl font-semibold text-default">{averagePreference}</p>
              </div>
              <div className="rounded-2xl border border-default bg-white/70 p-3 text-center shadow-sm">
                <p className="text-[11px] uppercase tracking-wide text-secondary">Sentiment</p>
                <p className="mt-1 text-xl font-semibold text-default">
                  {formatValue(displayProps.manager_summary?.sentiment?.overall)}
                </p>
              </div>
            </div>
          </div>

          {displayProps.image_url && (
            <div className="rounded-2xl border border-default bg-white/70 p-3 shadow-sm">
              <p className="text-[11px] uppercase tracking-[0.14em] text-secondary mb-2">
                Stimulus Image
              </p>
              <img
                src={displayProps.image_url}
                alt="Stimulus"
                className="max-h-56 w-full object-contain rounded-xl bg-surface"
              />
            </div>
          )}

          {simulationError && (
            <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3">
              <p className="text-sm text-danger">{simulationError}</p>
            </div>
          )}
        </header>

        <section className="relative space-y-3">
          <h3 className="text-lg font-semibold text-default">Refine Simulation</h3>
          <form
            onSubmit={handleRerunSimulation}
            className="rounded-2xl border border-default bg-white/75 p-4 shadow-sm space-y-3"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs uppercase tracking-wide text-secondary">
                  Target Audience
                </label>
                <input
                  value={editableTargetAudience}
                  onChange={(event) => setEditableTargetAudience(event.target.value)}
                  className="w-full rounded-lg border border-default bg-surface px-3 py-2 text-sm text-default outline-none focus:ring-2 focus:ring-info/40"
                  placeholder="Optional: leave blank to auto-generate"
                />
              </div>
              <div className="space-y-1">
                <label className="text-xs uppercase tracking-wide text-secondary">
                  Stimulus Image URL
                </label>
                <input
                  value={editableImageUrl}
                  onChange={(event) => setEditableImageUrl(event.target.value)}
                  className="w-full rounded-lg border border-default bg-surface px-3 py-2 text-sm text-default outline-none focus:ring-2 focus:ring-info/40"
                  placeholder="Optional"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs uppercase tracking-wide text-secondary">
                Stimulus Description
              </label>
              <textarea
                value={editableStimulus}
                onChange={(event) => setEditableStimulus(event.target.value)}
                className="w-full rounded-lg border border-default bg-surface px-3 py-2 text-sm text-default outline-none focus:ring-2 focus:ring-info/40 min-h-20"
                placeholder="Describe what you want analyzed"
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                type="submit"
                disabled={isRerunPending}
                className="rounded-lg bg-info text-white px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRerunPending ? "Rerunning..." : "Rerun Focus Group"}
              </button>
              <p className="text-xs text-secondary">
                Edit audience, stimulus, and image, then rerun in place.
              </p>
            </div>

            {rerunError && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2">
                <p className="text-sm text-danger">{rerunError}</p>
              </div>
            )}
          </form>
        </section>

        <section className="relative space-y-3">
          <h3 className="text-base font-semibold text-secondary/80">Background Summary</h3>
          <div className="opacity-55">
            <div className="rounded-2xl border border-default/70 bg-white/60 p-4 shadow-sm space-y-3">
              <p className="text-[10px] uppercase tracking-[0.16em] text-secondary">Sentiment Mix</p>
              {sentimentApplicable ? (
                <div className="rounded-xl border border-default/80 bg-surface p-3 space-y-2">
                  <div className="h-3 rounded-full overflow-hidden bg-default/10 flex">
                    <div
                      className="h-full bg-emerald-500"
                      style={{
                        width: `${(sentimentStats.positive / totalSentimentVotes) * 100}%`,
                      }}
                    />
                    <div
                      className="h-full bg-amber-400"
                      style={{
                        width: `${(sentimentStats.neutral / totalSentimentVotes) * 100}%`,
                      }}
                    />
                    <div
                      className="h-full bg-rose-500"
                      style={{
                        width: `${(sentimentStats.negative / totalSentimentVotes) * 100}%`,
                      }}
                    />
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="rounded-lg border border-default/70 px-2 py-1 bg-emerald-500/10">
                      <p className="text-secondary">Positive</p>
                      <p className="text-default font-semibold">{sentimentStats.positive}</p>
                    </div>
                    <div className="rounded-lg border border-default/70 px-2 py-1 bg-amber-400/10">
                      <p className="text-secondary">Neutral</p>
                      <p className="text-default font-semibold">{sentimentStats.neutral}</p>
                    </div>
                    <div className="rounded-lg border border-default/70 px-2 py-1 bg-rose-500/10">
                      <p className="text-secondary">Negative</p>
                      <p className="text-default font-semibold">{sentimentStats.negative}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="rounded-xl border border-default/80 bg-surface p-3">
                  <p className="text-xs text-secondary">
                    Sentiment is not applicable for objective UI tasks.
                  </p>
                </div>
              )}
            </div>
          </div>

          <details className="rounded-2xl border border-default/70 bg-white/65 p-4 shadow-sm">
            <summary className="cursor-pointer text-xs uppercase tracking-[0.16em] text-secondary select-none">
              Demographics
            </summary>
            <div className="mt-3">
              {demographics.length === 0 ? (
                <p className="text-sm text-secondary">No demographic breakdown returned.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {demographics.map((chart) => {
                    const maxValue = Math.max(...chart.rows.map((row) => row.value), 1);

                    return (
                      <div
                        key={chart.segment}
                        className="rounded-xl border border-default/80 bg-surface p-3 space-y-2"
                      >
                        <p className="text-xs font-semibold text-default">
                          {formatDemographicSegment(chart.segment)}
                        </p>

                        {chart.rows.map((row) => (
                          <div key={`${chart.segment}-${row.label}`} className="space-y-1">
                            <div className="flex items-center justify-between text-[11px] text-secondary">
                              <span>{row.label}</span>
                              <span>{row.value}</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-default/10 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-[#7aa7d1] to-[#f1a065]"
                                style={{ width: `${(row.value / maxValue) * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </details>
        </section>

        <section className="relative space-y-3">
          <h3 className="text-lg font-semibold text-default">Interactive Agent Roster</h3>
          {personas.length === 0 ? (
            <div className="rounded-xl border border-default bg-white/70 p-4">
              <p className="text-sm text-secondary">No personas were returned by the simulation.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
              {personas.map((persona) => {
                const isActive = persona.agent_id === activeAgentId;
                const sentiment = getPersonaSentiment(persona);
                const role = getPersonaRole(persona);
                const preference = getPersonaPreference(persona);
                const quote =
                  asString(pickFirst(persona, ["quote", "reaction_summary", "summary"])) ??
                  "No quote available.";
                const demographicTag =
                  asString(pickFirst(persona, ["gender_identity", "ethnicity"])) ?? null;

                return (
                  <button
                    key={persona.agent_id}
                    type="button"
                    onClick={() => handleSelectPersona(persona.agent_id)}
                    className={`text-left rounded-2xl border p-4 transition-all duration-200 cursor-pointer hover:-translate-y-0.5 ${sentimentToneClasses(
                      sentiment
                    )} ${
                      isActive
                        ? "ring-2 ring-info/40 shadow-md"
                        : "hover:shadow-sm"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-sm font-semibold text-default">{persona.name}</p>
                        <p className="text-[11px] text-secondary mt-0.5">Agent ID: {persona.agent_id}</p>
                      </div>
                      <span className="text-[10px] px-2 py-1 rounded-full border border-default bg-white/70 text-secondary uppercase tracking-wide">
                        {sentiment}
                      </span>
                    </div>

                    {role && <p className="text-xs text-secondary mt-2">{role}</p>}
                    {demographicTag && (
                      <p className="text-[11px] text-secondary mt-1">{demographicTag}</p>
                    )}

                    <div className="mt-2 space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-secondary">
                        <span>Preference</span>
                        <span>{Math.round(preference)}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/70 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-[#2f7ab6] to-[#f3924f]"
                          style={{ width: `${clampPercent(preference)}%` }}
                        />
                      </div>
                    </div>

                    <p className="text-xs text-secondary mt-2 line-clamp-3">{quote}</p>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <section className="relative space-y-3">
          <h3 className="text-lg font-semibold text-default">Persona Follow-up</h3>
          <div className="rounded-2xl border border-default bg-white/75 p-4 space-y-3 shadow-sm">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-wide text-secondary">Target Personas</p>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className={`text-xs rounded-full px-3 py-1 border ${
                      targetMode === "selected"
                        ? "bg-info text-white border-info"
                        : "bg-surface border-default text-secondary"
                    }`}
                    onClick={() => setTargetMode("selected")}
                  >
                    Selected
                  </button>
                  <button
                    type="button"
                    className={`text-xs rounded-full px-3 py-1 border ${
                      targetMode === "all"
                        ? "bg-info text-white border-info"
                        : "bg-surface border-default text-secondary"
                    }`}
                    onClick={() => setTargetMode("all")}
                  >
                    All Personas
                  </button>
                  <button
                    type="button"
                    className={`text-xs rounded-full px-3 py-1 border ${
                      targetMode === "segment"
                        ? "bg-info text-white border-info"
                        : "bg-surface border-default text-secondary"
                    }`}
                    onClick={() => setTargetMode("segment")}
                  >
                    By Segment
                  </button>
                </div>
              </div>

              {targetMode === "segment" && (
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-wide text-secondary">Segment Filter</p>
                  <div className="grid grid-cols-2 gap-2">
                    <select
                      value={segmentKey}
                      onChange={(event) => setSegmentKey(event.target.value as SegmentKey)}
                      className="rounded-lg border border-default bg-surface px-2 py-1.5 text-xs text-default"
                    >
                      {(Object.keys(SEGMENT_LABELS) as SegmentKey[]).map((key) => (
                        <option key={key} value={key}>
                          {SEGMENT_LABELS[key]}
                        </option>
                      ))}
                    </select>

                    <select
                      value={segmentValue}
                      onChange={(event) => setSegmentValue(event.target.value)}
                      className="rounded-lg border border-default bg-surface px-2 py-1.5 text-xs text-default"
                    >
                      {segmentValues.length === 0 ? (
                        <option value="">No values</option>
                      ) : (
                        segmentValues.map((value) => (
                          <option key={value} value={value}>
                            {value}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                </div>
              )}
            </div>

            <p className="text-sm text-secondary">
              {targetMode === "selected" && activePersona
                ? `Targeting selected persona: ${activePersona.name} (${activePersona.agent_id})`
                : targetMode === "selected"
                  ? "Select a persona from the roster."
                  : targetMode === "all"
                    ? `Targeting all personas (${targetedPersonas.length}).`
                    : `Targeting ${targetedPersonas.length} personas in ${SEGMENT_LABELS[segmentKey]} = ${segmentValue || "(none)"}.`}
            </p>

            <form className="flex flex-col sm:flex-row gap-2" onSubmit={handleSubmitFollowup}>
              <input
                value={questionInput}
                onChange={(event) => setQuestionInput(event.target.value)}
                placeholder="Ask follow-up to targeted personas"
                className="flex-1 rounded-lg border border-default bg-surface px-3 py-2 text-sm text-default outline-none focus:ring-2 focus:ring-info/40"
                disabled={isSubmitting || isFollowupPending || targetedPersonas.length === 0}
              />
              <button
                type="submit"
                disabled={
                  questionInput.trim().length === 0 ||
                  isSubmitting ||
                  isFollowupPending ||
                  targetedPersonas.length === 0
                }
                className="rounded-lg bg-info text-white px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting || isFollowupPending ? "Sending..." : "Ask Target Group"}
              </button>
            </form>

            {submitError && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2">
                <p className="text-sm text-danger">{submitError}</p>
              </div>
            )}

            {latestFollowups.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-wide text-secondary">
                  Latest Batch Responses ({latestFollowups.length})
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {latestFollowups.map((followup) => (
                    <div
                      key={`${followup.agent_id}-${followup.persona_name}`}
                      className="rounded-xl border border-default bg-surface px-4 py-3 space-y-2"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-default">{followup.persona_name}</p>
                        <p className="text-[11px] text-secondary">{followup.agent_id}</p>
                      </div>

                      {followup.error ? (
                        <p className="text-xs text-danger">{followup.error}</p>
                      ) : (
                        <p className="text-xs text-default leading-relaxed">{followup.answer}</p>
                      )}

                      {followup.memory_snippets.length > 0 && !followup.error && (
                        <details className="text-xs text-secondary">
                          <summary className="cursor-pointer">Memory snippets</summary>
                          <ul className="mt-1 space-y-1">
                            {followup.memory_snippets.map((snippet, index) => (
                              <li key={`${followup.agent_id}-memory-${index}`}>
                                {memorySnippetToText(snippet)}
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </McpUseProvider>
  );
};

export default FocusGroupResults;

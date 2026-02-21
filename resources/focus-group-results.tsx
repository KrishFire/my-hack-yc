import { McpUseProvider, useCallTool, useWidget, type WidgetMetadata } from "mcp-use/react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
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

const managerSummarySchema = z.object({
  cognitive_load_heatmap: z.record(z.string(), z.unknown()),
  sentiment: sentimentSchema,
  demographics: z.array(z.record(z.string(), z.unknown())),
});

export const propSchema = z.object({
  target_audience: z.string(),
  stimulus_description: z.string(),
  image_url: z.string().nullable(),
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

const DEFAULT_PROPS: FocusGroupProps = {
  target_audience: "",
  stimulus_description: "",
  image_url: null,
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

function pickFirst(record: Record<string, unknown>, keys: string[]): unknown {
  for (const key of keys) {
    if (key in record) {
      return record[key];
    }
  }

  return undefined;
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

const FocusGroupResults: React.FC = () => {
  const {
    props,
    isPending,
    state,
    setState,
    sendFollowUpMessage,
  } = useWidget<FocusGroupProps, FocusGroupState>(DEFAULT_PROPS);

  const {
    callToolAsync: callAskPersonaFollowup,
    isPending: isFollowupPending,
    isError: isFollowupError,
    data: followupData,
    error: followupError,
  } = useCallTool<
    { agent_id: string; question: string },
    { structuredContent?: FollowupStructuredContent }
  >("ask_persona_followup");

  const [questionInput, setQuestionInput] = useState("");
  const [latestFollowup, setLatestFollowup] =
    useState<FollowupStructuredContent | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const personas = useMemo(() => {
    return Array.isArray(props.personas) ? props.personas : [];
  }, [props.personas]);

  useEffect(() => {
    if (!state?.activeAgentId && personas.length > 0) {
      void setState({ activeAgentId: personas[0].agent_id });
    }
  }, [personas, setState, state?.activeAgentId]);

  useEffect(() => {
    if (followupData?.structuredContent) {
      setLatestFollowup(followupData.structuredContent);
      if (followupData.structuredContent.error) {
        setSubmitError(followupData.structuredContent.error);
      }
    }
  }, [followupData]);

  const activeAgentId = state?.activeAgentId ?? null;

  const activePersona =
    personas.find((persona) => persona.agent_id === activeAgentId) ?? null;

  const heatmapEntries = Object.entries(
    props.manager_summary?.cognitive_load_heatmap ?? {}
  );

  const sentimentOverall = props.manager_summary?.sentiment?.overall;
  const sentimentBreakdown =
    props.manager_summary?.sentiment?.breakdown ?? {};
  const sentimentEntries = Object.entries(sentimentBreakdown);

  const demographics = Array.isArray(props.manager_summary?.demographics)
    ? props.manager_summary.demographics
    : [];

  const simulationError = props.error;
  const followupErrorMessage =
    submitError ??
    (isFollowupError ? errorToMessage(followupError) : latestFollowup?.error ?? null);

  const handleSelectPersona = useCallback(
    (agentId: string) => {
      void setState((previous) => ({
        ...(previous ?? { activeAgentId: null }),
        activeAgentId: agentId,
      }));
    },
    [setState]
  );

  const handleSubmitFollowup = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();

      const selectedAgentId = activeAgentId;
      const trimmedQuestion = questionInput.trim();

      if (!selectedAgentId || trimmedQuestion.length === 0) {
        return;
      }

      setIsSubmitting(true);
      setSubmitError(null);
      setQuestionInput("");

      const hostPrompt = `Ask persona ${selectedAgentId} this follow-up question: ${trimmedQuestion}`;

      const [directResult, hostResult] = await Promise.allSettled([
        callAskPersonaFollowup({
          agent_id: selectedAgentId,
          question: trimmedQuestion,
        }),
        sendFollowUpMessage(hostPrompt),
      ]);

      if (directResult.status === "fulfilled") {
        const structuredContent = directResult.value.structuredContent;
        if (structuredContent) {
          setLatestFollowup(structuredContent);
          if (structuredContent.error) {
            setSubmitError(structuredContent.error);
          }
        }
      } else {
        setSubmitError((previous) =>
          appendError(
            previous,
            `Direct follow-up failed: ${errorToMessage(directResult.reason)}`
          )
        );
      }

      if (hostResult.status === "rejected") {
        setSubmitError((previous) =>
          appendError(
            previous,
            `Host follow-up failed: ${errorToMessage(hostResult.reason)}`
          )
        );
      }

      setIsSubmitting(false);
    },
    [
      activeAgentId,
      callAskPersonaFollowup,
      questionInput,
      sendFollowUpMessage,
      setQuestionInput,
    ]
  );

  if (isPending) {
    return (
      <McpUseProvider>
        <div className="rounded-2xl border border-default bg-surface-elevated p-6 space-y-4">
          <div className="h-6 w-64 rounded bg-default/15 animate-pulse" />
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
      <div className="rounded-2xl border border-default bg-surface-elevated p-5 sm:p-6 space-y-6">
        <header className="space-y-3">
          <h2 className="text-2xl font-semibold text-default">
            Synthetic Focus Group Results
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-xl border border-default bg-surface p-4">
              <p className="text-xs uppercase tracking-wide text-secondary mb-1">
                Target Audience
              </p>
              <p className="text-sm text-default">
                {props.target_audience || "Not provided"}
              </p>
            </div>
            <div className="rounded-xl border border-default bg-surface p-4">
              <p className="text-xs uppercase tracking-wide text-secondary mb-1">
                Stimulus
              </p>
              <p className="text-sm text-default">
                {props.stimulus_description || "Not provided"}
              </p>
            </div>
          </div>
          {props.image_url && (
            <div className="rounded-xl border border-default bg-surface p-3">
              <p className="text-xs uppercase tracking-wide text-secondary mb-2">
                Stimulus Image
              </p>
              <img
                src={props.image_url}
                alt="Stimulus"
                className="max-h-52 w-full object-contain rounded-lg bg-surface-elevated"
              />
            </div>
          )}
          {simulationError && (
            <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3">
              <p className="text-sm text-danger">{simulationError}</p>
            </div>
          )}
        </header>

        <section className="space-y-3">
          <h3 className="text-lg font-semibold text-default">Aggregation View</h3>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <div className="rounded-xl border border-default bg-surface p-4 space-y-2">
              <p className="text-xs uppercase tracking-wide text-secondary">
                Cognitive Load Heatmap
              </p>
              {heatmapEntries.length === 0 ? (
                <p className="text-sm text-secondary">No cognitive load data returned.</p>
              ) : (
                <div className="grid grid-cols-1 gap-2">
                  {heatmapEntries.map(([metric, value]) => (
                    <div
                      key={metric}
                      className="rounded-lg bg-surface-elevated border border-default px-3 py-2 flex items-center justify-between gap-2"
                    >
                      <span className="text-xs text-secondary">{metric}</span>
                      <span className="text-xs font-medium text-default">
                        {formatValue(value)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-default bg-surface p-4 space-y-2">
              <p className="text-xs uppercase tracking-wide text-secondary">Sentiment</p>
              <div className="rounded-lg bg-surface-elevated border border-default px-3 py-2">
                <p className="text-xs text-secondary">Overall</p>
                <p className="text-sm font-medium text-default">
                  {formatValue(sentimentOverall)}
                </p>
              </div>
              {sentimentEntries.length > 0 && (
                <div className="space-y-2">
                  {sentimentEntries.map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-lg border border-default px-3 py-2 flex items-center justify-between"
                    >
                      <span className="text-xs text-secondary">{label}</span>
                      <span className="text-xs font-medium text-default">
                        {formatValue(value)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-default bg-surface p-4 space-y-2">
              <p className="text-xs uppercase tracking-wide text-secondary">
                Demographics
              </p>
              {demographics.length === 0 ? (
                <p className="text-sm text-secondary">No demographic breakdown returned.</p>
              ) : (
                <div className="space-y-2">
                  {demographics.map((segment, index) => {
                    const title =
                      asString(
                        pickFirst(segment, ["segment", "label", "name", "group"])
                      ) ?? `Segment ${index + 1}`;

                    const detailPairs = Object.entries(segment).filter(
                      ([key]) => !["segment", "label", "name", "group"].includes(key)
                    );

                    return (
                      <div
                        key={`${title}-${index}`}
                        className="rounded-lg border border-default bg-surface-elevated px-3 py-2"
                      >
                        <p className="text-xs font-medium text-default">{title}</p>
                        {detailPairs.length > 0 ? (
                          <div className="mt-1 flex flex-wrap gap-2">
                            {detailPairs.map(([key, value]) => (
                              <span
                                key={key}
                                className="text-[11px] rounded-full bg-surface px-2 py-1 border border-default text-secondary"
                              >
                                {key}: {formatValue(value)}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-secondary mt-1">No additional details.</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="space-y-3">
          <h3 className="text-lg font-semibold text-default">Interactive Agent Roster</h3>
          {personas.length === 0 ? (
            <div className="rounded-xl border border-default bg-surface p-4">
              <p className="text-sm text-secondary">No personas were returned by the simulation.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
              {personas.map((persona) => {
                const isActive = persona.agent_id === activeAgentId;
                const subtitle = asString(
                  pickFirst(persona, ["role", "archetype", "occupation", "persona_type"])
                );

                return (
                  <button
                    key={persona.agent_id}
                    type="button"
                    onClick={() => handleSelectPersona(persona.agent_id)}
                    className={`text-left rounded-xl border px-4 py-3 transition-colors cursor-pointer ${
                      isActive
                        ? "border-info bg-info/10"
                        : "border-default bg-surface hover:bg-surface-elevated"
                    }`}
                  >
                    <p className="text-sm font-semibold text-default">{persona.name}</p>
                    <p className="text-xs text-secondary">Agent ID: {persona.agent_id}</p>
                    {subtitle && <p className="text-xs text-secondary mt-1">{subtitle}</p>}
                  </button>
                );
              })}
            </div>
          )}
        </section>

        <section className="space-y-3">
          <h3 className="text-lg font-semibold text-default">Persona Follow-up</h3>
          <div className="rounded-xl border border-default bg-surface p-4 space-y-3">
            <p className="text-sm text-secondary">
              {activePersona
                ? `Selected persona: ${activePersona.name} (${activePersona.agent_id})`
                : "Select a persona from the roster to ask a follow-up question."}
            </p>

            <form className="flex flex-col sm:flex-row gap-2" onSubmit={handleSubmitFollowup}>
              <input
                value={questionInput}
                onChange={(event) => setQuestionInput(event.target.value)}
                placeholder="Ask this persona a follow-up question"
                className="flex-1 rounded-lg border border-default bg-surface-elevated px-3 py-2 text-sm text-default outline-none focus:ring-2 focus:ring-info/40"
                disabled={!activePersona || isSubmitting || isFollowupPending}
              />
              <button
                type="submit"
                disabled={
                  !activePersona ||
                  questionInput.trim().length === 0 ||
                  isSubmitting ||
                  isFollowupPending
                }
                className="rounded-lg bg-info text-white px-4 py-2 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting || isFollowupPending ? "Sending..." : "Send"}
              </button>
            </form>

            {followupErrorMessage && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2">
                <p className="text-sm text-danger">{followupErrorMessage}</p>
              </div>
            )}

            {latestFollowup && (
              <div className="rounded-lg border border-default bg-surface-elevated px-4 py-3 space-y-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-secondary">Latest Response</p>
                  <p className="text-sm text-default mt-1">
                    {latestFollowup.answer || "No answer returned."}
                  </p>
                </div>

                {latestFollowup.memory_snippets.length > 0 && (
                  <div>
                    <p className="text-xs uppercase tracking-wide text-secondary mb-1">
                      Memory Snippets
                    </p>
                    <ul className="space-y-1">
                      {latestFollowup.memory_snippets.map((snippet, index) => (
                        <li key={`memory-${index}`} className="text-xs text-secondary">
                          {memorySnippetToText(snippet)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </McpUseProvider>
  );
};

export default FocusGroupResults;

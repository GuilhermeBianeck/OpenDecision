/** Context to decide about: text, a record with named fields, or a list of texts. */
export type StateValue = string | Record<string, unknown> | string[];

/** A candidate with optional text that separates it from its neighbours. Results key by label. */
export interface ChoiceOption {
  label: string;
  description?: string | null;
  not_for?: string | null;
  examples?: string[] | null;
}

/** A finite-choice request, evaluated locally by a resident HTTP server. */
export interface DecisionRequest {
  state: StateValue;
  question: string;
  choices: (string | ChoiceOption)[];
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  include_raw_scores?: boolean;
}

export interface DecisionResult {
  type: "choice";
  choice: string | null;
  /** Calibrated values when a profile is loaded; normalized scores otherwise. */
  probabilities: Record<string, number>;
  normalized_probabilities: Record<string, number>;
  calibrated_probabilities: Record<string, number> | null;
  /** Margin between the top two probabilities, not correctness probability. */
  confidence: number;
  top_probability: number;
  abstained: boolean;
  raw_scores: Record<string, number> | null;
  latency_ms: number;
  backend: string;
  model: string;
  metadata: Record<string, unknown>;
}

export interface BooleanRequest {
  state: StateValue;
  /** Read as a statement about the state. */
  question: string;
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  /** Abstain when the state neither supports nor contradicts the statement; NLI backends only. */
  unsupported_threshold?: number | null;
}

export interface BooleanResult {
  type: "boolean";
  value: boolean | null;
  /** Yes probability, even when abstained. */
  probability: number;
  /** p(state settles neither way) under statement scoring; null for binary choice. */
  unsupported: number | null;
  method: "statement" | "binary_choice";
  decision: DecisionResult;
}

/** Rate the state on ordered rubric levels, lowest first. */
export interface ScoreRequest {
  state: StateValue;
  question: string;
  levels: string[];
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  include_raw_scores?: boolean;
}

export interface ScoreResult {
  type: "score";
  /** Probability-weighted level index; may fall between two levels. */
  score: number;
  /** Most probable level index, or null when abstained. */
  level: number | null;
  /** Keyed by level index as a string, in rubric order. */
  probabilities: Record<string, number>;
  legend: Record<string, string>;
  confidence: number;
  abstained: boolean;
  decision: DecisionResult;
}

export interface ChoiceQuestion {
  type: "choice";
  question: string;
  choices: (string | ChoiceOption)[];
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  include_raw_scores?: boolean;
}

export interface BooleanQuestion {
  type: "boolean";
  statement: string;
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  unsupported_threshold?: number | null;
}

export interface ScoreQuestion {
  type: "score";
  question: string;
  levels: string[];
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  include_raw_scores?: boolean;
}

export type Question = ChoiceQuestion | BooleanQuestion | ScoreQuestion;

/** Independent typed questions about one state, keyed by your own ids (at most 128). */
export interface AskRequest {
  state: StateValue;
  questions: Record<string, Question>;
}

/** Discriminated on `type`, matching the question that produced it. */
export type Answer = DecisionResult | BooleanResult | ScoreResult;

export interface RankedChoice {
  choice: string;
  probability: number;
  raw_score: number | null;
}

export interface RankingResult {
  type: "ranking";
  ranking: RankedChoice[];
  decision: DecisionResult;
}

export interface MultiLabelRequest {
  state: StateValue;
  labels: string[];
  abstain_threshold?: number | null;
  margin_threshold?: number | null;
  unsupported_threshold?: number | null;
}

export interface ModelInfo {
  name: string;
  model_id: string;
  revision: string;
  license: string;
  family: string;
  parameters: number | string | null;
  weights_bytes: number | null;
  description: string;
}

export interface HealthResult {
  status: "ok";
  model: string;
  inference: "local";
  telemetry: false;
  /** Whether booleans are scored as statements (entailment vs contradiction). */
  supports_statements: boolean;
}

export interface ClientOptions {
  /** Defaults to the loopback server. No API key is needed. */
  baseUrl?: string;
  /** Per-request timeout, including reading the response body. Default 30 s. */
  timeoutMs?: number;
  /** Optional fetch implementation for embedding and testing. */
  fetch?: typeof globalThis.fetch;
}

export class OpenDecisionHTTPError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(status: number, body: unknown) {
    super(`OpenDecision HTTP request failed (${status})`);
    this.name = "OpenDecisionHTTPError";
    this.status = status;
    this.body = body;
  }
}

export class OpenDecisionTimeoutError extends Error {
  constructor(readonly timeoutMs: number) {
    super(`OpenDecision request timed out after ${timeoutMs} ms`);
    this.name = "OpenDecisionTimeoutError";
  }
}

/** Dependency-free, typed HTTP client. This does not load models in the browser. */
export class OpenDecision {
  readonly baseUrl: string;
  readonly timeoutMs: number;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8042").replace(/\/+$/, "");
    const url = new URL(this.baseUrl);
    if (!["http:", "https:"].includes(url.protocol)) {
      throw new TypeError("baseUrl must be an HTTP or HTTPS URL");
    }
    this.timeoutMs = options.timeoutMs ?? 30_000;
    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs <= 0) {
      throw new TypeError("timeoutMs must be a finite positive number");
    }
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(path: string, body?: unknown): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? {Accept: "application/json"} : {
          Accept: "application/json", "Content-Type": "application/json"
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await response.text();
      let data: unknown;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        if (!response.ok) throw new OpenDecisionHTTPError(response.status, text);
        throw new Error("OpenDecision server returned invalid JSON");
      }
      if (!response.ok) throw new OpenDecisionHTTPError(response.status, data);
      return data as T;
    } catch (error) {
      if (controller.signal.aborted) throw new OpenDecisionTimeoutError(this.timeoutMs);
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  health(): Promise<HealthResult> {
    return this.request("/health");
  }
  models(): Promise<{models: ModelInfo[]; active_model: string}> {
    return this.request("/v1/models");
  }
  choose(request: DecisionRequest): Promise<DecisionResult> {
    return this.request("/v1/decide", request);
  }
  chooseBatch(requests: DecisionRequest[]): Promise<DecisionResult[]> {
    return this.request("/v1/decide/batch", {requests});
  }
  rank(request: DecisionRequest): Promise<RankingResult> {
    return this.request("/v1/rank", request);
  }
  boolean(request: BooleanRequest): Promise<BooleanResult> {
    return this.request("/v1/boolean", request);
  }
  score(request: ScoreRequest): Promise<ScoreResult> {
    return this.request("/v1/score", request);
  }
  ask(request: AskRequest): Promise<Record<string, Answer>> {
    return this.request("/v1/ask", request);
  }
  multiLabel(request: MultiLabelRequest): Promise<Record<string, BooleanResult>> {
    return this.request("/v1/multi-label", request);
  }
}

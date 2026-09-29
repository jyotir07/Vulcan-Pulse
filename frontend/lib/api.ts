// Shapes mirror backend/api/schemas.py.

export type Method = "UPI" | "CARD" | "NETBANKING";

export type Intervention =
  | {
      type: "issuer_degradation";
      issuer: string;
      method: Method | null;
      success_rate_delta: number;
      start_hour?: number;
      duration_minutes?: number | null;
    }
  | { type: "gateway_outage"; gateway: string; start_hour?: number; duration_minutes?: number | null }
  | {
      type: "traffic_change";
      segment: Method | null;
      volume_delta: number;
      start_hour?: number;
      duration_minutes?: number | null;
    }
  | {
      type: "method_shift";
      from: Method;
      to: Method;
      percentage: number;
      issuer?: string | null;
      start_hour?: number;
      duration_minutes?: number | null;
    }
  | {
      type: "routing_change";
      source_gateway: string;
      target_gateway: string;
      traffic_percentage: number;
      start_hour?: number;
      duration_minutes?: number | null;
    };

export interface Scenario {
  interventions: Intervention[];
  window_date?: string | null;
}

export interface Metrics {
  transactions: number;
  success_rate: number;
  failure_rate: number;
  timeout_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  gmv: number;
  successful_gmv: number;
  failed_gmv: number;
}

export interface Impact {
  transactions_affected: number;
  success_rate_delta_pp: number;
  timeout_rate_delta_pp: number;
  failures_delta: number;
  gmv_at_risk: number;
  avg_latency_delta_ms: number;
  p95_latency_delta_ms: number;
}

export type Interval = Record<
  | "success_rate_delta_pp"
  | "timeout_rate_delta_pp"
  | "failures_delta"
  | "gmv_at_risk"
  | "avg_latency_delta_ms"
  | "p95_latency_delta_ms",
  [number, number]
> & { level: number };

export interface SegmentImpact {
  dimension: string;
  segment: string;
  baseline_transactions: number;
  counterfactual_transactions: number;
  baseline_success_rate: number;
  counterfactual_success_rate: number;
  success_rate_delta_pp: number;
  gmv_at_risk: number;
}

export interface SegmentContribution {
  dimension: string;
  segment: string;
  success_rate_delta_pp: number;
  failures_delta: number;
  gmv_at_risk: number;
}

export interface FactorContribution {
  factor: string;
  success_rate_delta_pp: number;
  gmv_at_risk: number;
}

export interface Meta {
  synthetic: boolean;
  disclaimer: string;
  predictor: string;
  model_version: string;
  window_date: string;
  cached: boolean;
  elapsed_ms: number;
}

export interface TimelinePoint {
  minute: number;
  baseline: number;
  counterfactual: number;
  true_baseline: number | null;
  true_counterfactual: number | null;
}

export interface SimulateResponse {
  meta: Meta;
  baseline: Metrics;
  counterfactual: Metrics;
  impact: Impact;
  interval: Interval | null;
  most_affected: Record<string, SegmentImpact[]>;
  attribution: Record<string, SegmentContribution[]>;
  factors: FactorContribution[] | null;
  explanation: { headline: string; facts: string[] };
  ground_truth: { baseline: Metrics; counterfactual: Metrics; impact: Impact } | null;
  timeline: TimelinePoint[];
}

export interface Mitigation {
  name: string;
  action: Intervention;
  success_rate: number;
  success_rate_recovered_pp: number;
  gmv_at_risk: number;
  gmv_recovered: number;
  label: string;
}

export interface MitigateResponse {
  meta: Meta;
  report: {
    scenario_success_rate: number;
    scenario_gmv_at_risk: number;
    baseline_success_rate: number;
    candidates: Mitigation[];
    label: string;
  };
}

export interface Overview {
  synthetic: boolean;
  disclaimer: string;
  window_date: string;
  available_dates: string[];
  transactions: number;
  success_rate: number;
  gmv: number;
  gmv_per_hour: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  methods: { method: Method; share: number; success_rate: number }[];
  issuers: string[];
  gateways: string[];
  predictors: string[];
  default_predictor: string;
}

export interface EcosystemGraph {
  window_date: string;
  nodes: { id: string; kind: string; label: string; volume: number; success_rate: number }[];
  edges: { source: string; target: string; volume: number; success_rate: number }[];
}

export type ParseResponse =
  | { status: "parsed"; scenario: Scenario; provider: string }
  | { status: "clarify"; message: string }
  | { status: "unavailable"; message: string };

export class ApiError extends Error {}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail =
        typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch {
      // Not JSON: keep the status line.
    }
    throw new ApiError(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  overview: (date?: string) => request<Overview>(`/overview${date ? `?date=${date}` : ""}`),
  graph: (date?: string) => request<EcosystemGraph>(`/ecosystem/graph${date ? `?date=${date}` : ""}`),
  simulate: (scenario: Scenario, predictor: string, factors = true) =>
    request<SimulateResponse>("/simulate", { scenario, predictor, factors, ground_truth: true }),
  mitigate: (scenario: Scenario, predictor: string) =>
    request<MitigateResponse>("/mitigate", { scenario, predictor, top: 6 }),
  parse: (text: string) => request<ParseResponse>("/parse", { text }),
};

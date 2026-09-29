"use client";

import type { Intervention, Method, Scenario } from "@/lib/api";
import { Card } from "./Card";

const METHODS: Method[] = ["UPI", "CARD", "NETBANKING"];

export const PRESETS: { label: string; interventions: Intervention[] }[] = [
  {
    label: "HDFC UPI degradation (−15 pp, evening)",
    interventions: [
      {
        type: "issuer_degradation",
        issuer: "HDFC",
        method: "UPI",
        success_rate_delta: -0.15,
        start_hour: 18,
        duration_minutes: 120,
      },
    ],
  },
  {
    label: "gateway_a outage (20 min at 19:00)",
    interventions: [
      { type: "gateway_outage", gateway: "gateway_a", start_hour: 19, duration_minutes: 20 },
    ],
  },
  {
    label: "Festival surge (+30% traffic, evening)",
    interventions: [
      { type: "traffic_change", segment: null, volume_delta: 0.3, start_hour: 18, duration_minutes: 240 },
    ],
  },
  {
    label: "20% of card payments move to UPI",
    interventions: [{ type: "method_shift", from: "CARD", to: "UPI", percentage: 0.2 }],
  },
  {
    label: "Reroute 25% of gateway_a to gateway_b",
    interventions: [
      { type: "routing_change", source_gateway: "gateway_a", target_gateway: "gateway_b", traffic_percentage: 0.25 },
    ],
  },
];

const TYPES: Intervention["type"][] = [
  "issuer_degradation",
  "gateway_outage",
  "traffic_change",
  "method_shift",
  "routing_change",
];

function defaults(type: Intervention["type"], issuers: string[], gateways: string[]): Intervention {
  switch (type) {
    case "issuer_degradation":
      return { type, issuer: issuers[0], method: "UPI", success_rate_delta: -0.15, start_hour: 18, duration_minutes: 120 };
    case "gateway_outage":
      return { type, gateway: gateways[0], start_hour: 19, duration_minutes: 30 };
    case "traffic_change":
      return { type, segment: null, volume_delta: 0.3, start_hour: 18, duration_minutes: 240 };
    case "method_shift":
      return { type, from: "CARD", to: "UPI", percentage: 0.2 };
    case "routing_change":
      return { type, source_gateway: gateways[0], target_gateway: gateways[1], traffic_percentage: 0.25 };
  }
}

const input = "rounded-md border border-slate-300 bg-white px-2 py-1 text-sm";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-slate-600">
      {label}
      {children}
    </label>
  );
}

function Percent({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
}) {
  return (
    <Field label={`${label}: ${Math.round(value * 100)}%`}>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={Math.round(value * 100)}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
      />
    </Field>
  );
}

function InterventionEditor({
  iv,
  issuers,
  gateways,
  onChange,
}: {
  iv: Intervention;
  issuers: string[];
  gateways: string[];
  onChange: (iv: Intervention) => void;
}) {
  const set = (patch: Partial<Intervention>) => onChange({ ...iv, ...patch } as Intervention);
  const timing = (
    <>
      <Field label="Start hour">
        <input
          className={input}
          type="number"
          min={0}
          max={23}
          value={iv.start_hour ?? 0}
          onChange={(e) => set({ start_hour: Number(e.target.value) })}
        />
      </Field>
      <Field label="Duration (min, empty = rest of day)">
        <input
          className={input}
          type="number"
          min={1}
          value={iv.duration_minutes ?? ""}
          onChange={(e) =>
            set({ duration_minutes: e.target.value === "" ? null : Number(e.target.value) })
          }
        />
      </Field>
    </>
  );

  switch (iv.type) {
    case "issuer_degradation":
      return (
        <>
          <Field label="Issuer">
            <select className={input} value={iv.issuer} onChange={(e) => set({ issuer: e.target.value })}>
              {issuers.map((i) => (
                <option key={i}>{i}</option>
              ))}
            </select>
          </Field>
          <Field label="Method">
            <select
              className={input}
              value={iv.method ?? ""}
              onChange={(e) => set({ method: (e.target.value || null) as Method | null })}
            >
              <option value="">All methods</option>
              {METHODS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Percent
            label="Success-rate drop (pp)"
            value={-iv.success_rate_delta}
            min={1}
            max={60}
            onChange={(v) => set({ success_rate_delta: -v })}
          />
          {timing}
        </>
      );
    case "gateway_outage":
      return (
        <>
          <Field label="Gateway">
            <select className={input} value={iv.gateway} onChange={(e) => set({ gateway: e.target.value })}>
              {gateways.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </select>
          </Field>
          {timing}
        </>
      );
    case "traffic_change":
      return (
        <>
          <Field label="Segment">
            <select
              className={input}
              value={iv.segment ?? ""}
              onChange={(e) => set({ segment: (e.target.value || null) as Method | null })}
            >
              <option value="">All traffic</option>
              {METHODS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Percent
            label="Volume change"
            value={iv.volume_delta}
            min={-90}
            max={200}
            onChange={(v) => set({ volume_delta: v })}
          />
          {timing}
        </>
      );
    case "method_shift":
      return (
        <>
          <Field label="From">
            <select className={input} value={iv.from} onChange={(e) => set({ from: e.target.value as Method })}>
              {METHODS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Field label="To">
            <select className={input} value={iv.to} onChange={(e) => set({ to: e.target.value as Method })}>
              {METHODS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Percent label="Share moved" value={iv.percentage} min={1} max={100} onChange={(v) => set({ percentage: v })} />
          {timing}
        </>
      );
    case "routing_change":
      return (
        <>
          <Field label="From gateway">
            <select
              className={input}
              value={iv.source_gateway}
              onChange={(e) => set({ source_gateway: e.target.value })}
            >
              {gateways.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </select>
          </Field>
          <Field label="To gateway">
            <select
              className={input}
              value={iv.target_gateway}
              onChange={(e) => set({ target_gateway: e.target.value })}
            >
              {gateways.map((g) => (
                <option key={g}>{g}</option>
              ))}
            </select>
          </Field>
          <Percent
            label="Share rerouted"
            value={iv.traffic_percentage}
            min={1}
            max={100}
            onChange={(v) => set({ traffic_percentage: v })}
          />
          {timing}
        </>
      );
  }
}

export function ScenarioBuilder({
  scenario,
  onChange,
  issuers,
  gateways,
  dates,
  predictors,
  predictor,
  onPredictor,
  onSimulate,
  busy,
}: {
  scenario: Scenario;
  onChange: (s: Scenario) => void;
  issuers: string[];
  gateways: string[];
  dates: string[];
  predictors: string[];
  predictor: string;
  onPredictor: (p: string) => void;
  onSimulate: () => void;
  busy: boolean;
}) {
  const update = (i: number, iv: Intervention) =>
    onChange({ ...scenario, interventions: scenario.interventions.map((x, j) => (j === i ? iv : x)) });

  return (
    <Card title="What-if builder" subtitle="Combine interventions; they apply together to one day">
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <Field label="Preset">
          <select
            className={input}
            value=""
            onChange={(e) => {
              const preset = PRESETS[Number(e.target.value)];
              if (preset) onChange({ ...scenario, interventions: preset.interventions });
            }}
          >
            <option value="">Choose a preset…</option>
            {PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Day (held-out test period)">
          <select
            className={input}
            value={scenario.window_date ?? ""}
            onChange={(e) => onChange({ ...scenario, window_date: e.target.value || null })}
          >
            <option value="">Default</option>
            {dates.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </Field>
        <Field label="Predictor">
          <select className={input} value={predictor} onChange={(e) => onPredictor(e.target.value)}>
            {predictors.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        </Field>
      </div>

      <div className="space-y-3">
        {scenario.interventions.map((iv, i) => (
          <div key={i} className="rounded-lg border border-slate-200 p-3">
            <div className="mb-2 flex items-center justify-between">
              <select
                className={`${input} font-medium`}
                value={iv.type}
                onChange={(e) =>
                  update(i, defaults(e.target.value as Intervention["type"], issuers, gateways))
                }
              >
                {TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.replace("_", " ")}
                  </option>
                ))}
              </select>
              {scenario.interventions.length > 1 && (
                <button
                  className="text-xs text-slate-500 hover:text-rose-600"
                  onClick={() =>
                    onChange({ ...scenario, interventions: scenario.interventions.filter((_, j) => j !== i) })
                  }
                >
                  Remove
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              <InterventionEditor iv={iv} issuers={issuers} gateways={gateways} onChange={(x) => update(i, x)} />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-50"
          onClick={() =>
            onChange({
              ...scenario,
              interventions: [...scenario.interventions, defaults("traffic_change", issuers, gateways)],
            })
          }
        >
          + Add intervention
        </button>
        <button
          className="rounded-md bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
          disabled={busy}
          onClick={onSimulate}
        >
          {busy ? "Simulating…" : "Simulate"}
        </button>
      </div>
    </Card>
  );
}

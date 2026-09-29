"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulateResponse } from "@/lib/api";
import { count, pct, pp, rupees, signedCount } from "@/lib/format";
import { Card, Stat } from "./Card";

const DIMENSION_LABEL: Record<string, string> = {
  merchant_category: "Merchant category",
  issuer: "Issuer",
  gateway: "Gateway",
  payment_method: "Method",
  city: "City",
  hour_bucket: "Hour",
};

function interval(r: SimulateResponse, key: "success_rate_delta_pp" | "gmv_at_risk", fmt: (v: number) => string) {
  if (!r.interval) return undefined;
  const [low, high] = r.interval[key];
  return `${Math.round(r.interval.level * 100)}% interval ${fmt(low)} to ${fmt(high)}`;
}

function BeforeAfter({ r }: { r: SimulateResponse }) {
  const data = [
    {
      name: "Current",
      predicted: r.baseline.success_rate * 100,
      truth: r.ground_truth ? r.ground_truth.baseline.success_rate * 100 : undefined,
    },
    {
      name: "Scenario",
      predicted: r.counterfactual.success_rate * 100,
      truth: r.ground_truth ? r.ground_truth.counterfactual.success_rate * 100 : undefined,
    },
  ];
  const values = data.flatMap((d) => [d.predicted, d.truth ?? d.predicted]);
  const low = Math.floor(Math.min(...values) - 1);
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis domain={[low, 100]} tickFormatter={(v) => `${v}%`} width={48} />
          <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="predicted" name={r.meta.predictor} fill="#6366f1" />
          {r.ground_truth && <Bar dataKey="truth" name="ground truth" fill="#94a3b8" />}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Waterfall({ r }: { r: SimulateResponse }) {
  const top = r.attribution.merchant_category ?? [];
  const rest = r.impact.gmv_at_risk - top.reduce((s, c) => s + c.gmv_at_risk, 0);
  const steps = [...top.map((c) => ({ name: c.segment, value: c.gmv_at_risk }))];
  if (Math.abs(rest) > 1) steps.push({ name: "other", value: rest });
  let running = 0;
  const data = steps.map((s) => {
    const base = s.value >= 0 ? running : running + s.value;
    running += s.value;
    return { name: s.name, base, value: Math.abs(s.value), sign: s.value >= 0 };
  });
  data.push({ name: "total", base: 0, value: Math.abs(r.impact.gmv_at_risk), sign: r.impact.gmv_at_risk >= 0 });
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" fontSize={11} interval={0} />
          <YAxis tickFormatter={(v) => rupees(Number(v))} width={72} fontSize={11} />
          <Tooltip formatter={(v, name) => (name === "base" ? null : rupees(Number(v)))} />
          <Bar dataKey="base" stackId="w" fill="transparent" />
          <Bar dataKey="value" stackId="w">
            {data.map((d) => (
              <Cell key={d.name} fill={d.name === "total" ? "#334155" : d.sign ? "#f43f5e" : "#10b981"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ImpactPanel({ result }: { result: SimulateResponse }) {
  const r = result;
  const bad = r.impact.success_rate_delta_pp < 0;
  return (
    <Card
      title="Predicted impact"
      subtitle={`${r.meta.predictor} on ${r.meta.window_date}${r.meta.cached ? " · cached" : ""} · ${(r.meta.elapsed_ms / 1000).toFixed(1)} s`}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat
          label="Success rate change"
          value={pp(r.impact.success_rate_delta_pp)}
          detail={interval(r, "success_rate_delta_pp", (v) => v.toFixed(2))}
          tone={bad ? "bad" : "good"}
        />
        <Stat
          label="GMV at risk"
          value={rupees(r.impact.gmv_at_risk)}
          detail={interval(r, "gmv_at_risk", rupees)}
          tone={r.impact.gmv_at_risk > 0 ? "bad" : "good"}
        />
        <Stat label="Extra failures" value={signedCount(r.impact.failures_delta)} />
        <Stat
          label="Payments affected"
          value={count(r.impact.transactions_affected)}
          detail={`of ${count(r.counterfactual.transactions)} in the day`}
        />
      </div>

      <div className="mt-4 rounded-lg bg-indigo-50 p-4 text-sm text-indigo-950">
        <div className="font-medium">{r.explanation.headline}</div>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {r.explanation.facts.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      </div>

      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <div>
          <h3 className="mb-1 text-sm font-medium text-slate-700">Success rate, current vs scenario</h3>
          <BeforeAfter r={r} />
        </div>
        <div>
          <h3 className="mb-1 text-sm font-medium text-slate-700">GMV at risk by merchant category</h3>
          <Waterfall r={r} />
        </div>
      </div>

      {r.factors && (
        <div className="mt-4">
          <h3 className="mb-2 text-sm font-medium text-slate-700">
            Where the impact comes from (Shapley split, as the {r.meta.predictor} sees it)
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {r.factors.map((f) => (
              <Stat
                key={f.factor}
                label={f.factor.replace("_", " ")}
                value={rupees(f.gmv_at_risk)}
                detail={pp(f.success_rate_delta_pp)}
              />
            ))}
          </div>
        </div>
      )}

      <div className="mt-4">
        <h3 className="mb-2 text-sm font-medium text-slate-700">Most affected segments</h3>
        <div className="grid gap-3 md:grid-cols-3">
          {["merchant_category", "issuer", "gateway"].map((d) => (
            <div key={d} className="rounded-lg border border-slate-100 p-3 text-sm">
              <div className="mb-1 text-xs font-medium uppercase text-slate-500">{DIMENSION_LABEL[d]}</div>
              {(r.most_affected[d] ?? []).length === 0 && <div className="text-slate-400">none above 0.5 pp</div>}
              {(r.most_affected[d] ?? []).map((s) => (
                <div key={s.segment} className="flex justify-between gap-2 tabular-nums">
                  <span>{s.segment}</span>
                  <span className="text-slate-500">
                    {pct(s.baseline_success_rate, 1)} → {pct(s.counterfactual_success_rate, 1)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

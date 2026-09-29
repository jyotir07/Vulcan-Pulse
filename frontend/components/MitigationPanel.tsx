"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MitigateResponse, Mitigation, SimulateResponse } from "@/lib/api";
import { pct, pp, rupees } from "@/lib/format";
import { Card, ErrorNote } from "./Card";

function Comparison({
  scenario,
  mitigated,
}: {
  scenario: SimulateResponse;
  mitigated: SimulateResponse;
}) {
  const rows = [
    {
      label: mitigated.meta.predictor,
      before: scenario.counterfactual.success_rate,
      after: mitigated.counterfactual.success_rate,
    },
  ];
  if (scenario.ground_truth && mitigated.ground_truth) {
    rows.push({
      label: "ground truth",
      before: scenario.ground_truth.counterfactual.success_rate,
      after: mitigated.ground_truth.counterfactual.success_rate,
    });
  }
  return (
    <table className="mt-2 w-full tabular-nums">
      <thead>
        <tr className="text-left text-xs uppercase text-slate-500">
          <th />
          <th className="text-right">No action</th>
          <th className="text-right">Mitigated</th>
          <th className="text-right">Recovered</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <td>{row.label}</td>
            <td className="text-right">{pct(row.before)}</td>
            <td className="text-right">{pct(row.after)}</td>
            <td className="text-right">{pp(100 * (row.after - row.before))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function MitigationPanel({
  report,
  busy,
  error,
  onSearch,
  onApply,
  applied,
  mitigated,
  scenarioResult,
}: {
  report: MitigateResponse | null;
  busy: boolean;
  error: string | null;
  onSearch: () => void;
  onApply: (m: Mitigation) => void;
  applied: Mitigation | null;
  mitigated: SimulateResponse | null;
  scenarioResult: SimulateResponse;
}) {
  const r = report?.report;
  const data = r
    ? [
        { name: "Current", rate: r.baseline_success_rate * 100 },
        { name: "No action", rate: r.scenario_success_rate * 100 },
        ...r.candidates.map((c, i) => ({ name: `Option ${i + 1}`, rate: c.success_rate * 100 })),
      ]
    : [];
  const low = data.length ? Math.floor(Math.min(...data.map((d) => d.rate)) - 0.5) : 0;

  return (
    <Card
      title="Mitigation"
      subtitle="Candidate reroutes and method steering, each re-simulated. Simulation output, not a guaranteed real-world result"
      action={
        <button
          className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-white hover:bg-slate-900 disabled:opacity-50"
          onClick={onSearch}
          disabled={busy}
        >
          {busy ? "Working…" : report ? "Search again" : "Find mitigations"}
        </button>
      }
    >
      {error && <ErrorNote message={error} />}
      {r && (
        <div className="grid gap-6 md:grid-cols-2">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis
                  domain={[low, "auto"]}
                  tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                  width={52}
                />
                <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                <Bar dataKey="rate" name="success rate">
                  {data.map((d, i) => (
                    <Cell
                      key={d.name}
                      fill={i === 0 ? "#94a3b8" : i === 1 ? "#f43f5e" : "#10b981"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <ol className="space-y-2 text-sm">
            {r.candidates.map((c, i) => (
              <li
                key={c.name}
                className="flex items-center justify-between gap-3 rounded-lg border border-slate-100 p-2"
              >
                <div>
                  <div className="font-medium">
                    Option {i + 1}: {c.name}
                  </div>
                  <div className="text-xs text-slate-500">
                    {pp(c.success_rate_recovered_pp)} · {rupees(c.gmv_recovered)} recovered
                  </div>
                </div>
                <button
                  className="rounded-md border border-slate-300 px-2 py-1 text-xs hover:bg-slate-50 disabled:opacity-50"
                  onClick={() => onApply(c)}
                  disabled={busy}
                >
                  Apply
                </button>
              </li>
            ))}
            {r.candidates.length === 0 && (
              <li className="text-slate-500">No candidate recovers anything.</li>
            )}
          </ol>
        </div>
      )}
      {applied && mitigated && (
        <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm">
          <div className="font-medium">Simulated again with: {applied.name}</div>
          <Comparison scenario={scenarioResult} mitigated={mitigated} />
        </div>
      )}
    </Card>
  );
}

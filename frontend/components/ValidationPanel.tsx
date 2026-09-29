import type { Impact, SimulateResponse } from "@/lib/api";
import { ms, pp, rupees, signedCount } from "@/lib/format";
import { Card } from "./Card";

type ImpactField = Exclude<keyof Impact, "transactions_affected" | "avg_latency_delta_ms">;

const ROWS: { key: ImpactField; label: string; fmt: (v: number) => string }[] = [
  { key: "success_rate_delta_pp", label: "Success rate change", fmt: (v) => pp(v) },
  { key: "timeout_rate_delta_pp", label: "Timeout rate change", fmt: (v) => pp(v) },
  { key: "failures_delta", label: "Extra failures", fmt: signedCount },
  { key: "gmv_at_risk", label: "GMV at risk", fmt: rupees },
  { key: "p95_latency_delta_ms", label: "P95 latency change", fmt: ms },
];

export function ValidationPanel({ result }: { result: SimulateResponse }) {
  const truth = result.ground_truth;
  if (!truth) return null;
  return (
    <Card
      title="Validation against ground truth"
      subtitle="The synthetic world's true process, re-run on the same payments with the same random draws"
    >
      <table className="w-full text-sm tabular-nums">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase text-slate-500">
            <th className="py-2">Impact</th>
            <th className="py-2 text-right">{result.meta.predictor}</th>
            <th className="py-2 text-right">Ground truth</th>
            <th className="py-2 text-right">Error</th>
            {result.interval && <th className="py-2 text-right">Truth in interval?</th>}
          </tr>
        </thead>
        <tbody>
          {ROWS.map(({ key, label, fmt }) => {
            const predicted = result.impact[key];
            const actual = truth.impact[key];
            const band = result.interval?.[key];
            const inside = band ? actual >= band[0] && actual <= band[1] : undefined;
            return (
              <tr key={key} className="border-b border-slate-100">
                <td className="py-2">{label}</td>
                <td className="py-2 text-right">{fmt(predicted)}</td>
                <td className="py-2 text-right">{fmt(actual)}</td>
                <td className="py-2 text-right text-slate-500">{fmt(predicted - actual)}</td>
                {result.interval && (
                  <td className={`py-2 text-right ${inside ? "text-emerald-600" : "text-rose-600"}`}>
                    {inside ? "yes" : "no"}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

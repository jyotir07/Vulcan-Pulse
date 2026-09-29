import type { Overview } from "@/lib/api";
import { count, ms, pct, rupees } from "@/lib/format";
import { Card, Stat } from "./Card";

export function OverviewTiles({ overview }: { overview: Overview }) {
  return (
    <Card
      title="Payment ecosystem health"
      subtitle={`Observed first attempts on ${overview.window_date}, a held-out day no model trained on`}
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Success rate" value={pct(overview.success_rate)} />
        <Stat label="GMV / hour" value={rupees(overview.gmv_per_hour)} detail={`${rupees(overview.gmv)} / day`} />
        <Stat label="Transactions" value={count(overview.transactions)} detail="first attempts / day" />
        <Stat label="Avg latency" value={ms(overview.avg_latency_ms)} detail={`P95 ${ms(overview.p95_latency_ms)}`} />
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        {overview.methods.map((m) => (
          <div key={m.method} className="rounded-lg border border-slate-100 p-3 text-sm">
            <div className="font-medium">{m.method}</div>
            <div className="text-slate-500">
              {pct(m.share, 0)} of volume · {pct(m.success_rate)} success
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

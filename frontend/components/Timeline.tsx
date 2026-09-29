"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimulateResponse } from "@/lib/api";
import { clock } from "@/lib/format";
import { Card } from "./Card";

export function Timeline({ result }: { result: SimulateResponse }) {
  const data = result.timeline.map((p) => ({
    minute: p.minute,
    baseline: p.baseline * 100,
    scenario: p.counterfactual * 100,
    truth: p.true_counterfactual === null ? undefined : p.true_counterfactual * 100,
  }));
  return (
    <Card
      title="Timeline"
      subtitle="Success rate per 5-minute bucket through the day. Realized truth is noisy: a bucket holds about a hundred payments"
    >
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="minute"
              type="number"
              domain={[0, 1440]}
              tickFormatter={clock}
              ticks={[0, 240, 480, 720, 960, 1200, 1440]}
            />
            <YAxis
              domain={[(min: number) => Math.floor(min / 5) * 5, 100]}
              allowDataOverflow
              tickFormatter={(v) => `${Math.round(v)}%`}
              width={44}
            />
            <Tooltip
              labelFormatter={(m) => clock(Number(m))}
              formatter={(v) => `${Number(v).toFixed(1)}%`}
            />
            <Legend />
            <Line dataKey="baseline" name="predicted, current" stroke="#94a3b8" dot={false} strokeWidth={1.5} />
            <Line dataKey="scenario" name="predicted, scenario" stroke="#6366f1" dot={false} strokeWidth={2} />
            {result.ground_truth && (
              <Line
                dataKey="truth"
                name="true, scenario"
                stroke="#f43f5e"
                dot={false}
                strokeWidth={1}
                strokeOpacity={0.7}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

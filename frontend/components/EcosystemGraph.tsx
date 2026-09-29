"use client";

import { ResponsiveContainer, Sankey, Tooltip } from "recharts";
import type { EcosystemGraph as Graph } from "@/lib/api";
import { count, pct } from "@/lib/format";
import { Card } from "./Card";

const KIND_COLOR: Record<string, string> = {
  merchant_category: "#6366f1",
  method: "#0ea5e9",
  issuer: "#f59e0b",
  gateway: "#10b981",
};

interface NodeShape {
  x: number;
  y: number;
  width: number;
  height: number;
  payload: { name: string; kind: string; success_rate: number; value: number };
}

function GraphNode({ x, y, width, height, payload }: NodeShape) {
  const right = payload.kind === "gateway";
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={KIND_COLOR[payload.kind]} rx={2} />
      {height > 9 && (
        <text
          x={right ? x - 6 : x + width + 6}
          y={y + height / 2}
          textAnchor={right ? "end" : "start"}
          dominantBaseline="middle"
          fontSize={11}
          fill="#334155"
        >
          {payload.name} · {pct(payload.success_rate, 1)}
        </text>
      )}
    </g>
  );
}

export function EcosystemGraph({ graph }: { graph: Graph }) {
  const index = new Map(graph.nodes.map((n, i) => [n.id, i]));
  const data = {
    nodes: graph.nodes.map((n) => ({ name: n.label, kind: n.kind, success_rate: n.success_rate })),
    links: graph.edges.map((e) => ({
      source: index.get(e.source)!,
      target: index.get(e.target)!,
      value: e.volume,
      success_rate: e.success_rate,
    })),
  };
  return (
    <Card
      title="Ecosystem map"
      subtitle="Merchant category → payment method → issuer → gateway, sized by volume, labelled with success rate"
    >
      <div className="mb-2 flex gap-4 text-xs text-slate-500">
        {Object.entries(KIND_COLOR).map(([kind, color]) => (
          <span key={kind} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: color }} />
            {kind.replace("_", " ")}
          </span>
        ))}
      </div>
      <div className="h-[520px]">
        <ResponsiveContainer width="100%" height="100%">
          <Sankey
            data={data}
            node={GraphNode as never}
            nodePadding={8}
            nodeWidth={10}
            link={{ stroke: "#94a3b8", strokeOpacity: 0.25 }}
            margin={{ top: 8, right: 150, bottom: 8, left: 8 }}
          >
            <Tooltip
              formatter={(value, _name, item) => {
                const rate = (item?.payload as { payload?: { success_rate?: number } })?.payload
                  ?.success_rate;
                return [
                  `${count(Number(value))} payments${rate !== undefined ? ` · ${pct(rate, 1)} success` : ""}`,
                  "",
                ];
              }}
            />
          </Sankey>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

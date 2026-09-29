"use client";

import { useState } from "react";
import { api, type ParseResponse, type Scenario } from "@/lib/api";
import { Card, ErrorNote } from "./Card";

export function NaturalLanguageInput({ onConfirm }: { onConfirm: (s: Scenario) => void }) {
  const [text, setText] = useState("What if HDFC UPI success rate drops by 15% for two hours at 6pm?");
  const [parsed, setParsed] = useState<ParseResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function parse() {
    setBusy(true);
    setError(null);
    setParsed(null);
    try {
      setParsed(await api.parse(text));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="Ask a what-if question"
      subtitle="An LLM turns the question into a structured scenario for you to confirm. It never predicts: every number comes from the engine"
    >
      <div className="flex gap-2">
        <input
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && parse()}
        />
        <button
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
          onClick={parse}
          disabled={busy || !text.trim()}
        >
          {busy ? "Parsing…" : "Parse"}
        </button>
      </div>
      {error && <div className="mt-3"><ErrorNote message={error} /></div>}
      {parsed?.status === "parsed" && (
        <div className="mt-3 rounded-lg border border-slate-200 p-3 text-sm">
          <div className="mb-2 text-slate-600">Parsed scenario ({parsed.provider}). Is this what you meant?</div>
          <pre className="overflow-x-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
            {JSON.stringify(parsed.scenario.interventions, null, 2)}
          </pre>
          <button
            className="mt-2 rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700"
            onClick={() => onConfirm(parsed.scenario)}
          >
            Yes, use this scenario
          </button>
        </div>
      )}
      {parsed && parsed.status !== "parsed" && (
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          {parsed.message}
        </div>
      )}
    </Card>
  );
}

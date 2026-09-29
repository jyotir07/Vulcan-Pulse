"use client";

import { useEffect, useState } from "react";
import {
  api,
  type EcosystemGraph as Graph,
  type MitigateResponse,
  type Mitigation,
  type Overview,
  type Scenario,
  type SimulateResponse,
} from "@/lib/api";
import { ErrorNote } from "./Card";
import { EcosystemGraph } from "./EcosystemGraph";
import { ImpactPanel } from "./ImpactPanel";
import { MitigationPanel } from "./MitigationPanel";
import { NaturalLanguageInput } from "./NaturalLanguageInput";
import { OverviewTiles } from "./OverviewTiles";
import { PRESETS, ScenarioBuilder } from "./ScenarioBuilder";
import { Timeline } from "./Timeline";
import { ValidationPanel } from "./ValidationPanel";

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function Dashboard() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [scenario, setScenario] = useState<Scenario>({ interventions: PRESETS[0].interventions });
  const [predictor, setPredictor] = useState("");
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  const [report, setReport] = useState<MitigateResponse | null>(null);
  const [applied, setApplied] = useState<Mitigation | null>(null);
  const [mitigated, setMitigated] = useState<SimulateResponse | null>(null);
  const [mitigating, setMitigating] = useState(false);
  const [mitError, setMitError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.overview(), api.graph()])
      .then(([o, g]) => {
        setOverview(o);
        setGraph(g);
        setPredictor(o.default_predictor);
      })
      .catch((e) => setLoadError(message(e)));
  }, []);

  async function simulate(next: Scenario = scenario) {
    setSimulating(true);
    setSimError(null);
    setReport(null);
    setApplied(null);
    setMitigated(null);
    try {
      setResult(await api.simulate(next, predictor));
    } catch (e) {
      setSimError(message(e));
    } finally {
      setSimulating(false);
    }
  }

  async function searchMitigations() {
    setMitigating(true);
    setMitError(null);
    try {
      setReport(await api.mitigate(scenario, predictor));
    } catch (e) {
      setMitError(message(e));
    } finally {
      setMitigating(false);
    }
  }

  async function apply(m: Mitigation) {
    setMitigating(true);
    setMitError(null);
    try {
      const withAction = { ...scenario, interventions: [...scenario.interventions, m.action] };
      setMitigated(await api.simulate(withAction, predictor, false));
      setApplied(m);
    } catch (e) {
      setMitError(message(e));
    } finally {
      setMitigating(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-5 px-4 py-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Vulcan Counterfactual</h1>
        <p className="text-slate-600">Ask what happens before you change the payment network.</p>
      </header>

      {loadError && <ErrorNote message={`Could not reach the API: ${loadError}`} />}
      {overview && <OverviewTiles overview={overview} />}
      {graph && <EcosystemGraph graph={graph} />}

      <NaturalLanguageInput
        onConfirm={(parsed) => {
          const next = { ...parsed, window_date: scenario.window_date };
          setScenario(next);
          simulate(next);
        }}
      />

      {overview && (
        <ScenarioBuilder
          scenario={scenario}
          onChange={setScenario}
          issuers={overview.issuers}
          gateways={overview.gateways}
          dates={overview.available_dates}
          predictors={overview.predictors}
          predictor={predictor}
          onPredictor={setPredictor}
          onSimulate={() => simulate()}
          busy={simulating}
        />
      )}

      {simError && <ErrorNote message={simError} />}
      {simulating && !result && (
        <div className="text-sm text-slate-500">
          Simulating… the first run on a day predicts the whole baseline, later runs reuse it.
        </div>
      )}
      {result && (
        <>
          <ImpactPanel result={result} />
          <ValidationPanel result={result} />
          <Timeline result={result} />
          <MitigationPanel
            report={report}
            busy={mitigating}
            error={mitError}
            onSearch={searchMitigations}
            onApply={apply}
            applied={applied}
            mitigated={mitigated}
            scenarioResult={result}
          />
        </>
      )}
    </div>
  );
}

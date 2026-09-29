"""Plain-language explanation of a simulated impact, built only from computed quantities.

Every sentence is a template filled from the scenario's transactions and the simulation's
numbers: traffic shares, where the at-risk GMV is concentrated, how much of the affected traffic
falls in peak hours, and how close gateways run to capacity. Nothing is generated freely, so the
explanation cannot name a cause the numbers do not show.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from backend.data.catalog import GATEWAYS, ISSUERS
from backend.data.generator import Dataset
from backend.services.attribution import SegmentContribution, top_contributors
from backend.simulation.engine import SimulationResult
from backend.simulation.interventions import Counterfactual
from backend.simulation.scenarios import (
    GatewayOutage,
    IssuerDegradation,
    MethodShift,
    RoutingChange,
    Scenario,
    TrafficChange,
)

# Load over capacity above which a gateway is described as near capacity.
NEAR_CAPACITY = 0.9
# The top segments are called concentrated when they hold at least this share of GMV at risk.
CONCENTRATION_SHARE = 0.5


class Explanation(BaseModel):
    headline: str
    facts: list[str]


def _rupees(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e7:
        return f"{sign}₹{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"{sign}₹{value / 1e5:,.2f} L"
    return f"{sign}₹{value:,.0f}"


def _span_text(iv) -> str:
    if iv.duration_minutes is None:
        return "from 00:00" if iv.start_hour == 0 else f"from {iv.start_hour:02d}:00"
    end = iv.start_hour * 60 + iv.duration_minutes
    return f"{iv.start_hour:02d}:00-{end // 60 % 24:02d}:{end % 60:02d}"


def _in_span(frame: pd.DataFrame, iv) -> np.ndarray:
    minute = frame["timestamp"].dt.hour.to_numpy() * 60 + frame["timestamp"].dt.minute.to_numpy()
    start = iv.start_hour * 60
    end = 24 * 60 if iv.duration_minutes is None else start + iv.duration_minutes
    return (minute >= start) & (minute < end)


def _intervention_fact(iv, cf: Counterfactual) -> str:
    base = cf.baseline
    span = _in_span(base, iv)
    if isinstance(iv, IssuerDegradation):
        issuer_id = [i.name for i in ISSUERS].index(iv.issuer)
        pool = span & (
            (base["payment_method"] == iv.method).to_numpy() if iv.method else np.ones_like(span)
        )
        hit = pool & (base["issuer_id"] == issuer_id).to_numpy()
        scope = f"{iv.method} " if iv.method else ""
        return (
            f"{iv.issuer} carries {hit.sum() / max(pool.sum(), 1):.0%} of {scope}payments "
            f"{_span_text(iv)} ({hit.sum():,} payments), and the scenario lowers its success "
            f"rate by {-iv.success_rate_delta * 100:.0f} pp."
        )
    if isinstance(iv, GatewayOutage):
        gateway_id = [g.name for g in GATEWAYS].index(iv.gateway)
        hit = span & (base["gateway_id"] == gateway_id).to_numpy()
        return (
            f"{iv.gateway} carries {hit.sum() / max(span.sum(), 1):.0%} of payments "
            f"{_span_text(iv)} ({hit.sum():,} payments); all of them fail while it is down."
        )
    if isinstance(iv, TrafficChange):
        segment = f"{iv.segment} " if iv.segment else ""
        change = len(cf.counterfactual) - len(base)
        return (
            f"{segment.strip() or 'All'} traffic {_span_text(iv)} changes by "
            f"{iv.volume_delta:+.0%}: {change:+,} payments in the window."
        )
    if isinstance(iv, MethodShift):
        who = f"{iv.issuer} " if iv.issuer else ""
        return (
            f"{iv.percentage:.0%} of {who}{iv.from_method} payments {_span_text(iv)} move to "
            f"{iv.to_method}."
        )
    if isinstance(iv, RoutingChange):
        return (
            f"{iv.traffic_percentage:.0%} of eligible {iv.source_gateway} traffic "
            f"{_span_text(iv)} moves to {iv.target_gateway}."
        )
    raise TypeError(f"unknown intervention {type(iv).__name__}")


def explain(
    dataset: Dataset,
    scenario: Scenario,
    cf: Counterfactual,
    result: SimulationResult,
    contributions: list[SegmentContribution],
) -> Explanation:
    impact = result.impact
    headline = (
        f"Success rate {result.baseline.success_rate:.2%} → "
        f"{result.counterfactual.success_rate:.2%} ({impact.success_rate_delta_pp:+.2f} pp), "
        f"{_rupees(impact.gmv_at_risk)} GMV at risk, "
        f"{impact.transactions_affected:,} payments affected."
    )
    facts = [_intervention_fact(iv, cf) for iv in scenario.interventions]

    total_risk = sum(c.gmv_at_risk for c in contributions if c.dimension == "merchant_category")
    top = top_contributors(contributions, "merchant_category")
    if total_risk > 0 and top:
        share = sum(c.gmv_at_risk for c in top) / total_risk
        names = ", ".join(c.segment for c in top)
        qualifier = "concentrated in" if share >= CONCENTRATION_SHARE else "spread, led by"
        facts.append(f"GMV at risk is {qualifier} {names} ({share:.0%} of the total).")

    affected = cf.counterfactual[cf.affected]
    if len(affected):
        facts.append(
            f"{affected['is_peak'].mean():.0%} of affected payments fall in the evening peak "
            "(18:00-22:00)."
        )
        busiest = (
            affected.groupby("gateway_id")["gateway_utilization"].max().sort_values(ascending=False)
        )
        gateway_id, peak_util = int(busiest.index[0]), float(busiest.iloc[0])
        before = cf.baseline.loc[cf.baseline["gateway_id"] == gateway_id, "gateway_utilization"]
        before_peak = float(before.max()) if len(before) else 0.0
        state = "near or over capacity" if peak_util >= NEAR_CAPACITY else "within capacity"
        facts.append(
            f"The busiest gateway for affected payments, {GATEWAYS[gateway_id].name}, peaks at "
            f"{peak_util:.0%} of capacity (baseline {before_peak:.0%}), {state}."
        )

    return Explanation(headline=headline, facts=facts)

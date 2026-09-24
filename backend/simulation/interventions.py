"""Turn a scenario into a changed world, seen two ways.

The engine side only uses observable data: the transactions, their observed state features,
gateway capacity recovered from observed load and utilization, and the scenario itself. The
ground-truth side additionally gets the scenario translated into latent episodes. Both sides
share the same modified first-attempt context, so they evaluate exactly the same transactions.

Interventions that change *which* transactions exist or how they are routed (traffic, method
shift, routing) are applied first, in scenario order. Interventions that change the state of an
issuer or gateway (degradation, outage) are applied last, so a rerouted payment sees the state of
its new gateway.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.data.dgp import CONTEXT_COLUMNS, first_attempt_context
from backend.data.generator import Dataset
from backend.data.latent import UTILIZATION_BUCKET_MINUTES, minute_index
from backend.data.state import HealthTables, health_tables
from backend.data.traffic import MINUTES_PER_DAY, PEAK_HOURS
from backend.simulation.scenarios import (
    GatewayOutage,
    IssuerDegradation,
    MethodShift,
    RoutingChange,
    Scenario,
    TrafficChange,
)

# Minutes replayed before the window so ground-truth load, retries and health are warm.
WARMUP_MINUTES = 30

_ISSUER_ID = {i.name: n for n, i in enumerate(ISSUERS)}
_GATEWAY_ID = {g.name: n for n, g in enumerate(GATEWAYS)}
_SUPPORTS = np.array([[m in g.supported_methods for m in METHODS] for g in GATEWAYS])
_ROUTING_SHARE = np.array([g.routing_share for g in GATEWAYS])


@dataclass(frozen=True)
class Window:
    date: date
    start_minute: int  # minutes since the dataset start
    end_minute: int


@dataclass(frozen=True)
class ObservedEcosystem:
    """Everything the engine is allowed to know about the ecosystem."""

    start: np.datetime64
    n_minutes: int
    festival_dates: list[date]
    health: HealthTables
    load: np.ndarray  # [gateway, bucket] observed attempts, first attempts and retries
    capacity: np.ndarray  # [gateway], recovered as load / utilization


@dataclass(frozen=True)
class Counterfactual:
    window: Window
    baseline: pd.DataFrame  # window first attempts with engine-computed features
    counterfactual: pd.DataFrame  # same, after the scenario
    affected: np.ndarray  # [counterfactual row] context or any feature changed
    oracle_baseline_context: pd.DataFrame  # warm-up + window, for the ground truth
    oracle_counterfactual_context: pd.DataFrame
    extra_episodes: pd.DataFrame  # scenario as latent episodes, for the ground truth only


def observe(dataset: Dataset) -> ObservedEcosystem:
    txns = dataset.transactions
    start = np.datetime64(dataset.config.start_date, "ms")
    n_minutes = dataset.config.n_days * MINUTES_PER_DAY
    n_buckets = -(-n_minutes // UTILIZATION_BUCKET_MINUTES)

    gateway = txns["gateway_id"].to_numpy()
    bucket = minute_index(txns["timestamp"], start) // UTILIZATION_BUCKET_MINUTES
    load = np.bincount(gateway * n_buckets + bucket, minlength=len(GATEWAYS) * n_buckets).reshape(
        len(GATEWAYS), n_buckets
    )

    capacity = np.zeros(len(GATEWAYS))
    utilization = txns["gateway_utilization"].to_numpy()
    for g in range(len(GATEWAYS)):
        rows = np.flatnonzero(gateway == g)
        implied = load[g, bucket[rows]] / utilization[rows]
        if not np.allclose(implied, implied[0], rtol=1e-9):
            raise ValueError(f"inconsistent observed utilization for gateway {g}")
        capacity[g] = implied[0]

    return ObservedEcosystem(
        start=start,
        n_minutes=n_minutes,
        festival_dates=dataset.festival_dates,
        health=health_tables(txns, start, n_minutes),
        load=load,
        capacity=capacity,
    )


def default_window_date(dataset: Dataset) -> date:
    """A representative normal day: the non-festival weekday with the median success rate."""
    first = dataset.transactions[dataset.transactions["attempt"] == 1]
    day = first["timestamp"].dt.date
    success = (first["transaction_status"] == "SUCCESS").groupby(day).mean()
    normal = success[
        [d.weekday() < 5 and d not in dataset.festival_dates for d in success.index]
    ].sort_values(kind="stable")
    return normal.index[len(normal) // 2]


def resolve_window(dataset: Dataset, scenario: Scenario) -> Window:
    day = scenario.window_date or default_window_date(dataset)
    offset = (day - dataset.config.start_date).days
    if not 0 <= offset < dataset.config.n_days:
        last = dataset.config.start_date + timedelta(days=dataset.config.n_days - 1)
        raise ValueError(f"window_date {day} is outside {dataset.config.start_date}..{last}")
    start = offset * MINUTES_PER_DAY
    return Window(date=day, start_minute=start, end_minute=start + MINUTES_PER_DAY)


def _span(window: Window, intervention) -> tuple[int, int]:
    start = window.start_minute + intervention.start_hour * 60
    if intervention.duration_minutes is None:
        return start, window.end_minute
    return start, min(start + intervention.duration_minutes, window.end_minute)


def engine_features(
    context: pd.DataFrame, observed: ObservedEcosystem, baseline_context: pd.DataFrame
) -> pd.DataFrame:
    """Observed-state features for a (possibly modified) context, computed without the DGP.

    Health comes from the observed trailing tables at each row's (issuer, method, gateway,
    minute). Utilization is the observed load plus the change in first-attempt load against the
    baseline context, over the recovered capacity. For an unmodified context this reproduces the
    observed features exactly.
    """
    minute = minute_index(context["timestamp"], observed.start)
    method = context["payment_method"].astype(str).map(METHODS.index).to_numpy()
    issuer = context["issuer_id"].to_numpy()
    gateway = context["gateway_id"].to_numpy()
    bucket = minute // UTILIZATION_BUCKET_MINUTES
    n_buckets = observed.load.shape[1]

    def counts(frame: pd.DataFrame) -> np.ndarray:
        b = minute_index(frame["timestamp"], observed.start) // UTILIZATION_BUCKET_MINUTES
        flat = frame["gateway_id"].to_numpy() * n_buckets + b
        return np.bincount(flat, minlength=len(GATEWAYS) * n_buckets).reshape(
            len(GATEWAYS), n_buckets
        )

    load = observed.load + counts(context) - counts(baseline_context)
    hour = context["timestamp"].dt.hour
    return context.assign(
        issuer_health=observed.health.issuer[issuer, method, minute],
        gateway_health=observed.health.gateway[gateway, minute],
        gateway_utilization=load[gateway, bucket] / observed.capacity[gateway],
        hour=hour,
        is_peak=hour.isin(PEAK_HOURS),
        is_festival=context["timestamp"].dt.date.isin(observed.festival_dates),
    )


def _rows_in(context: pd.DataFrame, observed: ObservedEcosystem, span: tuple[int, int]):
    minute = minute_index(context["timestamp"], observed.start)
    return (minute >= span[0]) & (minute < span[1])


def _apply_traffic_change(ctx, iv: TrafficChange, span, observed, rng, next_id):
    eligible = _rows_in(ctx, observed, span)
    if iv.segment is not None:
        eligible &= (ctx["payment_method"] == iv.segment).to_numpy()
    rows = np.flatnonzero(eligible)
    n_change = int(round(abs(iv.volume_delta) * len(rows)))
    if iv.volume_delta < 0:
        drop = rng.choice(rows, size=n_change, replace=False)
        return ctx.drop(index=ctx.index[drop]).reset_index(drop=True), next_id
    # New demand looks like existing demand in the same segment and time span.
    extra = ctx.iloc[rng.choice(rows, size=n_change, replace=True)].copy()
    extra["transaction_id"] = np.arange(next_id, next_id + n_change)
    return pd.concat([ctx, extra], ignore_index=True), next_id + n_change


def _apply_method_shift(ctx, iv: MethodShift, span, observed, rng):
    rows = np.flatnonzero(
        _rows_in(ctx, observed, span) & (ctx["payment_method"] == iv.from_method).to_numpy()
    )
    chosen = rng.choice(rows, size=int(round(iv.percentage * len(rows))), replace=False)
    ctx = ctx.copy()
    to = METHODS.index(iv.to_method)
    ctx.loc[ctx.index[chosen], "payment_method"] = iv.to_method

    # A payment on a gateway that can't take the new method is routed by the usual shares.
    gateway = ctx["gateway_id"].to_numpy()
    unsupported = chosen[~_SUPPORTS[gateway[chosen], to]]
    if len(unsupported):
        weights = _ROUTING_SHARE * _SUPPORTS[:, to]
        ctx.loc[ctx.index[unsupported], "gateway_id"] = rng.choice(
            len(GATEWAYS), size=len(unsupported), p=weights / weights.sum()
        )
    return ctx


def _apply_routing_change(ctx, iv: RoutingChange, span, observed, rng):
    source, target = _GATEWAY_ID[iv.source_gateway], _GATEWAY_ID[iv.target_gateway]
    method = ctx["payment_method"].astype(str).map(METHODS.index).to_numpy()
    rows = np.flatnonzero(
        _rows_in(ctx, observed, span)
        & (ctx["gateway_id"] == source).to_numpy()
        & _SUPPORTS[target, method]
    )
    chosen = rng.choice(rows, size=int(round(iv.traffic_percentage * len(rows))), replace=False)
    ctx = ctx.copy()
    ctx.loc[ctx.index[chosen], "gateway_id"] = target
    return ctx


def _episode(kind: str, target_id: int, method, span, magnitude: float) -> dict:
    return {
        "kind": kind,
        "target_id": target_id,
        "method": method,
        "start_minute": span[0],
        "end_minute": span[1],
        "magnitude": magnitude,
    }


def apply_scenario(
    dataset: Dataset, scenario: Scenario, observed: ObservedEcosystem
) -> Counterfactual:
    window = resolve_window(dataset, scenario)
    all_context = first_attempt_context(dataset.transactions)
    minute = minute_index(all_context["timestamp"], observed.start)
    in_window = (minute >= window.start_minute) & (minute < window.end_minute)
    in_warmup = (minute >= window.start_minute - WARMUP_MINUTES) & (minute < window.start_minute)
    base_ctx = all_context[in_window].reset_index(drop=True)
    warmup_ctx = all_context[in_warmup]

    ctx = base_ctx
    next_id = int(all_context["transaction_id"].max()) + 1
    episodes = []
    state_changes = []
    for index, iv in enumerate(scenario.interventions):
        rng = np.random.default_rng([dataset.config.seed, index])
        span = _span(window, iv)
        if isinstance(iv, TrafficChange):
            ctx, next_id = _apply_traffic_change(ctx, iv, span, observed, rng, next_id)
        elif isinstance(iv, MethodShift):
            ctx = _apply_method_shift(ctx, iv, span, observed, rng)
        elif isinstance(iv, RoutingChange):
            ctx = _apply_routing_change(ctx, iv, span, observed, rng)
        elif isinstance(iv, IssuerDegradation):
            magnitude = -iv.success_rate_delta
            episodes.append(
                _episode("issuer_degradation", _ISSUER_ID[iv.issuer], iv.method, span, magnitude)
            )
            state_changes.append((iv, span))
        elif isinstance(iv, GatewayOutage):
            episodes.append(_episode("gateway_outage", _GATEWAY_ID[iv.gateway], None, span, 1.0))
            state_changes.append((iv, span))

    baseline = engine_features(base_ctx, observed, base_ctx)
    counterfactual = engine_features(ctx, observed, base_ctx)
    for iv, span in state_changes:
        rows = _rows_in(counterfactual, observed, span)
        if isinstance(iv, IssuerDegradation):
            rows &= (counterfactual["issuer_id"] == _ISSUER_ID[iv.issuer]).to_numpy()
            if iv.method is not None:
                rows &= (counterfactual["payment_method"] == iv.method).to_numpy()
            # The scenario states the new success rate directly, so the engine applies it from
            # the first minute rather than waiting for the trailing window to notice.
            health = counterfactual["issuer_health"].to_numpy().copy()
            health[rows] = np.clip(health[rows] + iv.success_rate_delta, 0.0, 1.0)
            counterfactual = counterfactual.assign(issuer_health=health)
        else:
            rows &= (counterfactual["gateway_id"] == _GATEWAY_ID[iv.gateway]).to_numpy()
            health = counterfactual["gateway_health"].to_numpy().copy()
            health[rows] = 0.0
            counterfactual = counterfactual.assign(gateway_health=health)

    return Counterfactual(
        window=window,
        baseline=baseline,
        counterfactual=counterfactual,
        affected=_affected(baseline, counterfactual),
        oracle_baseline_context=pd.concat([warmup_ctx, base_ctx], ignore_index=True),
        oracle_counterfactual_context=pd.concat(
            [warmup_ctx, ctx[list(CONTEXT_COLUMNS)]], ignore_index=True
        ),
        extra_episodes=pd.DataFrame(
            episodes,
            columns=["kind", "target_id", "method", "start_minute", "end_minute", "magnitude"],
        ),
    )


_COMPARED = (
    "payment_method",
    "gateway_id",
    "issuer_health",
    "gateway_health",
    "gateway_utilization",
)


def _affected(baseline: pd.DataFrame, counterfactual: pd.DataFrame) -> np.ndarray:
    """Counterfactual rows that are new or differ from their baseline row in any input."""
    before = baseline.set_index("transaction_id")
    ids = counterfactual["transaction_id"]
    existing = ids.isin(before.index).to_numpy()
    affected = ~existing
    matched = before.loc[ids[existing]]
    after = counterfactual[existing]
    changed = np.zeros(existing.sum(), dtype=bool)
    for column in _COMPARED:
        a = after[column].to_numpy()
        b = matched[column].to_numpy()
        if column in ("payment_method", "gateway_id"):
            changed |= a.astype(str) != b.astype(str)
        else:
            changed |= ~np.isclose(a.astype(float), b.astype(float), rtol=0.0, atol=1e-12)
    affected[existing] = changed
    return affected

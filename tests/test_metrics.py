import numpy as np
import pandas as pd
import pytest

from backend.data.dgp import FAILURE_REASONS
from backend.simulation.metrics import (
    IMPACT_FIELDS,
    Predictions,
    compute_impact,
    compute_metrics,
    ensemble_mean,
    impact_interval,
    segment_impacts,
)


def _preds(p, timeout_share=0.0, latency=1000.0, sigma=0.0) -> Predictions:
    p = np.asarray(p, dtype=float)
    reasons = np.zeros((len(p), len(FAILURE_REASONS)))
    reasons[:, FAILURE_REASONS.index("UPI_TIMEOUT")] = timeout_share
    reasons[:, FAILURE_REASONS.index("ISSUER_DECLINED")] = 1.0 - timeout_share
    return Predictions(
        p_success=p,
        reason_probs=reasons,
        latency_median_ms=np.full(len(p), latency),
        latency_log_sigma=np.full(len(p), sigma),
    )


def _frame(amounts, gateways=None) -> pd.DataFrame:
    n = len(amounts)
    return pd.DataFrame(
        {
            "amount": amounts,
            "gateway": gateways or ["gateway_a"] * n,
            "issuer": ["HDFC"] * n,
            "payment_method": ["UPI"] * n,
            "merchant_category": ["travel"] * n,
            "city": ["Pune"] * n,
        }
    )


def test_metrics_are_expectations():
    m = compute_metrics(_frame([100.0, 300.0]), _preds([1.0, 0.5], timeout_share=0.4))
    assert m.transactions == 2
    assert m.success_rate == pytest.approx(0.75)
    assert m.gmv == pytest.approx(400.0)
    assert m.successful_gmv == pytest.approx(250.0)
    assert m.failed_gmv == pytest.approx(150.0)
    assert m.timeout_rate == pytest.approx(0.5 * 0.4 / 2)
    assert m.avg_latency_ms == pytest.approx(1000.0)
    assert m.p95_latency_ms == pytest.approx(1000.0)


def test_lognormal_latency_mean_and_p95():
    m = compute_metrics(_frame([1.0] * 2000), _preds(np.ones(2000), latency=1000.0, sigma=0.5))
    assert m.avg_latency_ms == pytest.approx(1000.0 * np.exp(0.125))
    assert m.p95_latency_ms == pytest.approx(1000.0 * np.exp(0.5 * 1.6449), rel=0.02)


def test_pure_volume_change_puts_no_gmv_at_risk():
    base = compute_metrics(_frame([100.0, 200.0]), _preds([0.9, 0.9]))
    doubled = compute_metrics(_frame([100.0, 200.0] * 2), _preds([0.9] * 4))
    impact = compute_impact(base, doubled, affected=2)
    assert impact.gmv_at_risk == pytest.approx(0.0, abs=1e-9)
    assert impact.success_rate_delta_pp == pytest.approx(0.0)
    assert impact.failures_delta == pytest.approx(0.2)


def test_degradation_gmv_at_risk_is_extra_failed_gmv():
    frame = _frame([100.0, 200.0])
    impact = compute_impact(
        compute_metrics(frame, _preds([1.0, 1.0])),
        compute_metrics(frame, _preds([0.5, 1.0])),
        affected=1,
    )
    assert impact.gmv_at_risk == pytest.approx(50.0)
    assert impact.success_rate_delta_pp == pytest.approx(-25.0)


def test_segments_follow_rerouted_rows():
    before = _frame([100.0, 100.0], gateways=["gateway_a", "gateway_a"])
    after = _frame([100.0, 100.0], gateways=["gateway_a", "gateway_b"])
    segments = {
        s.segment: s
        for s in segment_impacts(before, _preds([1.0, 1.0]), after, _preds([1.0, 0.5]))
        if s.dimension == "gateway"
    }
    assert segments["gateway_a"].counterfactual_transactions == 1
    assert segments["gateway_b"].baseline_transactions == 0
    assert segments["gateway_b"].counterfactual_success_rate == pytest.approx(0.5)


def test_predictions_validate_shapes_and_range():
    with pytest.raises(ValueError):
        _preds([1.2])
    with pytest.raises(ValueError):
        Predictions(
            p_success=np.ones(2),
            reason_probs=np.zeros((2, 3)),
            latency_median_ms=np.ones(2),
            latency_log_sigma=np.zeros(2),
        )


def _window(n, seed, p_shift=0.0, minutes=1440):
    """A day of transactions, spread over five-minute blocks with jittered counts."""
    rng = np.random.default_rng(seed)
    stamps = pd.to_datetime("2026-02-03") + pd.to_timedelta(
        np.sort(rng.integers(0, minutes, n)), unit="m"
    )
    frame = _frame(rng.lognormal(6.0, 1.0, n))
    frame.insert(0, "timestamp", stamps)
    p = np.clip(rng.beta(30.0, 1.0, n) - p_shift, 0.0, 1.0)
    return frame, _preds(p, timeout_share=0.2, latency=1000.0, sigma=0.2)


def test_block_bootstrap_reproduces_the_point_estimate():
    """Drawing every block once must give back exactly `compute_impact`, or the interval is
    centred on a different quantity than the one reported."""
    from backend.simulation.metrics import (
        _block_rows,
        _block_sums,
        _impact_from_sums,
        _LatencyQuantile,
        _pad_blocks,
    )

    base, base_preds = _window(4000, seed=1)
    cf, cf_preds = _window(3600, seed=2, p_shift=0.01)
    start = min(base["timestamp"].min(), cf["timestamp"].min())
    base_groups, base_blocks = _block_rows(base["timestamp"], start)
    cf_groups, cf_blocks = _block_rows(cf["timestamp"], start)
    n_blocks = max(len(base_groups), len(cf_groups))
    once = np.ones((1, n_blocks), dtype=np.int64)

    rebuilt = _impact_from_sums(
        once @ _pad_blocks(_block_sums(base, base_preds, base_groups), n_blocks),
        once @ _pad_blocks(_block_sums(cf, cf_preds, cf_groups), n_blocks),
        _LatencyQuantile.fit(base, base_preds, base_blocks, n_blocks).p95(once),
        _LatencyQuantile.fit(cf, cf_preds, cf_blocks, n_blocks).p95(once),
    )[0]
    direct = compute_impact(
        compute_metrics(base, base_preds), compute_metrics(cf, cf_preds), 400
    )
    for field, got in zip(IMPACT_FIELDS, rebuilt, strict=True):
        want = getattr(direct, field)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-6), field


def test_interval_contains_the_point_estimate():
    base, base_preds = _window(4000, seed=3)
    cf, cf_preds = _window(3600, seed=4, p_shift=0.01)
    interval = impact_interval(
        base, base_preds, cf, cf_preds, 400, member_impacts=[], n_bootstrap=60
    )
    assert interval.level == pytest.approx(0.90)
    for field in IMPACT_FIELDS:
        low, high = getattr(interval, field)
        assert low <= high, field


def test_interval_widens_with_ensemble_disagreement():
    """Member spread is model uncertainty, so disagreeing members must widen the interval."""
    base, base_preds = _window(3000, seed=5)
    cf, cf_preds = _window(3000, seed=6, p_shift=0.01)
    agree = impact_interval(base, base_preds, cf, cf_preds, 300, [], n_bootstrap=60)

    # Members that disagree about how bad the degradation is.
    mild = compute_impact(compute_metrics(base, base_preds), compute_metrics(cf, cf_preds), 300)
    harsh = compute_impact(
        compute_metrics(base, base_preds),
        compute_metrics(
            cf, _preds(np.clip(cf_preds.p_success - 0.05, 0, 1), timeout_share=0.2, sigma=0.2)
        ),
        300,
    )
    disagree = impact_interval(
        base, base_preds, cf, cf_preds, 300, [mild, harsh, mild], n_bootstrap=60
    )
    low, high = disagree.success_rate_delta_pp
    assert low < agree.success_rate_delta_pp[0]
    assert high > agree.success_rate_delta_pp[1]


def test_ensemble_mean_averages_members():
    a = _preds([1.0, 0.0], latency=1000.0, sigma=0.0)
    b = _preds([0.0, 1.0], latency=2000.0, sigma=0.0)
    mean = ensemble_mean([a, b])
    assert mean.p_success == pytest.approx([0.5, 0.5])
    # Latency medians average in log space: the geometric mean of 1000 and 2000, not 1500.
    assert mean.latency_median_ms == pytest.approx([1414.2136, 1414.2136])
    # A member's own spread plus the spread between members widens the combined spread.
    assert mean.latency_log_sigma == pytest.approx([0.5 * np.log(2.0)] * 2, rel=1e-6)
    assert ensemble_mean([a]).p_success == pytest.approx(a.p_success)

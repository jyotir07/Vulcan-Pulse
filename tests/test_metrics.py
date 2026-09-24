import numpy as np
import pandas as pd
import pytest

from backend.data.dgp import FAILURE_REASONS
from backend.simulation.metrics import (
    Predictions,
    compute_impact,
    compute_metrics,
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

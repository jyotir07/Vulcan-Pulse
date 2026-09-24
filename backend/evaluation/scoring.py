"""Error statistics for counterfactual estimates and transaction-level predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from backend.simulation.metrics import Predictions

# Impact fields scored against ground truth, with the smallest true magnitude for which a relative
# error or a sign is meaningful (below it the truth is essentially zero).
TARGETS = {
    "success_rate_delta_pp": 0.1,
    "timeout_rate_delta_pp": 0.05,
    "failures_delta": 10.0,
    "gmv_at_risk": 100_000.0,
    "p95_latency_delta_ms": 50.0,
}
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 7
CALIBRATION_BINS = 10


def _bootstrap_mae_ci(errors: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(np.abs(errors), size=(BOOTSTRAP_SAMPLES, len(errors)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def summarize_errors(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Error statistics per group of the long-format results (one row per estimate).

    `results` needs columns target, predicted and truth plus the `by` columns.
    """
    rows = []
    for keys, group in results.groupby(by, sort=True):
        target = group["target"].iloc[0]
        threshold = TARGETS[target]
        error = (group["predicted"] - group["truth"]).to_numpy()
        truth = group["truth"].to_numpy()
        material = np.abs(truth) >= threshold
        ci_low, ci_high = _bootstrap_mae_ci(error)
        rows.append(
            {
                **dict(zip(by, keys if isinstance(keys, tuple) else (keys,), strict=True)),
                "n": len(group),
                "mae": float(np.abs(error).mean()),
                "mae_ci_low": ci_low,
                "mae_ci_high": ci_high,
                "rmse": float(np.sqrt((error**2).mean())),
                "median_relative_error": (
                    float(np.median(np.abs(error[material]) / np.abs(truth[material])))
                    if material.any()
                    else None
                ),
                "sign_accuracy": (
                    float(
                        (
                            np.sign(group["predicted"].to_numpy()[material])
                            == np.sign(truth[material])
                        ).mean()
                    )
                    if material.any()
                    else None
                ),
                "n_material": int(material.sum()),
            }
        )
    return pd.DataFrame(rows)


def transaction_scores(
    success: np.ndarray,
    latency_ms: np.ndarray,
    reason_codes: np.ndarray,
    preds: Predictions,
) -> dict:
    """How well per-transaction predictions match realized first-attempt outcomes."""
    p = preds.p_success
    failed = ~success
    bins = pd.qcut(p, CALIBRATION_BINS, labels=False, duplicates="drop")
    calibration = (
        pd.DataFrame({"bin": bins, "predicted": p, "observed": success.astype(float)})
        .groupby("bin")
        .mean()
        .reset_index(drop=True)
    )
    reason_p = preds.reason_probs[np.flatnonzero(failed), reason_codes[failed]]
    log_latency = np.log(latency_ms)
    predicted_log = np.log(preds.latency_median_ms)
    return {
        "auc": float(roc_auc_score(success, p)),
        "brier": float(brier_score_loss(success, p)),
        "log_loss": float(log_loss(success, np.clip(p, 1e-6, 1 - 1e-6))),
        "mean_predicted": float(p.mean()),
        "mean_observed": float(success.mean()),
        "calibration": calibration.to_dict(orient="records"),
        "reason_log_loss": float(-np.log(np.clip(reason_p, 1e-6, 1.0)).mean()),
        "reason_top1_accuracy": float(
            (preds.reason_probs[failed].argmax(axis=1) == reason_codes[failed]).mean()
        ),
        "latency_log_mae": float(np.abs(log_latency - predicted_log).mean()),
        "latency_median_abs_error_ms": float(
            np.median(np.abs(latency_ms - preds.latency_median_ms))
        ),
    }

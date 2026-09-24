"""Independent ML baseline: one gradient-boosted tree model per task, no shared representation.

- success: binary classifier on every first attempt;
- failure reason: multiclass classifier trained on failed first attempts only, so it models the
  reason given a failure;
- latency: regressor on log latency; its residual spread per method, measured on a held-out
  slice of the training data, becomes the lognormal sigma.

Trees don't extrapolate: beyond the range of a feature seen in training, predictions stay at the
edge value. That limit is part of what the evaluation harness measures.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from backend.data.catalog import METHODS
from backend.data.dgp import FAILURE_REASONS
from backend.models.features import FeatureBuilder
from backend.simulation.metrics import Predictions

TREE_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 50,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "categorical_features": "from_dtype",
}
LATENCY_HOLDOUT_FRACTION = 0.1


@dataclass(frozen=True)
class IndependentModels:
    features: FeatureBuilder
    success: HistGradientBoostingClassifier
    reason: HistGradientBoostingClassifier
    latency: HistGradientBoostingRegressor
    latency_log_sigma: np.ndarray  # [method]
    name: str = "independent_ml"

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        customers: pd.DataFrame,
        merchants: pd.DataFrame,
        seed: int,
    ) -> "IndependentModels":
        """Fit on observed first attempts. `train` must not include any evaluation window."""
        features = FeatureBuilder(customers, merchants)
        x = features(train)
        failed = (train["transaction_status"] == "FAILED").to_numpy()

        success = HistGradientBoostingClassifier(**TREE_PARAMS, random_state=seed)
        success.fit(x, (~failed).astype(int))

        reason = HistGradientBoostingClassifier(**TREE_PARAMS, random_state=seed)
        reason.fit(x[failed], train["failure_reason"].cat.codes.to_numpy()[failed])

        log_latency = np.log(train["latency_ms"].to_numpy().astype(float))
        holdout = np.random.default_rng(seed).random(len(train)) < LATENCY_HOLDOUT_FRACTION
        latency = HistGradientBoostingRegressor(**TREE_PARAMS, random_state=seed)
        latency.fit(x[~holdout], log_latency[~holdout])
        residual = log_latency[holdout] - latency.predict(x[holdout])
        method = train["payment_method"].astype(str).map(METHODS.index).to_numpy()[holdout]
        sigma = np.array([residual[method == m].std() for m in range(len(METHODS))])

        return cls(
            features=features,
            success=success,
            reason=reason,
            latency=latency,
            latency_log_sigma=sigma,
        )

    def predict(self, frame: pd.DataFrame) -> Predictions:
        x = self.features(frame)
        reason_probs = np.zeros((len(frame), len(FAILURE_REASONS)))
        # Reasons never seen failing in training keep probability 0.
        reason_probs[:, self.reason.classes_] = self.reason.predict_proba(x)
        method = frame["payment_method"].astype(str).map(METHODS.index).to_numpy()
        return Predictions(
            p_success=self.success.predict_proba(x)[:, 1],
            reason_probs=reason_probs,
            latency_median_ms=np.exp(self.latency.predict(x)),
            latency_log_sigma=self.latency_log_sigma[method],
        )

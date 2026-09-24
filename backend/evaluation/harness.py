"""Run a scenario grid through every predictor and the ground truth.

A scenario counts as out of range when more than `OUT_OF_RANGE_SHARE` of the transactions it
affects have a state feature outside the central `SUPPORT_QUANTILES` of the training data. The
label comes from the data the models actually saw, not from the scenario's name.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.data.generator import Dataset
from backend.evaluation.scoring import TARGETS
from backend.simulation.engine import Predictor, ground_truth, simulate
from backend.simulation.interventions import ObservedEcosystem, apply_scenario
from backend.simulation.scenarios import Scenario

SUPPORT_FEATURES = ("issuer_health", "gateway_health", "gateway_utilization")
SUPPORT_QUANTILES = (0.001, 0.999)
OUT_OF_RANGE_SHARE = 0.05


@dataclass(frozen=True)
class Support:
    low: dict[str, float]
    high: dict[str, float]

    @classmethod
    def from_training(cls, train: pd.DataFrame) -> "Support":
        low, high = {}, {}
        for feature in SUPPORT_FEATURES:
            low[feature], high[feature] = np.quantile(train[feature], SUPPORT_QUANTILES)
        return cls(low=low, high=high)

    def outside(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame), dtype=bool)
        for feature in SUPPORT_FEATURES:
            values = frame[feature].to_numpy()
            out |= (values < self.low[feature]) | (values > self.high[feature])
        return out


def run_grid(
    dataset: Dataset,
    observed: ObservedEcosystem,
    windows: list[date],
    predictors: list[Predictor],
    grid: list[tuple[str, list[dict]]],
    support: Support,
    progress=None,
) -> pd.DataFrame:
    """Long-format results: one row per (scenario, window, predictor, target)."""
    rows = []
    for window in windows:
        for name, interventions in grid:
            scenario = Scenario.model_validate(
                {"interventions": interventions, "window_date": window}
            )
            cf = apply_scenario(dataset, scenario, observed)
            affected = cf.counterfactual[cf.affected]
            outside_share = float(support.outside(affected).mean()) if len(affected) else 0.0
            truth = ground_truth(dataset, cf).impact
            kinds = sorted({iv["type"] for iv in interventions})
            common = {
                "scenario": name,
                "type": kinds[0] if len(kinds) == 1 else "combined",
                "window": window.isoformat(),
                "range": "out_of_range" if outside_share > OUT_OF_RANGE_SHARE else "in_range",
                "outside_support_share": outside_share,
                "transactions_affected": truth.transactions_affected,
            }
            for predictor in predictors:
                predicted = simulate(dataset, cf, predictor).impact
                for target in TARGETS:
                    rows.append(
                        {
                            **common,
                            "predictor": predictor.name,
                            "target": target,
                            "predicted": getattr(predicted, target),
                            "truth": getattr(truth, target),
                        }
                    )
            if progress:
                progress(f"{window} {name}: {common['range']}")
    return pd.DataFrame(rows)

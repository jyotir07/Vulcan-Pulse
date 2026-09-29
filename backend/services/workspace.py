"""Everything a simulation session needs, loaded once: data, observed state, predictors.

Every predictor is trained on the same training days (the main suite), and a scenario without a
`window_date` runs on the last normal weekday of the held-out test period, so no predictor has
seen the day it is asked about and the ground truth comparison is fair.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from backend.data.generator import Dataset
from backend.data.io import load_dataset
from backend.models.independent import IndependentModels
from backend.models.rule_baseline import RuleBaseline
from backend.models.shared import SharedModel, default_model_dir
from backend.models.splits import TimeSplit, evaluation_windows, suite_training_rows, time_split
from backend.simulation.engine import BaselineCache, Predictor
from backend.simulation.interventions import (
    Counterfactual,
    ObservedEcosystem,
    apply_scenario,
    observe,
)
from backend.simulation.scenarios import Scenario

PREDICTORS = ("rule_baseline", "independent_ml", "shared_representation")


@dataclass
class Workspace:
    dataset: Dataset
    observed: ObservedEcosystem
    split: TimeSplit
    default_window: date
    predictors: dict[str, Predictor]
    cache: BaselineCache = field(default_factory=BaselineCache)

    @classmethod
    def load(
        cls,
        data_dir: Path,
        predictors: tuple[str, ...] = PREDICTORS,
        model_dir: Path | None = None,
        progress=None,
    ) -> "Workspace":
        dataset = load_dataset(data_dir)
        split = time_split(dataset)
        train = suite_training_rows(dataset, split, "main")
        loaded: dict[str, Predictor] = {}
        for name in predictors:
            if progress:
                progress(f"loading {name}")
            if name == "rule_baseline":
                loaded[name] = RuleBaseline.fit(train)
            elif name == "independent_ml":
                loaded[name] = IndependentModels.fit(
                    train, dataset.customers, dataset.merchants, dataset.config.seed
                )
            elif name == "shared_representation":
                shared = SharedModel.load(
                    model_dir or default_model_dir(), dataset.customers, dataset.merchants
                )
                if shared.train_days != [d.isoformat() for d in split.train_days]:
                    raise ValueError("the shared model was trained on different days")
                loaded[name] = shared
            else:
                raise ValueError(f"unknown predictor {name!r}; expected one of {PREDICTORS}")
        return cls(
            dataset=dataset,
            observed=observe(dataset),
            split=split,
            default_window=evaluation_windows(dataset, split)[-1],
            predictors=loaded,
        )

    def with_window(self, scenario: Scenario) -> Scenario:
        if scenario.window_date is not None:
            return scenario
        return scenario.model_copy(update={"window_date": self.default_window})

    def counterfactual(self, scenario: Scenario) -> Counterfactual:
        return apply_scenario(self.dataset, self.with_window(scenario), self.observed)

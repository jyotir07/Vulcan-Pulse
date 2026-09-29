"""Run the Phase 5 experiments: main comparison, cold start, distribution shift, calibration.

Each suite's rule baseline and independent models are fitted here on the suite's training rows;
its shared-representation ensemble is loaded from `artifacts/`, trained beforehand with
`python scripts/train.py --suite SUITE`. Results go to `experiments/<experiment>/results.{json,md}`
and every estimate to `experiments/scenario_results.csv`.

Usage:
    python scripts/run_experiments.py [--data DIR] [--out DIR] [--suites main cold_start ...]
                                      [--quick]
"""

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from backend.config import REPO_ROOT
from backend.data.io import load_dataset
from backend.evaluation.experiments import (
    Suite,
    calibration_summary,
    cold_start_grid,
    cold_start_summary,
    held_out_issuer_summary,
    main_grid,
    main_summary,
    run_suite,
    shift_grid,
    shift_summary,
)
from backend.evaluation.harness import OUT_OF_RANGE_SHARE, Support
from backend.evaluation.scoring import transaction_scores
from backend.models.independent import IndependentModels
from backend.models.rule_baseline import RuleBaseline
from backend.models.shared import SharedModel, default_model_dir
from backend.models.splits import (
    HOLDOUT_FRACTION,
    HOLDOUT_ISSUERS,
    SUITES,
    entity_holdout,
    evaluation_windows,
    first_attempts_on,
    normal_traffic,
    suite_training_rows,
    time_split,
)
from backend.simulation.interventions import observe
from backend.simulation.metrics import Predictions
from backend.simulation.oracle import true_success_probability

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"
DEFAULT_OUT = REPO_ROOT / "experiments"
STARTED = time.perf_counter()
TRANSACTION_KEYS = ("auc", "brier", "log_loss", "mean_predicted", "mean_observed")


def _log(message: str) -> None:
    print(f"[{time.perf_counter() - STARTED:6.0f}s] {message}", flush=True)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:,.{digits}f}"


def _load_suite(dataset, split, name: str) -> tuple[Suite, pd.DataFrame]:
    train = suite_training_rows(dataset, split, name)
    directory = default_model_dir(name)
    if not (directory / "model.json").exists():
        raise SystemExit(f"No {name} model at {directory}; run scripts/train.py --suite {name}.")
    _log(f"[{name}] fitting baselines on {len(train):,} first attempts")
    shared = SharedModel.load(directory, dataset.customers, dataset.merchants)
    if shared.train_days != [d.isoformat() for d in split.train_days]:
        raise SystemExit(f"The {name} model was trained on different days than this split.")
    if sorted(train["issuer_id"].unique().tolist()) != shared.known_issuers:
        raise SystemExit(f"The {name} model was trained on different issuers than this suite.")
    torch.set_num_threads(shared.config.threads)
    predictors = [
        RuleBaseline.fit(train),
        IndependentModels.fit(train, dataset.customers, dataset.merchants, dataset.config.seed),
        shared,
    ]
    holdout = entity_holdout(dataset) if name == "cold_start" else None
    return Suite(name, predictors, Support.from_training(train), holdout), train


def _scores(frame: pd.DataFrame, preds, true_p: np.ndarray, groups: dict) -> dict:
    """Transaction-level scores per row group, with the true probability as the ceiling."""
    success = (frame["transaction_status"] == "SUCCESS").to_numpy()
    latency = frame["latency_ms"].to_numpy().astype(float)
    reasons = frame["failure_reason"].cat.codes.to_numpy()
    out = {}
    for group, rows in groups.items():
        idx = np.flatnonzero(rows)
        scores = transaction_scores(success[idx], latency[idx], reasons[idx], preds.take(idx))
        truth = Predictions(
            p_success=true_p[idx],
            reason_probs=preds.reason_probs[idx],
            latency_median_ms=latency[idx],
            latency_log_sigma=np.zeros(len(idx)),
        )
        ceiling = transaction_scores(success[idx], latency[idx], reasons[idx], truth)
        out[group] = {
            "n": len(idx),
            **{k: scores[k] for k in TRANSACTION_KEYS},
            "true_probability_auc": ceiling["auc"],
            "true_probability_log_loss": ceiling["log_loss"],
        }
    return out


def _transaction_level(dataset, split, suites: dict[str, Suite], festival: date) -> dict:
    test = first_attempts_on(dataset, split.test_days)
    festival_rows = first_attempts_on(dataset, [festival])
    frame = pd.concat([test, festival_rows], ignore_index=True)
    true_p = true_success_probability(dataset).loc[frame["transaction_id"]].to_numpy()
    is_test = np.arange(len(frame)) < len(test)
    groups = {"cold_start": {}}
    groups["normal"] = {
        "test_normal_hours": is_test & normal_traffic(frame),
        "test_peak_hours": is_test & frame["is_peak"].to_numpy(),
        "festival_day": ~is_test,
    }
    if "cold_start" in suites:
        h = suites["cold_start"].holdout
        groups["cold_start"] = {
            "known": is_test & ~h.involved(frame),
            "held_out_merchant": is_test & h.merchant_rows(frame),
            "held_out_customer": is_test & h.customer_rows(frame),
            "held_out_issuer": is_test & h.issuer_rows(frame),
        }
    out = {}
    for name, suite in suites.items():
        kind = "cold_start" if name == "cold_start" else "normal"
        for predictor in suite.predictors:
            _log(f"[{name}] transaction-level scores for {predictor.name}")
            preds = predictor.predict(frame)
            out.setdefault(name, {})[predictor.name] = _scores(frame, preds, true_p, groups[kind])
    return out


def _table(frame: pd.DataFrame, keys: list[str], target: str = "success_rate_delta_pp") -> list:
    rows = frame[frame["target"] == target]
    lines = [
        "| " + " | ".join(keys) + " | n | MAE [95% CI] | RMSE | Median rel. error | Sign acc. |",
        "|" + "---|" * (len(keys) + 5),
    ]
    for _, r in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(str(r[k]) for k in keys)
            + f" | {r['n']} | {_fmt(r['mae'])} [{_fmt(r['mae_ci_low'])}, {_fmt(r['mae_ci_high'])}]"
            + f" | {_fmt(r['rmse'])} | {_fmt(r['median_relative_error'])}"
            + f" | {_fmt(r['sign_accuracy'], 2)} |"
        )
    return lines


def _pivot(frame: pd.DataFrame, index: list[str], target: str = "success_rate_delta_pp") -> list:
    """One MAE column per predictor, so the comparison reads across a row."""
    rows = frame[frame["target"] == target]
    wide = rows.pivot_table(index=index, columns="predictor", values="mae", aggfunc="first")
    counts = rows.groupby(index)["n"].first()
    predictors = list(wide.columns)
    lines = [
        "| " + " | ".join(index) + " | n | " + " | ".join(predictors) + " |",
        "|" + "---|" * (len(index) + 1 + len(predictors)),
    ]
    for key, row in wide.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        best = row.min()
        cells = [f"**{_fmt(v)}**" if v == best and len(predictors) > 1 else _fmt(v) for v in row]
        lines.append(
            "| "
            + " | ".join(map(str, key))
            + f" | {counts[key if len(key) > 1 else key[0]]} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def _write(out: Path, name: str, title: str, payload: dict, body: list[str]) -> None:
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    header = [
        f"# {title}",
        "",
        "Synthetic data; simulation output, not real payment performance. Generated by "
        "`python scripts/run_experiments.py`.",
        "",
    ]
    (directory / "results.md").write_text("\n".join(header + body) + "\n", encoding="utf-8")


def _transaction_table(scores: dict, groups: list[str]) -> list[str]:
    lines = [
        "| Suite | Predictor | Rows | n | AUC | Ceiling AUC | Log loss | Ceiling log loss | "
        "Mean p / observed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for suite, by_predictor in scores.items():
        for predictor, by_group in by_predictor.items():
            for group in groups:
                if group not in by_group:
                    continue
                s = by_group[group]
                lines.append(
                    f"| {suite} | {predictor} | {group} | {s['n']:,} | {_fmt(s['auc'], 4)} | "
                    f"{_fmt(s['true_probability_auc'], 4)} | {_fmt(s['log_loss'], 4)} | "
                    f"{_fmt(s['true_probability_log_loss'], 4)} | "
                    f"{_fmt(s['mean_predicted'], 4)} / {_fmt(s['mean_observed'], 4)} |"
                )
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument(
        "--quick", action="store_true", help="Every 4th scenario and one window, for a smoke run."
    )
    args = parser.parse_args()
    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"No dataset at {args.data}; run scripts/generate_data.py first.")

    dataset = load_dataset(args.data)
    split = time_split(dataset)
    normal_windows = evaluation_windows(dataset, split)
    festivals = [d for d in dataset.festival_dates if d in split.train_days]
    if not festivals:
        raise SystemExit("No festival day in the training period to test distribution shift on.")
    festival = festivals[-1]

    suites = {}
    for name in args.suites:
        suites[name], _ = _load_suite(dataset, split, name)

    plans = {
        "main": (main_grid(), {d: "normal" for d in normal_windows}),
        "cold_start": (cold_start_grid(), {d: "normal" for d in normal_windows}),
        "normal_only": (shift_grid(), {normal_windows[0]: "normal", festival: "festival"}),
    }
    if args.quick:
        plans = {
            k: (grid[::4], dict(list(windows.items())[-1:])) for k, (grid, windows) in plans.items()
        }

    transaction_level = _transaction_level(dataset, split, suites, festival)

    observed = observe(dataset)
    frames = []
    for name, suite in suites.items():
        grid, windows = plans[name]
        _log(f"[{name}] {len(grid)} scenarios x {len(windows)} windows")
        frames.append(run_suite(dataset, observed, suite, windows, grid, progress=_log))
    results = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "scenario_results.csv", index=False)

    meta = {
        "split": {
            "train": [d.isoformat() for d in split.train_days],
            "test": [d.isoformat() for d in split.test_days],
        },
        "normal_windows": [d.isoformat() for d in normal_windows],
        "festival_window": festival.isoformat(),
        "holdout": {
            "issuers": list(HOLDOUT_ISSUERS),
            "merchant_fraction": HOLDOUT_FRACTION,
            "customer_fraction": HOLDOUT_FRACTION,
        },
        "out_of_range_share": OUT_OF_RANGE_SHARE,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "quick": args.quick,
    }

    if {"main", "cold_start", "normal_only"} <= suites.keys():
        summary = main_summary(results)
        _write(
            args.out,
            "main",
            "Main experiment: does a shared representation improve counterfactual prediction?",
            {**meta, "summary": summary.to_dict(orient="records")},
            [
                "Success-rate MAE (pp) per category and range; best per row in bold. "
                "`new_merchant` and `new_customer` come from the cold-start suite, scored on the "
                "held-out entities' payments; `distribution_shift` from the normal-only suite on "
                f"an unseen festival day ({festival}).",
                "",
                *_pivot(summary, ["category", "range"]),
                "",
                "## GMV at risk MAE (₹)",
                "",
                *_pivot(summary, ["category", "range"], "gmv_at_risk"),
                "",
                "## Full statistics",
                "",
                *_table(summary, ["category", "range", "predictor"]),
            ],
        )
    if "cold_start" in suites:
        summary = cold_start_summary(results)
        issuers = held_out_issuer_summary(results)
        _write(
            args.out,
            "cold_start",
            "Cold start: known vs held-out merchants, customers and issuers",
            {
                **meta,
                "counterfactual_by_subset": summary.to_dict(orient="records"),
                "held_out_issuer_degradations": issuers.to_dict(orient="records"),
                "transaction_level": {"cold_start": transaction_level.get("cold_start", {})},
            },
            [
                f"Trained without {HOLDOUT_FRACTION:.0%} of merchants, {HOLDOUT_FRACTION:.0%} of "
                f"customers and the issuers {', '.join(HOLDOUT_ISSUERS)}.",
                "",
                "## Transaction level (test period)",
                "",
                "Ceiling columns score the true success probability on the same rows: groups "
                "differ in how predictable they are, so compare each model against its row's "
                "ceiling, not across rows.",
                "",
                *_transaction_table(
                    {"cold_start": transaction_level.get("cold_start", {})},
                    ["known", "held_out_merchant", "held_out_customer", "held_out_issuer"],
                ),
                "",
                "## Counterfactual success-rate MAE (pp) by payment subset",
                "",
                *_pivot(summary, ["subset"]),
                "",
                "## Degrading a held-out issuer vs a known one",
                "",
                *_pivot(issuers, ["category"]),
                "",
                "## Full statistics",
                "",
                *_table(summary, ["subset", "predictor"]),
            ],
        )
    if "normal_only" in suites:
        summary = shift_summary(results)
        tx = {k: v for k, v in transaction_level.items() if k in ("main", "normal_only")}
        _write(
            args.out,
            "distribution_shift",
            "Distribution shift: trained on normal traffic, tested on peak and festival traffic",
            {
                **meta,
                "counterfactual_by_window": summary.to_dict(orient="records"),
                "transaction_level": tx,
            },
            [
                "The normal-only suite never saw peak hours (18:00–22:00) or festival days. The "
                "main suite saw both, including this festival day, so its festival row is "
                "in-sample and only there for scale.",
                "",
                "## Transaction level",
                "",
                *_transaction_table(tx, ["test_normal_hours", "test_peak_hours", "festival_day"]),
                "",
                "## Counterfactual success-rate MAE (pp): normal day vs festival day",
                "",
                *_pivot(summary, ["window_kind"]),
                "",
                "## Full statistics",
                "",
                *_table(summary, ["window_kind", "predictor"]),
            ],
        )
    calibration = calibration_summary(results)
    if len(calibration):
        lines = [
            "Share of scenarios whose true impact falls inside the shared ensemble's 90% "
            "interval (ensemble spread plus a block bootstrap). A calibrated interval covers "
            "about 0.90.",
            "",
            "| Suite | Target | Range | n | Coverage | Median width | Median abs. error |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in calibration.iterrows():
            lines.append(
                f"| {r['suite']} | {r['target']} | {r['range']} | {r['n']} | "
                f"{_fmt(r['coverage'], 2)} | {_fmt(r['median_width'])} | "
                f"{_fmt(r['median_abs_error'])} |"
            )
        _write(
            args.out,
            "calibration",
            "Calibration of uncertainty intervals",
            {**meta, "coverage": calibration.to_dict(orient="records")},
            lines,
        )
    _log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

"""Train the shared-representation ensemble on the training days and save it.

Each member is checkpointed in the output directory as it finishes; rerunning the same command
after an interruption resumes from the members already saved.

`--suite` picks the training rows (`backend/models/splits.py`): `main` for the harness and API,
`cold_start` without held-out entities, `normal_only` without peak hours and festival days. The
last two feed `scripts/run_experiments.py`.

Usage:
    python scripts/train.py [--suite main|cold_start|normal_only] [--data DIR] [--config FILE]
                            [--out DIR]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from backend.config import DEFAULT_TRAIN_CONFIG, REPO_ROOT, load_train_config
from backend.data.io import load_dataset
from backend.models.shared import SharedModel, default_model_dir
from backend.models.splits import SUITES, suite_training_rows, time_split

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"


STARTED = time.perf_counter()


def _log(message: str) -> None:
    print(f"[{time.perf_counter() - STARTED:6.0f}s] {message}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--suite", choices=SUITES, default="main")
    parser.add_argument("--out", type=Path, help="Defaults to artifacts/shared_model[_SUITE].")
    args = parser.parse_args()
    out = args.out or default_model_dir(args.suite)
    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"No dataset at {args.data}; run scripts/generate_data.py first.")

    config = load_train_config(args.config)
    torch.set_num_threads(config.threads)
    dataset = load_dataset(args.data)
    split = time_split(dataset)
    train = suite_training_rows(dataset, split, args.suite)
    _log(
        f"Training {config.ensemble_size} members ({args.suite}) on {len(train):,} first "
        f"attempts ({split.train_days[0]}..{split.train_days[-1]})"
    )
    model = SharedModel.fit(
        train, dataset.customers, dataset.merchants, config, progress=_log, checkpoint_dir=out
    )
    model.save(out)
    _log(f"Wrote {out}")


if __name__ == "__main__":
    main()

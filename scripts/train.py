"""Train the shared-representation ensemble on the training days and save it.

Each member is checkpointed in the output directory as it finishes; rerunning the same command
after an interruption resumes from the members already saved.

Usage:
    python scripts/train.py [--data DIR] [--config FILE] [--out DIR]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from backend.config import DEFAULT_TRAIN_CONFIG, REPO_ROOT, load_train_config
from backend.data.io import load_dataset
from backend.models.shared import SharedModel
from backend.models.splits import first_attempts_on, time_split

DEFAULT_DATA = REPO_ROOT / "data" / "generated" / "default"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "shared_model"


STARTED = time.perf_counter()


def _log(message: str) -> None:
    print(f"[{time.perf_counter() - STARTED:6.0f}s] {message}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"No dataset at {args.data}; run scripts/generate_data.py first.")

    config = load_train_config(args.config)
    torch.set_num_threads(config.threads)
    dataset = load_dataset(args.data)
    split = time_split(dataset)
    train = first_attempts_on(dataset, split.train_days)
    _log(
        f"Training {config.ensemble_size} members on {len(train):,} first attempts "
        f"({split.train_days[0]}..{split.train_days[-1]})"
    )
    model = SharedModel.fit(
        train, dataset.customers, dataset.merchants, config, progress=_log, checkpoint_dir=args.out
    )
    model.save(args.out)
    _log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

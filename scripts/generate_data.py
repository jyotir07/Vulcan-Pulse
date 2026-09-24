import argparse
import time
from pathlib import Path

from backend.config import DEFAULT_ECOSYSTEM_CONFIG, REPO_ROOT, load_ecosystem_config
from backend.data.generator import generate_dataset
from backend.data.io import save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic payment ecosystem.")
    parser.add_argument("--config", type=Path, default=DEFAULT_ECOSYSTEM_CONFIG)
    parser.add_argument("--seed", type=int, help="Override the seed in the config.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "generated" / "default")
    args = parser.parse_args()

    config = load_ecosystem_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    started = time.perf_counter()
    dataset = generate_dataset(config)
    save_dataset(dataset, args.out)
    elapsed = time.perf_counter() - started

    print(f"Wrote {len(dataset.transactions):,} transactions to {args.out} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

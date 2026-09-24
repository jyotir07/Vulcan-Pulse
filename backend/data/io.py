import json
from datetime import date
from pathlib import Path

import pandas as pd

from backend.config import EcosystemConfig
from backend.data.generator import Dataset

TABLES = ("customers", "merchants", "issuers", "gateways", "cities", "transactions")


def save_dataset(dataset: Dataset, config: EcosystemConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(dataset, name).to_parquet(out_dir / f"{name}.parquet", index=False)
    manifest = {
        "config": config.model_dump(mode="json"),
        "festival_dates": [d.isoformat() for d in dataset.festival_dates],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_dataset(data_dir: Path) -> Dataset:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in TABLES}
    return Dataset(
        **tables,
        festival_dates=[date.fromisoformat(d) for d in manifest["festival_dates"]],
    )

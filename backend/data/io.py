import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import EcosystemConfig
from backend.data.generator import Dataset
from backend.data.latent import LatentParams

TABLES = ("customers", "merchants", "issuers", "gateways", "cities", "transactions")
LATENT_ARRAYS = (
    "issuer_reliability",
    "issuer_latency_factor",
    "issuer_load_sensitivity",
    "gateway_latency_factor",
    "gateway_capacity",
)


def save_dataset(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(dataset, name).to_parquet(out_dir / f"{name}.parquet", index=False)
    manifest = {
        "config": dataset.config.model_dump(mode="json"),
        "festival_dates": [d.isoformat() for d in dataset.festival_dates],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Ground truth lives in its own folder so it never gets loaded as a feature source.
    latent_dir = out_dir / "latent"
    latent_dir.mkdir(exist_ok=True)
    latent = dataset.latent
    latent.episodes.to_parquet(latent_dir / "episodes.parquet", index=False)
    params = {name: getattr(latent, name).tolist() for name in LATENT_ARRAYS}
    params["start"] = str(latent.start)
    params["n_minutes"] = latent.n_minutes
    (latent_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_dataset(data_dir: Path) -> Dataset:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_parquet(data_dir / f"{name}.parquet") for name in TABLES}

    latent_dir = data_dir / "latent"
    params = json.loads((latent_dir / "params.json").read_text(encoding="utf-8"))
    latent = LatentParams(
        start=np.datetime64(params["start"], "ms"),
        n_minutes=params["n_minutes"],
        episodes=pd.read_parquet(latent_dir / "episodes.parquet"),
        **{name: np.asarray(params[name]) for name in LATENT_ARRAYS},
    )
    return Dataset(
        config=EcosystemConfig.model_validate(manifest["config"]),
        **tables,
        festival_dates=[date.fromisoformat(d) for d in manifest["festival_dates"]],
        latent=latent,
    )

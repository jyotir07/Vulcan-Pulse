"""Rule-based baseline: historical rates scaled by health and load ratios.

This is the "simple hand-authored rules" comparison point. It never looks at the true outcome
process; everything is estimated from observed first attempts:

- success: the historical rate of the (issuer, method, gateway, 4-hour block) cell, shrunk toward
  the (issuer, method) rate and then the method rate so sparse cells don't swing, scaled by three
  ratio rules: current issuer health over that issuer's normal health, current gateway health
  over that gateway's normal health, and the historical success of the current utilization band
  over the unloaded success rate;
- failure reason: the historical reason mix per (method, gateway health low, issuer health low);
- latency: lognormal with the historical log-mean per (method, gateway, utilization band).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.data.catalog import GATEWAYS, ISSUERS, METHODS
from backend.data.dgp import FAILURE_REASONS
from backend.simulation.metrics import Predictions

HOUR_BLOCK = 4
N_BLOCKS = 24 // HOUR_BLOCK
SHRINKAGE = 50.0
# Bands above the first edge are "loaded"; the first band is the unloaded reference.
UTILIZATION_BANDS = (0.7, 0.8, 0.9, 1.0, 1.1)
LOW_GATEWAY_HEALTH = 0.8
LOW_ISSUER_HEALTH = 0.9
P_BOUNDS = (0.001, 0.999)


def _shrunk(successes: np.ndarray, counts: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return (successes + SHRINKAGE * prior) / (counts + SHRINKAGE)


def _codes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    utilization = frame["gateway_utilization"].to_numpy()
    return {
        "issuer": frame["issuer_id"].to_numpy(),
        "method": frame["payment_method"].astype(str).map(METHODS.index).to_numpy(),
        "gateway": frame["gateway_id"].to_numpy(),
        "block": frame["hour"].to_numpy() // HOUR_BLOCK,
        "band": np.searchsorted(UTILIZATION_BANDS, utilization, side="right"),
        "gateway_low": (frame["gateway_health"].to_numpy() < LOW_GATEWAY_HEALTH).astype(int),
        "issuer_low": (frame["issuer_health"].to_numpy() < LOW_ISSUER_HEALTH).astype(int),
    }


@dataclass(frozen=True)
class RuleBaseline:
    cell_rate: np.ndarray  # [issuer, method, gateway, block]
    # Mean observed health, so the health ratio averages 1 over normal history.
    normal_issuer_health: np.ndarray  # [issuer, method]
    normal_gateway_health: np.ndarray  # [gateway]
    band_factor: np.ndarray  # [band]
    reason_mix: np.ndarray  # [method, gateway_low, issuer_low, reason]
    log_latency_mean: np.ndarray  # [method, gateway, band]
    log_latency_sigma: np.ndarray  # [method]
    name: str = "rule_baseline"

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "RuleBaseline":
        """Fit on observed first attempts. `train` must not include the evaluation window."""
        c = _codes(train)
        success = (train["transaction_status"] == "SUCCESS").to_numpy().astype(float)
        n_i, n_m, n_g = len(ISSUERS), len(METHODS), len(GATEWAYS)

        def tally(index: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.bincount(index, weights=success, minlength=size),
                np.bincount(index, minlength=size).astype(float),
            )

        s_m, n_m_count = tally(c["method"], n_m)
        method_rate = s_m / np.maximum(n_m_count, 1.0)
        s_im, n_im = tally(c["issuer"] * n_m + c["method"], n_i * n_m)
        im_rate = _shrunk(s_im, n_im, np.tile(method_rate, n_i)).reshape(n_i, n_m)
        cell_index = ((c["issuer"] * n_m + c["method"]) * n_g + c["gateway"]) * N_BLOCKS + c[
            "block"
        ]
        s_cell, n_cell = tally(cell_index, n_i * n_m * n_g * N_BLOCKS)
        prior = np.broadcast_to(im_rate[:, :, None, None], (n_i, n_m, n_g, N_BLOCKS)).ravel()
        cell_rate = _shrunk(s_cell, n_cell, prior).reshape(n_i, n_m, n_g, N_BLOCKS)

        normal_issuer_health = (
            train.groupby([c["issuer"], c["method"]])["issuer_health"]
            .mean()
            .unstack()
            .reindex(index=range(n_i), columns=range(n_m))
            .to_numpy()
        )
        normal_gateway_health = (
            train.groupby(c["gateway"])["gateway_health"].mean().reindex(range(n_g)).to_numpy()
        )
        n_b = len(UTILIZATION_BANDS) + 1
        s_band, n_band = tally(c["band"], n_b)
        unloaded = s_band[0] / max(n_band[0], 1.0)
        band_factor = np.minimum(_shrunk(s_band, n_band, np.full(n_b, unloaded)) / unloaded, 1.0)

        failed = success == 0.0
        reason = train["failure_reason"].cat.codes.to_numpy()
        n_r = len(FAILURE_REASONS)
        method_mix = np.ones((n_m, n_r))  # add-one so no reason is ever impossible
        np.add.at(method_mix, (c["method"][failed], reason[failed]), 1.0)
        method_mix /= method_mix.sum(axis=1, keepdims=True)
        reason_counts = np.zeros((n_m, 2, 2, n_r))
        np.add.at(
            reason_counts,
            (
                c["method"][failed],
                c["gateway_low"][failed],
                c["issuer_low"][failed],
                reason[failed],
            ),
            1.0,
        )
        reason_mix = (reason_counts + SHRINKAGE * method_mix[:, None, None, :]) / (
            reason_counts.sum(axis=3, keepdims=True) + SHRINKAGE
        )

        log_latency = np.log(train["latency_ms"].to_numpy().astype(float))
        latency_index = (c["method"] * n_g + c["gateway"]) * n_b + c["band"]
        sums = np.bincount(latency_index, weights=log_latency, minlength=n_m * n_g * n_b)
        counts = np.bincount(latency_index, minlength=n_m * n_g * n_b).astype(float)
        method_mean = np.bincount(c["method"], weights=log_latency, minlength=n_m) / np.maximum(
            n_m_count, 1.0
        )
        method_prior = np.repeat(method_mean, n_g * n_b)
        log_latency_mean = ((sums + SHRINKAGE * method_prior) / (counts + SHRINKAGE)).reshape(
            n_m, n_g, n_b
        )
        residual_latency = log_latency - log_latency_mean.ravel()[latency_index]
        log_latency_sigma = np.sqrt(
            np.bincount(c["method"], weights=residual_latency**2, minlength=n_m)
            / np.maximum(n_m_count, 1.0)
        )

        return cls(
            cell_rate=cell_rate,
            normal_issuer_health=normal_issuer_health,
            normal_gateway_health=normal_gateway_health,
            band_factor=band_factor,
            reason_mix=reason_mix,
            log_latency_mean=log_latency_mean,
            log_latency_sigma=log_latency_sigma,
        )

    def predict(self, frame: pd.DataFrame) -> Predictions:
        c = _codes(frame)
        base = self.cell_rate[c["issuer"], c["method"], c["gateway"], c["block"]]
        issuer_ratio = (
            frame["issuer_health"].to_numpy() / self.normal_issuer_health[c["issuer"], c["method"]]
        )
        gateway_ratio = (
            frame["gateway_health"].to_numpy() / self.normal_gateway_health[c["gateway"]]
        )
        p = base * issuer_ratio * gateway_ratio * self.band_factor[c["band"]]
        return Predictions(
            p_success=np.clip(p, *P_BOUNDS),
            reason_probs=self.reason_mix[c["method"], c["gateway_low"], c["issuer_low"]],
            latency_median_ms=np.exp(self.log_latency_mean[c["method"], c["gateway"], c["band"]]),
            latency_log_sigma=self.log_latency_sigma[c["method"]],
        )

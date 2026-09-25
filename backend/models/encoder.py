"""Shared payment representation: every transaction is a set of field tokens.

Each field becomes one token: its value embedding plus a field-identity embedding.
- A categorical value is looked up in the field's own vocabulary.
- A numeric value, standardized on the training data, goes through a small per-field MLP. The
  MLP is piecewise linear, so values outside the training range still map somewhere sensible.
- A masked (or unknown) field uses the field's mask embedding instead of its value.

Self-attention blocks with no positional encoding (SAB) mix the tokens, so the output does not
depend on field order. Pooling by multihead attention (PMA) reduces the set to one vector. The
fields are few (about 20), so full attention is cheap and inducing points (ISAB) are unnecessary.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from backend.models.features import CATEGORIES, NUMERIC

N_CATEGORICAL = len(CATEGORIES)
N_NUMERIC = len(NUMERIC)
N_FIELDS = N_CATEGORICAL + N_NUMERIC
NUMERIC_HIDDEN = 16


@dataclass(frozen=True)
class FieldTokenizer:
    """Turns `FeatureBuilder` output into tensors; its statistics come from training data only."""

    numeric_mean: np.ndarray  # [numeric field]
    numeric_std: np.ndarray
    # Quantile bin edges per numeric field, the classes masked-field pretraining predicts.
    numeric_bin_edges: tuple[np.ndarray, ...]

    @classmethod
    def fit(cls, x: pd.DataFrame, n_bins: int) -> "FieldTokenizer":
        numeric = x[list(NUMERIC)].to_numpy(dtype=float)
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = tuple(np.unique(np.quantile(col, quantiles)) for col in numeric.T)
        return cls(
            numeric_mean=numeric.mean(axis=0),
            numeric_std=np.maximum(numeric.std(axis=0), 1e-6),
            numeric_bin_edges=edges,
        )

    @property
    def vocab_sizes(self) -> list[int]:
        return [len(v) for v in CATEGORIES.values()]

    @property
    def numeric_bin_counts(self) -> list[int]:
        return [len(e) + 1 for e in self.numeric_bin_edges]

    def __call__(self, x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        codes = np.stack([x[name].cat.codes.to_numpy() for name in CATEGORIES], axis=1)
        if (codes < 0).any():
            raise ValueError("categorical value outside the catalog")
        numeric = (x[list(NUMERIC)].to_numpy(dtype=float) - self.numeric_mean) / self.numeric_std
        return torch.from_numpy(codes.astype(np.int64)), torch.from_numpy(
            numeric.astype(np.float32)
        )

    def numeric_bins(self, x: pd.DataFrame) -> torch.Tensor:
        raw = x[list(NUMERIC)].to_numpy(dtype=float)
        bins = [
            np.searchsorted(edges, col, side="right")
            for edges, col in zip(self.numeric_bin_edges, raw.T, strict=True)
        ]
        return torch.from_numpy(np.stack(bins, axis=1).astype(np.int64))


class SetBlock(nn.Module):
    """Pre-norm multihead self-attention block (SAB) followed by a feed-forward layer."""

    def __init__(self, d_model: int, n_heads: int, ffn_multiplier: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.norm1(h)
        h = h + self.dropout(self.attn(q, q, q, need_weights=False)[0])
        return h + self.dropout(self.ffn(self.norm2(h)))


class PoolingByAttention(nn.Module):
    """PMA with one learned seed vector: a weighted summary of the token set."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        kv = self.norm(h)
        seed = self.seed.expand(h.shape[0], -1, -1)
        return self.attn(seed, kv, kv, need_weights=False)[0].squeeze(1)


class PaymentEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: list[int],
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        # One table for all categorical fields; each field's slice ends with its mask row.
        offsets = np.concatenate([[0], np.cumsum([v + 1 for v in vocab_sizes])[:-1]])
        self.register_buffer("cat_offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("mask_codes", torch.tensor(vocab_sizes, dtype=torch.long))
        self.cat_embedding = nn.Embedding(sum(v + 1 for v in vocab_sizes), d_model)
        self.num_w1 = nn.Parameter(torch.randn(N_NUMERIC, NUMERIC_HIDDEN))
        self.num_b1 = nn.Parameter(torch.zeros(N_NUMERIC, NUMERIC_HIDDEN))
        self.num_w2 = nn.Parameter(torch.randn(N_NUMERIC, NUMERIC_HIDDEN, d_model) * 0.1)
        self.num_mask = nn.Parameter(torch.randn(N_NUMERIC, d_model) * 0.02)
        self.field_embedding = nn.Parameter(torch.randn(N_FIELDS, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            SetBlock(d_model, n_heads, ffn_multiplier, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool = PoolingByAttention(d_model, n_heads)

    def tokens(
        self, codes: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Contextual token states [batch, field, d]. `mask` [batch, field] hides fields."""
        if mask is not None:
            codes = torch.where(mask[:, :N_CATEGORICAL], self.mask_codes, codes)
        cat = self.cat_embedding(codes + self.cat_offsets)
        hidden = torch.relu(numeric.unsqueeze(-1) * self.num_w1 + self.num_b1)
        num = torch.einsum("bfh,fhd->bfd", hidden, self.num_w2)
        if mask is not None:
            num = torch.where(mask[:, N_CATEGORICAL:, None], self.num_mask, num)
        h = torch.cat([cat, num], dim=1) + self.field_embedding
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(
        self, codes: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """One representation vector per transaction, [batch, d]."""
        return self.pool(self.tokens(codes, numeric, mask))

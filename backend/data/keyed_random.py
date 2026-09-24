"""Counter-based randomness: every draw is a pure function of (seed, key, stream).

The outcome process draws its randomness from the transaction id rather than from a sequential
generator, so a transaction gets the same draws no matter which other transactions exist. That
is what lets the oracle replay a counterfactual on identical noise.
"""

import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_STREAM_MULT = 0xD1B54A32D192ED03
_MASK_64 = (1 << 64) - 1


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX_1
    x = (x ^ (x >> np.uint64(27))) * _MIX_2
    return x ^ (x >> np.uint64(31))


def keyed_uniform(seed: int, keys: np.ndarray, stream: int) -> np.ndarray:
    """Uniform [0, 1) draws, one per key, independent across streams."""
    state = _splitmix64(np.full(len(keys), seed, dtype=np.uint64))
    # Wrapping multiply in Python ints: numpy warns on uint64 scalar overflow.
    state = _splitmix64(state ^ np.uint64((stream * _STREAM_MULT) & _MASK_64))
    state = _splitmix64(state ^ np.asarray(keys, dtype=np.uint64))
    return (state >> np.uint64(11)).astype(np.float64) * 2.0**-53


def keyed_normal(seed: int, keys: np.ndarray, stream: int) -> np.ndarray:
    """Standard normal draws via Box-Muller over two keyed uniform streams."""
    u1 = keyed_uniform(seed, keys, stream)
    u2 = keyed_uniform(seed, keys, stream + 1)
    return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

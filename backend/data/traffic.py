from datetime import date, timedelta

import numpy as np

from backend.config import EcosystemConfig

# Relative transaction intensity per hour of day: quiet overnight, lunch bump, evening peak.
HOURLY_PROFILE = (
    0.25, 0.15, 0.10, 0.08, 0.08, 0.12,
    0.30, 0.55, 0.80, 0.95, 1.00, 1.05,
    1.20, 1.15, 1.00, 0.95, 0.95, 1.05,
    1.25, 1.45, 1.55, 1.45, 1.05, 0.55,
)  # fmt: skip
PEAK_HOURS = (18, 19, 20, 21)
WEEKEND_MULTIPLIER = 1.10
MINUTES_PER_DAY = 1440


def pick_festival_dates(config: EcosystemConfig, rng: np.random.Generator) -> list[date]:
    offsets = rng.choice(config.n_days, size=config.n_festival_days, replace=False)
    return sorted(config.start_date + timedelta(days=int(d)) for d in offsets)


def minute_intensity(config: EcosystemConfig, festival_dates: list[date]) -> np.ndarray:
    """Relative expected volume for every minute in the simulated period."""
    hourly = np.repeat(np.asarray(HOURLY_PROFILE), 60)
    day_factors = np.ones(config.n_days)
    for d in range(config.n_days):
        day = config.start_date + timedelta(days=d)
        if day.weekday() >= 5:
            day_factors[d] *= WEEKEND_MULTIPLIER
        if day in festival_dates:
            day_factors[d] *= config.festival_traffic_multiplier
    return (day_factors[:, None] * hourly[None, :]).ravel()


def sample_timestamps(
    config: EcosystemConfig, festival_dates: list[date], rng: np.random.Generator
) -> np.ndarray:
    intensity = minute_intensity(config, festival_dates)
    expected = config.target_transactions * intensity / intensity.sum()
    counts = rng.poisson(expected)

    minute_index = np.repeat(np.arange(len(counts)), counts)
    offset_ms = minute_index * 60_000 + rng.integers(0, 60_000, size=len(minute_index))
    start = np.datetime64(config.start_date, "ms")
    return start + np.sort(offset_ms).astype("timedelta64[ms]")

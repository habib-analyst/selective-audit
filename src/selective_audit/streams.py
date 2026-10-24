"""Synthetic audit streams with planted segment structure.

A stream is T items. Each item has a segment label (observed by the auditor
for every item) and a hidden violation label. Segment violation rates vary by
regime:

- ``static``: heterogeneous rates, constant over time.
- ``burst``: low base rates; randomly chosen segments spike for short windows.
- ``drift``: the rate vector is permuted at fixed change points, so the "hot"
  segments move over time.

Rates are returned alongside the stream so the oracle and the analysis know
the ground truth (policies never see them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SEGMENTS = 8
TRAFFIC_SHARES = np.array([0.22, 0.18, 0.16, 0.13, 0.11, 0.09, 0.06, 0.05])
STATIC_RATES = np.array([0.25, 0.15, 0.10, 0.06, 0.04, 0.02, 0.01, 0.005])
BURST_BASE_RATES = np.array([0.030, 0.025, 0.020, 0.020, 0.015, 0.010, 0.010, 0.005])

REGIMES = ("static", "burst", "drift")


@dataclass
class Stream:
    segments: np.ndarray  # (T,) int segment id per item
    violations: np.ndarray  # (T,) 0/1 hidden label
    rates: np.ndarray  # (T, K) true violation rate of each item's segment over time
    regime: str
    seed: int

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def true_rate_of_item(self) -> np.ndarray:
        """True violation probability of each item, given its segment and time."""
        return self.rates[np.arange(len(self)), self.segments]


def _rates_static(T: int, rng: np.random.Generator) -> np.ndarray:
    return np.tile(STATIC_RATES, (T, 1))


def _rates_burst(T: int, rng: np.random.Generator) -> np.ndarray:
    # Bursts are sized to be *in-principle detectable* online: a burst lasting
    # 1000-2000 stream items puts ~50-440 items of the bursting segment in the
    # window (share-dependent), so a policy that watches every segment gets
    # enough in-window audits to react. (An earlier 300-800 item design put
    # as few as ~15 segment items in a window; no online policy could react
    # inside it, and the regime measured only parking luck.)
    rates = np.tile(BURST_BASE_RATES, (T, 1))
    t = 0
    while True:
        t += int(rng.integers(1200, 2500))  # gap between burst starts
        if t >= T:
            break
        seg = int(rng.integers(0, N_SEGMENTS))
        dur = int(rng.integers(1000, 2000))
        end = min(T, t + dur)
        rates[t:end, seg] = min(0.65, BURST_BASE_RATES[seg] * 15.0)
        t = end
    return rates


def _rates_drift(T: int, rng: np.random.Generator) -> np.ndarray:
    rates = np.empty((T, N_SEGMENTS))
    change_points = [0, T // 3, 2 * T // 3, T]
    current = STATIC_RATES.copy()
    for i in range(3):
        rates[change_points[i] : change_points[i + 1]] = current
        # derangement-style reshuffle: rotate by a random nonzero shift and
        # swap two positions, so the identity of hot segments really changes
        shift = int(rng.integers(1, N_SEGMENTS))
        current = np.roll(current, shift)
        a, b = rng.choice(N_SEGMENTS, size=2, replace=False)
        current[a], current[b] = current[b], current[a]
    return rates


_RATE_BUILDERS = {"static": _rates_static, "burst": _rates_burst, "drift": _rates_drift}


def generate_stream(regime: str, T: int = 20_000, seed: int = 0) -> Stream:
    if regime not in _RATE_BUILDERS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    rng = np.random.default_rng(seed)
    rates = _RATE_BUILDERS[regime](T, rng)
    segments = rng.choice(N_SEGMENTS, size=T, p=TRAFFIC_SHARES)
    p_item = rates[np.arange(T), segments]
    violations = (rng.random(T) < p_item).astype(np.int8)
    return Stream(
        segments=segments.astype(np.int64),
        violations=violations,
        rates=rates,
        regime=regime,
        seed=seed,
    )

"""Audit-allocation policies.

Every policy exposes the same interface to the engine:

- ``observe(segment, audited, violation)`` — called once per stream item, in
  order. ``violation`` is the revealed label if the item was audited, else
  ``None``. Segment labels are observed for all items.
- ``weights(rng)`` — called at each allocation epoch; returns a non-negative
  weight per segment. The engine converts weights to per-item audit
  probabilities with budget pacing, so policies compete at exactly equal
  budget.

Policies:

- :class:`UniformPolicy` — audit uniformly at random (the status quo).
- :class:`FixedProportionalPolicy` — pilot-estimate segment rates once, then
  allocate proportionally to those frozen estimates ("estimate once, fix").
- :class:`EpsilonGreedyPolicy` — classic epsilon-greedy bandit on
  undiscounted empirical rates.
- :class:`DriftUCBPolicy` — the drift-aware policy: UCB scores over
  per-segment rate estimates whose sufficient statistics decay
  exponentially (forgetting), with an empirical-Bayes prior anchored to the
  global audited rate, plus a per-segment change-point reset that wipes a
  segment's statistics when its recent audited outcomes contradict them.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .streams import N_SEGMENTS


class Policy:
    name = "base"

    def observe(self, segment: int, audited: bool, violation: int | None) -> None:
        pass

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class UniformPolicy(Policy):
    name = "uniform"

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        return np.ones(N_SEGMENTS)


class FixedProportionalPolicy(Policy):
    """Estimate rates from a uniform pilot, then freeze allocation ∝ rates."""

    name = "fixed_proportional"

    def __init__(self, pilot_steps: int = 1000):
        self.pilot_steps = pilot_steps
        self.items_seen = 0
        self.pilot_counts = np.zeros(N_SEGMENTS)
        self.pilot_hits = np.zeros(N_SEGMENTS)
        self._frozen: np.ndarray | None = None

    def observe(self, segment: int, audited: bool, violation: int | None) -> None:
        if self.items_seen < self.pilot_steps and audited:
            self.pilot_counts[segment] += 1
            self.pilot_hits[segment] += int(violation or 0)
        self.items_seen += 1

    def _freeze(self) -> np.ndarray:
        total_c = self.pilot_counts.sum()
        total_h = self.pilot_hits.sum()
        global_rate = (total_h + 1.0) / (total_c + 2.0)
        rates = np.where(
            self.pilot_counts > 0,
            (self.pilot_hits + 1.0) / (self.pilot_counts + 2.0),
            global_rate,
        )
        return rates

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        if self.items_seen < self.pilot_steps:
            return np.ones(N_SEGMENTS)
        if self._frozen is None:
            self._frozen = self._freeze()
        return self._frozen.copy()


class EpsilonGreedyPolicy(Policy):
    """Epsilon-greedy bandit over undiscounted per-segment empirical rates."""

    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(N_SEGMENTS)
        self.hits = np.zeros(N_SEGMENTS)

    def observe(self, segment: int, audited: bool, violation: int | None) -> None:
        if audited:
            self.counts[segment] += 1
            self.hits[segment] += int(violation or 0)

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        global_rate = (self.hits.sum() + 1.0) / (self.counts.sum() + 2.0)
        rates = np.where(
            self.counts > 0,
            (self.hits + 1.0) / (self.counts + 2.0),
            global_rate,
        )
        w = np.full(N_SEGMENTS, self.epsilon / N_SEGMENTS)
        w[int(np.argmax(rates))] += 1.0 - self.epsilon
        return w


class DriftUCBPolicy(Policy):
    """Drift-aware allocation: discounted UCB + change-point reset.

    Per-segment sufficient statistics (hits, counts) decay by ``gamma`` per
    elapsed item, so stale evidence fades with an effective memory of
    ~1/(1-gamma) items (~5000 items at the default: on the order of a few
    hundred audits at a 10% budget). Forgetting is deliberately the *slow*
    path — with audit budgets of 1-10%, aggressive forgetting leaves too few
    effective samples to rank segments at all (measured). Abrupt change is
    handled by the *fast* path, the change-point reset below.
    Scores are posterior means under an empirical-Bayes
    prior (prior mean = global decayed audited rate, strength ``prior_k``)
    plus a UCB exploration bonus on the *effective* (decayed) counts:

        score_s = mean_s + ucb_c * sqrt(ln(1 + items_seen) / (n_s + prior_k))

    Additionally, each segment keeps its last ``window`` audited outcomes;
    when that window's hit rate contradicts the segment's current estimate by
    more than ``z_thresh`` standard errors, the segment's statistics are
    reset to the prior — a burst or drift point in one segment does not have
    to wait for slow decay to be believed.

    Design note: an earlier variant used Thompson sampling over the same
    discounted posteriors. Under persistent forgetting the posteriors stay
    wide forever, and sampled upper tails diverted a large share of the
    budget to nearly-unexplored segments (measured: at/below uniform). The
    deterministic UCB score keeps exploration bounded and calibrated.
    """

    name = "drift_ucb"

    def __init__(
        self,
        gamma: float = 0.9998,
        prior_k: float = 10.0,
        ucb_c: float = 0.06,
        temperature: float = 0.02,
        floor_frac: float = 0.5,
        window: int = 25,
        z_thresh: float = 3.5,
    ):
        self.gamma = gamma
        self.prior_k = prior_k
        self.ucb_c = ucb_c
        self.temperature = temperature
        self.floor_frac = floor_frac
        self.window = window
        self.z_thresh = z_thresh
        self.counts = np.zeros(N_SEGMENTS)  # decayed effective counts
        self.hits = np.zeros(N_SEGMENTS)  # decayed effective hits
        self._pend_counts = np.zeros(N_SEGMENTS)
        self._pend_hits = np.zeros(N_SEGMENTS)
        self._items_since_fold = 0
        self.items_seen = 0
        self._arrivals = np.zeros(N_SEGMENTS)
        self._recent: list[deque] = [deque(maxlen=window) for _ in range(N_SEGMENTS)]
        self.resets = 0

    def observe(self, segment: int, audited: bool, violation: int | None) -> None:
        self._items_since_fold += 1
        self.items_seen += 1
        self._arrivals[segment] += 1
        if audited:
            self._pend_counts[segment] += 1
            self._pend_hits[segment] += int(violation or 0)
            self._recent[segment].append(int(violation or 0))

    def _fold(self) -> None:
        if self._items_since_fold:
            decay = self.gamma**self._items_since_fold
            self.counts *= decay
            self.hits *= decay
            self._items_since_fold = 0
        self.counts += self._pend_counts
        self.hits += self._pend_hits
        self._pend_counts[:] = 0
        self._pend_hits[:] = 0
        self._maybe_reset()

    def _prior_mean(self) -> float:
        return (self.hits.sum() + 1.0) / (self.counts.sum() + 2.0)

    def _est_mean(self, s: int, prior_mean: float) -> float:
        return (self.hits[s] + self.prior_k * prior_mean) / (
            self.counts[s] + self.prior_k
        )

    def _maybe_reset(self) -> None:
        prior_mean = self._prior_mean()
        for s in range(N_SEGMENTS):
            recent = self._recent[s]
            if len(recent) < self.window:
                continue
            est = self._est_mean(s, prior_mean)
            recent_mean = float(np.mean(recent))
            var = max(est * (1.0 - est), 1e-4)
            z = abs(recent_mean - est) / np.sqrt(var / self.window)
            if z > self.z_thresh:
                self.counts[s] = 0.0
                self.hits[s] = 0.0
                recent.clear()
                self.resets += 1

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        self._fold()
        prior_mean = self._prior_mean()
        means = np.array([self._est_mean(s, prior_mean) for s in range(N_SEGMENTS)])
        bonus = self.ucb_c * np.sqrt(
            np.log1p(self.items_seen) / (self.counts + self.prior_k)
        )
        scores = means + bonus
        # Allocate by softmax over scores. Detection-optimal allocation for
        # a known rate vector is a corner solution (fill from the hottest
        # segment down), not proportional-to-rate; a low-temperature softmax
        # approximates that fill while keeping allocation smooth in the
        # scores, so exploration bonuses and estimate updates shift budget
        # continuously instead of flipping a hard argmax.
        w = np.exp(scores / self.temperature)
        # Coverage floor: no segment's audit probability falls below
        # floor_frac x the uniform probability. Without a floor, a cold
        # segment is audited so rarely that a burst inside it is nearly
        # undetectable (too few in-window audits to move the estimate or
        # trip the change-point reset) — measured: pure softmax scored
        # *below uniform* on the burst regime. The floor is also standard
        # audit practice: every segment keeps some coverage.
        shares = (self._arrivals + 1.0) / (self._arrivals.sum() + N_SEGMENTS)
        denom = float(np.dot(shares, w))
        if denom > 0:
            w = np.maximum(w, self.floor_frac * denom)
        return w


POLICIES = {
    "uniform": UniformPolicy,
    "fixed_proportional": FixedProportionalPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
    "drift_ucb": DriftUCBPolicy,
}

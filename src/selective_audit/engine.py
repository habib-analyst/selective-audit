"""Simulation engine: run a policy over a stream at an exact audit budget.

Budget pacing: at each allocation epoch the engine converts the policy's
per-segment weights ``w`` into per-item audit probabilities

    q_s = clip(b * w_s / sum_j(share_j * w_j), 0, 1),   b = B_rem / N_rem

where ``share`` is the empirical traffic mix so far. Pacing guarantees every
policy spends (up to sampling noise and clipping) the same total budget B —
differences in detections are therefore allocation quality, not budget size.
Hard rules: never audit with B_rem = 0; if B_rem >= N_rem, audit everything
left (end-game catch-up).
"""

from __future__ import annotations

import numpy as np

from .policies import Policy
from .streams import N_SEGMENTS, Stream

EPOCH = 20


def run_policy(
    stream: Stream,
    policy: Policy,
    budget_frac: float,
    rng_seed: int = 12345,
) -> dict:
    T = len(stream)
    budget = int(budget_frac * T)
    rng = np.random.default_rng(rng_seed)
    segments = stream.segments
    violations = stream.violations

    b_rem = budget
    detections = 0
    audits = 0
    share_counts = np.zeros(N_SEGMENTS)
    t = 0
    while t < T:
        epoch_end = min(T, t + EPOCH)
        n_rem = T - t
        if b_rem <= 0:
            q = np.zeros(N_SEGMENTS)
        elif b_rem >= n_rem:
            q = np.ones(N_SEGMENTS)
        else:
            w = policy.weights(rng)
            shares = (share_counts + 1.0) / (share_counts.sum() + N_SEGMENTS)
            denom = float(np.dot(shares, w))
            if denom <= 0:
                q = np.full(N_SEGMENTS, b_rem / n_rem)
            else:
                q = np.clip((b_rem / n_rem) * w / denom, 0.0, 1.0)
        draws = rng.random(epoch_end - t)
        for i in range(t, epoch_end):
            seg = int(segments[i])
            share_counts[seg] += 1
            audited = False
            if b_rem > 0 and draws[i - t] < q[seg]:
                audited = True
                b_rem -= 1
                audits += 1
                detections += int(violations[i])
            policy.observe(seg, audited, int(violations[i]) if audited else None)
        t = epoch_end
    return {
        "detections": detections,
        "audits_used": audits,
        "budget": budget,
        "total_violations": int(violations.sum()),
    }


def oracle_detections(stream: Stream, budget_frac: float) -> int:
    """Clairvoyant reference: audit the B items with the highest true rates.

    The oracle knows the true segment rate of every item (but not the realized
    labels — it ranks by probability). No online policy can systematically
    beat it; regret = oracle - policy detections.
    """
    T = len(stream)
    budget = int(budget_frac * T)
    p = stream.true_rate_of_item
    top = np.argpartition(p, T - budget)[T - budget :]
    return int(stream.violations[top].sum())


def oracle_expected(stream: Stream, budget_frac: float) -> float:
    """Expected detections of the known-rates oracle: sum of the B largest
    true rates. Smooth (label-noise-free) regret reference."""
    T = len(stream)
    budget = int(budget_frac * T)
    p = stream.true_rate_of_item
    top = np.argpartition(p, T - budget)[T - budget :]
    return float(p[top].sum())

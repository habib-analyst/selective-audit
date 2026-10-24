"""Tests for selective-audit: streams, budget compliance, policy behavior."""

import numpy as np
import pytest

from selective_audit import (
    POLICIES,
    DriftUCBPolicy,
    FixedProportionalPolicy,
    generate_stream,
    oracle_expected,
    run_policy,
)
from selective_audit.streams import N_SEGMENTS, STATIC_RATES


# ---------- stream generator ----------


def test_stream_determinism():
    a = generate_stream("drift", T=2000, seed=42)
    b = generate_stream("drift", T=2000, seed=42)
    assert np.array_equal(a.segments, b.segments)
    assert np.array_equal(a.violations, b.violations)
    c = generate_stream("drift", T=2000, seed=43)
    assert not np.array_equal(a.violations, c.violations)


def test_static_rates_constant():
    s = generate_stream("static", T=500, seed=0)
    assert np.allclose(s.rates, STATIC_RATES)


def test_drift_rates_are_permutations_that_change():
    s = generate_stream("drift", T=3000, seed=0)
    for t in (0, 1500, 2999):
        assert np.allclose(np.sort(s.rates[t]), np.sort(STATIC_RATES))
    assert not np.allclose(s.rates[0], s.rates[1500])
    assert not np.allclose(s.rates[1500], s.rates[2999])


def test_burst_rates_spike_above_base():
    s = generate_stream("burst", T=20000, seed=1)
    assert s.rates.max() >= 0.30  # a real spike exists
    # and most rows are still at base rates (bursts are the exception)
    base = np.array([0.030, 0.025, 0.020, 0.020, 0.015, 0.010, 0.010, 0.005])
    frac_base = np.mean(np.all(np.isclose(s.rates, base), axis=1))
    assert frac_base > 0.5


def test_stream_labels_consistent_with_rates():
    s = generate_stream("static", T=20000, seed=3)
    # empirical violation rate of the hottest segment tracks its true rate
    mask = s.segments == 0
    assert abs(s.violations[mask].mean() - STATIC_RATES[0]) < 0.03


# ---------- engine / budget ----------


@pytest.mark.parametrize("policy_name", list(POLICIES))
def test_budget_compliance(policy_name):
    s = generate_stream("static", T=4000, seed=0)
    r = run_policy(s, POLICIES[policy_name](), 0.05, rng_seed=1)
    assert r["audits_used"] <= r["budget"]
    assert r["audits_used"] >= 0.90 * r["budget"]


def test_uniform_detection_rate_sanity():
    s = generate_stream("static", T=20000, seed=0)
    r = run_policy(s, POLICIES["uniform"](), 0.05, rng_seed=5)
    expected = r["budget"] * s.violations.mean()
    assert 0.6 * expected < r["detections"] < 1.5 * expected


def test_oracle_bounds_policies():
    s = generate_stream("static", T=8000, seed=0)
    ora = oracle_expected(s, 0.10)
    for cls in POLICIES.values():
        r = run_policy(s, cls(), 0.10, rng_seed=11)
        assert ora >= r["detections"] - 5


# ---------- headline behavior (fixed seeds, wide margins) ----------


def test_drift_ucb_beats_uniform_on_drift():
    du, un = [], []
    for seed in range(3):
        s = generate_stream("drift", T=8000, seed=seed)
        du.append(run_policy(s, DriftUCBPolicy(), 0.10, rng_seed=100 + seed)["detections"])
        un.append(run_policy(s, POLICIES["uniform"](), 0.10, rng_seed=100 + seed)["detections"])
    assert np.mean(du) > 1.15 * np.mean(un)


def test_drift_ucb_beats_uniform_on_burst():
    du, un = [], []
    for seed in range(3):
        s = generate_stream("burst", T=8000, seed=seed)
        du.append(run_policy(s, DriftUCBPolicy(), 0.05, rng_seed=100 + seed)["detections"])
        un.append(run_policy(s, POLICIES["uniform"](), 0.05, rng_seed=100 + seed)["detections"])
    assert np.mean(du) > 1.20 * np.mean(un)


# ---------- policy internals ----------


def test_change_point_reset_fires_on_flip():
    rng = np.random.default_rng(0)
    pol = DriftUCBPolicy()
    # segment 0 always violates for a long stretch...
    for i in range(300):
        pol.observe(0, True, 1)
        if i % 20 == 0:
            pol.weights(rng)
    # ...then flips to never violating
    for i in range(300):
        pol.observe(0, True, 0)
        if i % 20 == 0:
            pol.weights(rng)
    pol.weights(rng)
    assert pol.resets >= 1


def test_discount_decays_counts():
    rng = np.random.default_rng(0)
    pol = DriftUCBPolicy()
    for _ in range(100):
        pol.observe(1, True, 1)
    pol.weights(rng)
    before = pol.counts[1]
    assert before > 0
    for _ in range(20000):  # long silence: evidence should fade
        pol.observe(2, False, None)
    pol.weights(rng)
    assert pol.counts[1] < 0.25 * before


def test_fixed_proportional_freezes_after_pilot():
    rng = np.random.default_rng(0)
    pol = FixedProportionalPolicy(pilot_steps=100)
    for i in range(500):
        pol.observe(i % N_SEGMENTS, True, int(i % 3 == 0))
    w1 = pol.weights(rng)
    for i in range(500):
        pol.observe(i % N_SEGMENTS, True, 1)
    w2 = pol.weights(rng)
    assert np.array_equal(w1, w2)

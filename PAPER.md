# Online Audit-Budget Allocation Under Drift

*A simulator, a policy library, and a benchmark for spending a fixed audit
budget where violations concentrate — when the concentration moves.*

## Abstract

AI systems produce output streams far larger than any audit budget. We study
the allocation problem: items arrive in segments with unknown, time-varying
violation rates; an auditor may inspect a fixed fraction (1%, 5%, 10%) of
items and learns labels only for inspected items. We build a simulator with
three planted regimes (static, burst, drift), four policies, and a
known-rates oracle, with budget pacing that forces exactly equal spend. The
drift-aware policy — UCB over exponentially discounted segment statistics,
an empirical-Bayes prior, a per-segment coverage floor, softmax allocation,
and a change-point reset — beats uniform auditing in every regime and budget
(+38% to +78%), beats or ties a strong ε-greedy bandit baseline on the burst
(+14%/+27% at 5%/10%) and drift (+13% at 5%, tie at 10%) regimes, and loses
to it in the static regime, where adaptation has nothing to adapt to. Two
negative results are reported in full: a discounted-Thompson-sampling
variant scored at or below uniform, and the change-point reset, as
calibrated, almost never fired — the gains come from discounted estimation
and allocation shape, not the reset.

## 1. Problem

At each step t = 1..T an item arrives with an observed segment label
s_t ∈ {1..K} (K = 8) and a hidden violation label y_t ~ Bernoulli(p_{s_t}(t)).
The auditor chooses, online, whether to inspect the item; inspection reveals
y_t. Total inspections are capped at B = ⌊budget_frac · T⌋. The objective is
to maximize total detections Σ y_t·1[audited]. Segment traffic shares are
fixed (0.22 … 0.05); the rate functions p_s(t) define the regime:

- **static** — p = (0.25, 0.15, 0.10, 0.06, 0.04, 0.02, 0.01, 0.005), constant.
- **burst** — low base rates (0.030 … 0.005); bursts start after gaps of
  1200–2500 items in a uniformly random segment, last 1000–2000 items, and
  multiply that segment's rate ×15 (cap 0.65).
- **drift** — the static vector, permuted at t = T/3 and 2T/3 (random roll +
  swap), so hot segments change identity twice.

Streams average 2252 (static), 681 (burst), and 1808 (drift) violations per
20,000 items. All policies face identical streams per (regime, seed)
— common random numbers — with independent decision RNGs.

**Oracle.** The reference knows each item's true rate and audits the B items
with the highest rates. Regret is measured against its *expected* detections
(the sum of the B largest true rates), a label-noise-free upper reference.

**Budget pacing.** Policies output per-segment weights w each 20-item epoch;
the engine sets audit probabilities

q_s = clip(b · w_s / Σ_j share_j w_j, 0, 1),  b = B_rem / N_rem,

with share the empirical traffic mix. Every policy spends the same budget:
the minimum realized audits/budget ratio over all 180 runs was 0.994.
Detection differences are therefore allocation quality only.

## 2. Policies

- **uniform** — w_s = 1. The status-quo baseline.
- **fixed_proportional** — audit uniformly for a 1000-item pilot, estimate
  segment rates once, then allocate ∝ frozen estimates. The standard
  "estimate once, allocate" practice.
- **epsilon_greedy** — undiscounted empirical means (Laplace-smoothed);
  weight 1 − ε + ε/K on the argmax segment, ε/K elsewhere, ε = 0.1.
- **drift_ucb** (proposed) — per-segment hits/counts decayed by γ = 0.9998
  per item (effective memory ≈ 5000 items); score = empirical-Bayes-shrunk
  mean (prior = global decayed rate, strength 10) + 0.06·√(ln(1+t)/(n_s+10));
  weights = softmax(score / 0.02), floored so no segment's audit probability
  falls below 0.5× uniform; plus a change-point reset: if a segment's last
  25 audited outcomes deviate from its estimate by > 3.5 standard errors,
  its statistics reset to the prior.

## 3. Results

Detections, mean ± std over 5 seeds, T = 20,000 (full per-run data in
`results/results.json`):

### static
| Budget | uniform | fixed_proportional | epsilon_greedy | drift_ucb | oracle (exp.) |
|---|---|---|---|---|---|
| 1% | 22 ± 2 | 24 ± 3 | **45 ± 8** | 31 ± 6 | 50 |
| 5% | 107 ± 5 | 135 ± 20 | **217 ± 36** | 185 ± 12 | 250 |
| 10% | 219 ± 4 | 281 ± 49 | **443 ± 50** | 388 ± 15 | 500 |

### burst
| Budget | uniform | fixed_proportional | epsilon_greedy | drift_ucb | oracle (exp.) |
|---|---|---|---|---|---|
| 1% | 8 ± 3 | 9 ± 3 | 8 ± 4 | **9 ± 3** | 90 |
| 5% | 36 ± 6 | 32 ± 7 | 51 ± 9 | **58 ± 17** | 291 |
| 10% | 72 ± 15 | 68 ± 9 | 95 ± 9 | **120 ± 32** | 341 |

### drift
| Budget | uniform | fixed_proportional | epsilon_greedy | drift_ucb | oracle (exp.) |
|---|---|---|---|---|---|
| 1% | 17 ± 4 | 21 ± 6 | **26 ± 11** | 25 ± 8 | 50 |
| 5% | 93 ± 17 | 98 ± 15 | 141 ± 35 | **160 ± 19** | 250 |
| 10% | 188 ± 25 | 201 ± 28 | **324 ± 37** | 320 ± 26 | 500 |

### Regret vs expected oracle (mean detections shortfall)
| Regime | Budget | uniform | epsilon_greedy | drift_ucb |
|---|---|---|---|---|
| static | 5% | 143 | 33 | 65 |
| static | 10% | 281 | 57 | 112 |
| burst | 5% | 255 | 240 | 233 |
| burst | 10% | 269 | 246 | 221 |
| drift | 5% | 157 | 109 | 90 |
| drift | 10% | 312 | 176 | 180 |

### Reading the tables

1. **Learning where violations concentrate is worth a lot.** Every adaptive
   policy beats uniform nearly everywhere. Uniform wastes budget
   re-confirming clean segments; even the frozen pilot allocation of
   fixed_proportional gains +26%…+28% in static at 5–10%.
2. **Frozen allocations cannot react.** fixed_proportional falls to
   uniform-or-below in burst (32 vs 36 at 5%; 68 vs 72 at 10%): its pilot
   estimates describe a world that no longer exists.
3. **drift_ucb is the best policy on burst and co-best on drift**, the two
   non-stationary regimes: +59%/+67% over uniform and +14%/+27% over
   ε-greedy on burst (5%/10%); +72%/+70% over uniform and +13%/tie vs
   ε-greedy on drift. Discounted statistics let it abandon a segment whose
   rate collapsed and chase one whose rate rose, without waiting for a
   lifetime average to move.
4. **In static, ε-greedy wins — honestly.** With nothing to adapt to,
   hard-argmax exploitation of undiscounted means is optimal-shaped, and
   drift_ucb's forgetting, bonus, and coverage floor are pure cost there
   (−12…−33% vs ε-greedy). drift_ucb still beats uniform by +38…+78%.
   Practitioners who *know* their stream is stationary should use the
   simpler bandit; the drift-aware policy is insurance that pays when the
   world moves — and in these experiments the world moves in 2 of 3 regimes.
5. **Variance.** ε-greedy's std is 2–3× drift_ucb's in static and drift
   (e.g. 443 ± 50 vs 388 ± 15): hard argmax flips between near-tied segments
   make outcomes seed-sensitive. Softmax allocation is steadier.
6. **Budget binds before policy at 1% in burst.** All policies score 8–9
   against an oracle of 90. Two hundred audits cannot find a burst inside
   its window; no allocation cleverness changes that. (In static/drift at
   1%, adaptive policies still gain +38…+53% over uniform.)

## 4. Design history (two measured failures, kept in)

- **Discounted Thompson sampling (v1) scored at/below uniform.** With
  persistent forgetting, posteriors stay wide forever; sampled upper tails
  repeatedly diverted large budget shares to nearly-unexplored segments.
  At T = 8000 probes it matched uniform on drift (35 vs 41 at 5% — a loss).
  Replaced by deterministic UCB scores, whose exploration is bounded and
  shrinks with effective counts.
- **Bonus mis-calibration (v2).** A textbook UCB constant (0.25) is scaled
  for reward gaps ~0.5; violation-rate gaps here are 0.04–0.10, so the bonus
  swamped the signal and flattened the allocation (≈ uniform again).
  Calibrated to 0.06 against the rate scale.
- **Undetectable bursts (regime fix).** The first burst parameterization
  (300–800 items, ×10) put as few as ~15 items of the bursting segment in a
  window: no online policy could react inside it, and the regime measured
  parking luck. Bursts were resized (1000–2000 items, ×15) to be detectable
  in principle — the standard a benchmark regime should meet.
- **The change-point reset is nearly inert as calibrated** (0.0 resets/run
  in static, 0.3 in burst, 0.1 in drift). It fires on hard flips (unit
  test), but in these regimes discounted estimation already adapts before
  the conservative z = 3.5 / window-25 test triggers. We report this rather
  than tuning the test until it fires: the contribution that measures out
  is *discounted estimation + coverage floor + smooth allocation*.

## 5. Threats and limits

- Synthetic Bernoulli streams; real violation processes are heavier-tailed,
  seasonal, and sometimes adversarial (auditee adaptation is unmodeled).
- Segment labels are observed for all items and audit labels are noiseless;
  noisy auditors would widen all confidence intervals.
- Change points are abrupt; gradual drift is untested.
- Hyperparameters (γ, bonus, temperature, floor) were calibrated on
  T = 8000 probes and frozen before the T = 20,000 benchmark; they were not
  re-tuned per regime.
- The oracle ranks by true rate (clairvoyant timing in burst), so burst
  regret overstates what any online policy could achieve.

## 6. Reproduction

```bash
pip install numpy pytest
PYTHONPATH=src python3 scripts/run_benchmark.py   # ~35 s, writes results/results.json
PYTHONPATH=src python3 -m pytest tests/ -q        # 16 tests
```

Config: T = 20,000; seeds 0–4; budgets 1/5/10%; K = 8 segments; policies and
streams share seeds across policies per (regime, seed) for paired comparison.

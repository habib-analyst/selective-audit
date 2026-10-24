# selective-audit

**Online audit-budget allocation under drift: learn where violations concentrate, and spend the audit budget there.**

An AI system emits a stream of outputs. An auditor (human or automated) can
inspect only a small fraction — 1–10%. Violations are not spread evenly: they
concentrate in segments (model versions, locales, tools, user cohorts), and
the concentration *moves* — a bad deploy creates a burst in one segment, a
data shift moves the hot segment elsewhere. Uniform random auditing, the
status quo, spends most of its budget re-confirming clean segments.

This repo is a simulator + policy library + benchmark for that problem:

- **Simulator** — synthetic streams (T = 20,000 items, 8 segments) with planted
  structure in three regimes: `static` (fixed heterogeneous rates), `burst`
  (low base rates + strong temporary spikes in random segments), `drift`
  (the rate vector is permuted at change points, so hot segments move).
- **Policies** — `uniform` (random auditing), `fixed_proportional`
  (pilot-estimate rates once, allocate ∝ estimates, frozen), `epsilon_greedy`
  (classic ε-greedy bandit, ε = 0.1, undiscounted), and `drift_ucb`, the
  drift-aware policy: UCB scores over **exponentially discounted**
  per-segment statistics with an empirical-Bayes prior, a **coverage floor**
  (no segment drops below 0.5× the uniform audit probability), softmax
  allocation, and a per-segment **change-point reset**.
- **Benchmark** — equal budgets enforced by pacing (every policy spends the
  same number of audits, min realized/budget ratio 0.994), paired streams
  across policies, regret measured against a known-rates oracle.

## Headline results (detections, mean ± std over 5 seeds, T = 20,000)

| Regime | Budget | uniform | fixed_proportional | epsilon_greedy | **drift_ucb** | oracle (exp.) |
|---|---|---|---|---|---|---|
| static | 1% | 22 ± 2 | 24 ± 3 | **45 ± 8** | 31 ± 6 | 50 |
| static | 5% | 107 ± 5 | 135 ± 20 | **217 ± 36** | 185 ± 12 | 250 |
| static | 10% | 219 ± 4 | 281 ± 49 | **443 ± 50** | 388 ± 15 | 500 |
| burst | 1% | 8 ± 3 | 9 ± 3 | 8 ± 4 | **9 ± 3** | 90 |
| burst | 5% | 36 ± 6 | 32 ± 7 | 51 ± 9 | **58 ± 17** | 291 |
| burst | 10% | 72 ± 15 | 68 ± 9 | 95 ± 9 | **120 ± 32** | 341 |
| drift | 1% | 17 ± 4 | 21 ± 6 | **26 ± 11** | 25 ± 8 | 50 |
| drift | 5% | 93 ± 17 | 98 ± 15 | 141 ± 35 | **160 ± 19** | 250 |
| drift | 10% | 188 ± 25 | 201 ± 28 | **324 ± 37** | 320 ± 26 | 500 |

**What this shows**

- `drift_ucb` beats uniform auditing **everywhere**: +38…+78% in static,
  +59…+67% in burst (5–10%), +70…+72% in drift (5–10%).
- On the regimes it was built for it matches or beats the strong bandit
  baseline: **best on burst** (+14% / +27% over ε-greedy at 5% / 10%),
  tied on drift at 10%, ahead at 5%.
- ε-greedy also has 2–3× the variance of `drift_ucb` in static/drift —
  hard argmax flips are brittle.

## Honest limits (read before citing)

- **In the static regime, ε-greedy wins and it is not close at 1%**
  (45 vs 31). When nothing ever changes, undiscounted cumulative means +
  hard argmax exploitation is the right tool; the drift machinery
  (forgetting, exploration bonus, coverage floor) is insurance you pay for.
  `drift_ucb` still beats uniform by +38…+78% there.
- **At a 1% budget in the burst regime, nobody catches bursts** (all
  policies ≈ 8–9 detections vs oracle 90): 200 audits over 20,000 items
  cannot statistically detect a burst inside its window. Budget size, not
  policy choice, is the binding constraint there.
- **The change-point reset almost never fired** (0.0–0.3 resets per run at
  window 25, z = 3.5). The measured burst/drift advantage comes from the
  discounted statistics + UCB/softmax allocation + coverage floor, not the
  reset. The reset is unit-tested to fire on a hard flip, but as calibrated
  it is a safety net, not the engine.
- Synthetic streams only; segment labels are assumed observed for all items
  and violation labels noiseless when audited. Rates are Bernoulli with
  abrupt change points; gradual drift and adversarial adaptation are future
  work. Two design iterations are documented in PAPER.md (a Thompson-
  sampling variant that scored at/below uniform, and a burst regime whose
  first parameterization was undetectable-in-principle and was resized).

## Quickstart

```bash
pip install numpy pytest        # only dependencies
PYTHONPATH=src python3 scripts/run_benchmark.py   # writes results/results.json (~35 s)
PYTHONPATH=src python3 -m pytest tests/ -q        # 16 tests
```

## Layout

- `src/selective_audit/streams.py` — stream generator (3 regimes)
- `src/selective_audit/policies.py` — the four policies (one interface)
- `src/selective_audit/engine.py` — budget-paced simulation + oracle
- `scripts/run_benchmark.py` — full benchmark → `results/results.json`
- `PAPER.md` — full method, per-regime tables, design history, threats
- `KILLGATE.md` — prior-art search and why this shelf was empty

## Relation to prior work

Inspired by Gibbs et al., *Reliable AI-Assisted Decisions via Selective
Auditing* (arXiv:2610.31662), which decides per item — from a confidence
score — whether to audit, minimizing audit rate under an error-control
guarantee. This project addresses the orthogonal question that paper does
not: given a **fixed** audit budget and **no per-item confidence score**,
how should audits be allocated **across segments** whose violation rates
burst and drift? See `KILLGATE.md` for the full prior-art analysis.

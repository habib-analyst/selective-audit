# KILLGATE — selective-audit

**Claim under test:** an *online audit-budget allocation* benchmark — policies that learn,
per segment of an AI output stream, where violations concentrate and shift a fixed audit
budget there *under bursts and drift*, scored by detection yield at equal budget and by
regret against an oracle that knows the true segment rates — is not already shipped.

Search date: 2026-10-24. Tools: web search, arXiv abstract pages, GitHub repository search API.

## Closest prior art, and why each is a different deliverable

1. **Gibbs, Han, Tibshirani, Zrnic — "Reliable AI-Assisted Decisions via Selective
   Auditing", arXiv:2610.31662 (Oct 2026)** — the inspiring paper. It makes a *per-item*
   accept-vs-audit decision driven by a confidence/verifier score, minimizing the audit
   rate subject to a guarantee that the unaudited-error fraction stays below a threshold.
   It has **no segment structure, no fixed budget allocated across segments, no learned
   per-segment rates, and no drift/burst benchmark or oracle regret**. Our problem
   (where to spend a fixed budget across segments whose rates drift) is orthogonal.
2. **Das, Sharma, Nanavati, Shiragur, Deshpande — "Cost Efficient Fairness Audit Under
   Partial Feedback", arXiv:2510.03734** — bandit-style algorithms for estimating a
   classifier's *fairness metrics* cheaply when labels are costly. Objective is estimation
   cost for fairness quantities, not violation-detection yield; no segments under drift,
   no bursts, no detection benchmark.
3. **"Auditing Fairness by Betting" (sequential fairness tests)** — sequential hypothesis
   testing for group fairness. A test, not a budget-allocation policy.
4. **Verma, Manisha, Gujar — "Fairness Auditing: Lower Bounds on Company Manipulation",
   arXiv:2608.00568** — min-max lower bounds on post-audit manipulation under a budgeted
   auditor. Theory bounds, no allocation policy, no simulator/benchmark.
5. **Dang, Zhang, Zhu — ABARM / RASS (RL-based audit sampling for financial audit risk)**
   — reinforcement-learning sampling for *financial* misstatement risk on static financial
   datasets. Domain-specific risk scoring; not an open online allocation benchmark with
   planted segment/burst/drift regimes and oracle regret.
6. **"Adaptive Process Mining and Selective Monitoring for Algorithmic Auditing: A
   Survey", Mach. Learn. Knowl. Extr. 2026, 8, 234 (doi:10.3390/make8080234)** — a survey
   that names bandits/RL as one decision layer for selective auditing and explicitly
   states a pooled numerical benchmark is "not yet defensible" — i.e. the benchmark
   shelf this project fills is acknowledged to be empty.

## GitHub code search (2026-10-24)

| Query | Total hits | Relevant |
|---|---|---|
| `audit allocation bandit` | 0 | — |
| `selective audit bandit` | 0 | — |
| `audit sampling bandit simulator` | 0 | — |
| `drift bandit audit` | 1 | `varunpatil5050/StreamRank` — a short-video recommender; unrelated |

## Verdict

**PROCEED.** Bandit machinery for auditing exists in adjacent problems (fairness-metric
estimation, per-item confidence gating, financial risk sampling), but the specific
deliverable — an online, per-segment audit-budget allocator with forgetting/change-point
adaptation, benchmarked on planted static / burst / drifting-segment streams at equal
budgets against uniform, fixed-proportional and epsilon-greedy baselines and a
known-rates oracle — is not shipped as such. The novelty is the benchmark + the
drift-aware allocation policy measured on it, not the bandit idea itself.

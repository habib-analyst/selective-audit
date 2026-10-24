# Launch note — selective-audit (Rule 10 gate, 2026-10-24)
- Novelty/kill-gate: PASS — prior art adjacent but distinct (per-item confidence gating, fairness-audit bandits, RL financial sampling); a 2026 survey confirms no pooled drift-aware audit-allocation benchmark exists; GitHub search 0 relevant repos.
- Measured claims only: PASS — 180 benchmark runs (3 regimes x 3 budgets x 4 policies + oracle, 5 seeds, T=20,000); budgets enforced by pacing (min realized 0.994 of budget).
- Honest losses included: PASS — epsilon-greedy wins the static regime clearly; at 1% budget nobody catches bursts (oracle 90 vs ~9); the change-point reset almost never fired and the gains are attributed to discounted estimation + coverage floor, stated in PAPER.md. Two failed designs documented (discounted Thompson at/below uniform; first burst parameterization undetectable-in-principle).
- Tests: PASS — 16/16 pytest; benchmark reproduces in ~35 s, numpy-only.
- Cost: PASS — $0, CPU only.
- Secrets/PII: PASS — none; synthetic streams only.
- Identity: PASS — author habib-analyst.
- Remaining checklist items: N/A — no payments, forms, or external sends beyond the launch post (standing repo-post authorization).
LAUNCH GATE PASSED 2026-10-24.

"""selective-audit: online audit-budget allocation under drift.

A simulator + policy library + benchmark for the problem: an AI system emits a
stream of outputs segmented into slices; an auditor can inspect only a small
fraction (the budget); violations concentrate in segments whose rates burst
and drift. Policies decide online which items to audit.
"""

from .streams import Stream, generate_stream, REGIMES
from .policies import (
    UniformPolicy,
    FixedProportionalPolicy,
    EpsilonGreedyPolicy,
    DriftUCBPolicy,
    POLICIES,
)
from .engine import run_policy, oracle_detections, oracle_expected

__all__ = [
    "Stream",
    "generate_stream",
    "REGIMES",
    "UniformPolicy",
    "FixedProportionalPolicy",
    "EpsilonGreedyPolicy",
    "DriftUCBPolicy",
    "POLICIES",
    "run_policy",
    "oracle_detections",
    "oracle_expected",
]

"""Deduplication keys and "interestingness".

A self-learning system left alone will happily produce a million trivial
variants of ``x + 0 = x``. Two defences:

1. Canonicalisation: each engine reduces its results to a normal form before
   hashing them with ``stable_hash``, so disguised duplicates collapse to one
   key (``constants.canonical_pcf``, the simplified SymPy form in ``laws``,
   ``verify.lean.normalize_statement`` for theorems).
2. Interestingness: a score that rewards results that are short (elegant),
   surprising (hard for the current model), and different from what the
   knowledge base already holds. Engines use it to rank discoveries.
"""

from __future__ import annotations

import hashlib
import math


def stable_hash(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


def interestingness(complexity: float, difficulty: float = 0.5, novelty: float = 1.0,
                    precision_digits: float | None = None) -> float:
    """Score in roughly [0, 10].

    complexity        description length of the result (lower is more elegant)
    difficulty        in [0, 1]: how hard it was for the system (1 = rarely solved)
    novelty           in [0, 1]: 1 = nothing similar in the knowledge base
    precision_digits  for numerical conjectures: more verified digits, more trust
    """
    elegance = 1.0 / (1.0 + math.log1p(max(0.0, complexity)))
    trust = 1.0 if precision_digits is None else min(1.0, precision_digits / 100.0) ** 0.5
    return 10.0 * elegance * (0.4 + 0.6 * difficulty) * (0.3 + 0.7 * novelty) * trust

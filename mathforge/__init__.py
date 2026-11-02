"""MathForge: a self-improving mathematical discovery system.

The core loop, shared by every engine:

    propose  ->  attempt  ->  VERIFY  ->  store  ->  learn  ->  (repeat forever)

Nothing enters the knowledge base or the training data unless a verifier
(exact integer arithmetic, high-precision numerics, SymPy, or the Lean 4
kernel) has checked it. That is what lets the system learn from itself
without drifting into confident nonsense.
"""

__version__ = "0.1.0"

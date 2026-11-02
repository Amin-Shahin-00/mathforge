"""Checkable problem families for neural self-play, with exact verifiers.

Every family has
  generate(level, rng) -> problem dict  (constructed *backwards* so the
                                          answer is known or checkable), and
  check(problem, completion) -> bool     (a verifier; the model's own claim is
                                          never trusted)

Families
--------
integrate  find an antiderivative. Check: d/dx answer == integrand at random
           complex points (independent of how the answer is written).
solve      find all real roots. Built from chosen roots, optionally pushed
           through exp/shift substitutions. Check: same root set.
sum        closed form of sum_{k=1}^n p(k). Check: the answer is a polynomial
           of degree <= deg(p)+1 that agrees at deg(p)+2 points, which is a proof.
"""

from __future__ import annotations

import _thread
import contextlib
import random
import re
import signal
import threading
import time

import sympy as sp

from .safe_parse import UnsafeExpression, extract_answer, latex_to_text, normalize_answer, safe_parse

x, n, k = sp.symbols("x n k")
FAMILIES = ("integrate", "solve", "sum")

# ----------------------------------------------------------------------------- timeouts

class Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    if not hasattr(signal, "SIGALRM"):        # Windows
        with _time_limit_interrupt(seconds):
            yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise Timeout()

    try:
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
    except ValueError:                        # not the main thread: no hard limit
        yield
        return
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@contextlib.contextmanager
def _time_limit_interrupt(seconds: float):
    """Time limit without SIGALRM: a timer thread raises SIGINT in the main thread.

    Python runs signal handlers between bytecodes, so this stops pure-Python work
    such as SymPy just like SIGALRM does. Only our own timer's interrupt becomes a
    Timeout; a genuine Ctrl+C is passed on to the previous handler. If the timer
    fires just as the block finishes, the late interrupt is absorbed here instead of
    reaching that handler (where it would look like a stop request).
    """
    try:
        old = signal.getsignal(signal.SIGINT)
        state = {"armed": True, "fired": False, "handled": False}

        def handler(signum, frame):
            if state["fired"] and not state["handled"]:
                state["handled"] = True
                if state["armed"]:
                    raise Timeout()
                return
            if callable(old):
                old(signum, frame)
            elif old == signal.SIG_DFL:
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handler)
    except ValueError:                        # not the main thread: no hard limit
        yield
        return
    lock = threading.Lock()

    def fire():
        with lock:
            if state["armed"]:
                state["fired"] = True
                _thread.interrupt_main(signal.SIGINT)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        try:
            with lock:
                state["armed"] = False
                timer.cancel()
            deadline = time.monotonic() + 1.0
            while state["fired"] and not state["handled"] and time.monotonic() < deadline:
                time.sleep(0.001)             # let a pending interrupt reach our handler
        finally:
            signal.signal(signal.SIGINT, old)

# ----------------------------------------------------------------------------- generators

def _atom(rng: random.Random):
    a = rng.choice([1, 1, 2, 3, -1, -2])
    b = rng.randint(1, 4)
    p = rng.randint(2, 5)
    return rng.choice([
        x, x ** p, sp.exp(a * x), sp.sin(a * x), sp.cos(a * x), sp.log(x + b),
        sp.sqrt(x + b), 1 / (x + b), sp.atan(x), x * sp.exp(a * x), sp.log(x),
    ])


def gen_integrate(level: int, rng: random.Random) -> dict:
    for _ in range(50):
        F = _atom(rng)
        for _ in range(max(0, level // 2)):
            op = rng.random()
            g = _atom(rng)
            if op < 0.35:
                F = F + rng.choice([1, 2, 3, -1]) * g
            elif op < 0.75:
                F = F * g
            elif op < 0.9 and F.has(x) and sp.count_ops(F) < 6:
                F = rng.choice([sp.sin, sp.exp, sp.log, sp.sqrt, sp.atan])(F if F.is_polynomial(x) else x ** 2 + F)
            else:
                F = F / (x ** 2 + rng.randint(1, 3))
        try:
            with time_limit(5):
                f = sp.simplify(sp.diff(F, x))
        except Timeout:
            continue
        s = sp.sstr(f)
        if f != 0 and f.has(x) and len(s) < 60 + 25 * level:
            p = {"family": "integrate", "level": level, "integrand": s, "reference": sp.sstr(F)}
            return {**p, "text": problem_text(p)}
    return gen_integrate(max(1, level - 1), rng)


def gen_solve(level: int, rng: random.Random) -> dict:
    deg = 1 + min(3, (level + 1) // 3)
    R = 3 + level
    roots = set()
    while len(roots) < deg:
        if level >= 4 and rng.random() < 0.3:
            roots.add(sp.Rational(rng.randint(-R, R), rng.randint(2, 4)))
        else:
            roots.add(sp.Integer(rng.randint(-R, R)))
    t = sp.Symbol("t")
    lead = rng.choice([1, 1, 2, 3])
    P = lead * sp.prod([t - r for r in roots])
    sub = "id" if level < 6 else rng.choice(["id", "exp", "shift"])
    if sub == "exp":
        expr = P.subs(t, sp.exp(x))
        real_roots = [sp.log(r) for r in roots if r > 0]
    elif sub == "shift":
        s = rng.randint(1, 4)
        expr = P.subs(t, x + s)
        real_roots = [r - s for r in roots]
    else:
        expr = P.subs(t, x)
        real_roots = list(roots)
    expr = sp.expand(expr)
    if expr.has(sp.exp):
        expr = sp.expand(expr)
    if not real_roots:
        return gen_solve(level, rng)
    p = {"family": "solve", "level": level, "lhs": sp.sstr(expr), "roots": [sp.sstr(r) for r in real_roots]}
    return {**p, "text": problem_text(p)}


def gen_sum(level: int, rng: random.Random) -> dict:
    d = min(1 + level // 2, 7)
    coeffs = [rng.randint(-4, 4) for _ in range(d)] + [rng.choice([1, 2, 3, -1])]
    q = {"family": "sum", "level": level, "summand": sp.sstr(sum(c * k ** i for i, c in enumerate(coeffs))),
         "degree": d}
    return {**q, "text": problem_text(q)}


GENERATORS = {"integrate": gen_integrate, "solve": gen_solve, "sum": gen_sum}


def generate(family: str, level: int, seed: int) -> dict:
    rng = random.Random(seed)
    with time_limit(30):
        return GENERATORS[family](level, rng)

# ----------------------------------------------------------------------------- checkers

def _close(a: complex, b: complex, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_integrate(problem: dict, answer: str) -> bool:
    F = safe_parse(answer, ("x",))
    if F.has(sp.Integral) or not F.has(x):
        return False
    f = safe_parse(problem["integrand"], ("x",))
    # ln|u| and ln(u) have the same derivative u'/u wherever u != 0, so this rewrite is exact
    # and lets the usual ln|u| answers be checked at complex points.
    F = F.replace(lambda e: isinstance(e, sp.log) and isinstance(e.args[0], sp.Abs),
                  lambda e: sp.log(e.args[0].args[0]))
    var, im = x, 0.2
    if F.has(sp.Abs):
        # Any other |u| is not complex-differentiable (and must NOT be replaced by u, which
        # would accept |x - 3| as an antiderivative of 1): compare on the real line instead.
        var, im = sp.Symbol("x", real=True), 0.0
        F, f = F.subs(x, var), f.subs(x, var)
    dF = sp.diff(F, var)
    rng = random.Random(1234)
    good = 0
    for _ in range(20):
        pt = sp.Float(rng.uniform(0.3, 2.7)) + sp.I * sp.Float(rng.uniform(-im, im))
        try:
            a = complex(dF.subs(var, pt).evalf(30))
            b = complex(f.subs(var, pt).evalf(30))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if not _close(a, b, 1e-7):
            return False
        good += 1
        if good >= 6:
            return True
    return False


def check_solve(problem: dict, answer: str) -> bool:
    ans = answer.strip()
    if ans.startswith("(") and ans.endswith(")") and "," in ans:
        ans = "[" + ans[1:-1] + "]"         # set or tuple notation
    if not (ans.startswith("[") and ans.endswith("]")):
        ans = "[" + ans.replace("x =", "").replace("x=", "") + "]"
    parsed = safe_parse(ans, ("x",))
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    got = [complex(sp.N(v, 30)) for v in parsed]
    truth = [complex(sp.N(safe_parse(r, ()), 30)) for r in problem["roots"]]
    if len(got) != len(truth):
        return False
    return all(any(_close(g, t, 1e-9) for t in truth) for g in got) and \
        all(any(_close(g, t, 1e-9) for g in got) for t in truth)


def check_sum(problem: dict, answer: str) -> bool:
    S = sp.expand(safe_parse(answer, ("n",)))
    if not S.is_polynomial(n) or sp.Poly(S, n).degree() > problem["degree"] + 1:
        return False
    p = safe_parse(problem["summand"], ("k",))
    total = sp.Integer(0)
    for m in range(0, problem["degree"] + 3):
        if m > 0:
            total += p.subs(k, m)
        if sp.simplify(S.subs(n, m) - total) != 0:
            return False
    return True


CHECKERS = {"integrate": check_integrate, "solve": check_solve, "sum": check_sum}


def check(problem: dict, completion: str, timeout_s: int = 10) -> bool:
    ans = extract_answer(completion)
    if ans is None:
        return False
    try:
        with time_limit(timeout_s):
            ans = normalize_answer(ans, problem["family"])
            return bool(CHECKERS[problem["family"]](problem, ans))
    except (UnsafeExpression, Timeout, SyntaxError, TypeError, ValueError, ZeroDivisionError,
            AttributeError, sp.SympifyError, sp.PolynomialError, RecursionError, OverflowError):
        return False
    except Exception:  # noqa: BLE001 - any parser surprise counts as a wrong answer
        return False


def validate_proposed_integrand(text: str) -> str | None:
    """Accept a model-invented integration problem if it is well-formed."""
    try:
        with time_limit(5):
            f = safe_parse(text, ("x",))
            if not f.has(x) or f.has(sp.Integral) or sp.count_ops(f) > 40:
                return None
            v = complex(f.subs(x, sp.Rational(13, 10)).evalf(20))
            if v != v:  # NaN
                return None
            return sp.sstr(f)
    except Exception:  # noqa: BLE001
        return None


_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$|\$([^$]+?)\$", re.S)
_SHAPE = {"integrate": r"f\s*\(\s*x\s*\)\s*=\s*(.+)", "sum": r"p\s*\(\s*k\s*\)\s*=\s*(.+)",
          "solve": r"(.+?)\s*=\s*0"}


def extract_proposal(family: str, completion: str) -> str | None:
    """The invented problem in a proposer's answer.

    Asked for one ``PROBLEM: ...`` line, real maths models (Qwen2.5-Math) instead state
    the problem in LaTeX, ``\\( f(x) = \\frac{...}{...} \\)``, and go on to solve it. So:
    a PROBLEM line if there is one, else the first displayed or inline formula of the
    family's shape (f(x) = ..., ... = 0, p(k) = ...). Only text is extracted here; it
    still has to pass ``validate_proposed`` (safe parser, SymPy-computed ground truth).
    """
    m = re.search(r"PROBLEM:\s*(.+)", completion)
    if m:
        return m.group(1).strip().splitlines()[0]
    for mm in _MATH.finditer(completion):
        expr = " ".join(next(g for g in mm.groups() if g).split())
        g = re.fullmatch(_SHAPE[family], expr)
        if g:
            return g.group(1).strip().rstrip(".,")
    return None


def problem_text(p: dict) -> str:
    """The wording the solver sees (generated and model-invented problems alike)."""
    if p["family"] == "integrate":
        return f"Find an antiderivative of f(x) = {p['integrand']} with respect to x."
    if p["family"] == "solve":
        return (f"Find all real solutions x of the equation {p['lhs']} = 0. "
                f"Give the answer as a list, e.g. [1, -2].")
    return f"Find a closed form, as a polynomial in n, for the sum over k = 1..n of {p['summand']}."


def problem_spec(p: dict) -> str:
    """The one string that defines a problem (used for examples and deduplication)."""
    return p[{"integrate": "integrand", "solve": "lhs", "sum": "summand"}[p["family"]]]


def validate_proposed(family: str, text: str, max_degree: int = 4) -> dict | None:
    """Turn a model-invented problem into a checkable one, or None.

    The model only supplies the question. Everything the checker relies on is computed
    here by SymPy: the real roots of a proposed equation, the degree of a proposed
    summand. So a proposal can be useless, but it can never make a wrong answer pass.
    """
    text = text.strip().strip("`").strip()
    if "\\" in text or "{" in text:
        text = latex_to_text(text)
    try:
        if family == "integrate":
            f = validate_proposed_integrand(text)
            p = {"family": "integrate", "integrand": f} if f else None
        elif family == "solve":
            lhs, _, rhs = text.partition("=")
            with time_limit(5):
                P = sp.expand(safe_parse(lhs, ("x",)) - (safe_parse(rhs, ("x",)) if rhs.strip() else 0))
                if not P.is_polynomial(x):
                    return None
                poly = sp.Poly(P, x)
                if not 1 <= poly.degree() <= max_degree or not all(c.is_rational for c in poly.coeffs()):
                    return None
                roots = sorted(set(sp.real_roots(poly)), key=lambda r: float(r))
                if not roots:
                    return None
                # exact form when it is short, else 30 digits (the checker compares to 1e-9)
                strs = [sp.sstr(r) if not r.has(sp.CRootOf) else sp.sstr(sp.N(r, 30)) for r in roots]
            p = {"family": "solve", "lhs": sp.sstr(P), "roots": strs}
        elif family == "sum":
            with time_limit(5):
                q = sp.expand(safe_parse(text, ("k",)))
                if not q.is_polynomial(k) or not q.has(k):
                    return None
                d = sp.Poly(q, k).degree()
                if d > 7 or not all(c.is_rational for c in sp.Poly(q, k).coeffs()):
                    return None
            p = {"family": "sum", "summand": sp.sstr(q), "degree": d}
        else:
            return None
    except Exception:  # noqa: BLE001 - malformed proposals are simply rejected
        return None
    if p:
        p["text"] = problem_text(p)
    return p

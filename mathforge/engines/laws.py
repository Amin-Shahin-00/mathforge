"""Engine 2: rediscovering laws of nature from raw data, and learning a library.

Given only numbers (e.g. masses, distances, forces), find the formula that
generated them. The engine uses genetic-programming symbolic regression, plus
two learning mechanisms that make it improve over time:

1. Learned grammar. Operator frequencies in verified solutions become the
   sampling prior for new expressions (what kind of maths tends to work).

2. Library learning (in the spirit of DreamCoder). Sub-expressions that recur
   across solved laws are abstracted into new primitives with arguments, e.g.
   ``hyp(A, B) = sqrt(A^2 + B^2)`` or ``inv_sq(A, B) = A / B^2``. Later problems
   can use them as single building blocks, so the system effectively invents
   its own vocabulary and solves harder laws with shorter programs.

Free constants (as in PySR): besides the fixed constants 1, 2, 3, 1/2, trees can
contain ``("c", value)`` leaves whose values are fitted by least squares during the
search, so laws with constants inside them (rho*v^2/2 + 9.81*rho*h) are reachable.
Before verification the head snaps fitted values to short exact numbers (19.62 ->
981/50, 6.2831... -> 2*pi); the snapped formula is what gets verified and stored.

Verification (on the head, never trusted from workers):
  * held-out and out-of-distribution data (wider ranges than training) to
    rule out curve-fitting approximations, and
  * when the ground truth is known, a symbolic proof of affine equivalence:
    found f and true g satisfy grad(g) = alpha * grad(f) for a constant alpha.

Bring your own data: add CSV files under ``engines.laws.datasets`` and the
engine will hunt for formulas in them every round.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from fractions import Fraction

import numpy as np
import sympy as sp

from ..novelty import interestingness, stable_hash
from ..units import consistent, parse_units
from .base import Engine

# ----------------------------------------------------------------------------- primitives

UNARY = {
    "neg": np.negative,
    "sq": np.square,
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "inv": lambda x: 1.0 / x,
    "exp": lambda x: np.exp(np.clip(x, -50, 50)),
    "log": lambda x: np.log(np.abs(x) + 1e-300),
    "sin": np.sin,
    "cos": np.cos,
}
BINARY = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
}
def _sym_sqrt(x):
    # the search uses a protected sqrt(|x|); keep the symbolic form faithful to it
    return sp.sqrt(x) if x.is_nonnegative else sp.sqrt(sp.Abs(x))


def _sym_log(x):
    return sp.log(x) if x.is_positive else sp.log(sp.Abs(x))


SYM_UNARY = {
    "neg": lambda x: -x, "sq": lambda x: x ** 2, "sqrt": _sym_sqrt, "inv": lambda x: 1 / x,
    "exp": sp.exp, "log": _sym_log, "sin": sp.sin, "cos": sp.cos,
}
SYM_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
}
CONSTANTS = [1.0, 2.0, 3.0, 0.5]
MAX_FIT = 4          # trees with more free constants than this are not tuned (parsimony)
# Starting prior over operators (arithmetic is common in physics, trig less so).
# The engine multiplies these by weights it learns from verified solutions.
PRIOR = {"add": 3.0, "sub": 2.0, "mul": 3.0, "div": 2.5, "sq": 2.0, "sqrt": 2.0, "inv": 1.0,
         "neg": 0.5, "exp": 0.8, "log": 0.6, "sin": 0.6, "cos": 0.6}

# ----------------------------------------------------------------------------- benchmark
# (name, variables, sympy formula string, sampling ranges, difficulty tier)

LAWS = [
    ("spring_energy",       ["k", "x"],           "k*x**2/2",                               [(1, 5), (1, 5)], 1),
    ("kinetic_energy",      ["m", "v"],           "m*v**2/2",                               [(1, 5), (1, 5)], 1),
    ("centripetal_force",   ["m", "v", "r"],      "m*v**2/r",                               [(1, 5), (1, 5), (1, 5)], 2),
    ("ideal_gas",           ["n", "T", "V"],      "8.314*n*T/V",                            [(1, 5), (1, 5), (1, 5)], 2),
    ("newton_gravity",      ["m1", "m2", "r"],    "6.674e-11*m1*m2/r**2",                   [(1, 5), (1, 5), (1, 5)], 2),
    ("pendulum_period",     ["L", "g"],           "2*pi*sqrt(L/g)",                         [(1, 5), (1, 5)], 2),
    ("pythagoras",          ["a", "b"],           "sqrt(a**2 + b**2)",                      [(1, 5), (1, 5)], 3),
    ("escape_velocity",     ["M", "r"],           "sqrt(2*6.674e-11*M/r)",                  [(1, 5), (1, 5)], 3),
    ("thin_lens",           ["d1", "d2"],         "d1*d2/(d1 + d2)",                        [(1, 5), (1, 5)], 3),
    ("kepler_third_law",    ["a", "M"],           "2*pi*sqrt(a**3/(6.674e-11*M))",          [(1, 5), (1, 5)], 3),
    ("gaussian",            ["x"],                "exp(-x**2/2)/sqrt(2*pi)",                [(-3, 3)], 3),
    ("doppler",             ["f", "v", "c"],      "f*c/(c - v)",                            [(1, 5), (1, 2), (3, 10)], 4),
    ("distance_3d",         ["x", "y", "z"],      "sqrt(x**2 + y**2 + z**2)",               [(1, 5), (1, 5), (1, 5)], 4),
    ("projectile_range",    ["v", "th", "g"],     "v**2*sin(2*th)/g",                       [(1, 5), (0.1, 1.4), (1, 5)], 4),
    ("lorentz_factor",      ["v", "c"],           "1/sqrt(1 - v**2/c**2)",                  [(1, 2), (3, 10)], 4),
    ("relativistic_energy", ["m", "v", "c"],      "m*c**2/sqrt(1 - v**2/c**2)",             [(1, 5), (1, 2), (3, 10)], 5),
    ("coulomb_superpos",    ["q", "r1", "r2"],    "q/r1**2 + q/r2**2",                      [(1, 5), (1, 5), (1, 5)], 5),
    ("damped_oscillation",  ["t", "w"],           "exp(-t/2)*cos(w*t)",                     [(0, 4), (1, 3)], 5),
    ("bernoulli_pressure",  ["rho", "v", "h"],    "rho*v**2/2 + 9.81*rho*h",                [(1, 5), (1, 5), (1, 5)], 5),
    ("planck_like",         ["x"],                "x**3/(exp(x) - 1)",                      [(0.5, 5)], 6),
]


# Units of each law's variables (for dimensional analysis, see mathforge/units.py).
LAW_UNITS = {
    "spring_energy": ["kg/s^2", "m"], "kinetic_energy": ["kg", "m/s"],
    "centripetal_force": ["kg", "m/s", "m"], "ideal_gas": ["mol", "K", "m^3"],
    "newton_gravity": ["kg", "kg", "m"], "pendulum_period": ["m", "m/s^2"], "pythagoras": ["m", "m"],
    "escape_velocity": ["kg", "m"], "thin_lens": ["m", "m"], "kepler_third_law": ["m", "kg"],
    "gaussian": ["1"], "doppler": ["Hz", "m/s", "m/s"], "distance_3d": ["m", "m", "m"],
    "projectile_range": ["m/s", "1", "m/s^2"], "lorentz_factor": ["m/s", "m/s"],
    "relativistic_energy": ["kg", "m/s", "m/s"], "coulomb_superpos": ["C", "m", "m"],
    "damped_oscillation": ["s", "Hz"], "bernoulli_pressure": ["kg/m^3", "m/s", "m"],
    "planck_like": ["1"],
}


def law_dims(law_idx: int) -> list:
    return [parse_units(u) for u in LAW_UNITS[LAWS[law_idx][0]]]


def law_symbols(law_idx: int):
    """Symbols with positivity assumptions where the physics guarantees them."""
    _, vars_, _, ranges, _ = LAWS[law_idx]
    return [sp.Symbol(v, positive=True) if lo > 0 else sp.Symbol(v, real=True)
            for v, (lo, _) in zip(vars_, ranges)]


def law_truth(law_idx: int, syms):
    # rational=True: a literal such as 9.81 means exactly 981/100. As a Float it would
    # block exact cancellation in the proof (9.81 * 50/981 is not quite 1/2 in floats).
    return sp.sympify(LAWS[law_idx][2], locals={s.name: s for s in syms}, rational=True)


def sample_law_data(law_idx: int, n: int, seed: int, noise: float = 0.0, widen: float = 1.0):
    name, vars_, formula, ranges, _ = LAWS[law_idx]
    rng = np.random.default_rng(seed)
    cols = []
    for lo, hi in ranges:
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        lo2, hi2 = mid - half * widen, mid + half * widen
        if lo > 0:
            lo2 = max(lo2, lo * 0.5)   # stay in the physical domain
        if name in ("lorentz_factor", "relativistic_energy", "doppler"):
            lo2, hi2 = lo, hi          # keep v < c
        cols.append(rng.uniform(lo2, hi2, n))
    X = np.array(cols)
    syms = sp.symbols(vars_)
    f = sp.lambdify(syms, sp.sympify(formula), "numpy")
    y = np.asarray(f(*X), dtype=float) * np.ones(n)
    if noise:
        y = y + noise * np.std(y) * rng.standard_normal(n)
    return X, y

# ----------------------------------------------------------------------------- trees

def to_tuple(x):
    if isinstance(x, list):
        return tuple(to_tuple(v) for v in x)
    return x


def evaluate(node, X, lib, args=None, consts=None):
    """Evaluate a tree on data X (one row per variable).

    ``consts``: optional iterator of values for the tree's ``("c", ...)`` leaves in
    pre-order, used while fitting them (avoids rebuilding the tree per trial value).
    """
    t = node[0]
    if t == "v":
        return X[node[1]]
    if t == "k":
        return np.full(X.shape[1], node[1])
    if t == "c":
        return np.full(X.shape[1], node[1] if consts is None else next(consts))
    if t == "a":
        return args[node[1]]
    if t == "u":
        return UNARY[node[1]](evaluate(node[2], X, lib, args, consts))
    if t == "b":
        return BINARY[node[1]](evaluate(node[2], X, lib, args, consts), evaluate(node[3], X, lib, args, consts))
    if t == "m":
        vals = [evaluate(c, X, lib, args, consts) for c in node[2:]]
        return evaluate(lib[node[1]]["body"], X, lib, vals)
    raise ValueError(t)


def to_sympy(node, syms, lib, args=None):
    t = node[0]
    if t == "v":
        return syms[node[1]]
    if t == "k":
        return sp.nsimplify(node[1])
    if t == "c":
        # a snapped constant carries its exact form; a raw fitted one stays a float
        return sp.sympify(node[2]) if len(node) > 2 else sp.Float(node[1], 15)
    if t == "a":
        return args[node[1]]
    if t == "u":
        return SYM_UNARY[node[1]](to_sympy(node[2], syms, lib, args))
    if t == "b":
        return SYM_BINARY[node[1]](to_sympy(node[2], syms, lib, args), to_sympy(node[3], syms, lib, args))
    if t == "m":
        vals = [to_sympy(c, syms, lib, args) for c in node[2:]]
        return to_sympy(lib[node[1]]["body"], syms, lib, vals)
    raise ValueError(t)


def children_idx(node):
    t = node[0]
    if t == "u":
        return (2,)
    if t == "b":
        return (2, 3)
    if t == "m":
        return tuple(range(2, len(node)))
    return ()


def size(node) -> int:
    return 1 + sum(size(node[i]) for i in children_idx(node))


def subtrees(node, path=()):
    yield path, node
    for i in children_idx(node):
        yield from subtrees(node[i], path + (i,))


def replace_at(node, path, new):
    if not path:
        return new
    i = path[0]
    lst = list(node)
    lst[i] = replace_at(node[i], path[1:], new)
    return tuple(lst)


def constants_of(node) -> list[float]:
    """Values of the free-constant leaves, in the pre-order ``evaluate`` consumes them."""
    return [n[1] for _, n in subtrees(node) if n[0] == "c"]


def with_constants(node, values):
    """Copy of ``node`` with its free-constant leaves set to ``values`` (pre-order)."""
    it = iter(values)

    def go(n):
        if n[0] == "c":
            return ("c", float(next(it)))
        ci = children_idx(n)
        return tuple(go(n[i]) if i in ci else n[i] for i in range(len(n))) if ci else n

    return go(node)


def round_floats(expr: sp.Expr, digits: int = 6) -> sp.Expr:
    """Fitted constants printed to ``digits`` significant digits (display only)."""
    return expr.xreplace({f: sp.Float(f, digits) for f in expr.atoms(sp.Float)})


def ops_used(node, lib) -> list[str]:
    out = []
    for _, n in subtrees(node):
        if n[0] in ("u", "b"):
            out.append(n[1])
        elif n[0] == "m":
            out.append(f"m{n[1]}")
    return out


class TreeGen:
    def __init__(self, nvars, lib, op_weights, rng: random.Random, fit_constants: bool = False,
                 const_prob: float = 0.08):
        self.nvars, self.lib, self.rng = nvars, lib, rng
        self.fit = fit_constants
        self.const_prob = const_prob
        names = list(UNARY) + list(BINARY) + [f"m{i}" for i in range(len(lib))]
        self.names = names
        self.weights = [PRIOR.get(n, 1.0) * op_weights.get(n, 1.0) for n in names]

    def terminal(self):
        r = self.rng.random()
        if r < (0.75 if self.fit else 0.8):
            return ("v", self.rng.randrange(self.nvars))
        if self.fit and r < 0.75 + self.const_prob:   # a free constant; its start value is only a guess
            return ("c", self.rng.choice(CONSTANTS) * self.rng.choice([1.0, -1.0]))
        return ("k", self.rng.choice(CONSTANTS))

    def random(self, depth: int, full: bool = False):
        if depth <= 0 or (not full and self.rng.random() < 0.3):
            return self.terminal()
        op = self.rng.choices(self.names, weights=self.weights)[0]
        if op in UNARY:
            return ("u", op, self.random(depth - 1, full))
        if op in BINARY:
            return ("b", op, self.random(depth - 1, full), self.random(depth - 1, full))
        idx = int(op[1:])
        return ("m", idx) + tuple(self.random(depth - 1, full) for _ in range(self.lib[idx]["arity"]))


def fitness(node, X, y, lib):
    """Linear-scaled NMSE: y ~ alpha * f(X) + beta. Returns (nmse, alpha, beta)."""
    with np.errstate(all="ignore"):
        f = evaluate(node, X, lib)
        f = np.asarray(f, dtype=float)
        if f.shape != y.shape or not np.all(np.isfinite(f)):
            return math.inf, 0.0, 0.0
        fm, ym = f.mean(), y.mean()
        df = f - fm
        var_f = float(np.dot(df, df))
        if var_f < 1e-24 * len(f) * max(1.0, fm * fm):
            return 1.0, 0.0, ym
        alpha = float(np.dot(df, y - ym) / var_f)
        beta = float(ym - alpha * fm)
        r = y - (alpha * f + beta)
        var_y = float(np.var(y)) or 1.0
        nmse = float(np.mean(r * r) / var_y)
        if not math.isfinite(nmse):
            return math.inf, 0.0, 0.0
        return nmse, alpha, beta


def score(nmse: float, sz: int, parsimony: float) -> float:
    return math.log10(nmse + 1e-15) + parsimony * sz


def fit_constants(node, X, y, lib, max_nfev: int = 60):
    """Least-squares fit of a tree's free constants (Levenberg-Marquardt).

    The global scale and offset stay in closed form (variable projection): for each
    trial value of the constants, alpha and beta are the exact linear fit, so the
    optimiser only searches the constants that sit inside the formula. Returns the
    tree with the fitted values, or the tree unchanged if that is not an improvement.
    """
    theta0 = np.asarray(constants_of(node), dtype=float)
    if not 0 < len(theta0) <= MAX_FIT:
        return node
    from scipy.optimize import least_squares

    yc = y - y.mean()
    norm = (float(np.std(y)) or 1.0) * math.sqrt(len(y))

    def resid(theta):
        with np.errstate(all="ignore"):
            f = np.asarray(evaluate(node, X, lib, consts=iter(theta)), dtype=float)
            if f.shape != y.shape:
                f = np.broadcast_to(f, y.shape)
            if not np.isfinite(f).all():
                return np.full(len(y), 1e3)
            df = f - f.mean()
            var_f = float(df @ df)
            r = yc if var_f < 1e-300 else yc - (float(df @ yc) / var_f) * df
            return r / norm

    try:
        r0 = resid(theta0)
        sol = least_squares(resid, theta0, method="lm", max_nfev=max_nfev, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    except Exception:  # noqa: BLE001 - a failed fit just keeps the old constants
        return node
    if not np.all(np.isfinite(sol.x)) or float(sol.fun @ sol.fun) >= float(r0 @ r0):
        return node
    return with_constants(node, sol.x)

# ----------------------------------------------------------------------------- worker

def laws_worker(task: dict) -> dict:
    rng = random.Random(task["seed"])
    lib = [{"body": to_tuple(m["body"]), "arity": m["arity"]} for m in task["library"]]
    X = np.asarray(task["X"], dtype=float)
    y = np.asarray(task["y"], dtype=float)
    n = X.shape[1]
    idx = np.arange(n)
    rng_np = np.random.default_rng(task["seed"])
    rng_np.shuffle(idx)
    tr, va = idx[: int(0.75 * n)], idx[int(0.75 * n):]
    Xt, yt, Xv, yv = X[:, tr], y[tr], X[:, va], y[va]
    fit = bool(task.get("fit_constants", False))
    gen = TreeGen(X.shape[0], lib, task["op_weights"], rng, fit_constants=fit,
                  const_prob=task.get("const_prob", 0.08))
    P, G, pars = task["population"], task["generations"], task["parsimony"]
    max_size = task.get("max_size", 40)
    var_dims = task.get("units")             # None: no dimensional analysis
    units_penalty = task.get("units_penalty", 2.0)

    pop = [gen.random(rng.randint(1, 5), full=rng.random() < 0.5) for _ in range(P)]
    pop += [to_tuple(s) for s in task.get("seeds", [])]
    cache: dict = {}
    tuned: set = set()
    shape_best: dict = {}      # tree with its constants blanked -> (score, best fitted version)
    n_tune = max(4, P // 50)
    small = task.get("tune_small", 12)
    use_cache = task.get("shape_cache", True)

    def shape(n):
        if n[0] == "c":
            return ("c",)
        ci = children_idx(n)
        return tuple(shape(n[i]) if i in ci else n[i] for i in range(len(n))) if ci else n

    def tune(scored):
        """Lamarckian constant fitting: fit the free constants of the most promising
        untuned trees, of every small one (cheap, and a small tree with a guessed
        constant looks hopeless until its constant is fitted: exp(c*x) with c = 1
        loses to any polynomial), and of a few random ones. The fitted values are
        written back into the population, so offspring inherit them."""
        if not fit:
            return
        idx = [i for i, (_, t) in enumerate(scored)
               if t not in tuned and any(n[0] == "c" for _, n in subtrees(t))]
        idx.sort(key=lambda i: scored[i][0])
        pick = set(idx[:n_tune]) | {i for i in idx if size(scored[i][1]) <= small}
        rest = [i for i in idx if i not in pick]
        for i in sorted(pick) + rng.sample(rest, min(len(rest), n_tune)):
            key = shape(scored[i][1])
            hit = shape_best.get(key) if use_cache else None   # a third of fits repeat a shape
            t2 = hit[1] if hit else fit_constants(scored[i][1], Xt, yt, lib, max_nfev=60 if i in pick else 30)
            tuned.update((scored[i][1], t2))
            scored[i] = (eval_one(t2), t2)
            if hit is None or scored[i][0] < hit[0]:
                shape_best[key] = scored[i]

    def eval_one(ind):
        v = cache.get(ind)
        if v is None:
            if size(ind) > max_size:
                v = math.inf
            else:
                nm, _, _ = fitness(ind, Xt, yt, lib)
                v = score(nm, size(ind), pars) if math.isfinite(nm) else math.inf
                if var_dims is not None and math.isfinite(v) and not consistent(ind, var_dims, lib):
                    v += units_penalty
            if len(cache) < 200_000:
                cache[ind] = v
        return v

    def point_mutation(a):
        nodes = [(p, n) for p, n in subtrees(a) if n[0] in ("u", "b")]
        if not nodes:
            return gen.random(2)
        p, n = rng.choice(nodes)
        ops = list(UNARY) if n[0] == "u" else list(BINARY)
        return replace_at(a, p, (n[0], rng.choice(ops)) + tuple(n[2:]))

    target = score(task["success_nmse"] * 1e-2, 1, 0)
    scored = [(eval_one(t), t) for t in pop]
    tune(scored)
    best = min(scored, key=lambda s: s[0])
    stale, patience = 0, task.get("patience", 10)
    for g in range(G):
        scored.sort(key=lambda s: s[0])
        elite = scored[: max(2, P // 20)]
        new = [t for _, t in elite]
        seen = set(new)

        def tourn():
            return min(rng.sample(scored, 5), key=lambda s: s[0])[1]

        attempts = 0
        while len(new) < P and attempts < 4 * P:
            attempts += 1
            r = rng.random()
            a = tourn()
            if r < 0.55:     # crossover
                b = tourn()
                pa, _ = rng.choice(list(subtrees(a)))
                _, sb = rng.choice(list(subtrees(b)))
                child = replace_at(a, pa, sb)
            elif r < 0.75:   # subtree mutation
                pa, _ = rng.choice(list(subtrees(a)))
                child = replace_at(a, pa, gen.random(rng.randint(0, 3)))
            elif r < 0.87:   # point mutation (swap an operator, keep the shape)
                child = point_mutation(a)
            elif r < 0.95:   # hoist (simplifies)
                _, child = rng.choice(list(subtrees(a)))
            else:            # wrap in a unary
                child = ("u", rng.choice(list(UNARY)), a)
            if child in seen:  # duplicate elimination keeps the population diverse
                continue
            seen.add(child)
            new.append(child)
        if stale >= patience:  # stagnation: keep the elite, re-seed the rest
            new = new[: len(elite)] + [gen.random(rng.randint(1, 5), full=rng.random() < 0.5)
                                       for _ in range(P - len(elite))]
            stale = 0
        scored = [(eval_one(t), t) for t in new]
        tune(scored)
        cur = min(scored, key=lambda s: s[0])
        if cur[0] < best[0] - 1e-9:
            best, stale = cur, 0
        else:
            stale += 1
        if best[0] < target + pars * size(best[1]):
            break

    tree = best[1]
    if fit:   # polish: exact data deserve constants accurate to many digits (for snapping)
        tree = fit_constants(tree, Xt, yt, lib, max_nfev=600)
    nm_t, alpha, beta = fitness(tree, Xt, yt, lib)
    nm_v, _, _ = fitness(tree, Xv, yv, lib)
    return {"law": task["law"], "tree": tree, "nmse_train": nm_t, "nmse_val": nm_v,
            "alpha": alpha, "beta": beta, "size": size(tree), "generations": g + 1}

# ----------------------------------------------------------------------------- verification

def bases_positive_on_domain(expr: sp.Expr, syms, X: np.ndarray) -> bool:
    """True if every base of a non-integer power in ``expr`` is > 0 at all sampled points.

    Under that condition (a*b)**p == a**p * b**p and similar rewrites are valid, which
    is what ``force=True`` in SymPy's power simplification assumes.
    """
    for base in {p.base for p in expr.atoms(sp.Pow) if not p.exp.is_integer}:
        try:
            vals = np.asarray(sp.lambdify(syms, base, "numpy")(*X), dtype=float) * np.ones(X.shape[1])
        except Exception:  # noqa: BLE001
            return False
        if not np.all(vals > 0):
            return False
    return True


def affine_equivalent_symbolic(f: sp.Expr, g: sp.Expr, syms, domain_X: np.ndarray | None = None) -> bool:
    """True if g = alpha*f + beta for constants alpha, beta (via gradient ratios).

    With ``domain_X`` (sampled points of the physical domain), a ratio SymPy cannot
    reduce is retried with powers combined as if their bases were positive, but only
    after checking that every such base is positive at all sampled points.
    """
    ratios = []
    for s in syms:
        df, dg = sp.diff(f, s), sp.diff(g, s)
        if df == 0 and dg == 0:
            continue
        if df == 0 or dg == 0:
            return False
        r = sp.simplify(dg / df)
        if r.free_symbols & set(syms) and domain_X is not None \
                and bases_positive_on_domain(r, syms, domain_X):
            r = sp.simplify(sp.powsimp(sp.powdenest(r, force=True), force=True))
        if r.free_symbols & set(syms):
            return False
        ratios.append(r)
    if not ratios:
        return False
    return all(sp.simplify(r - ratios[0]) == 0 for r in ratios[1:])


def strip_abs_on_domain(expr: sp.Expr, syms, X: np.ndarray) -> sp.Expr | None:
    """Replace Abs(u) by u or -u when u keeps one sign at every sampled domain point.

    Protected operations (sqrt|x|, log|x|) leave Abs in found formulas; on the
    physical domain they usually reduce to a plain expression, which can then be
    compared symbolically. Returns None if some Abs argument changes sign.
    """
    reps = {}
    for a in expr.atoms(sp.Abs):
        u = a.args[0]
        try:
            vals = np.asarray(sp.lambdify(syms, u, "numpy")(*X), dtype=float) * np.ones(X.shape[1])
        except Exception:  # noqa: BLE001
            return None
        if np.all(vals > 0):
            reps[a] = u
        elif np.all(vals < 0):
            reps[a] = -u
        else:
            return None
    return expr.xreplace(reps) if reps else expr


def prove_equivalence(f: sp.Expr, law_idx: int, X: np.ndarray) -> str | None:
    """Try to prove that ``f`` is the true law up to alpha*f + beta.

    Returns "symbolic", "symbolic (on the physical domain)" or None. The domain
    version uses sign facts read off the sampled points ``X`` (signs of Abs arguments,
    positivity of fractional-power bases), so it is a proof on that domain.
    """
    syms = law_symbols(law_idx)
    g = law_truth(law_idx, syms)
    if affine_equivalent_symbolic(f, g, syms):
        return "symbolic"
    f2 = strip_abs_on_domain(f, syms, X)
    if f2 is not None and affine_equivalent_symbolic(f2, g, syms, domain_X=X):
        return "symbolic (on the physical domain)"
    return None


def verify_law(law_idx, tree, lib, threshold, seed) -> tuple[bool, str, float]:
    """Out-of-distribution numeric test, then symbolic proof against ground truth."""
    X, y = sample_law_data(law_idx, 2000, seed + 999, widen=2.0)
    nm, _, _ = fitness(tree, X, y, lib)
    if not (nm < threshold):
        return False, "failed-ood", nm
    try:
        how = prove_equivalence(to_sympy(tree, law_symbols(law_idx), lib), law_idx, X)
        if how:
            return True, how, nm
    except Exception:  # noqa: BLE001
        pass
    # Numerically exact far outside the training range: accept, flagged as such.
    return (nm < threshold * 1e-2), "numeric-ood", nm


def _convergents(x: float, max_den: int):
    """Continued-fraction convergents of x (its best rational approximations)."""
    h, h1, k, k1 = math.floor(x), 1, 1, 0
    yield Fraction(h, k)
    frac = x - math.floor(x)
    for _ in range(40):
        if frac < 1e-15:
            return
        x = 1 / frac
        a = math.floor(x)
        frac = x - a
        h, h1, k, k1 = a * h + h1, h, a * k + k1, k
        if k > max_den:
            return
        yield Fraction(h, k)


def snap_value(v: float, rel_tol: float = 1e-7, max_cost: int = 7) -> sp.Expr | None:
    """The shortest exact number within ``rel_tol`` of a fitted value, or None.

    Candidates: a decimal with at most 6 significant digits (9.81, 6.674e-11), a
    fraction with denominator <= 1000 (1/3), or such a fraction times pi (2*pi).
    The shortest written form wins, and it must be short (at most ``max_cost``
    digits): every number is close to *some* long integer, so snapping 1658324077453.2
    to 1658324077453 would only disguise a fitted constant as an exact one. This only
    *proposes* an exact value: the snapped formula still has to pass the full verifier.
    """
    if not math.isfinite(v):
        return None
    if v == 0:
        return sp.Integer(0)
    tol = rel_tol * abs(v)
    best = None                                   # (cost, expr)
    for digits in range(1, 7):
        s = f"{v:.{digits}g}"
        if abs(float(s) - v) <= tol:
            fr = Fraction(s)
            best = (digits, sp.Rational(fr.numerator, fr.denominator))
            break
    for mult, extra in ((1, 0), (math.pi, 1)):
        for fr in _convergents(v / mult, 1000 if mult == 1 else 100):
            if abs(float(fr) * mult - v) <= tol:
                cost = len(str(abs(fr.numerator))) + len(str(fr.denominator)) + extra
                if best is None or cost < best[0]:
                    best = (cost, sp.Rational(fr.numerator, fr.denominator) * (sp.pi if extra else 1))
                break
    return best[1] if best and best[0] <= max_cost else None


def normalize_affine(expr: sp.Expr) -> sp.Expr:
    """Drop what the fitted scale and offset absorb anyway, so only inner constants remain.

    * an overall factor and an additive constant (alpha * f + beta), and
    * a common scale of a fraction's numerator and denominator: both are divided by
      one coefficient of the denominator, so d1*d2/(1.4e8*d1 + 1.4e8*d2) becomes
      d1*d2/(d1 + 0.99999999*d2) / 1.4e8, whose constants are short (or junk shows).
    """
    num, den = sp.fraction(sp.together(expr))
    if den != 1:
        den = sp.expand(den)
        first = den.as_ordered_terms()[0] if isinstance(den, sp.Add) else den
        c, _ = first.as_coeff_Mul()
        if c.is_number and c != 0 and c != 1:
            num, den = sp.expand(num / c), sp.expand(den / c)
        expr = num / den
    rest = expr.as_coeff_Add()[1]
    if rest != 0:
        expr = rest
    coeff, rest = expr.as_coeff_Mul()
    if coeff != 0 and rest != 1:
        expr = rest
    return expr


def expr_fit(expr: sp.Expr, syms, X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """NMSE of y ~ alpha * expr + beta on (X, y), as in ``fitness`` but for a SymPy
    expression. Returns (nmse, alpha, beta); nmse is inf if expr can't be evaluated."""
    try:
        with np.errstate(all="ignore"):
            f = np.asarray(sp.lambdify(syms, expr, "numpy")(*X), dtype=float) * np.ones(len(y))
    except Exception:  # noqa: BLE001
        return math.inf, 0.0, 0.0
    if not np.all(np.isfinite(f)):
        return math.inf, 0.0, 0.0
    A = np.vstack([f, np.ones_like(f)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    nm = float(np.mean((A @ (alpha, beta) - y) ** 2) / (np.var(y) or 1.0))
    return (nm if math.isfinite(nm) else math.inf), float(alpha), float(beta)


def check_claim(law_idx, tree, lib, threshold, seed) -> dict:
    """Head-side check of a worker's claimed law.

    Returns {"ok", "how", "nmse_ood", "expr" (the formula that was verified), "alpha", "beta"}.
    Shared by ``LawsEngine.ingest`` and the benchmark, so both judge results identically.

    Trees without fitted constants go through ``verify_law`` exactly as before. With
    fitted constants the formula is cleaned up first, at the SymPy level where constants
    have been merged (c1*c2 -> one coefficient):
      1. drop negligible constants: zero any float whose removal keeps the formula
         numerically exact out of distribution (the search likes to hide junk terms
         such as + 4.8e-9*n that barely change the fit);
      2. remove the overall scale and offset (``normalize_affine``);
      3. snap the remaining floats to short exact values (19.62 -> 981/50).
    A formula all of whose constants snapped may be accepted by proof or numerically.
    One with a constant that has no short exact form (often junk riding on a huge
    constant, e.g. d1*d2*(d1 + 652377492)/(d1 + d2)) is accepted only with a proof.
    """
    syms = law_symbols(law_idx)
    X, y = sample_law_data(law_idx, 2000, seed + 999, widen=2.0)     # the sample verify_law uses
    if not any(n[0] == "c" for _, n in subtrees(tree)):
        ok, how, nm = verify_law(law_idx, tree, lib, threshold, seed)
        _, alpha, beta = fitness(tree, X, y, lib)
        return {"ok": ok, "how": how if ok else "failed", "nmse_ood": nm,
                "expr": sp.simplify(to_sympy(tree, syms, lib)), "alpha": alpha, "beta": beta}
    raw = sp.simplify(to_sympy(tree, syms, lib))
    nm_raw = expr_fit(raw, syms, X, y)[0]
    fail = {"ok": False, "how": "failed", "nmse_ood": nm_raw, "expr": raw, "alpha": 0.0, "beta": 0.0}
    if not nm_raw < threshold:
        return fail
    cleaned, nm_cur = raw, nm_raw
    for f in sorted(raw.atoms(sp.Float), key=lambda a: abs(float(a))):
        trial = cleaned.xreplace({f: sp.Integer(0)})
        nm_t = expr_fit(trial, syms, X, y)[0]
        if nm_t < max(threshold * 1e-2, 10 * nm_cur):
            cleaned, nm_cur = sp.simplify(trial), nm_t
    norm = sp.simplify(normalize_affine(cleaned))
    snaps = {f: snap_value(float(f)) for f in norm.atoms(sp.Float)}
    exact = all(v is not None for v in snaps.values())
    expr = sp.simplify(norm.xreplace(snaps)) if exact and snaps else norm
    nm, alpha, beta = expr_fit(expr, syms, X, y)
    if not nm < threshold:
        return fail
    try:
        how = prove_equivalence(expr, law_idx, X)
    except Exception:  # noqa: BLE001
        how = None
    if how or (exact and nm < threshold * 1e-2):
        return {"ok": True, "how": how or "numeric-ood", "nmse_ood": nm, "expr": expr,
                "alpha": alpha, "beta": beta}
    return fail

# ----------------------------------------------------------------------------- library learning

def abstract_subtree(node):
    """Turn a concrete subtree into a template with argument placeholders."""
    mapping: dict[int, int] = {}

    def go(n):
        if n[0] == "v":
            if n[1] not in mapping:
                mapping[n[1]] = len(mapping)
            return ("a", mapping[n[1]])
        if n[0] in ("k", "a"):
            return n
        ci = children_idx(n)
        return tuple(go(n[i]) if i in ci else n[i] for i in range(len(n)))

    body = go(node)
    return body, len(mapping)

# ----------------------------------------------------------------------------- engine

def law_task(c: dict, law_idx: int, X, y, seed: int, lib: list, op_weights: dict,
             units: list | None = None) -> dict:
    """One search task for ``laws_worker``. Shared by the engine and the benchmark so
    both always search with exactly the same settings. Benchmark laws bring their own
    units; for datasets pass ``units`` (one entry per input column, None = unknown)."""
    if c.get("units", True):
        units = law_dims(law_idx) if law_idx >= 0 else units
    else:
        units = None
    return {"law": law_idx, "X": np.asarray(X).tolist(), "y": np.asarray(y).tolist(), "seed": seed,
            "library": lib, "op_weights": op_weights,
            "population": c["population"], "generations": c["generations"],
            "parsimony": 0.02, "success_nmse": c["success_nmse"],
            "fit_constants": c.get("fit_constants", True), "const_prob": c.get("const_prob", 0.08),
            "units": units, "units_penalty": c.get("units_penalty", 2.0)}


class LawsEngine(Engine):
    name = "laws"
    worker_fn = staticmethod(laws_worker)

    def __init__(self, cfg, full_cfg, store, rng):
        super().__init__(cfg, full_cfg, store, rng)
        self.curriculum.max_level = max(t for *_, t in LAWS)
        self.curriculum.target, self.curriculum.band = 0.5, 0.2
        self.library: list[dict] = []
        self.op_weights: dict[str, float] = {}
        self.solved: dict[str, dict] = {}      # law name -> best verified solution
        self.attempts: Counter = Counter()
        self.datasets = self._load_datasets(cfg.get("datasets") or [])
        self.dataset_best: dict[str, dict] = {}

    # --------------------------------------------------------------- datasets
    @staticmethod
    def _load_datasets(specs):
        out = []
        for spec in specs:
            with open(spec["path"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            target = spec["target"]
            features = [c for c in rows[0] if c != target]
            X = np.array([[float(r[c]) for r in rows] for c in features])
            y = np.array([float(r[target]) for r in rows])
            units = spec.get("units") or {}
            dims = [parse_units(units.get(c)) for c in features] if units else None   # unlisted: unknown
            out.append({"name": spec.get("name", spec["path"]), "vars": features, "X": X, "y": y,
                        "units": dims})
        return out

    # --------------------------------------------------------------- loop
    def _pick_law(self) -> int:
        level = self.curriculum.sample_level(self.rng)
        pool = [i for i, l in enumerate(LAWS) if l[4] <= level]
        unsolved = [i for i in pool if LAWS[i][0] not in self.solved]
        if unsolved and self.rng.random() < 0.8:
            pool = unsolved
        return self.rng.choice(pool)

    def make_tasks(self, round_: int) -> list[dict]:
        c = self.cfg
        lib = [{"body": m["body"], "arity": m["arity"]} for m in self.library]
        tasks = []
        for _ in range(c["tasks_per_round"]):
            li = self._pick_law()
            seed = self.rng.randrange(1 << 30)
            X, y = sample_law_data(li, c["samples"], seed, noise=c["noise"])
            self.attempts[LAWS[li][0]] += 1
            tasks.append(law_task(c, li, X, y, seed, lib, self.op_weights))
        for ds in self.datasets:
            t = law_task(c, -1, ds["X"], ds["y"], self.rng.randrange(1 << 30), lib, self.op_weights,
                         units=ds.get("units"))
            t.update({"dataset": ds["name"], "generations": c["generations"] * 2,
                      "seeds": [self.dataset_best[ds["name"]]["tree"]] if ds["name"] in self.dataset_best else []})
            tasks.append(t)
        return tasks

    def ingest(self, results: list[dict], round_: int) -> dict:
        solved_now, new, attempted = 0, 0, 0
        for res in results:
            tree = to_tuple(res["tree"])
            if res["law"] == -1:
                self._ingest_dataset(res, tree, round_)
                continue
            attempted += 1
            li = res["law"]
            name, vars_, formula, _, tier = LAWS[li]
            if not (res["nmse_val"] < self.cfg["success_nmse"]):
                continue
            chk = check_claim(li, tree, self.library, self.cfg["success_nmse"], round_)
            if not chk["ok"]:
                continue
            solved_now += 1
            how, nm, expr, alpha, beta = chk["how"], chk["nmse_ood"], chk["expr"], chk["alpha"], chk["beta"]
            prev = self.solved.get(name)
            if prev is None or res["size"] < prev["size"]:
                self.solved[name] = {"tree": tree, "size": res["size"], "expr": sp.sstr(expr)}
                for op in ops_used(tree, self.library):
                    self.op_weights[op] = self.op_weights.get(op, 1.0) + 1.0
            statement = (f"{name}: y = {alpha:.6g} * ({sp.sstr(expr)})"
                         + (f" + {beta:.3g}" if abs(beta) > 1e-9 * max(1, abs(alpha)) else "")
                         + f"   [truth: {formula}]")
            key = stable_hash("law", name, sp.srepr(expr))
            difficulty = 1.0 - 1.0 / (1 + self.attempts[name])
            if self.store.add_discovery(self.name, key, "law", statement,
                                        {"law": name, "expr": sp.sstr(expr), "alpha": alpha, "beta": beta,
                                         "verified": how, "nmse_ood": nm, "size": res["size"],
                                         "library_size": len(self.library), "tier": tier},
                                        interestingness(res["size"], difficulty, 1.0 if prev is None else 0.3),
                                        round_):
                new += 1
        self.curriculum.update(solved_now, max(1, attempted))
        return {"attempted": attempted, "solved": solved_now, "new_discoveries": new,
                "laws_solved_total": len(self.solved), "laws_total": len(LAWS)}

    def _ingest_dataset(self, res, tree, round_):
        ds = next(d for d in self.datasets if d["name"] == res["dataset"])
        cur = self.dataset_best.get(ds["name"])
        if cur is None or res["nmse_val"] < cur["nmse_val"]:
            syms = sp.symbols(ds["vars"])
            expr = round_floats(sp.simplify(to_sympy(tree, syms, self.library)))
            self.dataset_best[ds["name"]] = {"tree": tree, "nmse_val": res["nmse_val"], "expr": sp.sstr(expr)}
            if res["nmse_val"] < 1e-6:
                self.store.add_discovery(
                    self.name, stable_hash("dataset", ds["name"], sp.srepr(expr)), "empirical",
                    f"{ds['name']}: y = {res['alpha']:.6g} * ({sp.sstr(expr)}) + {res['beta']:.6g}"
                    f"   [held-out NMSE {res['nmse_val']:.2e}]",
                    {"dataset": ds["name"], "expr": sp.sstr(expr), "nmse_val": res["nmse_val"]},
                    interestingness(res["size"], 0.7, 1.0), round_)

    # --------------------------------------------------------------- learning
    def learn(self, round_: int) -> dict:
        """Library learning: abstract recurring sub-expressions into new primitives."""
        counts: Counter = Counter()
        users: dict = {}
        bodies: dict = {}
        for name, sol in self.solved.items():
            seen = set()
            for _, st in subtrees(sol["tree"]):
                sz = size(st)
                if sz < 3 or sz > 9 or st[0] in ("v", "k", "c"):
                    continue
                if any(n[0] == "c" for _, n in subtrees(st)):
                    continue   # a fitted constant belongs to one law, not to a reusable primitive
                body, arity = abstract_subtree(st)
                if arity == 0 or arity > 3:
                    continue
                if all(body[i][0] == "a" for i in children_idx(body)):
                    continue   # just one primitive applied to arguments: nothing to learn
                k = repr(body)
                if k in seen:
                    continue
                seen.add(k)
                counts[k] += sz - 1          # compression gained per use
                users.setdefault(k, set()).add(name)
                bodies[k] = (body, arity)
        existing = {repr(m["body"]) for m in self.library}
        existing_pretty = {m.get("pretty") for m in self.library}
        added = 0
        for k, gain in counts.most_common():
            if len(self.library) >= self.cfg["max_library"]:
                break
            if k in existing:
                continue
            if len(users[k]) >= 2 or (gain >= 5 and len(self.solved) >= 3):
                body, arity = bodies[k]
                args = sp.symbols(f"A0:{arity}", real=True)
                expr = sp.simplify(to_sympy(body, [], self.library, list(args)))
                if expr.free_symbols != set(args):
                    continue      # e.g. A0/A0: simplifies away an argument, useless as a primitive
                pretty = sp.sstr(expr)
                if pretty in existing_pretty:
                    continue      # same function as an existing primitive, different spelling
                self.library.append({"body": body, "arity": arity, "pretty": pretty,
                                     "users": sorted(users[k]), "round": round_})
                self.op_weights[f"m{len(self.library) - 1}"] = 2.0
                existing.add(k)
                existing_pretty.add(pretty)
                added += 1
        return {"library_size": len(self.library), "library_added": added}

    # --------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update({"library": self.library, "op_weights": self.op_weights, "solved": self.solved,
                  "attempts": dict(self.attempts), "dataset_best": self.dataset_best})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        if not d:
            return
        self.library = [{**m, "body": to_tuple(m["body"])} for m in d.get("library", [])]
        self.op_weights = d.get("op_weights", {})
        self.solved = {k: {**v, "tree": to_tuple(v["tree"])} for k, v in d.get("solved", {}).items()}
        self.attempts = Counter(d.get("attempts", {}))
        self.dataset_best = {k: {**v, "tree": to_tuple(v["tree"])} for k, v in d.get("dataset_best", {}).items()}

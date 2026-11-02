"""Engine 1: discovering formulas for fundamental constants.

Object of study: polynomial continued fractions (PCFs)

    L = a(0) + b(1) / (a(1) + b(2) / (a(2) + b(3) / (a(3) + ...)))

with integer polynomials a(n), b(n). Classic examples the engine can find by
itself:

    4/pi  = 1 + 1^2/(3 + 2^2/(5 + 3^2/(7 + ...)))      a(n)=2n+1, b(n)=n^2
    e - 2 = 1/(1 + 1/(2 + 2/(3 + 3/(4 + ...))))         a(n)=n,    b(n)=n
    6/zeta(3) (Apery)                                   a(n)=34n^3+51n^2+27n+5, b(n)=-n^6

Pipeline per candidate (all exact or arbitrary-precision, no floats):
  1. Evaluate convergents p_n/q_n with exact integer recurrences.
  2. Keep it only if it converges to >= ``screen_digits`` digits.
  3. Discard rational limits (trivial).
  4. Integer relation search (PSLQ) for p + q*K - r*L - s*K*L = 0, i.e.
     L = (p + q K)/(r + s K), against each target constant K.
  5. Re-evaluate at much greater depth and confirm the relation to
     ``verify_digits`` digits. Only then is it a discovery.

Results are *conjectures verified to N digits*, which is exactly the status of
the Ramanujan Machine discoveries; many later received proofs.

Learning: a cross-entropy-method proposal distribution over polynomial
degrees and coefficient values is fitted to verified hits, and successful
formulas seed local "family" searches (formulas come in families).
Open targets such as zeta(5) and Euler's gamma are included on purpose.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict

import mpmath as mpm
import sympy as sp

from ..novelty import interestingness, stable_hash
from .base import Engine

# ----------------------------------------------------------------------------- constants

def target_values(dps: int) -> dict[str, mpm.mpf]:
    with mpm.workdps(dps + 10):
        vals = {
            "pi": +mpm.pi, "e": +mpm.e, "ln2": mpm.log(2), "zeta3": mpm.zeta(3),
            "catalan": +mpm.catalan, "sqrt2": mpm.sqrt(2), "phi": +mpm.phi,
            "euler_gamma": +mpm.euler, "zeta5": mpm.zeta(5), "pi2": mpm.pi ** 2,
            "ln3": mpm.log(3), "zeta2": mpm.zeta(2), "sqrt3": mpm.sqrt(3),
        }
    return vals


SYMBOLS = {
    "pi": sp.pi, "e": sp.E, "ln2": sp.log(2), "zeta3": sp.zeta(3), "catalan": sp.Catalan,
    "sqrt2": sp.sqrt(2), "phi": sp.GoldenRatio, "euler_gamma": sp.EulerGamma,
    "zeta5": sp.zeta(5), "pi2": sp.pi ** 2, "ln3": sp.log(3), "zeta2": sp.zeta(2),
    "sqrt3": sp.sqrt(3),
}

# ----------------------------------------------------------------------------- core math

def poly_eval(coeffs: list[int], n: int) -> int:
    v = 0
    for c in reversed(coeffs):
        v = v * n + c
    return v


def poly_mul(p: list[int], q: list[int]) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] += x * y
    return out


def apery_like(u: list[int], s: int, odd: bool) -> tuple[list[int], list[int]]:
    """The structured family behind Apery's fractions.

    a(n) = (2n+1) * P(n(n+1)) if ``odd`` else P(n(n+1)), with P(m) = u[0] + u[1] m + ...,
    and b(n) = s * n^(2 deg a). Examples:
      u = [5, 17], s = -1, odd  ->  a = 34n^3+51n^2+27n+5, b = -n^6   (6/zeta(3))
      u = [3, 11], s = 1, even  ->  a = 11n^2+11n+3,       b = n^4    (5/zeta(2))
    The symmetry a(-1-n) = -a(n) (odd) or a(n) (even) is what these classics share;
    searching the few integers u, s instead of every coefficient makes degree 5 (the
    natural next step, for zeta(5)) searchable at all.
    """
    P = [0] * (2 * len(u) - 1)
    mp = [1]                                        # n(n+1) to the power j
    for c in u:
        for i, x in enumerate(mp):
            P[i] += c * x
        mp = poly_mul(mp, [0, 1, 1])
    a = poly_mul([1, 2], P) if odd else P
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a, [0] * (2 * (len(a) - 1)) + [s]


def convergents(a: list[int], b: list[int], depth: int, checkpoints: tuple[int, ...]):
    """Exact convergents p_n/q_n at the requested depths. Returns {n: (p, q)}.

    Returns None if b(n) == 0 for some 1 <= n <= depth (the fraction terminates,
    so its value is rational and uninteresting).
    """
    p_prev, q_prev = 1, 0
    p, q = poly_eval(a, 0), 1
    out = {}
    cps = set(checkpoints)
    for n in range(1, depth + 1):
        an, bn = poly_eval(a, n), poly_eval(b, n)
        if bn == 0:
            return None
        p, p_prev = an * p + bn * p_prev, p
        q, q_prev = an * q + bn * q_prev, q
        if n in cps:
            out[n] = (p, q)
            # keep integers small: divide out the common factor occasionally
        if n % 64 == 0:
            g = math.gcd(math.gcd(p, p_prev), math.gcd(q, q_prev))
            if g > 1:
                p, p_prev, q, q_prev = p // g, p_prev // g, q // g, q_prev // g
    return out


def evaluate_pcf(a: list[int], b: list[int], depth: int, dps: int):
    """Return (value, digits_of_convergence) or (None, 0)."""
    n1 = max(8, int(depth * 0.8))
    cv = convergents(a, b, depth, (n1, depth))
    if cv is None:
        return None, 0
    (p0, q0), (p1, q1) = cv[n1], cv[depth]
    if q0 == 0 or q1 == 0:
        return None, 0
    with mpm.workdps(dps + 10):
        x0 = mpm.mpf(p0) / q0
        x1 = mpm.mpf(p1) / q1
        diff = abs(x1 - x0)
        scale = max(mpm.mpf(1), abs(x1))
        if diff == 0:
            digits = dps
        else:
            digits = float(-mpm.log10(diff / scale))
        return x1, max(0.0, min(digits, float(dps)))


def find_relation(L, K, digits: float, maxcoeff: int):
    """PSLQ for p + qK - rL - sKL = 0 with the relation genuinely involving L and K."""
    with mpm.workdps(int(digits) + 5):
        tol = mpm.mpf(10) ** (-(int(digits) - 3))
        rel = mpm.pslq([1, K, -L, -K * L], tol=tol, maxcoeff=maxcoeff, maxsteps=4000)
    if rel is None:
        return None
    p, q, r, s = (int(x) for x in rel)
    if q == 0 and s == 0:   # does not involve K
        return None
    if r == 0 and s == 0:   # does not involve L
        return None
    if q * r - p * s == 0:  # degenerate Moebius map (L would be constant / rational)
        return None
    g = math.gcd(math.gcd(p, q), math.gcd(r, s))
    p, q, r, s = p // g, q // g, r // g, s // g
    if (r, s) < (0, 0):
        p, q, r, s = -p, -q, -r, -s
    return p, q, r, s


def is_rational(L, digits: float) -> bool:
    if abs(L) < mpm.mpf(10) ** (-(int(digits) - 2)):
        return True   # converges to 0
    with mpm.workdps(int(digits) + 5):
        tol = mpm.mpf(10) ** (-(int(digits) - 3))
        rel = mpm.pslq([1, L], tol=tol, maxcoeff=10 ** 6, maxsteps=2000)
    return rel is not None


def verify_candidate(a, b, target, rel, depth, digits) -> float:
    """Recompute at high depth/precision. Returns the number of digits confirmed."""
    p, q, r, s = rel
    L, conv = evaluate_pcf(a, b, depth, digits + 20)
    if L is None:
        return 0.0
    with mpm.workdps(digits + 30):
        K = target_values(digits + 30)[target]
        resid = abs(p + q * K - r * L - s * K * L)
        if resid == 0:
            rel_digits = float(digits + 20)
        else:
            rel_digits = float(-mpm.log10(resid / max(mpm.mpf(1), abs(L), abs(K * L))))
    return min(conv, rel_digits)


def convergence_profile(a: list[int], b: list[int], ref_depth: int = 2000, ref_dps: int = 1200) -> dict:
    """How fast a fraction converges, and how well its convergents approximate the limit.

    digits_per_term  slope of -log10|p_n/q_n - L| in n (exponential convergence is the
                     useful kind; Apery's zeta(3) fraction gains about 1.5 digits a term)
    delta            irrationality-measure estimate, as in the Ramanujan Machine papers:
                     |L - p_n/q_n| = q_n^-(1 + delta) with q_n the *reduced* denominator.
                     If delta > 0 holds for all large n, L is irrational (Apery's proof
                     for zeta(3) rests on delta ~ 0.08). Measured at the deepest n where
                     the reference value is still accurate.

    The reference limit is the depth-``ref_depth`` convergent at ``ref_dps`` digits.
    Returns {} if the fraction is degenerate.
    """
    L, ref_digits = evaluate_pcf(a, b, ref_depth, ref_dps)
    if L is None:
        return {}
    grid = []
    n = 4
    while n < ref_depth // 2:
        grid.append(n)
        n *= 2
    cv = convergents(a, b, grid[-1], tuple(grid)) or {}
    pts = []
    with mpm.workdps(ref_dps + 10):
        for n in grid:
            p, q = cv.get(n, (0, 0))
            if q == 0:
                continue
            err = abs(mpm.mpf(p) / q - L)
            d = float(ref_digits) if err == 0 else float(-mpm.log10(err))
            if d > 0.8 * ref_digits:
                break
            g = math.gcd(p, q)
            qr = abs(q // g)
            delta = -1 - (-d * math.log(10)) / math.log(qr) if qr > 1 else None
            pts.append((n, d, delta))
    if len(pts) < 2:
        return {}
    (n1, d1, _), (n2, d2, delta) = pts[-2], pts[-1]
    return {"digits_per_term": round((d2 - d1) / (n2 - n1), 4), "delta": None if delta is None else round(delta, 4),
            "delta_at": n2}


def describe(a: list[int], b: list[int], target: str, rel) -> tuple[str, str]:
    n = sp.Symbol("n")
    pa = sp.expand(sum(c * n ** i for i, c in enumerate(a)))
    pb = sp.expand(sum(c * n ** i for i, c in enumerate(b)))
    p, q, r, s = rel
    K = SYMBOLS[target]
    value = sp.simplify((p + q * K) / (r + s * K))
    return f"PCF[a(n) = {pa}, b(n) = {pb}] = {sp.sstr(value)}", sp.sstr(value)


# ----------------------------------------------------------------------------- equivalence

def shift_poly(c: list[int], s: int) -> list[int]:
    """Coefficients of p(n + s) given those of p(n)."""
    out = [0] * len(c)
    for i, ci in enumerate(c):
        if ci:
            for j in range(i + 1):
                out[j] += ci * math.comb(i, j) * s ** (i - j)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def canonical_pcf(a: list[int], b: list[int], max_shift: int = 6) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One representative per family of trivially equivalent fractions.

    These transformations change a PCF's limit only by a rational Moebius map,
    so they are the same formula in disguise:
      * shift n -> n + s           (start the same fraction s steps later/earlier)
      * a -> k a, b -> k^2 b       (limit scales by k; k = -1 flips the sign)
    The representative is the smallest-coefficient member of the family.
    """
    best = None
    for s in range(-max_shift, max_shift + 1):
        a2, b2 = shift_poly(a, s), shift_poly(b, s)
        g = 0
        for x in a2:
            g = math.gcd(g, x)
        for k in range(g, 1, -1):
            if g % k == 0 and all(x % (k * k) == 0 for x in b2):
                a2 = [x // k for x in a2]
                b2 = [x // (k * k) for x in b2]
                break
        if a2[-1] < 0 or (a2[-1] == 0 and len(a2) == 1 and a2[0] < 0):
            a2 = [-x for x in a2]
        key = (sum(abs(x) for x in a2 + b2), len(a2) + len(b2), tuple(a2), tuple(b2))
        if best is None or key < best:
            best = key
    return best[2], best[3]


# ----------------------------------------------------------------------------- meet-in-the-middle index
# Precompute L = (p + qK)/(r + sK) for every target K and all small integer
# coefficients, keyed by L rounded to 1e-8. Identifying a limit then costs a
# dictionary lookup instead of a PSLQ run (the Ramanujan Machine's MITM trick);
# every hit is still confirmed at full precision before it counts.

_MITM_CACHE: dict = {}


def mitm_table(targets: tuple[str, ...], C: int) -> dict:
    key = (targets, C)
    if key in _MITM_CACHE:
        return _MITM_CACHE[key]
    vals = target_values(30)
    table: dict[int, object] = {}
    rng = range(-C, C + 1)
    for ti, t in enumerate(targets):
        K = float(vals[t])
        for r in rng:
            for s_ in rng:
                if (r, s_) == (0, 0) or r < 0 or (r == 0 and s_ < 0):
                    continue          # one sign per projective class
                den = r + s_ * K
                if abs(den) < 1e-9:
                    continue
                for p in rng:
                    for q in rng:
                        if q * r - p * s_ == 0 or (q == 0 and s_ == 0):
                            continue  # degenerate, or does not involve K
                        if math.gcd(math.gcd(p, q), math.gcd(r, s_)) != 1:
                            continue
                        v = (p + q * K) / den
                        if abs(v) > 1e6:
                            continue
                        k = round(v * 1e8)
                        code = (ti, p, q, r, s_)
                        prev = table.get(k)
                        if prev is None:
                            table[k] = code
                        elif isinstance(prev, list):
                            prev.append(code)
                        else:
                            table[k] = [prev, code]
    _MITM_CACHE[key] = table
    return table


def mitm_lookup(L, table, targets, tv, digits: float):
    Lf = float(L)
    if not math.isfinite(Lf) or abs(Lf) > 1e6:
        return None
    k0 = round(Lf * 1e8)
    with mpm.workdps(int(digits) + 5):
        tol = mpm.mpf(10) ** (-(int(digits) - 3))
        for k in (k0, k0 - 1, k0 + 1):
            e = table.get(k)
            if e is None:
                continue
            for ti, p, q, r, s_ in (e if isinstance(e, list) else [e]):
                K = tv[targets[ti]]
                resid = abs(p + q * K - r * L - s_ * K * L)
                if resid < tol * max(mpm.mpf(1), abs(L), abs(K * L)):
                    return targets[ti], (p, q, r, s_)
    return None


# ----------------------------------------------------------------------------- worker

def constants_worker(task: dict) -> dict:
    """Screen a batch of PCFs, identify limits, verify hits. Runs on any worker."""
    targets: tuple[str, ...] = tuple(task["targets"])
    sdig = task["screen_digits"]
    tv = target_values(sdig + 20)
    table = mitm_table(targets, task.get("mitm_coeff", 6))
    rng = random.Random(task.get("seed", 0))
    hits, converged, failed, pslq_runs = [], 0, 0, 0
    for a, b in task["candidates"]:
        try:
            L, digits = evaluate_pcf(a, b, task["depth"], sdig + 15)
            if L is None or digits < sdig:
                continue
            converged += 1
            if is_rational(L, digits):
                continue
            found = mitm_lookup(L, table, targets, tv, digits)
            if found is None and rng.random() < task.get("pslq_fraction", 0.1):
                pslq_runs += 1
                for t in targets:   # slower, but finds larger coefficients
                    rel = find_relation(L, tv[t], digits, task["maxcoeff"])
                    if rel is not None:
                        found = (t, rel)
                        break
            if found is None:
                continue
            t, rel = found
            vd = verify_candidate(a, b, t, rel, task["verify_depth"], task["verify_digits"])
            if vd >= task["verify_digits"] - 1:
                hits.append({"a": a, "b": b, "target": t, "rel": list(rel), "digits": vd,
                             "value": mpm.nstr(L, 30)})
        except (ValueError, ZeroDivisionError, OverflowError):
            failed += 1   # numerically degenerate candidate; skip it
    return {"hits": hits, "converged": converged, "tested": len(task["candidates"]),
            "failed": failed, "pslq_runs": pslq_runs}


# ----------------------------------------------------------------------------- engine

class ConstantsEngine(Engine):
    name = "constants"
    worker_fn = staticmethod(constants_worker)

    MAX_DEG_A, MAX_DEG_B = 3, 6

    def __init__(self, cfg, full_cfg, store, rng):
        super().__init__(cfg, full_cfg, store, rng)
        # Level = size of the search space (coefficient range and degrees); see ingest().
        self.curriculum.max_level = 20
        self.deg_weights: dict[str, float] = {}
        self.coef_weights: dict[str, dict[int, float]] = {}
        self.seeds: list[tuple[list[int], list[int]]] = []
        self.seen_values: dict[str, int] = {}
        self.tested: OrderedDict[str, None] = OrderedDict()
        self.tested_raw: OrderedDict[str, None] = OrderedDict()
        self.dry_rounds = 0
        self.structured_seeds: list[list] = []    # [u, s, odd] of Apery-like hits
        self._structured_params: dict[str, list] = {}   # this round's candidates -> [u, s, odd]

    # --------------------------------------------------------------- proposal model
    def coef_range(self) -> int:
        return 2 + self.curriculum.level

    def _sample_degrees(self, level: int) -> tuple[int, int]:
        options = [(da, db) for da in range(1, self.MAX_DEG_A + 1)
                   for db in range(0, self.MAX_DEG_B + 1)
                   if db <= 2 * da and da + db <= 2 + level]
        if not options:
            options = [(1, 2)]
        w = [self.deg_weights.get(f"{da},{db}", 1.0) for da, db in options]
        return self.rng.choices(options, weights=w)[0]

    def _sample_coef(self, slot: str, R: int) -> int:
        learned = self.coef_weights.get(slot)
        values = list(range(-R, R + 1))
        w = [1.0 + (learned or {}).get(v, 0.0) for v in values]
        return self.rng.choices(values, weights=w)[0]

    def _random_candidate(self, level: int):
        R = 2 + level
        da, db = self._sample_degrees(level)
        a = [self._sample_coef(f"a{i}", R) for i in range(da + 1)]
        b = [self._sample_coef(f"b{j}", R) for j in range(db + 1)]
        if a[-1] == 0:
            a[-1] = self.rng.choice([1, 2])
        if b[-1] == 0:
            b[-1] = self.rng.choice([-1, 1])
        return a, b

    def _structured_candidate(self, level: int):
        """An Apery-like fraction (see ``apery_like``); half the time a neighbour of a hit.

        Degrees unlock with the level: up to 3 at level 1, 4 at level 3, 5 from level 5.
        """
        if self.structured_seeds and self.rng.random() < 0.5:
            u, s, odd = self.rng.choice(self.structured_seeds)
            u = list(u)
            if self.rng.random() < 0.8:
                i = self.rng.randrange(len(u))
                u[i] += self.rng.choice([-2, -1, 1, 2])
            else:
                s = -s if self.rng.random() < 0.5 else s * self.rng.choice([2, 4])
            if u[-1] == 0:
                return None
        else:
            top = min(self.cfg.get("structured_max_deg", 5), 3 + (level - 1) // 2)
            options = list(range(2, top + 1))
            d = self.rng.choices(options, weights=[self.deg_weights.get(f"s{x}", 1.0) for x in options])[0]
            odd = d % 2 == 1
            k = (d - 1) // 2 if odd else d // 2
            R = 4 * (2 + level)
            u = [self._sample_coef(f"s{d}u{j}", R) for j in range(k)] + [self.rng.randint(1, R)]
            s = self.rng.choice([-1, 1]) * self.rng.choice([1, 1, 1, 2, 4, 8, 16])
        a, b = apery_like(u, s, odd)
        self._structured_params[repr((a, b))] = [u, s, odd]
        return a, b

    def _mutate(self, a, b, R):
        a, b = list(a), list(b)
        kind = self.rng.random()
        if kind < 0.6:          # nudge one coefficient
            tgt = a if self.rng.random() < 0.5 else b
            i = self.rng.randrange(len(tgt))
            tgt[i] += self.rng.choice([-2, -1, 1, 2])
        elif kind < 0.8 and len(self.seeds) > 1:   # recombine two successful formulas
            a2, b2 = self.rng.choice(self.seeds)
            if self.rng.random() < 0.5:
                a = list(a2)
            else:
                b = list(b2)
        else:                   # grow or shrink a polynomial by one degree
            tgt = a if self.rng.random() < 0.5 else b
            if self.rng.random() < 0.5 and len(tgt) <= (self.MAX_DEG_A if tgt is a else self.MAX_DEG_B):
                tgt.append(self.rng.choice([-1, 1, 2]))
            elif len(tgt) > 1:
                tgt.pop()
        while len(a) > 1 and a[-1] == 0:
            a.pop()
        while len(b) > 1 and b[-1] == 0:
            b.pop()
        if not any(b) or any(abs(x) > 10 * R + 50 for x in a + b):
            return None
        return a, b

    def _novel_candidate(self, cand) -> bool:
        if len(cand[0]) == 1 and len(cand[1]) == 1:
            return False   # constant a and b: periodic fraction, always a quadratic irrational
        raw = repr(cand)
        if raw in self.tested_raw:            # cheap check first
            return False
        self.tested_raw[raw] = None
        key = repr(canonical_pcf(*cand))      # never test two disguises of one fraction
        if key in self.tested:
            return False
        self.tested[key] = None
        for d in (self.tested_raw, self.tested):
            if len(d) > 2_000_000:
                d.popitem(last=False)
        return True

    # --------------------------------------------------------------- loop
    def make_tasks(self, round_: int) -> list[dict]:
        c = self.cfg
        tasks = []
        self._structured_params = {}
        for _ in range(c["tasks_per_round"]):
            level = self.curriculum.sample_level(self.rng)
            R = 2 + level
            structured = self.rng.random() < c.get("structured_fraction", 0.0)
            cands = []
            tries = 0
            misses = 0
            while len(cands) < c["candidates_per_task"] and tries < 20 * c["candidates_per_task"]:
                tries += 1
                if misses > 3 * c["candidates_per_task"]:
                    break                     # this level's search space is used up
                if structured:
                    cand = self._structured_candidate(level)
                elif self.seeds and self.rng.random() > c["explore_fraction"]:
                    cand = self._mutate(*self.rng.choice(self.seeds), R)
                else:
                    cand = self._random_candidate(level)
                if cand and self._novel_candidate(cand):
                    cands.append(cand)
                    misses = 0
                else:
                    misses += 1
            if not cands:
                continue
            tasks.append({
                "candidates": cands, "targets": c["targets"], "depth": c["depth"],
                "screen_digits": c["screen_digits"], "verify_depth": c["verify_depth"],
                "verify_digits": c["verify_digits"], "maxcoeff": c["max_relation_coeff"],
                "mitm_coeff": c.get("mitm_coeff", 6),
                # nearly every structured fraction converges, and their limits tend to need
                # relations with larger coefficients (30/pi^2) than the hash table holds
                "pslq_fraction": c.get("structured_pslq_fraction", 0.3) if structured
                else c.get("pslq_fraction", 0.1),
                "seed": self.rng.randrange(1 << 30),
            })
        return tasks

    def ingest(self, results: list[dict], round_: int) -> dict:
        new, verified, tested, converged, productive = 0, 0, 0, 0, 0
        for res in results:
            tested += res["tested"]
            converged += res["converged"]
            task_new = 0
            for h in res["hits"]:
                a, b, t, rel = h["a"], h["b"], h["target"], tuple(h["rel"])
                # Trust but verify: re-check on the head with an independent evaluation.
                digits = verify_candidate(a, b, t, rel, self.cfg["verify_depth"],
                                          self.cfg["verify_digits"])
                if digits < self.cfg["verify_digits"] - 1:
                    continue
                verified += 1
                statement, value_expr = describe(a, b, t, rel)
                cluster = f"{t}:{value_expr}"
                novelty = 1.0 if cluster not in self.seen_values else 0.35
                complexity = sum(abs(x) for x in a + b) ** 0.5 + len(a) + len(b) + sum(abs(x) for x in rel) ** 0.5
                score = interestingness(complexity, difficulty=0.8, novelty=novelty,
                                        precision_digits=digits)
                ca, cb = canonical_pcf(a, b)
                key = stable_hash("pcf", ca, cb)
                if self.store.has_key(key):
                    continue
                if self.store.add_discovery(self.name, key, "conjecture", statement,
                                            {"a": a, "b": b, "canonical": [list(ca), list(cb)],
                                             "target": t, "relation": list(rel),
                                             "digits": round(digits, 1), "value": h["value"],
                                             "closed_form": value_expr, **convergence_profile(a, b)},
                                            score, round_):
                    new += 1
                    task_new += 1
                    self.seen_values[cluster] = self.seen_values.get(cluster, 0) + 1
                    self._learn_from(a, b, weight=novelty)
                    params = self._structured_params.get(repr((a, b)))
                    if params is not None:          # an Apery-like hit: search its family
                        self.structured_seeds = (self.structured_seeds + [params])[-500:]
                        u = params[0]
                        d = 2 * (len(u) - 1) + (1 if params[2] else 0)
                        self.deg_weights[f"s{d}"] = self.deg_weights.get(f"s{d}", 1.0) + 2.0 * novelty
                        for j, cu in enumerate(u[:-1]):
                            w = self.coef_weights.setdefault(f"s{d}u{j}", {})
                            w[cu] = w.get(cu, 0.0) + 2.0 * novelty
            productive += 1 if task_new else 0
        # Curriculum for a *search* problem: the level is the size of the search space.
        # Grow it when the current space is used up or has stopped yielding new results.
        requested = self.cfg["tasks_per_round"] * self.cfg["candidates_per_task"]
        self.dry_rounds = 0 if new else self.dry_rounds + 1
        if (tested < 0.5 * requested or self.dry_rounds >= 3) and self.curriculum.level < self.curriculum.max_level:
            self.curriculum.level += 1
            self.dry_rounds = 0
        return {"tested": tested, "converged": converged, "verified": verified,
                "new_discoveries": new, "productive_tasks": productive, "coef_range": self.coef_range()}

    # --------------------------------------------------------------- learning (CEM)
    def _learn_from(self, a, b, weight: float = 1.0, lr: float = 2.0):
        dk = f"{len(a) - 1},{len(b) - 1}"
        self.deg_weights[dk] = self.deg_weights.get(dk, 1.0) + lr * weight
        for i, c in enumerate(a):
            d = self.coef_weights.setdefault(f"a{i}", {})
            d[c] = d.get(c, 0.0) + lr * weight
        for j, c in enumerate(b):
            d = self.coef_weights.setdefault(f"b{j}", {})
            d[c] = d.get(c, 0.0) + lr * weight
        self.seeds.append((list(a), list(b)))
        if len(self.seeds) > 5000:
            self.seeds = self.seeds[-5000:]

    def learn(self, round_: int) -> dict:
        # Slow decay so the proposal never collapses onto a single family.
        for d in self.coef_weights.values():
            for k in list(d):
                d[k] *= 0.995
        return {"seeds": len(self.seeds)}

    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update({"deg_weights": self.deg_weights, "dry_rounds": self.dry_rounds,
                  "coef_weights": {k: {str(c): w for c, w in v.items()} for k, v in self.coef_weights.items()},
                  "seeds": self.seeds[-5000:], "seen_values": self.seen_values,
                  "structured_seeds": self.structured_seeds})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        if not d:
            return
        self.deg_weights = d.get("deg_weights", {})
        self.coef_weights = {k: {int(c): w for c, w in v.items()} for k, v in d.get("coef_weights", {}).items()}
        self.seeds = [(list(a), list(b)) for a, b in d.get("seeds", [])]
        self.seen_values = d.get("seen_values", {})
        self.dry_rounds = d.get("dry_rounds", 0)
        self.structured_seeds = d.get("structured_seeds", [])

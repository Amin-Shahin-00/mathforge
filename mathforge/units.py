"""Dimensional analysis for expression trees (as in AI Feynman).

Physical laws are dimensionally consistent: you cannot add a length to a time or
take the sine of a mass. Knowing the units of the input columns therefore rules
out most of the expression space before any data is looked at.

Units are exponent vectors over the SI base dimensions (M, L, T, Theta, N, I).
Inference runs bottom-up over a tree:

* variables have their declared units; fixed constants (1, 2, 3, 1/2) are
  dimensionless;
* a *fitted* constant is a wildcard: it may carry any unit (the 19.62 in
  rho*(v^2 + 19.62*h) is really 2*g, in m/s^2), so anything it multiplies or is
  added to is adjustable;
* add/sub need equal units; exp/log/sin/cos need a dimensionless argument;
* the output unit is free, because the fitted global scale alpha can carry units
  (Newton's G, the gas constant R).

The search adds a penalty to inconsistent trees instead of forbidding them, so an
exact but (by these rules) inconsistent formula still beats any approximation.
"""

from __future__ import annotations

import re

BASE = ("kg", "m", "s", "K", "mol", "A")
DERIVED = {
    "N": {"kg": 1, "m": 1, "s": -2}, "J": {"kg": 1, "m": 2, "s": -2}, "W": {"kg": 1, "m": 2, "s": -3},
    "Pa": {"kg": 1, "m": -1, "s": -2}, "Hz": {"s": -1}, "C": {"A": 1, "s": 1},
    "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
}
ZERO = (0.0,) * len(BASE)
WILD = None          # a subtree whose unit can still be chosen freely (contains a fitted constant)


class Inconsistent(Exception):
    pass


def parse_units(text: str | None) -> tuple | None:
    """'kg*m/s^2' -> exponent vector. '' or '1' -> dimensionless; None -> unknown (wildcard)."""
    if text is None:
        return WILD
    s = str(text).replace(" ", "").replace("**", "^")
    if s in ("", "1", "-"):
        return ZERO
    vec = dict.fromkeys(BASE, 0.0)
    sign = 1
    for tok in re.findall(r"[*/]|[^*/]+", s):
        if tok in "*/":
            sign = 1 if tok == "*" else -1
            continue
        m = re.fullmatch(r"([A-Za-z]+)(?:\^(-?[0-9.]+))?", tok)
        if not m:
            if tok == "1":
                continue
            raise ValueError(f"cannot read unit {tok!r} in {text!r}")
        name, p = m.group(1), float(m.group(2) or 1)
        parts = {name: 1} if name in BASE else DERIVED.get(name)
        if parts is None:
            raise ValueError(f"unknown unit {name!r} in {text!r} (use {', '.join(BASE + tuple(DERIVED))})")
        for b, e in parts.items():
            vec[b] += sign * p * e
        sign = 1
    return tuple(vec[b] for b in BASE)


def _same(a, b) -> bool:
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def dims(node, var_dims, lib, args=None):
    """Units of a tree (a vector, or WILD). Raises Inconsistent."""
    t = node[0]
    if t == "v":
        return var_dims[node[1]]
    if t == "k":
        return ZERO
    if t == "c":
        return WILD
    if t == "a":
        return args[node[1]]
    if t == "u":
        d = dims(node[2], var_dims, lib, args)
        op = node[1]
        if op in ("exp", "log", "sin", "cos"):
            if d is not WILD and not _same(d, ZERO):
                raise Inconsistent(op)
            return ZERO
        if d is WILD or op == "neg":
            return d
        k = {"sq": 2.0, "sqrt": 0.5, "inv": -1.0}[op]
        return tuple(k * x for x in d)
    if t == "b":
        a = dims(node[2], var_dims, lib, args)
        b = dims(node[3], var_dims, lib, args)
        op = node[1]
        if op in ("add", "sub"):
            if a is WILD:
                return b
            if b is WILD:
                return a
            if not _same(a, b):
                raise Inconsistent(op)
            return a
        if a is WILD or b is WILD:
            return WILD
        return tuple(x + y if op == "mul" else x - y for x, y in zip(a, b))
    if t == "m":
        vals = [dims(c, var_dims, lib, args) for c in node[2:]]
        return dims(lib[node[1]]["body"], var_dims, lib, vals)
    raise ValueError(t)


def consistent(node, var_dims, lib) -> bool:
    try:
        dims(node, var_dims, lib)
        return True
    except Inconsistent:
        return False

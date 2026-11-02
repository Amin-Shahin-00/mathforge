import random

import sympy as sp

from mathforge.engines.laws import (LAWS, LawsEngine, abstract_subtree, affine_equivalent_symbolic,
                                    bases_positive_on_domain, check_claim, constants_of, evaluate,
                                    fit_constants, law_symbols, laws_worker, sample_law_data, snap_value,
                                    to_sympy, to_tuple, verify_law, with_constants)
from mathforge.store import Store


def law_index(name):
    return next(i for i, l in enumerate(LAWS) if l[0] == name)


def solve(name, seed=1, lib=None, pop=300, gens=40):
    li = law_index(name)
    X, y = sample_law_data(li, 200, seed)
    return li, laws_worker({"law": li, "X": X.tolist(), "y": y.tolist(), "seed": seed,
                            "library": lib or [], "op_weights": {}, "population": pop,
                            "generations": gens, "parsimony": 0.02, "success_nmse": 1e-10})


def test_rediscovers_and_proves_newtons_gravity():
    li, r = solve("newton_gravity")
    assert r["nmse_val"] < 1e-10
    ok, how, _ = verify_law(li, r["tree"], [], 1e-10, 0)
    assert ok and how == "symbolic"


def test_rediscovers_keplers_third_law():
    li, r = solve("kepler_third_law", seed=9)
    ok, how, _ = verify_law(li, r["tree"], [], 1e-10, 0)
    assert ok


def test_symbolic_check_rejects_wrong_law():
    a, b = sp.symbols("a b", positive=True)
    assert affine_equivalent_symbolic(3 * sp.sqrt(a**2 + b**2) + 1, sp.sqrt(a**2 + b**2), [a, b])
    assert not affine_equivalent_symbolic(a + b, sp.sqrt(a**2 + b**2), [a, b])


def test_out_of_distribution_check_rejects_approximations():
    # a polynomial that fits exp(-x^2/2) roughly on the training range but not outside
    li = law_index("gaussian")
    approx = ("b", "sub", ("k", 1.0), ("b", "mul", ("k", 0.5), ("u", "sq", ("v", 0))))
    ok, how, _ = verify_law(li, approx, [], 1e-10, 0)
    assert not ok


def test_library_abstraction_and_macro_evaluation():
    tree = ("u", "sqrt", ("b", "add", ("u", "sq", ("v", 0)), ("u", "sq", ("v", 1))))
    body, arity = abstract_subtree(tree)
    assert arity == 2
    lib = [{"body": body, "arity": 2}]
    import numpy as np
    X = np.array([[3.0, 5.0], [4.0, 12.0]])
    macro_call = ("m", 0, ("v", 0), ("v", 1))
    assert np.allclose(evaluate(macro_call, X, lib), [5.0, 13.0])
    A0, A1 = sp.symbols("A0 A1", real=True)
    assert sp.simplify(to_sympy(body, [], lib, [A0, A1]) - sp.sqrt(A0**2 + A1**2)) == 0


def test_symbolic_form_is_faithful_to_protected_sqrt():
    # The search evaluates sqrt(|u|), so the printed formula must say so when u can be
    # negative; on the physical domain (v < c) the Abs is removed and the law is proved.
    li = law_index("lorentz_factor")                     # 1/sqrt(1 - v^2/c^2), vars (v, c)
    syms = law_symbols(li)
    v, c = ("v", 0), ("v", 1)
    one = ("k", 1.0)
    right = ("b", "div", one, ("u", "sqrt", ("b", "sub", one, ("b", "div", ("u", "sq", v), ("u", "sq", c)))))
    assert to_sympy(right, syms, []).has(sp.Abs)
    ok, how, _ = verify_law(li, right, [], 1e-10, 0)
    assert ok and how == "symbolic (on the physical domain)"
    # 1/sqrt(c^2 - v^2) is off by a factor of c: it must be rejected
    wrong = ("b", "div", one, ("u", "sqrt", ("b", "sub", ("u", "sq", c), ("u", "sq", v))))
    ok, _, _ = verify_law(li, wrong, [], 1e-10, 0)
    assert not ok


def test_domain_proof_combines_powers_only_when_bases_are_positive():
    # The form the search actually found in a 40-round run: sqrt(|1/(1 - v^2/c^2)|).
    # Its gradient ratios reduce to 1/(sqrt(c^2-v^2)*sqrt(1/(c^2-v^2))), which is 1 only
    # because c > v on the domain.
    li = law_index("lorentz_factor")
    syms = law_symbols(li)
    v, c, one = ("v", 0), ("v", 1), ("k", 1.0)
    found = ("u", "sqrt", ("b", "div", one, ("b", "sub", one, ("b", "div", ("u", "sq", v), ("u", "sq", c)))))
    ok, how, _ = verify_law(li, found, [], 1e-10, 0)
    assert ok and how == "symbolic (on the physical domain)"
    # the positivity guard itself
    import numpy as np
    x = sp.Symbol("x", real=True)
    assert bases_positive_on_domain(sp.sqrt(x + 2), [x], np.array([[0.5, 1.0, 3.0]]))
    assert not bases_positive_on_domain(sp.sqrt(x - 2), [x], np.array([[0.5, 1.0, 3.0]]))


def test_engine_learns_a_library(tmp_path, small_cfg):
    cfg = small_cfg("engines.laws.tasks_per_round=6")
    store = Store(tmp_path / "db.sqlite")
    eng = LawsEngine(cfg["engines"]["laws"], cfg, store, random.Random(0))
    results = []
    for name in ["escape_velocity", "pendulum_period", "kinetic_energy", "centripetal_force"]:
        results.append(solve(name, seed=3, pop=400, gens=60)[1])
    stats = eng.ingest(results, 1)
    assert stats["solved"] >= 3
    learned = eng.learn(1)
    assert learned["library_size"] >= 1
    assert all(m["arity"] >= 1 for m in eng.library)
    # state survives a checkpoint
    eng2 = LawsEngine(cfg["engines"]["laws"], cfg, store, random.Random(0))
    eng2.load_state_dict(eng.state_dict())
    assert eng2.library == eng.library and eng2.solved.keys() == eng.solved.keys()


# ----------------------------------------------------------------------------- fitted constants (R4)

def test_snap_value_finds_short_exact_forms():
    assert snap_value(19.6200000031) == sp.Rational(981, 50)
    assert snap_value(6.2831853071795) == 2 * sp.pi
    assert snap_value(0.33333333334) == sp.Rational(1, 3)
    assert snap_value(6.674e-11 * (1 + 1e-10)) == sp.Rational(6674, 10 ** 14)
    assert snap_value(float("nan")) is None


def test_fit_constants_recovers_an_inner_constant():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (2, 100))
    y = 3 * (X[0] ** 2 + 19.62 * X[1]) + 1          # outer scale and offset are free anyway
    tree = ("b", "add", ("u", "sq", ("v", 0)), ("b", "mul", ("c", 1.0), ("v", 1)))
    fitted = fit_constants(tree, X, y, [])
    assert abs(constants_of(fitted)[0] - 19.62) < 1e-8
    assert constants_of(with_constants(fitted, [2.5])) == [2.5]


def test_search_fits_a_constant_the_fixed_ones_cannot_express():
    import numpy as np
    X = np.random.default_rng(1).uniform(0, 3, (1, 200))
    y = np.exp(-0.37 * X[0])
    r = laws_worker({"law": -1, "X": X.tolist(), "y": y.tolist(), "seed": 0, "library": [], "op_weights": {},
                     "population": 200, "generations": 20, "parsimony": 0.02, "success_nmse": 1e-10,
                     "fit_constants": True})
    assert r["nmse_val"] < 1e-12                      # exact: impossible with 1, 2, 3, 1/2 alone ...
    tree = to_tuple(r["tree"])
    assert constants_of(tree)                         # ... so fitted constants did it
    x = sp.Symbol("x0")
    f = sp.lambdify(x, sp.simplify(to_sympy(tree, [x], [])))
    # however it is written, f = a*exp(-0.37 x) + b, so this ratio of differences is exp(-0.37)
    assert abs((f(3.0) - f(2.0)) / (f(2.0) - f(1.0)) - float(sp.exp(-0.37))) < 1e-6


def test_check_claim_merges_and_snaps_constants_then_proves():
    # How a real search wrote Bernoulli: rho/(c1/v)/(c2/v) + rho*h, with only c1*c2 = 19.62
    # identifiable. Neither constant is a short number; their product is.
    li = law_index("bernoulli_pressure")
    rho, v, h = ("v", 0), ("v", 1), ("v", 2)
    found = ("b", "add", ("b", "div", ("b", "div", rho, ("b", "div", ("c", 5.598841364534773), v)),
                          ("b", "div", ("c", 3.504296464672257), v)), ("b", "mul", rho, h))
    chk = check_claim(li, found, [], 1e-10, 0)
    assert chk["ok"] and chk["how"] == "symbolic"
    assert not chk["expr"].atoms(sp.Float)
    # a slightly wrong constant is not a law
    wrong = ("b", "add", ("b", "mul", ("c", 0.052), ("b", "mul", rho, ("u", "sq", v))), ("b", "mul", rho, h))
    assert not check_claim(li, wrong, [], 1e-10, 0)["ok"]


def test_check_claim_drops_negligible_junk_terms():
    # T*n/V + 4.8e-9*n fits the ideal gas law to NMSE ~1e-16, but it is not the law:
    # it must be stored without the junk term, and proved.
    li = law_index("ideal_gas")
    n, T, V = ("v", 0), ("v", 1), ("v", 2)
    junk = ("b", "add", ("b", "div", ("b", "mul", T, n), V), ("b", "mul", ("c", 4.7556e-9), n))
    chk = check_claim(li, junk, [], 1e-10, 0)
    assert chk["ok"] and chk["how"] == "symbolic"
    syms = law_symbols(li)
    assert sp.simplify(chk["expr"] - syms[1] * syms[0] / syms[2]) == 0


def test_junk_behind_huge_constants_is_rejected_but_scale_is_not():
    # Seen in benchmark runs: numerically exact to 1e-16, yet not the law.
    li = law_index("thin_lens")
    d1, d2 = ("v", 0), ("v", 1)
    junk = ("b", "div", ("b", "mul", ("b", "mul", d1, d2), ("b", "add", d1, ("c", 652377492.3))), ("b", "add", d1, d2))
    assert not check_claim(li, junk, [], 1e-10, 0)["ok"]
    assert snap_value(652377492.3) is None and snap_value(1658324077453.2) is None   # no "free" long integers
    # the true law written with a common scale c1 ~ c2 ~ 1.4e8: normalised, snapped, proved
    scaled = ("b", "div", ("b", "mul", d1, d2),
              ("b", "add", ("b", "mul", ("c", 142159816.2), d1), ("b", "mul", ("c", 142159815.3), d2)))
    chk = check_claim(li, scaled, [], 1e-10, 0)
    assert chk["ok"] and chk["how"] == "symbolic"
    # an overall factor with no short form is irrelevant (alpha absorbs it)
    lb = law_index("bernoulli_pressure")
    rho, v, h = ("v", 0), ("v", 1), ("v", 2)
    bern = ("b", "mul", ("c", 2.8035e-6), ("b", "mul", rho, ("b", "add", ("b", "mul", ("c", 19.62000000001), h),
                                                             ("u", "sq", v))))
    assert check_claim(lb, bern, [], 1e-10, 0)["how"] == "symbolic"


def test_library_never_learns_fitted_constants(tmp_path, small_cfg):
    cfg = small_cfg()
    eng = LawsEngine(cfg["engines"]["laws"], cfg, Store(tmp_path / "db.sqlite"), random.Random(0))
    with_c = ("b", "mul", ("c", 19.62), ("u", "sq", ("v", 1)))
    hyp = ("u", "sqrt", ("b", "add", ("u", "sq", ("v", 0)), ("u", "sq", ("v", 2))))
    eng.solved = {"a": {"tree": ("b", "add", with_c, hyp), "size": 11},
                  "b": {"tree": ("b", "sub", with_c, hyp), "size": 11}}
    eng.learn(1)
    assert eng.library                                        # the shared sqrt(A0^2 + A1^2) is learned
    assert all("'c'" not in repr(m["body"]) for m in eng.library)

# ----------------------------------------------------------------------------- units (R5)

def test_units_parse_and_consistency():
    from mathforge.units import WILD, consistent, dims, parse_units
    assert parse_units("N") == parse_units("kg*m/s^2") == parse_units("kg * m * s**-2")
    assert parse_units("1") == parse_units("") and parse_units(None) is WILD
    rho, v, h = parse_units("kg/m^3"), parse_units("m/s"), parse_units("m")
    U = [rho, v, h]
    V0, V1, V2 = ("v", 0), ("v", 1), ("v", 2)
    assert not consistent(("b", "add", V1, V2), U, [])                    # m/s + m
    assert not consistent(("u", "sin", V2), U, [])                        # sin(metres)
    assert consistent(("u", "sin", ("b", "div", V2, V2)), U, [])
    # rho*v^2 + c*rho*h: the fitted constant c carries the units of g (m/s^2)
    bern = ("b", "add", ("b", "mul", V0, ("u", "sq", V1)), ("b", "mul", ("c", 19.6), ("b", "mul", V0, V2)))
    assert consistent(bern, U, [])
    assert not consistent(("b", "add", ("b", "mul", V0, ("u", "sq", V1)), ("b", "mul", ("k", 2.0), ("b", "mul", V0, V2))), U, [])
    assert dims(("u", "sqrt", ("b", "mul", V2, V2)), U, []) == h
    # library primitives are checked through their bodies
    lib = [{"body": ("b", "add", ("a", 0), ("a", 1)), "arity": 2}]
    assert not consistent(("m", 0, V1, V2), U, lib)
    assert consistent(("m", 0, V2, V2), U, lib)


def test_every_benchmark_law_declares_units():
    from mathforge.engines.laws import LAW_UNITS, law_dims
    for i, law in enumerate(LAWS):
        assert len(LAW_UNITS[law[0]]) == len(law[1])
        law_dims(i)                                                       # parses


def test_symbols_respect_physics():
    syms = law_symbols(law_index("projectile_range"))
    assert all(s.is_positive for s in syms)
    syms = law_symbols(law_index("gaussian"))
    assert syms[0].is_positive is None and syms[0].is_real

import time

import pytest
import sympy as sp

from mathforge.verify import lean as leanv
from mathforge.verify.safe_parse import UnsafeExpression, extract_answer, safe_parse
from mathforge.verify.sympy_tasks import (FAMILIES, Timeout, check, generate, time_limit,
                                          validate_proposed_integrand)

# ----------------------------------------------------------------------------- safe parsing

@pytest.mark.parametrize("evil", [
    "__import__('os').system('ls')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "exec('1')",
    "lambda: 1",
    "x if 1 else 2",
])
def test_safe_parse_refuses_code(evil):
    with pytest.raises(UnsafeExpression):
        safe_parse(evil, ("x",))


def test_safe_parse_accepts_maths():
    x = sp.Symbol("x")
    assert safe_parse("x^2*exp(x)/2", ("x",)) == x**2 * sp.exp(x) / 2
    assert safe_parse("2x + ln(x)", ("x",)) == 2 * x + sp.log(x)
    assert safe_parse("[1, -3/2]", ("x",)) == [1, sp.Rational(-3, 2)]


def test_extract_answer():
    assert extract_answer("work...\nANSWER: x**2") == "x**2"
    assert extract_answer("so \\boxed{x^3/3}") == "x^3/3"
    assert extract_answer("no answer here") is None

# ----------------------------------------------------------------------------- sympy families

@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", [1, 4, 8])
def test_generated_problems_accept_truth_and_reject_noise(family, level):
    p = generate(family, level, seed=level * 31 + len(family))
    if family == "integrate":
        good = p["reference"]
    elif family == "solve":
        good = "[" + ", ".join(p["roots"]) + "]"
    else:
        k, n = sp.symbols("k n")
        good = sp.sstr(sp.expand(sp.summation(sp.sympify(p["summand"]), (k, 1, n))))
    assert check(p, f"Reasoning...\nANSWER: {good}")
    assert not check(p, "ANSWER: x**7 + 1")
    assert not check(p, "no final answer")


def test_integration_accepts_any_constant_and_equivalent_forms():
    p = {"family": "integrate", "integrand": "2*sin(x)*cos(x)"}
    assert check(p, "ANSWER: sin(x)**2")
    assert check(p, "ANSWER: -cos(x)**2 + 5")       # differs by a constant
    assert check(p, "ANSWER: -cos(2*x)/2")
    assert not check(p, "ANSWER: Integral(2*sin(x)*cos(x), x)")


def test_absolute_values_are_handled_soundly():
    # ln|u| is fine: same derivative as ln(u)
    assert check({"family": "integrate", "integrand": "1/(x - 3)"}, "ANSWER: log(Abs(x - 3))")
    # but |x - 3| is NOT an antiderivative of 1 on the checked interval (its slope is -1 there);
    # an earlier version dropped every Abs and wrongly accepted this
    assert not check({"family": "integrate", "integrand": "1"}, "ANSWER: Abs(x - 3)")
    assert not check({"family": "integrate", "integrand": "2*x"}, "ANSWER: Abs(x**2 - 9)")


def test_solve_requires_all_roots():
    p = {"family": "solve", "roots": ["1", "2"]}
    assert check(p, "ANSWER: [2, 1]")
    assert not check(p, "ANSWER: [1]")
    assert not check(p, "ANSWER: [1, 2, 3]")


def test_time_limit_stops_runaway_work():
    # SIGALRM on Linux/macOS, a timer interrupt on Windows: either way a hard limit
    t0 = time.monotonic()
    with pytest.raises(Timeout):
        with time_limit(1):
            while True:
                sum(range(1000))
    assert time.monotonic() - t0 < 5
    with time_limit(1):        # a block that finishes in time leaves no stray interrupt behind
        pass
    time.sleep(1.3)


def test_pathological_answer_is_cut_off():
    t0 = time.monotonic()
    assert not check({"family": "sum", "summand": "k", "degree": 1}, "ANSWER: (n + 1)**200000", timeout_s=1)
    assert time.monotonic() - t0 < 6


def test_proposed_problems_get_their_ground_truth_from_sympy():
    from mathforge.verify.sympy_tasks import validate_proposed
    p = validate_proposed("solve", "x^3 = 2")
    assert p["lhs"] == "x**3 - 2" and p["roots"] == ["2**(1/3)"]
    assert check(p, "ANSWER: [2**(1/3)]") and not check(p, "ANSWER: [1.26]")
    assert validate_proposed("solve", "2*x**2 - 3 = 0")["roots"] == ["-sqrt(6)/2", "sqrt(6)/2"]
    assert validate_proposed("solve", "x**2 + 1 = 0") is None          # no real solutions
    assert validate_proposed("solve", "exp(x) - 1") is None            # not a polynomial
    s = validate_proposed("sum", "3*k**2 - k + 1")
    assert s["degree"] == 2 and check(s, "ANSWER: n**3 + n**2 + n") and not check(s, "ANSWER: n**3 + n")
    assert validate_proposed("sum", "5") is None and validate_proposed("sum", "k**9") is None
    assert validate_proposed("solve", "__import__('os').system('x')") is None


def test_proposals_in_the_style_real_models_write_them():
    # Real Qwen2.5-Math-1.5B proposer outputs: no PROBLEM line, the problem in LaTeX, then a solution
    from mathforge.verify.sympy_tasks import extract_proposal, validate_proposed
    integ = ("To create a new integration problem ... We'll focus on the function \\( f(x) = "
             "\\frac{(x + 3)\\cos(x) - 1}{x + 3} \\).\n\nThe integrand in SymPy syntax is:\n\\[ f(x) = ...")
    solve = ("Sure, here's a new polynomial equation for you to solve:\n\n\\[\n2x^3 - 5x^2 + 3x - 1 = 0\n\\]\n\n"
             "The solutions ... ```python\nimport sympy as sp\n```")
    summ = "Let's define the polynomial \\( p(k) = k^3 - k \\).\n\nTo verify that the sum of \\( p(k) \\) ..."
    p = validate_proposed("integrate", extract_proposal("integrate", integ))
    assert p and sp.simplify(sp.sympify(p["integrand"]) - (sp.cos(sp.Symbol("x")) - 1 / (sp.Symbol("x") + 3))) == 0
    p = validate_proposed("solve", extract_proposal("solve", solve))
    assert p and p["lhs"] == "2*x**3 - 5*x**2 + 3*x - 1" and len(p["roots"]) == 1
    p = validate_proposed("sum", extract_proposal("sum", summ))
    assert p and p["summand"] == "k**3 - k" and p["degree"] == 3
    assert extract_proposal("sum", "PROBLEM: 2*k + 1\nmore text") == "2*k + 1"      # the asked-for format wins
    assert extract_proposal("solve", "no formula here") is None


def test_validate_proposed_integrand():
    assert validate_proposed_integrand("x*exp(x)") == "x*exp(x)"
    assert validate_proposed_integrand("5") is None
    assert validate_proposed_integrand("__import__('os')") is None

# ----------------------------------------------------------------------------- Lean

def test_parse_completion_with_helpers():
    comp = """Plan: use a helper.
```lean4
import Mathlib
open Real

lemma helper (a b : ℝ) : 0 ≤ (a - b) ^ 2 := sq_nonneg _

theorem am_gm (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by
  nlinarith [helper a b]
```"""
    prelude, proof = leanv.parse_completion(comp, "theorem am_gm (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2")
    assert prelude.startswith("lemma helper") and "import" not in prelude
    assert proof.startswith("by") and "nlinarith" in proof


def test_parse_statement():
    st = leanv.parse_statement("```lean\nlemma foo (n : ℕ) : n ≤ n ^ 2 + 1 := by sorry\n```")
    assert st == "theorem foo (n : ℕ) : n ≤ n ^ 2 + 1"


@pytest.mark.parametrize("cheat", ["sorry", "by\n  native_decide", "by exact?\n#exit",
                                   "by\n  admit", "by simp\n  axiom bad : False"])
def test_lean_escape_hatches_rejected_without_running_lean(cheat, tmp_path):
    res = leanv.check_batch([{"statement": "theorem t : 1 = 1", "proof": cheat}], str(tmp_path))
    assert not res[0]["ok"] and res[0]["error"] == "forbidden token"


def test_prelude_cannot_define_things(tmp_path):
    item = {"statement": "theorem t : (2 : ℕ) + 2 = 4", "proof": "by rfl",
            "prelude": "instance : Add ℕ := ⟨fun _ _ => 4⟩"}
    res = leanv.check_batch([item], str(tmp_path))
    assert not res[0]["ok"] and "prelude" in res[0]["error"]


def test_lean_batch_attribution_and_evidence(fake_lake):
    items = [
        {"statement": "theorem a1 (n : ℕ) : n + 0 = n", "proof": "by simp"},
        {"statement": "theorem a2 (n : ℕ) : n * 1 = n", "proof": "by\n  bad_tactic"},
        {"statement": "theorem a3 (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
         "prelude": "lemma helper (a b : ℝ) : 0 ≤ (a - b) ^ 2 := sq_nonneg _",
         "proof": "by nlinarith [helper a b]"},
    ]
    res = leanv.check_batch(items, fake_lake)
    assert [r["ok"] for r in res] == [True, False, True]
    assert "unknown tactic" in res[1]["error"]


def test_lean_crash_never_counts_as_success(fake_lake):
    items = [{"statement": "theorem c1 : 1 = 1", "proof": "by rfl -- CRASH"},
             {"statement": "theorem c2 : 2 = 2", "proof": "by rfl"}]
    res = leanv.check_batch(items, fake_lake)
    # the crash gives no evidence for either; c2 is re-checked alone and passes
    assert res[0]["ok"] is False
    assert res[1]["ok"] is True


def test_wellformed_mode_allows_sorry(fake_lake):
    out = leanv.lean_worker({"items": [{"statement": "theorem w (n : ℕ) : n = n"}],
                             "mode": "wellformed", "project_dir": fake_lake, "timeout_s": 30})
    assert out["ok"] == [True]


def test_load_theorems(tmp_path):
    f = tmp_path / "t.lean"
    f.write_text("/- comment -/\ntheorem x1 (n : ℕ) : n ≤ n := by\n  sorry\n\nlemma x2 : True := trivial\n",
                 encoding="utf-8")
    rows = leanv.load_theorems(str(f))
    assert [r["name"] for r in rows] == ["x1", "x2"]
    assert rows[0]["statement"] == "theorem x1 (n : ℕ) : n ≤ n"


# ----------------------------------------------------------------------------- answers in LaTeX

@pytest.mark.parametrize("problem,completion", [
    ({"family": "integrate", "integrand": "x*exp(x)"}, r"So the answer is \boxed{x e^{x} - e^{x} + C}"),
    ({"family": "integrate", "integrand": "2*x"}, "ANSWER: F(x) = x^2 + C"),
    ({"family": "integrate", "integrand": "1/x"}, r"\boxed{\ln\left|x\right| + C}"),
    ({"family": "integrate", "integrand": "1/(x**2+3)"},
     r"\boxed{\frac{1}{\sqrt{3}}\arctan\left(\frac{x}{\sqrt{3}}\right)}"),
    ({"family": "solve", "roots": ["2", "-3"]}, r"\boxed{x = 2 \text{ or } x = -3}"),
    ({"family": "solve", "roots": ["2", "-3"]}, r"\boxed{\{-3, 2\}}"),
    ({"family": "sum", "summand": "k", "degree": 1}, r"\boxed{\frac{n(n+1)}{2}}"),
])
def test_latex_answers_are_understood(problem, completion):
    assert check(problem, completion)


def test_latex_wrong_answers_still_fail():
    assert not check({"family": "integrate", "integrand": "x*exp(x)"}, r"\boxed{x e^{x} + C}")
    assert not check({"family": "sum", "summand": "k", "degree": 1}, r"\boxed{\frac{n^2}{2}}")

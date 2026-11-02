"""Optional: the Lean verifier against a real Lean 4 + Mathlib install (roadmap R2).

Needs ``scripts/setup_lean.sh`` (or ``setup_lean.ps1`` on Windows) to have been run,
so that ``lake`` is found (on PATH or in ~/.elan/bin) and ``lean_workspace`` (or
$MF_LEAN_DIR) holds a built Mathlib project. Each Lean run loads Mathlib, so this
is slow. Run with:

    MF_SLOW=1 pytest tests/test_lean_real.py -q
"""

import os
import shutil
from pathlib import Path

import pytest

from mathforge.verify import lean as leanv

WORKSPACE = Path(os.environ.get("MF_LEAN_DIR", Path(__file__).resolve().parents[1] / "lean_workspace"))
pytestmark = [
    pytest.mark.skipif(os.environ.get("MF_SLOW") != "1", reason="set MF_SLOW=1 to run"),
    pytest.mark.skipif(not shutil.which(leanv.lake_executable()) and not Path(leanv.lake_executable()).exists(),
                       reason="lake not found (run scripts/setup_lean.*)"),
    pytest.mark.skipif(not (WORKSPACE / "lakefile.toml").exists() and not (WORKSPACE / "lakefile.lean").exists(),
                       reason=f"no Lean workspace at {WORKSPACE}"),
]
WS = str(WORKSPACE)


def test_known_proofs_are_accepted_and_wrong_ones_rejected():
    items = [
        {"statement": "theorem add_comm_nat (a b : ℕ) : a + b = b + a", "proof": "by omega"},
        {"statement": "theorem two_mul_le_sq_add_sq (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
         "proof": "by nlinarith [sq_nonneg (a - b)]"},
        {"statement": "theorem two_mul_le_sq_add_sq (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
         "proof": "by linarith"},                                   # wrong: linarith can't do this
        {"statement": "theorem sq_shift_nonneg (x : ℝ) : 0 ≤ x ^ 2 - 2 * x + 1",
         "proof": "by nlinarith [sq_nonneg (x - 1)]"},
        {"statement": "theorem prime_101 : Nat.Prime 101", "proof": "by norm_num"},
    ]
    res = leanv.check_batch(items, WS, timeout_s=600)
    assert [r["ok"] for r in res] == [True, True, False, True, True], res
    assert "error" in res[2]["error"]                               # the error was attributed to item 2


def test_escape_hatches_never_reach_lean_or_never_pass():
    st = "theorem t (a b : ℕ) : a + b = b + a"
    for cheat in ["sorry", "by\n  native_decide", "by\n  admit", "by omega /- unterminated"]:
        assert not leanv.check_batch([{"statement": st, "proof": cheat}], WS, timeout_s=600)[0]["ok"]


def test_axiom_reports_match_real_lean_output():
    # both message formats the verifier relies on, as printed by this Lean version
    out, timed_out = leanv._run_lean(
        leanv.HEADER + "theorem mf_a : (2 : ℕ) + 2 = 4 := by rfl\n"
        "theorem mf_b (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (a - b)]\n"
        "#print axioms mf_a\n#print axioms mf_b\n", WS, 600)
    assert not timed_out
    assert {m.group(1) for m in leanv.NO_AXIOMS_RE.finditer(out)} == {"mf_a"}, out
    found = {m.group(1): m.group(2) for m in leanv.AXIOMS_RE.finditer(out)}
    assert set(found) == {"mf_b"} and {a.strip() for a in found["mf_b"].split(",")} <= leanv.STD_AXIOMS, out


def test_helper_lemmas_and_statement_modes():
    item = {"statement": "theorem am_gm2 (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2",
            "prelude": "lemma helper (a b : ℝ) : 0 ≤ (a - b) ^ 2 := sq_nonneg _",
            "proof": "by nlinarith [helper a b]"}
    assert leanv.check_batch([item], WS, timeout_s=600)[0]["ok"]
    ok = leanv.lean_worker({"items": [{"statement": "theorem w (n : ℕ) : n ≤ n ^ 2 + 1"},
                                      {"statement": "theorem w2 (n : ℕ) : n ≤ undefined_thing"}],
                            "mode": "wellformed", "project_dir": WS, "timeout_s": 600})["ok"]
    assert ok == [True, False]                                      # statements must elaborate
    triv = leanv.lean_worker({"items": [{"statement": "theorem tr (a b : ℕ) : a + b = b + a"}],
                              "mode": "trivial", "project_dir": WS, "timeout_s": 600})["ok"]
    assert triv == [True]                                           # closed by a stock tactic

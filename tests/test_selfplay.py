"""End-to-end tests of the self-play engine with the scripted MockPolicy.

The mock is not a learner; these tests exercise the plumbing (proposing,
solving, verifying, trace collection, the training trigger, adapter
versioning, evaluation, checkpointing) without needing a GPU.
"""

import random

from mathforge.distributed import Executor
from mathforge.engines.selfplay import SelfPlayEngine
from mathforge.store import Store


def make_engine(small_cfg, tmp_path, *overrides):
    cfg = small_cfg("engines.selfplay.enabled=true", "engines.selfplay.policy.backend=mock",
                    "engines.selfplay.problems_per_round=12", "engines.selfplay.policy.samples_per_problem=4",
                    "engines.selfplay.train_every=1", "engines.selfplay.min_new_traces=5",
                    "engines.selfplay.eval_every=2", "engines.selfplay.eval_per_level=1",
                    "engines.selfplay.eval_levels=[1,4]", *overrides)
    store = Store(tmp_path / "db.sqlite")
    return SelfPlayEngine(cfg["engines"]["selfplay"], cfg, store, random.Random(0)), store, cfg


def test_sympy_selfplay_loop(small_cfg, tmp_path):
    eng, store, cfg = make_engine(small_cfg, tmp_path)
    ex = Executor("serial")
    s1 = eng.run_round(ex, 1)
    assert s1["problems"] >= 10
    assert 0 < s1["solve_rate"] <= 1
    assert s1["new_traces"] > 0
    assert s1.get("trained") == 1 and s1["adapter_version"] == 1     # expert-iteration step
    s2 = eng.run_round(ex, 2)
    assert "eval_pass1" in s2 and 0 <= s2["eval_pass1"] <= 1
    assert s2["proposals"] == 3 and s2["proposals_valid"] >= 1     # model-invented problems
    roles = {t["meta"]["role"] for t in store.traces("selfplay")}
    assert "solver" in roles
    # checkpoint round-trip keeps the adapter version and curricula
    eng2, _, _ = make_engine(small_cfg, tmp_path)
    eng2.load_state_dict(eng.state_dict())
    assert eng2.version == eng.version and eng2.eval_set == eng.eval_set


def test_proposer_invents_problems_in_every_family(small_cfg, tmp_path):
    from mathforge.verify.sympy_tasks import check
    eng, store, _ = make_engine(small_cfg, tmp_path, "engines.selfplay.train_every=100",
                                "engines.selfplay.eval_every=0")
    ex = Executor("serial")
    for r in range(1, 3):
        eng.run_round(ex, r)
        assert {f for f, *_ in eng._proposals} == {"integrate", "solve", "sum"}
    invented = [d for d in store.discoveries("selfplay", limit=100) if d["kind"] == "solution"]
    assert {d["payload"]["problem"]["family"] for d in invented} >= {"solve", "sum"}
    # what is stored is enough to re-verify it (mathforge verify does exactly this)
    assert all(check(d["payload"]["problem"], "ANSWER: " + d["payload"]["answer"]) for d in invented)


def test_grpo_warm_starts_with_expert_iteration_then_uses_group_advantages(small_cfg, tmp_path):
    import json
    import statistics
    eng, store, cfg = make_engine(small_cfg, tmp_path, "engines.selfplay.trainer.algorithm=grpo",
                                  "engines.selfplay.trainer.grpo.min_rollouts=8")
    ex = Executor("serial")
    s1 = eng.run_round(ex, 1)
    assert s1["trained"] == 1 and s1["new_rollouts"] > 0
    assert "MOCK_ADAPTER" in "".join(p.name for p in (tmp_path / "run" / "adapters" / "v1").iterdir())
    assert "expert_iteration" in (tmp_path / "run" / "adapters" / "v1" / "MOCK_ADAPTER.txt").read_text()
    s2 = eng.run_round(ex, 2)
    assert s2["trained"] == 1 and s2["grpo_groups"] >= 1 and eng.version == 2
    assert "grpo" in (tmp_path / "run" / "adapters" / "v2" / "MOCK_ADAPTER.txt").read_text()
    rows = [json.loads(l) for l in (tmp_path / "run" / "adapters" / "v2" / "train.jsonl").read_text().splitlines()]
    by_prompt: dict = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], []).append(r["advantage"])
    assert any(min(a) < 0 < max(a) for a in by_prompt.values())     # failures pushed down, successes up
    assert all(abs(statistics.fmean(a)) < 1e-6 for a in by_prompt.values() if len(a) > 1)
    assert not store.rollouts("selfplay", only_unused=True) or s2["new_rollouts"]


def test_lean_selfplay_loop(small_cfg, tmp_path, fake_lake):
    eng, store, cfg = make_engine(small_cfg, tmp_path, "engines.selfplay.verifier=lean",
                                  f"engines.selfplay.lean.project_dir={fake_lake}",
                                  "engines.selfplay.min_new_traces=1000")
    ex = Executor("serial")
    s = eng.run_round(ex, 1)
    assert s["lean_attempted"] > 0
    assert s["lean_proved_total"] >= 1
    # the mock proposes `n + a = a + n`, which the trivial-tactic filter must reject
    assert s["conjectures_proposed"] == 3 and s["conjectures_accepted"] == 0
    theorems = [d for d in store.discoveries("selfplay") if d["kind"] == "theorem"]
    assert theorems and theorems[0]["payload"]["verified"] == "lean-kernel"


def test_lean_repair_and_lemma_library(small_cfg, tmp_path, fake_lake):
    import re
    eng, store, cfg = make_engine(small_cfg, tmp_path, "engines.selfplay.verifier=lean",
                                  f"engines.selfplay.lean.project_dir={fake_lake}",
                                  "engines.selfplay.min_new_traces=1000", "engines.selfplay.proposer_fraction=0")
    ex = Executor("serial")
    s1 = eng.run_round(ex, 1)
    # failed attempts got Lean's error back and some were fixed
    assert s1["lean_repair_attempted"] > 0 and s1["lean_repaired"] >= 1
    assert any(t["meta"]["role"] == "repair" for t in store.traces("selfplay"))
    s2 = eng.run_round(ex, 2)
    assert s2["lean_library_used"] > 0          # proved theorems are offered as lemmas
    # a proof that used library lemmas is stored with them, so it can be re-checked on its own
    with_lemmas = [d for d in store.discoveries("selfplay", limit=100) if d["payload"].get("lemmas_used")]
    assert with_lemmas
    for d in with_lemmas:
        for name in d["payload"]["lemmas_used"]:
            assert re.search(rf"theorem {re.escape(name)}\b", d["payload"]["prelude"])
    # the dependency closure lists every lemma once
    code, order = eng._library_code([t for t in eng.theorems if t["solved"]])
    assert len(order) == len(set(order))


def test_theorem_loaders(tmp_path):
    from mathforge.verify.lean import load_theorems
    j = tmp_path / "minif2f.jsonl"
    j.write_text('{"name": "a", "split": "valid", "formal_statement": "theorem a : 1 = 1 := by sorry"}\n'
                 '{"name": "b", "split": "test", "formal_statement": "theorem b : 2 = 2 := by sorry"}\n',
                 encoding="utf-8")
    assert [t["name"] for t in load_theorems(str(j), "test")] == ["b"]
    assert len(load_theorems(str(j))) == 2
    d = tmp_path / "putnam"
    d.mkdir()
    (d / "p1.lean").write_text("theorem putnam_1 (n : ℕ) : n ≤ n ^ 2 + 1 := by\n  sorry\n", encoding="utf-8")
    (d / "p2.lean").write_text("abbrev putnam_2_solution : ℕ := sorry\n"
                               "theorem putnam_2 : putnam_2_solution = 3 := by\n  sorry\n", encoding="utf-8")
    rows = load_theorems(str(d))
    assert [t["name"] for t in rows] == ["putnam_1"]      # needs its own definition: skipped

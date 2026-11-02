import random

from mathforge.engines.constants import (ConstantsEngine, apery_like, constants_worker, convergence_profile,
                                         describe, evaluate_pcf, find_relation, is_rational, target_values,
                                         verify_candidate)
from mathforge.store import Store

TARGETS = ["pi", "e", "ln2", "zeta3", "catalan", "sqrt2", "phi", "euler_gamma", "zeta5", "pi2"]


def task(cands, pslq_fraction=0.0):
    return {"candidates": cands, "targets": TARGETS, "depth": 300, "screen_digits": 18,
            "verify_depth": 2000, "verify_digits": 50, "maxcoeff": 200,
            "mitm_coeff": 6, "pslq_fraction": pslq_fraction, "seed": 0}


def test_rediscovers_classic_formulas():
    res = constants_worker(task([([1, 2], [0, 0, 1]),               # 4/pi (Brouncker)
                                 ([0, 1], [0, 1]),                  # 1/(e-1)
                                 ([5, 27, 51, 34], [0, 0, 0, 0, 0, 0, -1])]))  # Apery: 6/zeta(3)
    found = {(h["target"], tuple(h["rel"])) for h in res["hits"]}
    assert ("pi", (4, 0, 0, 1)) in found
    assert ("zeta3", (6, 0, 0, 1)) in found
    assert any(t == "e" for t, _ in found)
    assert all(h["digits"] >= 49 for h in res["hits"])


def test_closed_forms_are_readable():
    stmt, value = describe([1, 2], [0, 0, 1], "pi", (4, 0, 0, 1))
    assert value == "4/pi"
    assert "2*n + 1" in stmt and "n**2" in stmt


def test_pslq_path_finds_relations_too():
    L, digits = evaluate_pcf([1, 2], [0, 0, 1], 300, 40)
    rel = find_relation(L, target_values(40)["pi"], digits, 200)
    assert rel == (4, 0, 0, 1)


def test_terminating_and_rational_fractions_are_rejected():
    assert evaluate_pcf([1, 1], [3, -1], 50, 30) == (None, 0)      # b(3) = 0 terminates
    L, d = evaluate_pcf([2], [1], 300, 30)                           # periodic -> quadratic irrational
    assert L is not None
    L2, d2 = evaluate_pcf([1, 0], [0, 0], 10, 30)                    # b(1)=0 -> None
    assert L2 is None
    import mpmath
    with mpmath.workdps(40):
        assert is_rational(mpmath.mpf(3) / 7, 25)
        assert not is_rational(+mpmath.pi, 25)


def test_verification_rejects_a_wrong_relation():
    assert verify_candidate([1, 2], [0, 0, 1], "pi", (4, 0, 0, 1), 1500, 40) >= 39
    assert verify_candidate([1, 2], [0, 0, 1], "pi", (4, 1, 0, 1), 1500, 40) < 5


def test_apery_family_contains_aperys_fractions():
    assert apery_like([5, 17], -1, True) == ([5, 27, 51, 34], [0, 0, 0, 0, 0, 0, -1])      # 6/zeta(3)
    assert apery_like([3, 11], 1, False) == ([3, 11, 11], [0, 0, 0, 0, 1])                   # 5/zeta(2)
    a, b = apery_like([1, 2, 3], -1, True)
    assert len(a) - 1 == 5 and b == [0] * 10 + [-1]                                         # zeta(5) shape


def test_pslq_identifies_aperys_zeta2_fraction():
    # 5/zeta(2) = 30/pi^2: coefficient 30 is beyond the hash table, PSLQ finds it
    res = constants_worker(task([([3, 11, 11], [0, 0, 0, 0, 1])], pslq_fraction=1.0))
    assert [(h["target"], tuple(h["rel"])) for h in res["hits"]] == [("pi2", (30, 0, 0, 1))]


def test_convergence_profile_matches_theory():
    import math
    apery = convergence_profile([5, 27, 51, 34], [0, 0, 0, 0, 0, 0, -1])
    # Apery's convergents gain 8*log10(1 + sqrt 2) digits per term
    assert abs(apery["digits_per_term"] - 8 * math.log10(1 + math.sqrt(2))) < 0.05
    assert apery["delta"] > 0                         # the irrationality-proof signature
    assert convergence_profile([1, 2], [0, 0, 1])["delta"] < 0     # 4/pi: no proof from this one


def test_structured_tasks_and_family_seeding(tmp_path, small_cfg):
    cfg = small_cfg("engines.constants.structured_fraction=1.0", "engines.constants.tasks_per_round=1")
    eng = ConstantsEngine(cfg["engines"]["constants"], cfg, Store(tmp_path / "db.sqlite"), random.Random(0))
    t = eng.make_tasks(1)[0]
    assert t["pslq_fraction"] == cfg["engines"]["constants"]["structured_pslq_fraction"]
    for a, b in t["candidates"]:
        assert len(a) - 1 <= 3 and len(b) - 1 == 2 * (len(a) - 1) and not any(b[:-1])   # level 1: deg <= 3
    # an Apery-like hit seeds a family search around it, and survives a checkpoint
    a, b = apery_like([3, 11], 1, False)
    eng._structured_params[repr((a, b))] = [[3, 11], 1, False]
    eng.ingest([constants_worker(task([(a, b)], pslq_fraction=1.0))], 1)
    assert eng.structured_seeds == [[[3, 11], 1, False]]
    eng2 = ConstantsEngine(cfg["engines"]["constants"], cfg, Store(tmp_path / "db2.sqlite"), random.Random(0))
    eng2.load_state_dict(eng.state_dict())
    assert eng2.structured_seeds == eng.structured_seeds
    stored = eng.store.discoveries("constants")[0]["payload"]
    assert stored["digits_per_term"] > 1 and "delta" in stored


def test_report_groups_formulas_into_classes(tmp_path, small_cfg):
    from mathforge.report import constant_classes
    cfg = small_cfg()
    eng = ConstantsEngine(cfg["engines"]["constants"], cfg, Store(tmp_path / "db.sqlite"), random.Random(0))
    eng.ingest([constants_worker(task([([1, 2], [0, 0, 1]), ([3, 11, 11], [0, 0, 0, 0, 1])], pslq_fraction=1.0))], 1)
    text = "\n".join(constant_classes(eng.store))
    assert "| pi |" in text and "| pi2 |" in text


def test_engine_round_trip(tmp_path, small_cfg):
    cfg = small_cfg()
    store = Store(tmp_path / "db.sqlite")
    eng = ConstantsEngine(cfg["engines"]["constants"], cfg, store, random.Random(0))
    eng._learn_from([1, 2], [0, 0, 1])
    tasks = eng.make_tasks(1)
    assert len(tasks) == 2 and all(t["candidates"] for t in tasks)
    stats = eng.ingest([constants_worker(task([([1, 2], [0, 0, 1])]))], 1)
    assert stats["new_discoveries"] == 1
    again = eng.ingest([constants_worker(task([([1, 2], [0, 0, 1])]))], 2)
    assert again["new_discoveries"] == 0          # deduplicated
    state = eng.state_dict()
    eng2 = ConstantsEngine(cfg["engines"]["constants"], cfg, store, random.Random(0))
    eng2.load_state_dict(state)
    assert eng2.seeds and eng2.coef_weights == eng.coef_weights

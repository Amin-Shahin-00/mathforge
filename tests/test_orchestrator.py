from mathforge.cli import main
from mathforge.orchestrator import Orchestrator
from mathforge.store import Store


def test_runs_checkpoints_and_resumes(small_cfg, tmp_path):
    cfg = small_cfg()
    orch = Orchestrator(cfg)
    orch.run(2)
    orch.close()
    store = Store(tmp_path / "run" / "mathforge.db")
    assert store.load_state("round:constants,laws") == 2
    assert store.load_state("engine:laws") is not None
    assert (tmp_path / "run" / "discoveries.md").exists()
    assert (tmp_path / "run" / "metrics.csv").exists()
    store.close()

    orch = Orchestrator(cfg)                    # resume
    assert orch.round == 2
    orch.run(1)
    orch.close()
    store = Store(tmp_path / "run" / "mathforge.db")
    assert store.load_state("round:constants,laws") == 3
    assert store.count("laws") >= 1


def test_scientific_notation_is_a_number_in_configs(tmp_path):
    # PyYAML (YAML 1.1) reads 1e-10 as a *string*; every laws task then failed at runtime
    from mathforge.config import load_config
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("engines:\n  laws:\n    success_nmse: 1e-10\n  selfplay:\n    trainer:\n      lr: 2e-5\n",
                        encoding="utf-8")
    cfg = load_config(str(cfg_file), [f"run.dir={tmp_path}", "engines.constants.verify_digits=5e1"])
    assert cfg["engines"]["laws"]["success_nmse"] == 1e-10
    assert cfg["engines"]["selfplay"]["trainer"]["lr"] == 2e-5
    assert cfg["engines"]["constants"]["verify_digits"] == 50.0
    assert load_config(None, [f"run.dir={tmp_path}", "run.dir=runs/x"])["run"]["dir"] == "runs/x"   # strings stay


def test_cli_status_discoveries_and_verify(small_cfg, tmp_path, capsys):
    cfg = small_cfg("engines.constants.enabled=false")
    orch = Orchestrator(cfg)
    orch.run(1)
    orch.close()
    run_dir = str(tmp_path / "run")
    main(["status", "--run-dir", run_dir])
    main(["discoveries", "--run-dir", run_dir, "--top", "5"])
    out = capsys.readouterr().out
    assert "laws" in out
    store = Store(tmp_path / "run" / "mathforge.db")
    first = store.discoveries("laws", limit=1)[0]["id"]
    store.close()
    try:
        main(["verify", str(first), "--run-dir", run_dir])
    except SystemExit as e:
        assert e.code == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_cli_fit(tmp_path, capsys):
    import numpy as np

    rng = np.random.default_rng(0)
    m, v = rng.uniform(1, 5, 150), rng.uniform(1, 5, 150)
    path = tmp_path / "energy.csv"
    with open(path, "w") as f:
        f.write("m,v,E\n")
        for a, b in zip(m, v):
            f.write(f"{a},{b},{0.5 * a * b * b}\n")
    main(["fit", str(path), "--target", "E", "--population", "200", "--generations", "30", "--restarts", "1"])
    out = capsys.readouterr().out
    assert "m*v**2" in out


def _decay_csv(path):
    import numpy as np
    rng = np.random.default_rng(1)
    m, t = rng.uniform(1, 5, 150), rng.uniform(0, 3, 150)
    with open(path, "w") as f:
        f.write("m,t,y\n")
        for a, b in zip(m, t):
            f.write(f"{a},{b},{2.3 * a / (b + 1.7)}\n")      # 1.7: not buildable from 1, 2, 3, 1/2


def test_cli_fit_fits_constants_and_takes_units(tmp_path, capsys):
    path = tmp_path / "decay.csv"
    _decay_csv(path)
    main(["fit", str(path), "--target", "y", "--population", "300", "--generations", "40", "--restarts", "1",
          "--units", "m=kg,t=s"])
    out = capsys.readouterr().out
    assert "t+1.7" in out.replace(" ", "")


def test_dataset_units_reach_the_search(tmp_path, small_cfg):
    import random

    from mathforge.engines.laws import LawsEngine
    from mathforge.units import parse_units
    path = tmp_path / "decay.csv"
    _decay_csv(path)
    cfg = small_cfg(f"engines.laws.datasets=[{{path: '{path.as_posix()}', target: y, units: {{t: s}}}}]")
    eng = LawsEngine(cfg["engines"]["laws"], cfg, Store(tmp_path / "db.sqlite"), random.Random(0))
    task = [t for t in eng.make_tasks(1) if t["law"] == -1][0]
    assert task["units"] == [None, parse_units("s")]          # m unknown (wildcard), t in seconds

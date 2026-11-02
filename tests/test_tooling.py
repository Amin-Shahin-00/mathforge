"""Research tooling: benchmark, ablation runner, live dashboard, experiment tracking."""

import json
import sys
import threading
import types
import urllib.request
from http.server import ThreadingHTTPServer

from mathforge.ablate import parse_vary, run_ablation, table, variants
from mathforge.bench import bench_laws, summarize_laws
from mathforge.config import load_config
from mathforge.dashboard import make_handler, snapshot
from mathforge.distributed import Executor
from mathforge.orchestrator import Orchestrator

FAST = ["compute.backend=serial", "engines.constants.enabled=false", "engines.laws.tasks_per_round=2",
        "engines.laws.population=120", "engines.laws.generations=10"]


def test_bench_laws_reports_every_run(tmp_path):
    cfg = load_config(None, [f"run.dir={tmp_path}", "engines.laws.population=150", "engines.laws.generations=15"])
    rows = bench_laws(cfg, [0, 1], Executor("serial"), only=["kinetic_energy", "spring_energy"])
    assert len(rows) == 4 and all(r["solved"] for r in rows)
    text = summarize_laws(rows)
    assert "2/2 laws solved" in text and "kinetic_energy" in text


def test_parse_vary_and_variants():
    vary = parse_vary(["engines.laws.units=true,false", "engines.laws.population=[100, 200]"])
    assert vary == [("engines.laws.units", [True, False]), ("engines.laws.population", [100, 200])]
    assert len(variants(vary)) == 4 and variants([]) == [{}]


def test_ablation_runs_each_variant_and_resumes(tmp_path):
    logs = []
    vary = parse_vary(["engines.laws.units=true,false"])
    rows = run_ablation(None, FAST, vary, [0], 1, tmp_path, log=logs.append)
    assert [r["variant"] for r in rows] == ["units=true", "units=false"]
    assert all("laws.results" in r and "laws.laws_solved_total" in r for r in rows)
    assert "| units=true | 1 |" in table(rows)
    logs.clear()
    run_ablation(None, FAST, vary, [0], 1, tmp_path, log=logs.append)       # nothing left to do
    assert all("already done" in m for m in logs)


def test_dashboard_serves_the_page_and_live_state(tmp_path):
    cfg = load_config(None, [f"run.dir={tmp_path}"] + FAST)
    orch = Orchestrator(cfg)
    orch.run(2)
    orch.close()
    db = tmp_path / "mathforge.db"
    snap = snapshot(db)
    assert snap["round"] == 2 and "laws_solved_total" in snap["engines"]["laws"]["curves"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(db))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert "<title>MathForge live</title>" in page
        assert state["round"] == 2 and state["recent"]
    finally:
        server.shutdown()
        server.server_close()


def test_tracking_is_optional_and_resumes_the_same_run(tmp_path, monkeypatch):
    calls = {"init": [], "log": []}

    class FakeRun:
        def log(self, values, step):
            calls["log"].append((step, values))

        def finish(self):
            calls["finish"] = True

    fake = types.SimpleNamespace(util=types.SimpleNamespace(generate_id=lambda: "abc123"),
                                 init=lambda **kw: calls["init"].append(kw) or FakeRun())
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setitem(sys.modules, "mlflow", None)          # not installed: must only warn
    cfg = load_config(None, [f"run.dir={tmp_path}", "run.loggers=[wandb, mlflow]"] + FAST)
    orch = Orchestrator(cfg)
    orch.run(1)
    orch.close()
    assert calls["init"][0]["id"] == "abc123" and calls["finish"]
    step, values = calls["log"][0]
    assert step == 1 and "laws/solved" in values
    orch = Orchestrator(cfg)                                   # resume: same W&B run id
    orch.close()
    assert calls["init"][1]["id"] == "abc123"

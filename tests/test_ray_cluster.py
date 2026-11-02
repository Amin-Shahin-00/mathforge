"""Optional: the Ray backend on a simulated multi-node cluster (roadmap R3).

``ray.cluster_utils.Cluster`` starts real separate Ray nodes on this machine. A worker
node is removed while a round is running; the round must complete without hanging,
with lost tasks either re-run elsewhere (retries) or counted as errors. Needs ray;
run with:

    MF_SLOW=1 pytest tests/test_ray_cluster.py -q
"""

import os
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MF_SLOW") != "1", reason="set MF_SLOW=1 to run")


def slow_task(task: dict) -> dict:
    time.sleep(task["sleep"])
    return {"ok": task["i"]}


@pytest.fixture
def cluster():
    ray = pytest.importorskip("ray")
    from ray.cluster_utils import Cluster

    # pytest imports this file under a name Ray's worker processes can't import: ship it by value
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    c = Cluster(initialize_head=True, head_node_args={"num_cpus": 1})
    worker = c.add_node(num_cpus=4)
    c.wait_for_nodes()
    ray.init(address=c.address, log_to_driver=False)
    timers: list = []
    try:
        yield c, worker, timers
    finally:
        for t in timers:
            t.cancel()
        ray.shutdown()
        c.shutdown()


def _kill_later(c, node, after: float, timers: list):
    t = threading.Timer(after, lambda: c.remove_node(node, allow_graceful=False))
    timers.append(t)
    t.start()


def _node_lost(err: str) -> bool:
    return any(w in err for w in ("node", "Node", "died", "WorkerCrashed", "unavailable"))


def test_lost_node_without_retries_gives_counted_errors(cluster):
    from mathforge.distributed import Executor
    c, worker, timers = cluster
    ex = Executor("ray", ray_address=c.address, ray_max_retries=0)
    assert ex.workers == 5
    _kill_later(c, worker, 2.0, timers)
    t0 = time.time()
    out = ex.map(slow_task, [{"i": i, "sleep": 4} for i in range(10)])
    assert len(out) == 10 and time.time() - t0 < 180
    errors = [r["error"] for r in out if "error" in r]
    oks = [r["ok"] for r in out if "error" not in r]
    assert errors and oks                          # the head node's share finished
    assert all(_node_lost(e) for e in errors), errors


def test_lost_node_with_retries_completes_everything(cluster):
    from mathforge.distributed import Executor
    c, worker, timers = cluster
    ex = Executor("ray", ray_address=c.address, ray_max_retries=3)
    _kill_later(c, worker, 2.0, timers)
    out = ex.map(slow_task, [{"i": i, "sleep": 3} for i in range(6)])
    assert all("ok" in r for r in out), out
    assert sorted(r["ok"] for r in out) == list(range(6))       # re-run on the surviving node


def test_orchestrator_round_survives_losing_a_node(cluster, tmp_path):
    from mathforge.config import load_config
    from mathforge.orchestrator import Orchestrator
    from mathforge.store import Store
    c, worker, timers = cluster
    cfg = load_config(None, [f"run.dir={tmp_path}", "compute.backend=ray", f"compute.ray_address={c.address}",
                             "compute.ray_max_retries=0", "engines.constants.enabled=false",
                             "engines.laws.tasks_per_round=10", "engines.laws.population=300",
                             "engines.laws.generations=120", "engines.laws.success_nmse=1e-300"])
    orch = Orchestrator(cfg)            # success_nmse=1e-300: no early stop, every task runs ~10 s
    _kill_later(c, worker, 5.0, timers)
    orch.run(1)
    orch.close()
    store = Store(tmp_path / "mathforge.db")
    errors = store.metric_series("laws", "errors")
    tasks = store.metric_series("laws", "tasks")
    assert store.load_state("round:laws") == 1 and tasks[0][1] == 10
    assert 1 <= errors[0][1] < 10       # the tasks on the lost node, and only those
    print(f"\nround completed: {errors[0][1]:.0f} of 10 tasks lost with the node, counted as errors")

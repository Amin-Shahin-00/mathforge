"""Compute backends: run worker functions on one machine or across a cluster.

Every engine expresses its heavy work as ``fn(task: dict) -> dict`` where ``fn``
is a top-level (picklable) function. The executor fans tasks out and collects
results. Three backends:

serial  in-process, for debugging and tests
local   all CPU cores of this machine (multiprocessing)
ray     any number of machines: start a Ray cluster, point ``ray_address`` at it

Failures in a single task never kill the run: they come back as
``{"error": "..."}`` and are counted in metrics.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import traceback
from typing import Callable, Iterable


def _worker_init():
    # Ctrl+C reaches the whole process group; only the coordinator should react,
    # so it can finish the current step and save before exiting.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _safe_call(args):
    fn, task = args
    try:
        return fn(task)
    except Exception as e:  # noqa: BLE001 - a worker must never crash the loop
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5)}


class Executor:
    def __init__(self, backend: str = "local", workers: int = 0, ray_address: str = "auto",
                 ray_max_retries: int = 3):
        self.backend = backend
        self.workers = workers or os.cpu_count() or 1
        self._pool = None
        self._ray = None
        if backend == "local":
            # forkserver: workers never inherit the head's GPU/vLLM threads (fork is unsafe there)
            ctx = mp.get_context("spawn" if os.name == "nt" else "forkserver")
            if os.name != "nt":
                ctx.set_forkserver_preload(["numpy", "sympy", "mpmath"])
            self._pool = ctx.Pool(self.workers, initializer=_worker_init, maxtasksperchild=200)
        elif backend == "ray":
            import ray  # optional dependency

            if not ray.is_initialized():
                try:
                    ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
                except ConnectionError:
                    # No cluster running: start a local one so the same config works anywhere.
                    ray.init(ignore_reinit_error=True, log_to_driver=False)
            self._ray = ray
            # a task lost with its machine is re-run elsewhere up to max_retries times;
            # after that it comes back as an error entry (see map)
            self._remote = ray.remote(max_retries=ray_max_retries)(_safe_call)
            self.workers = int(ray.cluster_resources().get("CPU", 1))
        elif backend != "serial":
            raise ValueError(f"unknown backend {backend!r}")

    def map(self, fn: Callable[[dict], dict], tasks: Iterable[dict]) -> list[dict]:
        items = [(fn, t) for t in tasks]
        if not items:
            return []
        if self.backend == "serial":
            return [_safe_call(x) for x in items]
        if self.backend == "local":
            return list(self._pool.imap(_safe_call, items, chunksize=1))
        refs = [self._remote.remote(x) for x in items]   # Ray retries tasks lost with a node
        out = []
        for ref in refs:
            try:
                out.append(self._ray.get(ref))
            except Exception as e:  # noqa: BLE001 - e.g. a spot machine vanished for good
                out.append({"error": f"{type(e).__name__}: {e}"})
        return out

    def describe(self) -> str:
        return f"{self.backend} x{self.workers}"

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except KeyboardInterrupt:
                self._pool.terminate()
            self._pool = None

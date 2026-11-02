"""The interface every discovery engine implements."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

from ..curriculum import Curriculum
from ..distributed import Executor
from ..store import Store


class Engine:
    """One self-improving loop over a family of mathematical objects.

    Subclasses provide:
      make_tasks(round)      -> list of picklable task dicts (the "propose" step)
      worker_fn              top-level function task -> result (the "attempt" step,
                             runs on workers; it must verify what it claims)
      ingest(results, round) re-verify claims on the head, dedupe, store
      learn(round)           update the engine's own model from verified data
    """

    name = "base"
    worker_fn: Callable[[dict], dict] | None = None

    def __init__(self, cfg: dict, full_cfg: dict, store: Store, rng: random.Random):
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.store = store
        self.rng = rng
        c = full_cfg["curriculum"]
        self.curriculum = Curriculum(c["target_success"], c["band"], c["step"],
                                     c["min_level"], c["max_level"])

    # ----------------------------------------------------------------- loop
    def run_round(self, executor: Executor, round_: int) -> dict:
        t0 = time.time()
        tasks = self.make_tasks(round_)
        results = executor.map(type(self).worker_fn, tasks)
        errors = sum(1 for r in results if "error" in r)
        if errors:   # counted in the metrics too, but say *why*, or a config typo looks like bad luck
            first = next(r for r in results if "error" in r)
            logging.getLogger("mathforge").warning(
                f"[round {round_}] {self.name}: {errors}/{len(tasks)} tasks failed, e.g. {first['error']}")
        stats = self.ingest([r for r in results if "error" not in r], round_)
        stats.update(self.learn(round_))
        stats["errors"] = errors
        stats["tasks"] = len(tasks)
        stats["seconds"] = round(time.time() - t0, 2)
        stats["level"] = self.curriculum.level
        return stats

    def make_tasks(self, round_: int) -> list[dict]:
        raise NotImplementedError

    def ingest(self, results: list[dict], round_: int) -> dict:
        raise NotImplementedError

    def learn(self, round_: int) -> dict:
        return {}

    # ----------------------------------------------------------------- checkpoint
    def state_dict(self) -> dict:
        return {"curriculum": self.curriculum.state_dict()}

    def load_state_dict(self, d: dict | None) -> None:
        if d:
            self.curriculum.load_state_dict(d.get("curriculum"))

    def close(self) -> None:
        pass

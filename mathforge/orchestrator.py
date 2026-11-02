"""The forever loop: run every enabled engine round after round, checkpointing
as it goes, so the process can be stopped (Ctrl+C, SIGTERM, reboot) and
resumed without losing anything it has learned.
"""

from __future__ import annotations

import logging
import random
import signal
import sys
import time
from pathlib import Path

import yaml

from .distributed import Executor
from .engines.constants import ConstantsEngine
from .engines.laws import LawsEngine
from .engines.selfplay import SelfPlayEngine
from .report import write_report
from .store import Store
from .tracking import Tracking

ENGINES = {"constants": ConstantsEngine, "laws": LawsEngine, "selfplay": SelfPlayEngine}
SUMMARY_KEYS = ["new_discoveries", "verified", "tested", "solved", "laws_solved_total",
                "library_size", "solve_rate", "pass_at_k", "eval_pass1", "lean_proved_total",
                "adapter_version", "trained", "train_failed", "level", "errors", "seconds"]


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("mathforge")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    path = str((run_dir / "mathforge.log").resolve())
    for h in list(log.handlers):
        # one process can run several runs in turn (ablations): log to the current one only
        if isinstance(h, logging.FileHandler) and h.baseFilename != path:
            log.removeHandler(h)
            h.close()
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


def fmt_stats(stats: dict) -> str:
    parts = []
    for k in SUMMARY_KEYS:
        if k in stats:
            v = stats[k]
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) and k != "seconds" else f"{k}={v}")
    return "  ".join(parts)


class Orchestrator:
    def __init__(self, cfg: dict, executor: Executor | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg["run"]["dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.run_dir)
        self.store = Store(self.run_dir / "mathforge.db")
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        names = [n for n in ENGINES if cfg["engines"][n].get("enabled")]
        if not names:
            raise SystemExit("No engines enabled. Enable at least one under 'engines:' in the config.")
        self.round_key = "round:" + ",".join(sorted(names))
        self.round = int(self.store.load_state(self.round_key, 0))
        seed = cfg["run"]["seed"]
        self.engines = []
        for name in names:
            rng = random.Random(f"{seed}:{name}:{self.round}")
            eng = ENGINES[name](cfg["engines"][name], cfg, self.store, rng)
            eng.load_state_dict(self.store.load_state(f"engine:{name}"))
            self.engines.append(eng)
        self.executor = executor or Executor(cfg["compute"]["backend"], cfg["compute"]["workers"],
                                             cfg["compute"]["ray_address"],
                                             cfg["compute"].get("ray_max_retries", 3))
        self.tracking = Tracking(cfg, self.store, self.log)
        self._stop = False

    # ----------------------------------------------------------------- control
    def _install_signals(self):
        def handler(signum, frame):  # noqa: ARG001
            if self._stop:
                raise KeyboardInterrupt
            self._stop = True
            self.log.info("Stop requested: finishing the current step, then saving. "
                          "Press Ctrl+C again to abort immediately.")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def checkpoint(self):
        for eng in self.engines:
            self.store.save_state(f"engine:{eng.name}", eng.state_dict())
        self.store.save_state(self.round_key, self.round)

    # ----------------------------------------------------------------- loop
    def run(self, max_rounds: int | None = None) -> None:
        max_rounds = self.cfg["run"]["max_rounds"] if max_rounds is None else max_rounds
        self._install_signals()
        self.log.info(f"MathForge starting at round {self.round + 1} | engines: "
                      f"{', '.join(e.name for e in self.engines)} | compute: {self.executor.describe()} | "
                      f"run dir: {self.run_dir}")
        done = 0
        try:
            while not self._stop and (max_rounds == 0 or done < max_rounds):
                self.round += 1
                t0 = time.time()
                for eng in self.engines:
                    if self._stop:
                        break
                    try:
                        stats = eng.run_round(self.executor, self.round)
                    except Exception:  # noqa: BLE001 - one engine failing must not stop the others
                        self.log.exception(f"[round {self.round}] engine {eng.name} crashed; continuing")
                        stats = {"crashed": 1}
                    self.store.log_metrics(self.round, eng.name, stats)
                    self.tracking.log_round(self.round, eng.name, stats)
                    self.log.info(f"[round {self.round}] {eng.name:<9} {fmt_stats(stats)}")
                done += 1
                if self.round % self.cfg["run"]["checkpoint_every"] == 0:
                    self.checkpoint()
                if self.round % self.cfg["run"]["report_every"] == 0:
                    write_report(self.store, self.run_dir)
                self.log.info(f"[round {self.round}] done in {time.time() - t0:.1f}s | "
                              f"knowledge base: {self.store.count()} verified results")
        finally:
            self.checkpoint()
            write_report(self.store, self.run_dir)
            self.log.info(f"Saved. Resume any time with the same command. Report: "
                          f"{self.run_dir / 'discoveries.md'}")

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def close(self):
        for eng in self.engines:
            eng.close()
        self.tracking.close()
        self.executor.close()
        self.store.close()
        for h in list(self.log.handlers):          # release the log file (Windows locks open files)
            if isinstance(h, logging.FileHandler):
                self.log.removeHandler(h)
                h.close()

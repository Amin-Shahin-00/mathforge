"""Optional experiment tracking: send every round's metrics to W&B and/or MLflow.

    run:
      loggers: [wandb, mlflow]
      wandb_project: mathforge
      mlflow_experiment: mathforge

Neither package is a dependency. If one is missing or fails to start, a warning is
logged and the loop carries on: tracking must never stop a discovery run. The W&B
run id is kept in the run's database, so a resumed run continues the same W&B run.
"""

from __future__ import annotations

import logging
from pathlib import Path


class Tracking:
    def __init__(self, cfg: dict, store, log: logging.Logger):
        self.log = log
        self.sinks: list[tuple[str, object]] = []
        rc = cfg["run"]
        name = Path(rc["dir"]).name
        for kind in rc.get("loggers") or []:
            try:
                if kind == "wandb":
                    import wandb

                    run_id = store.load_state("tracking:wandb_id") or wandb.util.generate_id()
                    store.save_state("tracking:wandb_id", run_id)
                    run = wandb.init(project=rc.get("wandb_project", "mathforge"), name=name, id=run_id,
                                     resume="allow", config=cfg)
                    self.sinks.append(("wandb", run))
                elif kind == "mlflow":
                    import mlflow

                    mlflow.set_experiment(rc.get("mlflow_experiment", "mathforge"))
                    run_id = store.load_state("tracking:mlflow_id")
                    run = mlflow.start_run(run_id=run_id, run_name=None if run_id else name)
                    store.save_state("tracking:mlflow_id", run.info.run_id)
                    if not run_id:
                        mlflow.log_params({k: str(v)[:250] for k, v in _flatten(cfg).items()})
                    self.sinks.append(("mlflow", mlflow))
                else:
                    log.warning(f"[tracking] unknown logger {kind!r} (use wandb or mlflow)")
            except Exception as e:  # noqa: BLE001 - tracking is optional
                log.warning(f"[tracking] {kind} disabled: {type(e).__name__}: {e}")

    def log_round(self, round_: int, engine: str, stats: dict) -> None:
        values = {f"{engine}/{k}": float(v) for k, v in stats.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for kind, sink in list(self.sinks):
            try:
                if kind == "wandb":
                    sink.log(values, step=round_)
                else:
                    sink.log_metrics({k.replace("/", "."): v for k, v in values.items()}, step=round_)
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"[tracking] {kind} failed and is now off: {e}")
                self.sinks.remove((kind, sink))

    def close(self) -> None:
        for kind, sink in self.sinks:
            try:
                sink.finish() if kind == "wandb" else sink.end_run()
            except Exception:  # noqa: BLE001
                pass
        self.sinks = []


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out

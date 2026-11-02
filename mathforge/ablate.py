"""Ablation runner: configs x seeds -> one results table.

    mathforge ablate -c configs/laptop.yaml --rounds 10 --seeds 3 \\
        --vary engines.laws.fit_constants=true,false --vary engines.laws.units=true,false

Every combination of the ``--vary`` values is run once per seed, each in its own run
directory under ``--out-dir`` (so an interrupted ablation resumes where it stopped:
finished runs are not repeated). The table reports, per variant, the mean and spread
over seeds of each engine's final metrics, and is written as Markdown and CSV.

Unlike ``mathforge bench`` (cold-start search power on fixed tasks), this measures the
whole system, learning included, over a number of rounds.
"""

from __future__ import annotations

import csv
import itertools
import re
import statistics
from pathlib import Path

import yaml

from .config import yaml_load

FINAL =["laws_solved_total", "library_size", "eval_pass1", "lean_proved_total", "coef_range", "level"]


def parse_vary(items: list[str]) -> list[tuple[str, list]]:
    """'a.b=x,y' or 'a.b=[x, y]' -> ('a.b', [x, y]) with YAML-typed values."""
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--vary must look like key=v1,v2, got {item!r}")
        key, raw = item.split("=", 1)
        parsed = yaml_load(raw)
        if isinstance(parsed, list):
            values = parsed
        else:
            values = [yaml_load(v) for v in raw.split(",")]
        out.append((key.strip(), values))
    return out


def variants(vary: list[tuple[str, list]]) -> list[dict]:
    keys = [k for k, _ in vary]
    return [dict(zip(keys, combo)) for combo in itertools.product(*[v for _, v in vary])] or [{}]


def _label(v: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]}={json_short(x)}" for k, x in v.items()) or "base"


def json_short(x) -> str:
    return yaml.safe_dump(x, default_flow_style=True).strip().removesuffix("...").strip()


def _slug(v: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_=.-]+", "_", "_".join(f"{k.split('.')[-1]}={json_short(x)}"
                                                      for k, x in v.items())) or "base"


def summarize_run(db_path: Path) -> dict[str, float]:
    """Final metrics of one run: per engine, the result count, the sum of
    new_discoveries and seconds, and the last value of the FINAL metrics."""
    from .store import Store

    store = Store(db_path)
    try:
        out: dict[str, float] = {}
        engines = [r[0] for r in store.db.execute("SELECT DISTINCT engine FROM metrics")]
        for e in engines:
            out[f"{e}.results"] = store.count(e)
            for name, agg in (("new_discoveries", "SUM"), ("seconds", "SUM")):
                v = store.db.execute(f"SELECT {agg}(value) FROM metrics WHERE engine=? AND name=?",
                                     (e, name)).fetchone()[0]
                if v is not None:
                    out[f"{e}.{name}"] = v
            for name in FINAL:
                s = store.metric_series(e, name)
                if s:
                    out[f"{e}.{name}"] = s[-1][1]
        return out
    finally:
        store.close()


def run_ablation(config: str | None, overrides: list[str], vary: list[tuple[str, list]], seeds: list[int],
                 rounds: int, out_dir: Path, log=print) -> list[dict]:
    from .config import load_config
    from .orchestrator import Orchestrator
    from .store import Store

    rows = []
    for v in variants(vary):
        for seed in seeds:
            run_dir = out_dir / _slug(v) / f"seed{seed}"
            sets = overrides + [f"{k}={json_short(x)}" for k, x in v.items()] + \
                [f"run.seed={seed}", f"run.dir={run_dir.as_posix()}", "run.max_rounds=0"]
            cfg = load_config(config, sets)
            names = sorted(n for n in cfg["engines"] if cfg["engines"][n].get("enabled"))
            db = run_dir / "mathforge.db"
            done = 0
            if db.exists():
                st = Store(db)
                done = int(st.load_state("round:" + ",".join(names), 0))
                st.close()
            if done < rounds:
                log(f"[ablate] {_label(v)} seed {seed}: rounds {done + 1}..{rounds} in {run_dir}")
                orch = Orchestrator(cfg)
                try:
                    orch.run(rounds - done)
                finally:
                    orch.close()
                if orch.stop_requested:            # Ctrl+C ends the whole ablation, not just one run
                    log("[ablate] stopped; run the same command again to continue")
                    return rows
            else:
                log(f"[ablate] {_label(v)} seed {seed}: already done ({done} rounds)")
            rows.append({"variant": _label(v), "seed": seed, **summarize_run(db)})
    return rows


def table(rows: list[dict]) -> str:
    metrics = sorted({k for r in rows for k in r if k not in ("variant", "seed")})
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    lines = ["| variant | seeds | " + " | ".join(metrics) + " |", "|---|---:|" + "---:|" * len(metrics)]
    for name, rs in by_variant.items():
        cells = []
        for m in metrics:
            vals = [r[m] for r in rs if m in r]
            if not vals:
                cells.append("-")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3g}")
            else:
                cells.append(f"{statistics.mean(vals):.3g} ± {statistics.stdev(vals):.2g}")
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_cli(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_ablation(args.config, args.set, parse_vary(args.vary), list(range(args.seeds)), args.rounds, out_dir)
    text = table(rows)
    print(text)
    (out_dir / "ablation.md").write_text(text + "\n", encoding="utf-8")
    keys = ["variant", "seed"] + sorted({k for r in rows for k in r if k not in ("variant", "seed")})
    with open(out_dir / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {out_dir / 'ablation.md'} and ablation.csv")

"""Benchmarks: an engine's raw search power on fixed tasks.

The forever loop is the wrong place to judge a change to a search algorithm: its
library, learned grammar and curriculum all depend on everything that happened
before. A benchmark runs every task from a cold start with fixed seeds and a fixed
budget, so two versions of the code (or two configs) can be compared directly:

    mathforge bench laws --seeds 3
    mathforge bench laws --seeds 3 --set engines.laws.fit_constants=false

Claims are judged by the same head-side check the engine uses before storing.
"""

from __future__ import annotations

import json
import statistics
import time

import sympy as sp

from .distributed import Executor


def _timed(task: dict) -> dict:
    from .engines.laws import laws_worker

    t0 = time.perf_counter()
    out = laws_worker(task)
    out["seconds"] = time.perf_counter() - t0
    return out


def bench_laws(cfg: dict, seeds: list[int], executor: Executor, only: list[str] | None = None) -> list[dict]:
    """Search every benchmark law once per seed; return one row per (law, seed)."""
    from .engines.laws import (LAWS, check_claim, law_symbols, law_task, round_floats, sample_law_data,
                               to_sympy, to_tuple)

    c = cfg["engines"]["laws"]
    picked = [i for i, law in enumerate(LAWS) if not only or law[0] in only]
    tasks = []
    for li in picked:
        for s in seeds:
            X, y = sample_law_data(li, c["samples"], s, noise=c["noise"])
            tasks.append(law_task(c, li, X, y, s, [], {}))
    rows = []
    for task, res in zip(tasks, executor.map(_timed, tasks)):
        li = task["law"]
        name, _, truth, _, tier = LAWS[li]
        row = {"law": name, "tier": tier, "seed": task["seed"], "truth": truth, "solved": False,
               "how": "", "expr": "", "error": res.get("error", "")}
        if not row["error"]:
            tree = to_tuple(res["tree"])
            row.update({"nmse_val": res["nmse_val"], "size": res["size"], "seconds": res["seconds"],
                        "generations": res["generations"]})
            expr = None
            if res["nmse_val"] < c["success_nmse"]:
                chk = check_claim(li, tree, [], c["success_nmse"], task["seed"])
                row.update({"solved": chk["ok"], "how": chk["how"] if chk["ok"] else ""})
                expr = chk["expr"]
            try:
                if expr is None:
                    expr = sp.simplify(to_sympy(tree, law_symbols(li), []))
                row["expr"] = sp.sstr(round_floats(expr))
            except Exception:  # noqa: BLE001 - display only
                row["expr"] = "?"
        rows.append(row)
    return rows


def summarize_laws(rows: list[dict]) -> str:
    """Markdown table: one line per law, then totals."""
    by_law: dict[str, list[dict]] = {}
    for r in rows:
        by_law.setdefault(r["law"], []).append(r)
    lines = ["| law | tier | solved | proof | median s | found |", "|---|---:|---:|---|---:|---|"]
    n_laws = n_runs = n_ok = 0
    secs = []
    for law, rs in by_law.items():
        ok = [r for r in rs if r["solved"]]
        hows = sorted({r["how"] for r in ok})
        times = [r["seconds"] for r in rs if "seconds" in r]
        secs += times
        found = (ok[0] if ok else min(rs, key=lambda r: r.get("nmse_val", 1e9)))["expr"]
        lines.append(f"| {law} | {rs[0]['tier']} | {len(ok)}/{len(rs)} | {', '.join(hows) or '-'} | "
                     f"{statistics.median(times) if times else 0:.1f} | `{found[:60]}` |")
        n_laws += bool(ok)
        n_runs += len(rs)
        n_ok += len(ok)
    lines.append("")
    lines.append(f"**{n_laws}/{len(by_law)} laws solved at least once; {n_ok}/{n_runs} runs solved; "
                 f"{sum(secs):.0f} s of search in total ({statistics.median(secs) if secs else 0:.1f} s median per run).**")
    return "\n".join(lines)


def run_cli(args) -> None:
    from .config import load_config

    cfg = load_config(args.config, args.set)
    ex = Executor(cfg["compute"]["backend"] if args.workers != 1 else "serial", args.workers or 0)
    try:
        t0 = time.time()
        seeds = list(range(args.first_seed, args.first_seed + args.seeds))
        rows = bench_laws(cfg, seeds, ex, args.only.split(",") if args.only else None)
    finally:
        ex.close()
    print(summarize_laws(rows))
    print(f"\nWall time {time.time() - t0:.0f} s on {ex.describe()}.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, default=str)
        print(f"Rows written to {args.out}")

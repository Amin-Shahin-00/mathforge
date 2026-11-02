"""Command-line interface.

    mathforge run         start (or resume) the forever loop
    mathforge status      what the system knows and how it is learning
    mathforge discoveries list verified results
    mathforge verify ID   independently re-verify one stored result
    mathforge fit FILE    find a formula for your own CSV data
    mathforge bench laws  cold-start benchmark of the laws search (fixed seeds)
    mathforge ablate      config variants x seeds -> results table
    mathforge dashboard   live web page for a running (or finished) run
    mathforge report      write discoveries.md / metrics.csv / learning_curves.png
    mathforge export      dump the knowledge base as JSON Lines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run_dir(args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    from .config import load_config

    return Path(load_config(args.config, args.set)["run"]["dir"])


def _store(args):
    from .store import Store

    db = _run_dir(args) / "mathforge.db"
    if not db.exists():
        sys.exit(f"No run found at {db.parent}. Start one with: mathforge run")
    return Store(db)


# ----------------------------------------------------------------------------- commands

def cmd_run(args):
    from .config import load_config
    from .orchestrator import Orchestrator

    cfg = load_config(args.config, args.set)
    if args.only:
        keep = set(args.only.split(","))
        for name in cfg["engines"]:
            cfg["engines"][name]["enabled"] = name in keep
    orch = Orchestrator(cfg)
    try:
        orch.run(args.rounds)
    finally:
        orch.close()


def cmd_status(args):
    store = _store(args)
    print(f"Run directory: {_run_dir(args)}")
    print(f"Verified results: {store.count()}\n")
    for engine in ("constants", "laws", "selfplay"):
        n = store.count(engine)
        state = store.load_state(f"engine:{engine}")
        if state is None and n == 0:
            continue
        print(f"== {engine}: {n} results")
        rows = store.db.execute(
            "SELECT name, value, round FROM metrics m WHERE engine=? AND round="
            "(SELECT MAX(round) FROM metrics WHERE engine=?)", (engine, engine)).fetchall()
        if rows:
            print(f"   last round {rows[0][2]}: " + ", ".join(
                f"{k}={v:.3g}" for k, v, _ in rows if k not in ("tasks",)))
        if state:
            if engine == "laws":
                print(f"   laws solved: {len(state.get('solved', {}))}, library primitives: "
                      f"{len(state.get('library', []))}")
            if engine == "selfplay":
                print(f"   model adapter version: v{state.get('version', 0)}")
                if state.get("theorems"):
                    t = state["theorems"]
                    print(f"   Lean theorems proved: {sum(1 for x in t if x['solved'])}/{len(t)}")
            cur = state.get("curriculum", {})
            if cur:
                print(f"   curriculum level: {cur.get('level')}")
        print()


def cmd_discoveries(args):
    store = _store(args)
    rows = store.discoveries(args.engine, limit=args.top, order=args.order)
    if not rows:
        print("Nothing yet. Let it run a little longer.")
    for d in rows:
        print(f"[{d['id']:>5}] {d['engine']:<9} {d['kind']:<10} score={d['score']:.2f}  round {d['round']}")
        print(f"        {d['statement']}")


def cmd_verify(args):
    store = _store(args)
    row = store.db.execute("SELECT engine, kind, payload, statement FROM discoveries WHERE id=?",
                           (args.id,)).fetchone()
    if not row:
        sys.exit(f"No discovery with id {args.id}")
    engine, kind, payload, statement = row[0], row[1], json.loads(row[2]), row[3]
    print(statement)
    ok, detail = False, ""
    if engine == "constants":
        from .engines.constants import verify_candidate

        digits = verify_candidate(payload["a"], payload["b"], payload["target"], tuple(payload["relation"]),
                                  args.depth, args.digits)
        ok, detail = digits >= args.digits - 1, f"{digits:.1f} digits agree at depth {args.depth}"
    elif engine == "laws" and kind == "law":
        import numpy as np
        import sympy as sp

        from .engines.laws import LAWS, law_symbols, prove_equivalence, sample_law_data

        li = next(i for i, l in enumerate(LAWS) if l[0] == payload["law"])
        syms = law_symbols(li)
        f = sp.sympify(payload["expr"], locals={s.name: s for s in syms})
        X, y = sample_law_data(li, 5000, 4242, widen=3.0)
        how = prove_equivalence(f, li, X)
        if how:
            ok, detail = True, f"{how} proof of equivalence with the true law"
        else:
            fx = np.asarray(sp.lambdify(syms, f, "numpy")(*X), dtype=float) * np.ones(len(y))
            A = np.vstack([fx, np.ones_like(fx)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            nmse = float(np.mean((A @ coef - y) ** 2) / np.var(y))
            ok, detail = nmse < 1e-12, f"out-of-distribution NMSE {nmse:.2e}"
    elif engine == "selfplay" and kind == "theorem":
        from .config import load_config
        from .verify.lean import check_batch

        cfg = load_config(args.config, args.set)
        lc = cfg["engines"]["selfplay"]["lean"]
        res = check_batch([{"statement": payload["statement"], "proof": payload["proof"],
                            "prelude": payload.get("prelude", "")}],
                          str(Path(lc["project_dir"]).resolve()), lc["timeout_s"], lc.get("header") or None)[0]
        ok, detail = res["ok"], res["error"] or "accepted by the Lean kernel"
    elif engine == "selfplay":
        from .verify.sympy_tasks import check

        if kind == "solution" and "problem" in payload:
            ok = check(payload["problem"], "ANSWER: " + payload["answer"])
        elif kind == "solution":            # stored before every family had a proposer
            problem = {"family": "integrate", "integrand": payload["integrand"]}
            ok = check(problem, "ANSWER: " + payload["antiderivative"])
        else:
            ok = check(payload["problem"], "ANSWER: " + payload["answer"])
        detail = "re-checked with the SymPy verifier"
    else:
        detail = "empirical result: re-run `mathforge fit` on the dataset to reproduce"
    print(("VERIFIED  " if ok else "NOT VERIFIED  ") + detail)
    sys.exit(0 if ok else 1)


def cmd_fit(args):
    import csv

    import numpy as np
    import sympy as sp

    from .config import DEFAULTS
    from .distributed import Executor
    from .engines.laws import law_task, laws_worker, round_floats, to_sympy, to_tuple

    with open(args.file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.target not in rows[0]:
        sys.exit(f"Column {args.target!r} not found. Columns: {', '.join(rows[0])}")
    feats = [c for c in rows[0] if c != args.target]
    X = np.array([[float(r[c]) for r in rows] for c in feats])
    y = np.array([float(r[args.target]) for r in rows])
    c = {**DEFAULTS["engines"]["laws"], "population": args.population, "generations": args.generations,
         "success_nmse": 1e-12, "fit_constants": not args.no_constants}
    units = None
    if args.units:                      # e.g. "m=kg,v=m/s"; columns not listed are unknown
        from .units import parse_units

        given = dict(kv.split("=", 1) for kv in args.units.split(","))
        units = [parse_units(given.get(f)) for f in feats]
    tasks = [law_task(c, -1, X, y, s, [], {}, units=units) for s in range(args.restarts)]
    ex = Executor("local" if args.restarts > 1 else "serial")
    try:
        res = [r for r in ex.map(laws_worker, tasks) if "error" not in r]
    finally:
        ex.close()
    syms = sp.symbols(feats)
    seen = set()
    print(f"Best formulas for {args.target} (held-out normalised error, lower is better):")
    for r in sorted(res, key=lambda r: (r["nmse_val"], r["size"])):
        e = round_floats(sp.simplify(to_sympy(to_tuple(r["tree"]), syms, [])))
        s = f"{args.target} = {r['alpha']:.6g} * ({e}) + {r['beta']:.6g}"
        if s in seen:
            continue
        seen.add(s)
        print(f"  NMSE {r['nmse_val']:.2e}  size {r['size']:>2}   {s}")


def cmd_bench(args):
    from .bench import run_cli

    run_cli(args)


def cmd_ablate(args):
    from .ablate import run_cli

    run_cli(args)


def cmd_dashboard(args):
    from .dashboard import serve

    serve(_run_dir(args), args.host, args.port)


def cmd_report(args):
    from .report import write_report

    print(write_report(_store(args), _run_dir(args)))


def cmd_export(args):
    store = _store(args)
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for d in store.discoveries(None, limit=10 ** 9, order="old"):
        out.write(json.dumps(d, default=str) + "\n")
    if args.out:
        out.close()
        print(f"Wrote {args.out}")


# ----------------------------------------------------------------------------- parser

def main(argv=None):
    p = argparse.ArgumentParser(prog="mathforge", description="A self-improving mathematical discovery system.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp_, run_dir=True):
        sp_.add_argument("--config", "-c", help="YAML config (see configs/)")
        sp_.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="override a config value, e.g. --set engines.constants.depth=500")
        if run_dir:
            sp_.add_argument("--run-dir", help="run directory (default: from config)")

    r = sub.add_parser("run", help="start or resume the forever loop")
    common(r, run_dir=False)
    r.add_argument("--rounds", type=int, default=None, help="stop after N rounds (default: config, 0 = forever)")
    r.add_argument("--only", help="comma-separated engines to run, e.g. constants,laws")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="summary of the knowledge base and learning progress")
    common(s)
    s.set_defaults(fn=cmd_status)

    d = sub.add_parser("discoveries", help="list verified results")
    common(d)
    d.add_argument("--engine", choices=["constants", "laws", "selfplay"])
    d.add_argument("--top", type=int, default=20)
    d.add_argument("--order", choices=["score", "new", "old"], default="score")
    d.set_defaults(fn=cmd_discoveries)

    v = sub.add_parser("verify", help="independently re-verify a stored result")
    common(v)
    v.add_argument("id", type=int)
    v.add_argument("--depth", type=int, default=6000, help="continued-fraction depth (constants)")
    v.add_argument("--digits", type=int, default=100, help="digits to confirm (constants)")
    v.set_defaults(fn=cmd_verify)

    f = sub.add_parser("fit", help="discover a formula for a CSV dataset")
    f.add_argument("file")
    f.add_argument("--target", required=True, help="column to predict")
    f.add_argument("--population", type=int, default=1000)
    f.add_argument("--generations", type=int, default=150)
    f.add_argument("--restarts", type=int, default=8)
    f.add_argument("--no-constants", action="store_true", help="don't fit free numeric constants")
    f.add_argument("--units", help="units of input columns for dimensional analysis, e.g. m=kg,v=m/s")
    f.set_defaults(fn=cmd_fit)

    b = sub.add_parser("bench", help="benchmark an engine from a cold start on fixed seeds")
    common(b, run_dir=False)
    b.add_argument("engine", choices=["laws"])
    b.add_argument("--seeds", type=int, default=3, help="runs per task")
    b.add_argument("--first-seed", type=int, default=0,
                   help="seeds first..first+N-1; tune on one range, confirm on another")
    b.add_argument("--only", help="comma-separated task names, e.g. bernoulli_pressure,doppler")
    b.add_argument("--workers", type=int, default=0, help="0 = all cores, 1 = serial")
    b.add_argument("--out", help="also write every run as JSON")
    b.set_defaults(fn=cmd_bench)

    ab = sub.add_parser("ablate", help="run config variants x seeds and tabulate the results")
    common(ab, run_dir=False)
    ab.add_argument("--vary", action="append", default=[], metavar="KEY=V1,V2",
                    help="a config key and its alternatives, e.g. engines.laws.units=true,false (repeatable)")
    ab.add_argument("--seeds", type=int, default=3)
    ab.add_argument("--rounds", type=int, default=10)
    ab.add_argument("--out-dir", default="runs/ablation")
    ab.set_defaults(fn=cmd_ablate)

    db_ = sub.add_parser("dashboard", help="live web dashboard for a run")
    common(db_)
    db_.add_argument("--host", default="127.0.0.1")
    db_.add_argument("--port", type=int, default=8765)
    db_.set_defaults(fn=cmd_dashboard)

    rep = sub.add_parser("report", help="write discoveries.md, metrics.csv, learning_curves.png")
    common(rep)
    rep.set_defaults(fn=cmd_report)

    e = sub.add_parser("export", help="export all discoveries as JSON Lines")
    common(e)
    e.add_argument("--out")
    e.set_defaults(fn=cmd_export)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

"""Human-readable outputs: discoveries.md, metrics.csv and (if matplotlib is
installed) learning_curves.png, all written into the run directory."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from .store import Store

ENGINE_TITLES = {
    "constants": "Formulas for fundamental constants (continued fractions)",
    "laws": "Laws rediscovered from data",
    "selfplay": "Neural self-play (verified solutions and theorems)",
}
CURVES = [
    ("constants", "new_discoveries", "Constants: new formulas per round", True),
    ("laws", "laws_solved_total", "Laws: benchmark laws solved", False),
    ("laws", "library_size", "Laws: size of learned library", False),
    ("selfplay", "eval_pass1", "Self-play: held-out pass@1", False),
    ("selfplay", "solve_rate", "Self-play: training solve rate", False),
    ("selfplay", "lean_proved_total", "Lean: theorems proved", False),
]


def write_report(store: Store, run_dir: str | Path, top: int = 25) -> Path:
    run_dir = Path(run_dir)
    lines = ["# MathForge discoveries", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Every entry below passed a verifier; "
             "the `verified` field says which one.", ""]
    total = store.count()
    lines.append(f"**{total} verified results** in the knowledge base.")
    lines.append("")
    for engine, title in ENGINE_TITLES.items():
        n = store.count(engine)
        if not n:
            continue
        lines += [f"## {title}", "", f"{n} results. Top {min(top, n)} by interestingness score:", ""]
        for d in store.discoveries(engine, limit=top):
            p = d["payload"]
            how = p.get("verified") or (f"{p['digits']} digits" if "digits" in p else d["kind"])
            lines.append(f"- **[{d['id']}]** `{d['statement']}`  ")
            lines.append(f"  score {d['score']:.2f} · round {d['round']} · verified: {how}")
        lines.append("")
        if engine == "constants":
            lines += constant_classes(store)
        if engine == "laws":
            lib = store.load_state("engine:laws", {}).get("library", [])
            if lib:
                lines += ["### Primitives the system invented for itself (library learning)", ""]
                for i, m in enumerate(lib):
                    users = ", ".join(m.get("users", []))
                    lines.append(f"- `m{i}(A0..A{m['arity'] - 1}) = {m.get('pretty', '?')}`  (abstracted from: {users})")
                lines.append("")
    (run_dir / "discoveries.md").write_text("\n".join(lines), encoding="utf-8")
    _write_metrics(store, run_dir)
    _plot(store, run_dir)
    return run_dir / "discoveries.md"


def constant_classes(store: Store) -> list[str]:
    """Group continued-fraction formulas by the constant they evaluate.

    Every limit of the form (p + qK)/(r + sK) is a rational Moebius transform of K, so
    all formulas for one target constant are equivalent in that sense; what tells
    them apart is speed. The class lists how many there are and its fastest member
    (most digits per term) and best irrationality-measure estimate delta.
    """
    rows = store.discoveries("constants", limit=10 ** 9, order="old")
    classes: dict[str, list[dict]] = {}
    for d in rows:
        classes.setdefault(d["payload"].get("target", "?"), []).append(d)
    if not classes:
        return []
    out = ["### Equivalence classes (one per constant; the fastest member is the useful one)", "",
           "| constant | formulas | distinct values | fastest: digits/term | best delta | fastest formula |",
           "|---|---:|---:|---:|---:|---|"]
    for t, ds in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        values = {d["payload"].get("closed_form") for d in ds}
        timed = [d for d in ds if d["payload"].get("digits_per_term") is not None]
        fast = max(timed, key=lambda d: d["payload"]["digits_per_term"]) if timed else None
        deltas = [d["payload"]["delta"] for d in ds if d["payload"].get("delta") is not None]
        speed = f"{fast['payload']['digits_per_term']:.3f}" if fast else "-"
        delta = f"{max(deltas):.4f}" if deltas else "-"
        formula = f"`{fast['statement']}`" if fast else "-"
        out.append(f"| {t} | {len(ds)} | {len(values)} | {speed} | {delta} | {formula} |")
    out.append("")
    return out


def _write_metrics(store: Store, run_dir: Path) -> None:
    rows = store.db.execute("SELECT round, engine, name, value FROM metrics ORDER BY round").fetchall()
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "engine", "metric", "value"])
        w.writerows(rows)


def _plot(store: Store, run_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    series = [(t, store.metric_series(e, m), cum) for e, m, t, cum in CURVES]
    series = [(t, s, cum) for t, s, cum in series if len(s) >= 2]
    if not series:
        return
    fig, axes = plt.subplots(len(series), 1, figsize=(8, 2.6 * len(series)), squeeze=False)
    for ax, (title, s, cum) in zip(axes[:, 0], series):
        xs = [r for r, _ in s]
        ys = [v for _, v in s]
        if cum:
            acc, ys2 = 0.0, []
            for v in ys:
                acc += v
                ys2.append(acc)
            ys, title = ys2, title + " (cumulative)"
        ax.plot(xs, ys, color="#3b6ea8", linewidth=1.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=120)
    plt.close(fig)

"""Lean 4 verifier: the gold standard. If this module says a proof is correct,
the Lean kernel has checked it against the exact statement we asked for.

How a batch is checked
----------------------
1. Each candidate becomes a uniquely named top-level theorem with OUR statement
   text (the model never gets to restate the theorem, so it cannot weaken it).
   Optional helper lemmas written by the model ("prelude") are allowed, but
   only ``theorem``/``lemma`` declarations, renamed to unique names, so they
   cannot introduce definitions, notation or instances that change what the
   statement means.
2. Text-level screening rejects escape hatches (``sorry``, ``admit``,
   ``axiom``, ``native_decide``, ``implemented_by``, ``extern``, ``unsafe``,
   ``#exit``, kernel-skipping options, macros, ...).
3. Lean compiles the file. Errors are attributed to items by line ranges.
4. Positive evidence: the file ends with ``#print axioms`` for every item. An
   item is accepted only if Lean actually reported it AND it depends on
   nothing beyond the standard axioms (propext, Classical.choice, Quot.sound).
   This closes tricks such as an unterminated comment that silently swallows
   the rest of the file, or a crash that produces no error lines.
5. Items that are error-free but lack evidence (for example because another
   item broke the file) are re-checked alone.

Set up the workspace once with ``scripts/setup_lean.sh``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid

# autoImplicit off (as in Mathlib): otherwise an unknown name in a statement silently
# becomes a universally quantified variable, so `n ≤ undefined_thing` would "type-check"
# and a model's typo would change what a conjecture means. Found by tests/test_lean_real.py.
HEADER = """import Mathlib
import Aesop
set_option maxHeartbeats 400000
set_option autoImplicit false
open BigOperators Real Nat Topology Rat

"""
STD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
FORBIDDEN = re.compile(
    r"\b(sorry|admit|axiom|native_decide|implemented_by|extern|unsafe|opaque|elab|macro|"
    r"macro_rules|syntax|notation|infix|infixl|infixr|prefix|postfix|run_tac|run_cmd|"
    r"run_elab|skipKernelTC|ofReduceBool|declare_syntax_cat)\b|#exit|#eval")
PRELUDE_BANNED = re.compile(
    r"^(instance|def|abbrev|structure|class|inductive|attribute|local|scoped|variable|namespace|"
    r"section|end|universe|noncomputable|set_option|import|@\[)", re.M)
ERROR_LINE = re.compile(r":(\d+):(\d+): error")
AXIOMS_RE = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]", re.S)
NO_AXIOMS_RE = re.compile(r"'([^']+)' does not depend on any axioms")
CODE_BLOCK = re.compile(r"```[ \t]*(?:lean4?|Lean4?)?[^\n]*\n(.*?)```", re.S)
DECL = re.compile(r"(?m)^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?(theorem|lemma)\s+([^\s(:{\[]+)")

# ----------------------------------------------------------------------------- text utilities

def clean_statement(statement: str) -> str:
    """``theorem name binders : type`` with any proof part removed."""
    s = statement.strip()
    if ":=" in s:
        s = s[: s.index(":=")].rstrip()
    return s


def statement_name(statement: str) -> str | None:
    m = re.match(r"\s*(?:theorem|lemma)\s+([^\s(:{\[]+)", statement)
    return m.group(1) if m else None


def normalize_statement(statement: str) -> str:
    s = re.sub(r"^\s*(theorem|lemma)\s+\S+", "", clean_statement(statement))
    return re.sub(r"\s+", " ", s).strip()


def strip_block_comments(text: str) -> str | None:
    """Remove /- ... -/ comments (incl. doc comments). None if unbalanced."""
    out = re.sub(r"/-.*?-/", "", text, flags=re.S)
    if "/-" in out or "-/" in out:
        return None
    return out


def assemble(statement: str, proof: str) -> str:
    proof = proof.strip()
    if proof.startswith(":="):
        proof = proof[2:].strip()
    if not proof.startswith("by"):
        proof = ("by\n" if "\n" in proof else "by ") + proof
    lines = proof.splitlines()
    body = "\n".join([lines[0]] + ["  " + l for l in lines[1:]])
    return f"{clean_statement(statement)} := {body}"


def _rename(statement: str, new_name: str) -> str:
    return re.sub(r"^\s*(theorem|lemma)\s+\S+", f"theorem {new_name}", clean_statement(statement), count=1)


def parse_completion(completion: str, statement: str) -> tuple[str, str]:
    """Split a model answer into (helper-lemma prelude, proof of our theorem).

    Takes the last Lean code block; finds our theorem by name (falling back to
    the last theorem in the block); everything before it, minus header lines,
    is the prelude. Returns ("", "") if no proof can be found.
    """
    blocks = CODE_BLOCK.findall(completion)
    code = blocks[-1] if blocks else completion
    decls = list(DECL.finditer(code))
    if not decls:
        return "", code.strip() if blocks else ""
    name = statement_name(statement)
    main = next((d for d in reversed(decls) if d.group(2) == name), decls[-1])
    prelude_lines = [l for l in code[: main.start()].splitlines()
                     if not re.match(r"\s*(import|set_option maxHeartbeats|open\s+[^\n]*$)", l)
                     or re.match(r"\s*open\s+.*\bin\s*$", l)]
    rest = code[main.start():]
    if ":=" not in rest:
        return "", ""
    proof = rest[rest.index(":=") + 2:].strip()
    return "\n".join(prelude_lines).strip(), proof


def parse_statement(completion: str) -> str | None:
    """Extract a ``theorem name ... :`` statement proposed by a model."""
    blocks = CODE_BLOCK.findall(completion)
    code = blocks[-1] if blocks else completion
    m = DECL.search(code)
    if not m:
        return None
    rest = code[m.start():]
    if ":=" not in rest:
        return None
    st = rest[: rest.index(":=")].strip()
    st = re.sub(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)?lemma\b", "theorem", st)
    return st if len(st) < 1500 else None


DEFINES = re.compile(r"(?m)^\s*(?:noncomputable\s+)?(?:def|abbrev)\s+([^\s(:{\[]+)")


def load_theorems(path: str, split: str | None = None) -> list[dict]:
    """Read target theorems.

    * ``.jsonl``: one theorem per line, ``{"name", "statement"}`` or the miniF2F /
      DeepSeek-Prover layout ``{"name", "formal_statement", "split"}``; ``split``
      keeps only rows of that split (e.g. "test" or "valid").
    * a ``.lean`` file, or a directory of them (PutnamBench's layout): every theorem.
      Statements that mention a ``def``/``abbrev`` of their own file are skipped,
      because definitions can't be sent to the verifier (they could change what a
      statement means), so those statements would not even parse.
    """
    out = []
    if os.path.isdir(path):
        for root, _, files in sorted(os.walk(path)):
            for name in sorted(files):
                if name.endswith(".lean"):
                    out += load_theorems(os.path.join(root, name))
        return out
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".jsonl"):
        for i, line in enumerate(l for l in text.splitlines() if l.strip()):
            row = json.loads(line)
            if split and row.get("split") not in (None, split):
                continue
            st = clean_statement(row.get("formal_statement") or row["statement"])
            out.append({"name": row.get("name") or statement_name(st) or f"t{i}", "statement": st})
    else:
        text = strip_block_comments(text) or text
        local_defs = DEFINES.findall(text)
        for m in DECL.finditer(text):
            rest = text[m.start():]
            if ":=" in rest:
                st = rest[: rest.index(":=")].strip()
                if any(re.search(rf"(?<![\w.']){re.escape(d)}(?![\w'])", st) for d in local_defs):
                    continue
                out.append({"name": m.group(2), "statement": st})
    return out

# ----------------------------------------------------------------------------- checking

def _prepare(item: dict, name: str, tag: str, allow_sorry: bool) -> tuple[str | None, str]:
    statement = item["statement"]
    proof = item.get("proof", "") or ""
    prelude = item.get("prelude", "") or ""
    if allow_sorry:
        if proof.strip() != "sorry" or FORBIDDEN.search(statement):
            return None, "forbidden token"
        return assemble(_rename(statement, name), "sorry") + "\n\n", ""
    if not proof.strip():
        return None, "no proof found"
    proof_c, prelude_c = strip_block_comments(proof), strip_block_comments(prelude)
    if proof_c is None or prelude_c is None:
        return None, "unbalanced comment"
    if FORBIDDEN.search(statement + "\n" + prelude_c + "\n" + proof_c):
        return None, "forbidden token"
    prelude = prelude_c.strip()
    if prelude:
        for line in prelude.splitlines():
            if not line.strip() or line[:1].isspace() or line.lstrip().startswith("--"):
                continue
            if re.match(r"open\s+.*\bin\s*$", line):
                continue
            if not re.match(r"(@\[[^\]]*\]\s*)?(private\s+|protected\s+)?(theorem|lemma)\b", line):
                return None, "prelude may only contain lemmas"
        if PRELUDE_BANNED.search(prelude):
            return None, "prelude may only contain lemmas"
        for m in DECL.finditer(prelude):
            old = m.group(2)
            new = f"{old}_{tag}"
            pat = re.compile(rf"(?<![\w.']){re.escape(old)}(?![\w'])")
            prelude, proof_c = pat.sub(new, prelude), pat.sub(new, proof_c)
        prelude = re.sub(r"(?m)^lemma\b", "theorem", prelude)
    main = assemble(_rename(statement, name), proof_c)
    return (prelude + "\n\n" if prelude else "") + main + "\n\n", ""


def lake_executable() -> str:
    """``lake`` from PATH, else from elan's own bin directory (setup_lean.ps1 installs
    elan without touching PATH unless asked to)."""
    found = shutil.which("lake")   # honours PATHEXT, so lake.exe / lake.cmd are found on Windows
    if found:
        return found
    home = os.environ.get("ELAN_HOME") or os.path.join(os.path.expanduser("~"), ".elan")
    found = shutil.which("lake", path=os.path.join(home, "bin"))
    return found or "lake"


def _run_lean(text: str, project_dir: str, timeout_s: int) -> tuple[str, bool]:
    tmp = os.path.join(project_dir, ".mf_tmp")
    os.makedirs(tmp, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".lean", dir=tmp)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        # Lean always writes UTF-8 (ℝ, ℕ, ...); don't let the Windows locale codec decode it.
        proc = subprocess.run([lake_executable(), "env", "lean", path], cwd=project_dir,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s)
        return proc.stdout + "\n" + proc.stderr, False
    except subprocess.TimeoutExpired:
        return "", True
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def check_batch(items: list[dict], project_dir: str, timeout_s: int = 300,
                header: str | None = None, allow_sorry: bool = False,
                recheck: bool = True) -> list[dict]:
    """items: [{"statement", "proof", "prelude"?}] -> [{"ok": bool, "error": str}]"""
    header = header or HEADER
    if not header.endswith("\n\n"):
        header = header.rstrip("\n") + "\n\n"
    results = [{"ok": False, "error": ""} for _ in items]
    tag = uuid.uuid4().hex[:6]
    blocks, spans, names = [], [None] * len(items), [None] * len(items)
    line = header.count("\n") + 1
    for i, it in enumerate(items):
        name = f"mf_{i}_{tag}"
        block, err = _prepare(it, name, f"{i}{tag}", allow_sorry)
        if block is None:
            results[i]["error"] = err
            continue
        names[i] = name
        spans[i] = (line, line + block.count("\n") - 1)
        line += block.count("\n")
        blocks.append(block)
    if not blocks:
        return results
    evidence = "".join(f"#print axioms {n}\n" for n in names if n)
    out, timed_out = _run_lean(header + "".join(blocks) + evidence, project_dir, timeout_s)
    if timed_out:
        for i in range(len(items)):
            if names[i]:
                results[i]["error"] = "timeout"
        return results

    msgs = out.splitlines()
    first = min(s[0] for s in spans if s)
    bad: dict[int, str] = {}
    workspace_error = ""
    for j, m in enumerate(msgs):
        e = ERROR_LINE.search(m)
        if not e:
            continue
        ln = int(e.group(1))
        if ln < first:
            workspace_error = workspace_error or "\n".join(msgs[j:j + 3])[:400]
        for i, s in enumerate(spans):
            if s and s[0] <= ln <= s[1]:
                bad.setdefault(i, "\n".join(msgs[j:j + 4])[:600])
    axioms: dict[str, set[str]] = {}
    for m in AXIOMS_RE.finditer(out):
        axioms[m.group(1)] = {a.strip() for a in m.group(2).replace("\n", " ").split(",") if a.strip()}
    for m in NO_AXIOMS_RE.finditer(out):
        axioms[m.group(1)] = set()
    allowed = STD_AXIOMS | ({"sorryAx"} if allow_sorry else set())

    uncertain = []
    for i in range(len(items)):
        if not names[i]:
            continue
        if workspace_error:
            results[i]["error"] = "workspace error (is Mathlib built?): " + workspace_error
        elif i in bad:
            results[i]["error"] = bad[i]
        elif names[i] not in axioms:
            uncertain.append(i)
            results[i]["error"] = "no evidence that the theorem was checked"
        elif not axioms[names[i]] <= allowed:
            results[i]["error"] = "uses non-standard axioms: " + ", ".join(sorted(axioms[names[i]] - allowed))
        else:
            results[i] = {"ok": True, "error": ""}
    if uncertain and recheck and len(items) > 1:
        for i in uncertain:
            results[i] = check_batch([items[i]], project_dir, timeout_s, header, allow_sorry, recheck=False)[0]
    return results


# Each alternative must *close* the goal: `first` stops at the first tactic that merely
# succeeds, and norm_num/simp succeed whenever they make progress (found on real Lean).
TRIVIAL_TACTIC = "first | " + " | ".join(
    f"({t}; done)" for t in ("rfl", "decide", "norm_num", "simp", "omega", "positivity", "linarith", "nlinarith"))


def lean_worker(task: dict) -> dict:
    """Executor entry point.

    task = {"items": [...], "project_dir": str, "timeout_s": int, "header": str|None,
            "mode": "proof" | "wellformed" | "trivial"}
    """
    mode = task.get("mode", "proof")
    pd, to, hd = task["project_dir"], task["timeout_s"], task.get("header")
    if mode == "wellformed":
        items = [{"statement": it["statement"], "proof": "sorry"} for it in task["items"]]
        return {"ok": [r["ok"] for r in check_batch(items, pd, to, hd, allow_sorry=True)]}
    if mode == "trivial":
        items = [{"statement": it["statement"], "proof": TRIVIAL_TACTIC} for it in task["items"]]
        return {"ok": [r["ok"] for r in check_batch(items, pd, to, hd)]}
    res = check_batch(task["items"], pd, to, hd)
    return {"ok": [r["ok"] for r in res], "errors": [r["error"] for r in res]}

"""Engine 3: neural self-play with verified rewards.

A language model plays two roles against itself:

  PROPOSER  invents new problems (integrals, or Lean theorem statements)
  SOLVER    attempts every problem many times

A verifier (SymPy-based exact checks, or the Lean 4 kernel) grades every
attempt. Then:

* Verified solutions become training data for the solver (expert iteration:
  the model is fine-tuned only on its own *correct* work, so it can't learn
  its own mistakes).
* The proposer is rewarded for *learnable* problems: ones the solver solves
  sometimes but not always (0 < pass rate < 1; as a reward, 4 r (1 - r)).
  Trivial and impossible inventions earn nothing, so the proposer learns to
  generate problems right at the frontier of the solver's ability, a
  self-generated curriculum. It invents problems in every family; whatever the
  checker needs (the roots of an invented equation, ...) is computed by SymPy.
* Training is expert iteration (supervised, on verified successes), optionally
  followed by GRPO: reinforcement learning on groups of attempts at the same
  prompt, with the verifier's verdict as the reward, so failures are pushed down
  as well as successes pulled up.
* Generated (non-invented) problems follow an adaptive difficulty curriculum
  per problem family.
* A fixed held-out evaluation set is re-run every few rounds so you can see
  whether the system is genuinely getting better, not just busier.

Lean mode works on theorem statements (a starter set, your own file, e.g.
miniF2F-style, and conjectures the model invents) and produces machine-checked
proofs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from importlib import resources

from ..curriculum import Curriculum
from ..models.policy import PolicyPool
from ..novelty import interestingness, stable_hash
from ..verify import lean as leanv
from ..verify.safe_parse import extract_answer
from ..verify.sympy_tasks import (FAMILIES, check, extract_proposal, generate, problem_spec,
                                  validate_proposed)
from .base import Engine

log = logging.getLogger("mathforge")

SOLVER_SYMPY = (
    "Solve the following problem. Reason step by step, then give the final answer on the last line "
    "in exactly this form:\nANSWER: <answer in SymPy syntax, e.g. x**2*exp(x)/2 or [1, -3/2]>\n\n"
    "Problem: {text}"
)
_INVENT = "Invent a new practice problem for yourself. These are {what} at the edge of what you can currently {verb}:\n{{examples}}\n\n"
PROPOSER_SYMPY = {
    "integrate": _INVENT.format(what="integration problems", verb="solve") +
    "Invent ONE new integrand that is a little harder but still has an elementary antiderivative. "
    "Do not copy the examples. Output exactly one line:\nPROBLEM: <integrand f(x) in SymPy syntax>",
    "solve": _INVENT.format(what="equations", verb="solve") +
    "Invent ONE new polynomial equation in x with integer coefficients that is a little harder but whose "
    "real solutions can still be found exactly. Do not copy the examples. Output exactly one line:\n"
    "PROBLEM: <polynomial in x in SymPy syntax> = 0",
    "sum": _INVENT.format(what="sums", verb="find closed forms for") +
    "Invent ONE new polynomial p(k) with integer coefficients whose sum over k = 1..n is a little harder "
    "to find. Do not copy the examples. Output exactly one line:\nPROBLEM: <p(k) in SymPy syntax>",
}
EXAMPLE_FMT = {"integrate": "- f(x) = {}", "solve": "- {} = 0", "sum": "- sum over k = 1..n of {}"}
SOLVER_LEAN = (
    "Complete the following Lean 4 code:\n\n```lean4\n{header}{statement} := by\n  sorry\n```\n\n{lemmas}"
    "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof "
    "plan outlining the main proof steps and strategies. Then give the complete Lean 4 code, "
    "including the theorem, in a single ```lean4 code block."
)
LEMMAS_LEAN = ("These lemmas are already proved and available; you may use them by name:\n\n"
               "```lean4\n{code}\n```\n\n")
REPAIR_LEAN = (
    "This Lean 4 proof attempt does not compile:\n\n```lean4\n{header}{attempt}\n```\n\n"
    "Lean reported:\n```\n{error}\n```\n\n"
    "Fix the proof. Give the complete corrected Lean 4 code, including the theorem, in a single "
    "```lean4 code block."
)
PROPOSER_LEAN = (
    "Invent a new theorem. Here are theorems you have proved in Lean 4 with Mathlib:\n\n{examples}\n\n"
    "State ONE new theorem that is true, non-trivial, related to these but not already listed. "
    "Use Lean 4 Mathlib syntax and give it with a placeholder proof:\n"
    "```lean4\ntheorem new_conjecture ... := by sorry\n```"
)

# ----------------------------------------------------------------------------- worker functions

def gen_worker(task: dict) -> dict:
    return generate(task["family"], task["level"], task["seed"])


def sympy_verify_worker(task: dict) -> dict:
    return {"ok": [check(p, c) for p, c in task["items"]]}

# ----------------------------------------------------------------------------- engine

class SelfPlayEngine(Engine):
    name = "selfplay"

    def __init__(self, cfg, full_cfg, store, rng):
        super().__init__(cfg, full_cfg, store, rng)
        self.mode = cfg["verifier"]
        self.run_dir = full_cfg["run"]["dir"]
        self.use_ray = full_cfg["compute"]["backend"] == "ray"
        c = full_cfg["curriculum"]
        self.fam_curr = {f: Curriculum(c["target_success"], c["band"], c["step"], c["min_level"],
                                       c["max_level"]) for f in FAMILIES}
        self.pool: PolicyPool | None = None
        self.version = 0
        self.adapter_path: str | None = None
        self.last_train = 0
        self.frontier: list[dict] = []          # {"family", "spec", "rate"}: solved sometimes
        self.best_level = {f: 0 for f in FAMILIES}
        self.eval_set: list[dict] | None = None
        self._prop_stats = (0, 0)
        self._proposals: list[tuple] = []       # this round: (family, prompt, completion, problem|None)
        self._prop_offset = 0
        lc = cfg["lean"]
        self.lean_dir = os.path.abspath(lc["project_dir"])
        self.lean_header = lc.get("header") or leanv.HEADER
        self.theorems: list[dict] = []
        if self.mode == "lean":
            self._init_theorems()

    # --------------------------------------------------------------- policy
    def _ensure_policy(self):
        if self.pool is None:
            pc = self.cfg["policy"]
            self.pool = PolicyPool(pc, self.cfg["trainer"]["lora_r"], self.use_ray,
                                   self.full_cfg["run"]["seed"])
            if self.adapter_path or pc["backend"] == "mock":
                self.pool.set_adapter(self.adapter_path, self.version)

    def run_round(self, executor, round_: int) -> dict:
        t0 = time.time()
        self._ensure_policy()
        if self.mode == "sympy":
            stats = self._round_sympy(executor, round_)
        elif self.mode == "lean":
            stats = self._round_lean(executor, round_)
        else:
            raise ValueError(f"unknown verifier {self.mode!r}")
        stats.update(self._maybe_train(round_))
        if self.mode == "sympy" and self.cfg["eval_every"] and round_ % self.cfg["eval_every"] == 0:
            stats.update(self._evaluate(executor))
        stats["adapter_version"] = self.version
        stats["seconds"] = round(time.time() - t0, 2)
        stats["level"] = max(cur.level for cur in self.fam_curr.values())
        return stats

    # =============================================================== SymPy mode
    def _round_sympy(self, executor, round_):
        c, pc = self.cfg, self.cfg["policy"]
        n_total = c["problems_per_round"]
        n_prop = int(n_total * c["proposer_fraction"])
        specs = []
        for _ in range(n_total - n_prop):
            fam = self.rng.choice(FAMILIES)
            specs.append({"family": fam, "level": self.fam_curr[fam].sample_level(self.rng),
                          "seed": self.rng.randrange(1 << 30)})
        problems = [p for p in executor.map(gen_worker, specs) if "error" not in p]
        problems += self._propose_sympy(n_prop)
        prompts = [SOLVER_SYMPY.format(text=p["text"]) for p in problems]
        comps = self.pool.generate(prompts, pc["samples_per_problem"], pc["temperature"],
                                   pc["max_new_tokens"])
        oks = self._verify_sympy(executor, problems, comps)
        return self._ingest_sympy(problems, prompts, comps, oks, round_)

    def _verify_sympy(self, executor, problems, comps):
        flat = [(i, j) for i in range(len(problems)) for j in range(len(comps[i]))]
        items = [(problems[i], comps[i][j]) for i, j in flat]
        tasks = [{"items": items[s:s + 32]} for s in range(0, len(items), 32)]
        flat_ok: list[bool] = []
        for t, r in zip(tasks, executor.map(sympy_verify_worker, tasks)):
            flat_ok += r.get("ok", [False] * len(t["items"]))
        oks = [[False] * len(cs) for cs in comps]
        for (i, j), ok in zip(flat, flat_ok):
            oks[i][j] = bool(ok)
        return oks

    def _propose_sympy(self, n: int) -> list[dict]:
        """The model invents n problems, spread over the families in turn."""
        self._proposals = []
        if n <= 0:
            self._prop_stats = (0, 0)
            return []
        examples: dict[str, list[str]] = {}
        for f in FAMILIES:
            ex = [e["spec"] for e in self.frontier if e["family"] == f][-100:]
            while len(ex) < 4:
                ex.append(problem_spec(generate(f, self.fam_curr[f].level, self.rng.randrange(1 << 30))))
            examples[f] = ex
        fams, prompts = [], []
        for i in range(n):
            f = FAMILIES[(self._prop_offset + i) % len(FAMILIES)]
            shown = self.rng.sample(examples[f], 4)
            fams.append(f)
            prompts.append(PROPOSER_SYMPY[f].format(examples="\n".join(EXAMPLE_FMT[f].format(e) for e in shown)))
        self._prop_offset += n
        outs = self.pool.generate(prompts, 1, 1.0, 160)
        problems = []
        seen = {(f, e) for f, ex in examples.items() for e in ex}      # copying an example earns nothing
        for f, p, o in zip(fams, prompts, outs):
            text = extract_proposal(f, o[0])
            prob = validate_proposed(f, text) if text else None
            if prob and (f, problem_spec(prob)) in seen:
                prob = None
            if prob:
                seen.add((f, problem_spec(prob)))
                prob.update({"level": self.fam_curr[f].level, "proposed": True, "proposer_prompt": p,
                             "proposer_completion": o[0].strip()})
                problems.append(prob)
            self._proposals.append((f, p, o[0], prob))
        self._prop_stats = (n, len(problems))
        return problems

    def _ingest_sympy(self, problems, prompts, comps, oks, round_):
        fam_s, fam_a = Counter(), Counter()
        traces, new_disc, n_correct, n_samples, any_solved, learnable = [], 0, 0, 0, 0, 0
        rollouts: list[tuple] = []
        prop_rate: dict[int, float] = {}
        for i, (p, prompt, cs, ok) in enumerate(zip(problems, prompts, comps, oks)):
            k, s = len(cs), sum(ok)
            rate = s / k if k else 0.0
            n_correct += s
            n_samples += k
            any_solved += 1 if s else 0
            if not p.get("proposed"):
                fam_s[p["family"]] += s
                fam_a[p["family"]] += k
            correct = sorted({c.strip() for c, o in zip(cs, ok) if o}, key=len)[:2]
            for comp in correct:
                traces.append((prompt, comp, {"role": "solver", "family": p["family"],
                                              "level": p["level"], "pass_rate": rate}))
            if 0 < s < k:      # a group with a signal: some attempts right, some wrong
                rollouts += [(f"r{round_}:p{i}", prompt, c, float(o)) for c, o in zip(cs, ok)]
            if 0 < rate < 1:
                self.frontier.append({"family": p["family"], "spec": problem_spec(p), "rate": rate})
            if p.get("proposed"):
                prop_rate[id(p)] = rate
                if 0 < rate < 1:
                    learnable += 1
                    traces.append((p["proposer_prompt"], p["proposer_completion"],
                                   {"role": "proposer", "pass_rate": rate, "reward": 4 * rate * (1 - rate)}))
                if s:
                    ans = extract_answer(correct[0]) or ""
                    if self._store_invented(p, ans, s, k, round_):
                        new_disc += 1
            elif s and p["level"] > self.best_level[p["family"]]:
                self.best_level[p["family"]] = p["level"]
                ans = extract_answer(correct[0]) or ""
                if self.store.add_discovery(
                        self.name, stable_hash("milestone", p["family"], p["level"]), "milestone",
                        f"First {p['family']} problem solved at level {p['level']}: {p['text']}  ->  {ans}",
                        {"family": p["family"], "level": p["level"], "problem": p, "answer": ans,
                         "adapter_version": self.version},
                        interestingness(len(ans) / 8, 1 - rate, 0.5), round_):
                    new_disc += 1
        self.frontier = self.frontier[-300:]
        for f in FAMILIES:
            if fam_a[f]:
                self.fam_curr[f].update(fam_s[f], fam_a[f])
        new_traces = self.store.add_traces(self.name, traces, round_)
        # The proposer's group: every proposal of the round, rewarded 4 r (1 - r) by the
        # solver's pass rate r on it (0 for an invalid or copied proposal).
        props = [(p, c, 4 * prop_rate[id(prob)] * (1 - prop_rate[id(prob)]) if prob is not None
                  and id(prob) in prop_rate else 0.0) for _, p, c, prob in self._proposals]
        if len({r for *_, r in props}) > 1:
            rollouts += [(f"r{round_}:proposer", p, c, r) for p, c, r in props]
        new_rollouts = self.store.add_rollouts(self.name, rollouts, round_) if self._algorithm() == "grpo" else 0
        out = {"problems": len(problems), "solve_rate": n_correct / max(1, n_samples),
               "pass_at_k": any_solved / max(1, len(problems)), "proposals": self._prop_stats[0],
               "proposals_valid": self._prop_stats[1], "proposals_learnable": learnable,
               "new_traces": new_traces, "new_rollouts": new_rollouts, "new_discoveries": new_disc}
        out.update({f"level_{f}": self.fam_curr[f].level for f in FAMILIES})
        return out

    def _store_invented(self, p: dict, ans: str, s: int, k: int, round_: int) -> bool:
        """A problem the model invented and then solved (checked by our verifier)."""
        f, spec = p["family"], problem_spec(p)
        tail = f"   (problem invented by the model; solved in {s}/{k} attempts)"
        if f == "integrate":
            key, statement = stable_hash("selfplay-int", spec), f"∫ {spec} dx = {ans} + C" + tail
        elif f == "solve":
            key, statement = stable_hash("selfplay-solve", spec), f"{spec} = 0  has real solutions x in {ans}" + tail
        else:
            key, statement = stable_hash("selfplay-sum", spec), f"sum_(k=1..n) {spec} = {ans}" + tail
        problem = {key_: p[key_] for key_ in ("family", "integrand", "lhs", "roots", "summand", "degree") if key_ in p}
        payload = {"problem": problem, "answer": ans, "pass_rate": s / k, "adapter_version": self.version}
        if f == "integrate":
            payload.update({"integrand": spec, "antiderivative": ans})
        return self.store.add_discovery(self.name, key, "solution", statement, payload,
                                        interestingness(len(ans) / 8, 1 - s / k, 1.0), round_)

    def _algorithm(self) -> str:
        return self.cfg["trainer"].get("algorithm", "expert_iteration")

    def _build_eval_set(self, executor):
        specs = [{"family": f, "level": lvl, "seed": 1_000_003 * lvl + 7_919 * i + 97 * fi}
                 for fi, f in enumerate(FAMILIES) for lvl in self.cfg["eval_levels"]
                 for i in range(self.cfg["eval_per_level"])]
        self.eval_set = [p for p in executor.map(gen_worker, specs) if "error" not in p]

    def _evaluate(self, executor) -> dict:
        if self.eval_set is None:
            self._build_eval_set(executor)
        prompts = [SOLVER_SYMPY.format(text=p["text"]) for p in self.eval_set]
        comps = self.pool.generate(prompts, 1, 0.0, self.cfg["policy"]["max_new_tokens"])
        oks = self._verify_sympy(executor, self.eval_set, comps)
        out = {"eval_pass1": sum(o[0] for o in oks) / max(1, len(oks))}
        for f in FAMILIES:
            sel = [o[0] for p, o in zip(self.eval_set, oks) if p["family"] == f]
            if sel:
                out[f"eval_pass1_{f}"] = sum(sel) / len(sel)
        return out

    # =============================================================== Lean mode
    def _init_theorems(self):
        rows = []
        with resources.files("mathforge.data").joinpath("lean_starter.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({"name": r["name"], "statement": leanv.clean_statement(r["statement"])})
        path = self.cfg["lean"].get("theorem_file")
        if path:
            rows += leanv.load_theorems(path, self.cfg["lean"].get("theorem_split") or None)
        seen = set()
        for r in rows:
            key = leanv.normalize_statement(r["statement"])
            if key in seen:
                continue
            seen.add(key)
            self.theorems.append({"name": r["name"], "statement": r["statement"], "source": "given",
                                  "solved": False, "attempts": 0, "proof": None})

    def _lean_task(self, items, mode="proof"):
        lc = self.cfg["lean"]
        return {"items": items, "mode": mode, "project_dir": self.lean_dir,
                "timeout_s": lc["timeout_s"], "header": self.lean_header}

    def _lean_map(self, executor, items, mode="proof", chunk=16) -> tuple[list[bool], list[str]]:
        """Check items in batches of ``chunk`` per Lean process: (ok flags, error messages)."""
        tasks = [self._lean_task(items[s:s + chunk], mode) for s in range(0, len(items), chunk)]
        oks: list[bool] = []
        errors: list[str] = []
        for t, r in zip(tasks, executor.map(leanv.lean_worker, tasks)):
            n = len(t["items"])
            oks += r.get("ok", [False] * n)
            errors += r.get("errors") or [r.get("error", "")] * n
        return oks, errors

    # --------------------------------------------------------------- lemma library
    def _library_for(self, statement: str) -> list[dict]:
        """Proved theorems that look relevant to ``statement`` (most shared identifiers),
        offered to the solver as lemmas it may cite: the Lean analogue of the laws
        engine's learned primitives."""
        lc = self.cfg["lean"]
        if not lc.get("use_library", True):
            return []
        ident = re.compile(r"[A-Za-z_][\w.']*")
        words = set(ident.findall(statement))
        cands = [t for t in self.theorems if t["solved"] and t.get("proof") and t["statement"] != statement
                 and len(t["proof"]) + len(t.get("prelude") or "") < 2000]
        cands.sort(key=lambda t: -len(words & set(ident.findall(t["statement"]))))
        return cands[: lc.get("library_size", 3)]

    def _library_code(self, lemmas: list[dict]) -> tuple[str, list[str]]:
        """Lean code of the lemmas and everything they were proved with, each once, in
        dependency order (a lemma may itself rest on earlier library lemmas)."""
        by_name = {t["name"]: t for t in self.theorems if t["solved"]}
        order: list[str] = []

        def visit(name, depth=0):
            t = by_name.get(name)
            if t is None or name in order or depth > 20:
                return
            for dep in t.get("lib_used") or []:
                visit(dep, depth + 1)
            order.append(name)

        for t in lemmas:
            visit(t["name"])
        parts = []
        for name in order:
            t = by_name[name]
            parts += ([t["prelude"]] if t.get("prelude") else []) + [leanv.assemble(t["statement"], t["proof"])]
        return "\n\n".join(parts), order

    def _propose_lean(self, executor, n, round_) -> tuple[int, int]:
        if n <= 0:
            return 0, 0
        proved = [t["statement"] for t in self.theorems if t["solved"]] or \
            [t["statement"] for t in self.theorems]
        prompts = [PROPOSER_LEAN.format(examples="\n\n".join(
            self.rng.sample(proved, min(4, len(proved))))) for _ in range(n)]
        outs = self.pool.generate(prompts, 1, 1.0, 400)
        known = {leanv.normalize_statement(t["statement"]) for t in self.theorems}
        cands = []
        for p, o in zip(prompts, outs):
            st = leanv.parse_statement(o[0])
            if not st:
                continue
            st = re.sub(r"^\s*theorem\s+\S+", f"theorem conj_r{round_}_{len(cands)}", st)
            norm = leanv.normalize_statement(st)
            if norm in known:
                continue
            known.add(norm)
            cands.append((st, p, o[0]))
        if not cands:
            return n, 0
        wf, _ = self._lean_map(executor, [{"statement": s} for s, _, _ in cands], "wellformed")
        cands = [c for c, ok in zip(cands, wf) if ok]
        if not cands:
            return n, 0
        trivial, _ = self._lean_map(executor, [{"statement": s} for s, _, _ in cands], "trivial")
        accepted = 0
        for (st, p, o), triv in zip(cands, trivial):
            if triv:
                continue
            accepted += 1
            self.theorems.append({"name": leanv.statement_name(st), "statement": st,
                                  "source": "proposed", "solved": False, "attempts": 0,
                                  "proof": None, "proposer_prompt": p, "proposer_completion": o,
                                  "round": round_})
        return n, accepted

    def _round_lean(self, executor, round_):
        c, pc = self.cfg, self.cfg["policy"]
        n_total = c["problems_per_round"]
        n_prop, n_acc = self._propose_lean(executor, int(n_total * c["proposer_fraction"]), round_)
        open_ = [t for t in self.theorems if not t["solved"] and not t.get("abandoned")]
        self.rng.shuffle(open_)
        open_.sort(key=lambda t: t["attempts"])
        done = [t for t in self.theorems if t["solved"]]
        n_rehearse = min(len(done), max(1, n_total // 10)) if done else 0
        chosen = open_[: n_total - n_rehearse] + self.rng.sample(done, n_rehearse)
        if not chosen:
            return {"lean_open": 0, "lean_proved_total": len(done)}
        libs = [self._library_code(self._library_for(t["statement"])) for t in chosen]
        prompts = [SOLVER_LEAN.format(header=self.lean_header, statement=t["statement"],
                                      lemmas=LEMMAS_LEAN.format(code=code) if code else "")
                   for t, (code, _) in zip(chosen, libs)]
        comps = self.pool.generate(prompts, pc["samples_per_problem"], pc["temperature"],
                                   pc["max_new_tokens"])
        items, index = [], []
        for ti, (t, cs) in enumerate(zip(chosen, comps)):
            uniq: dict[tuple[str, str], int] = {}
            for j, comp in enumerate(cs):
                prelude, proof = leanv.parse_completion(comp, t["statement"])
                key = (prelude, proof)
                if key not in uniq:
                    uniq[key] = len(items)
                    items.append({"statement": t["statement"], "proof": proof, "own_prelude": prelude,
                                  "prelude": "\n\n".join(x for x in (libs[ti][0], prelude) if x)})
                index.append((ti, j, uniq[key]))
        verdict, errors = self._lean_map(executor, items)
        oks = [[False] * len(cs) for cs in comps]
        for ti, j, it in index:
            oks[ti][j] = verdict[it]

        traces, new_disc, solved_now, n_correct, n_samples = [], 0, 0, 0, 0
        for ti, (t, prompt, cs, ok) in enumerate(zip(chosen, prompts, comps, oks)):
            t["attempts"] += 1
            k, s = len(cs), sum(ok)
            rate = s / k if k else 0.0
            n_correct += s
            n_samples += k
            good = sorted({cmp.strip() for cmp, o in zip(cs, ok) if o}, key=len)[:2]
            for comp in good:
                traces.append((prompt, comp, {"role": "solver", "theorem": t["name"], "pass_rate": rate}))
            if t["source"] == "proposed" and 0 < rate < 1 and not t.get("proposer_rewarded"):
                t["proposer_rewarded"] = True
                traces.append((t["proposer_prompt"], t["proposer_completion"],
                               {"role": "proposer", "pass_rate": rate}))
            if s and not t["solved"]:
                solved_now += 1
                new_disc += self._record_proof(t, good[0], libs[ti], rate, round_)
        repaired, repair_new, repair_tried, repair_traces = self._repair_lean(
            executor, chosen, libs, items, index, verdict, errors, round_)
        solved_now += repaired
        new_disc += repair_new
        traces += repair_traces
        for t in chosen:
            if t["source"] == "proposed" and not t["solved"] and t["attempts"] >= 5:
                t["abandoned"] = True
        new_traces = self.store.add_traces(self.name, traces, round_)
        return {"lean_attempted": len(chosen), "lean_solved_now": solved_now,
                "lean_proved_total": sum(1 for t in self.theorems if t["solved"]),
                "lean_pool": len(self.theorems), "solve_rate": n_correct / max(1, n_samples),
                "lean_repair_attempted": repair_tried, "lean_repaired": repaired,
                "lean_library_used": sum(1 for code, _ in libs if code),
                "conjectures_proposed": n_prop, "conjectures_accepted": n_acc,
                "new_traces": new_traces, "new_discoveries": new_disc}

    def _record_proof(self, t: dict, completion: str, lib: tuple[str, list[str]], rate: float, round_: int) -> int:
        """Mark a theorem proved and store it with everything needed to re-check it."""
        t["solved"] = True
        own_prelude, proof = leanv.parse_completion(completion, t["statement"])
        t.update({"proof": proof, "prelude": own_prelude, "lib_used": lib[1]})
        prelude = "\n\n".join(x for x in (lib[0], own_prelude) if x)
        full = (prelude + "\n\n" if prelude else "") + leanv.assemble(t["statement"], proof)
        novelty = 1.0 if t["source"] == "proposed" else 0.6
        return int(self.store.add_discovery(
            self.name, stable_hash("lean", leanv.normalize_statement(t["statement"])), "theorem",
            full, {"statement": t["statement"], "proof": proof, "prelude": prelude, "lemmas_used": lib[1],
                   "source": t["source"], "pass_rate": rate, "attempt_rounds": t["attempts"],
                   "adapter_version": self.version, "verified": "lean-kernel"},
            interestingness(len(proof) / 40, 1 - rate, novelty), round_))

    def _repair_lean(self, executor, chosen, libs, items, index, verdict, errors, round_):
        """Second chance for every theorem whose attempts all failed: show the model one
        failed attempt with Lean's error message and let it fix the proof.

        Returns (theorems repaired, new discoveries, repair prompts sent, training traces).
        A successful repair is trained on as (repair prompt -> fixed proof), so the model
        also learns to read Lean's errors.
        """
        n = self.cfg["lean"].get("repair_samples", 0)
        if not n:
            return 0, 0, 0, []
        targets = []
        for ti, t in enumerate(chosen):
            if t["solved"]:
                continue
            tried = [it for t2, _, it in index if t2 == ti]
            # a real Lean error (not a screening rejection such as "forbidden token")
            failed = next((it for it in tried if not verdict[it] and ": error" in errors[it]), None)
            if failed is not None:
                targets.append((ti, failed))
        if not targets:
            return 0, 0, 0, []
        pc = self.cfg["policy"]
        prompts = []
        for ti, it in targets:
            attempt = ((items[it]["own_prelude"] + "\n\n") if items[it]["own_prelude"] else "") + \
                leanv.assemble(items[it]["statement"], items[it]["proof"])
            prompts.append(REPAIR_LEAN.format(header=self.lean_header, attempt=attempt,
                                              error=errors[it][:1500]))
        comps = self.pool.generate(prompts, n, pc["temperature"], pc["max_new_tokens"])
        ritems, rindex = [], []
        for (ti, _), cs in zip(targets, comps):
            for comp in cs:
                prelude, proof = leanv.parse_completion(comp, chosen[ti]["statement"])
                ritems.append({"statement": chosen[ti]["statement"], "proof": proof,
                               "prelude": "\n\n".join(x for x in (libs[ti][0], prelude) if x)})
                rindex.append(ti)
        rver, _ = self._lean_map(executor, ritems)
        repaired, new, traces = 0, 0, []
        k = 0
        for (ti, _), prompt, cs in zip(targets, prompts, comps):
            for comp in cs:
                if rver[k]:
                    traces.append((prompt, comp.strip(), {"role": "repair", "theorem": chosen[ti]["name"]}))
                    if not chosen[ti]["solved"]:
                        repaired += 1
                        new += self._record_proof(chosen[ti], comp, libs[ti], 1 / len(cs), round_)
                k += 1
        return repaired, new, len(targets), traces

    # =============================================================== expert iteration
    def _maybe_train(self, round_: int) -> dict:
        """Every ``train_every`` rounds: an expert-iteration step (supervised, from the base
        model, on all verified traces), or, with ``trainer.algorithm: grpo`` once a first
        adapter exists, a GRPO step that continues from the current adapter."""
        if round_ - self.last_train < self.cfg["train_every"]:
            return {}
        if self._algorithm() == "grpo" and self.version > 0:
            return self._grpo_step(round_)
        return self._expert_iteration_step(round_)

    def _expert_iteration_step(self, round_: int) -> dict:
        c = self.cfg
        if self.store.unused_trace_count(self.name) < c["min_new_traces"]:
            return {}
        unused_ids = [r["id"] for r in self.store.traces(self.name, limit=10 ** 7, only_unused=True)]
        rows = self.store.traces(self.name, limit=c["max_traces"])
        data = "\n".join(json.dumps({"prompt": r["problem"], "completion": r["solution"]}) for r in rows)
        out = self._train(round_, {"algorithm": "expert_iteration"}, data,
                          f"expert-iteration step on {len(rows)} verified traces")
        if out.get("trained"):
            self.store.mark_used(unused_ids)
            out["train_examples"] = len(rows)
        return out

    def _grpo_step(self, round_: int) -> dict:
        """Group-relative policy optimisation on the attempts since the last step.

        Each group holds attempts at one prompt; an attempt's advantage is its reward
        (the verifier's verdict, or 4 r (1 - r) for a proposal) minus the group mean,
        over the group's spread. Groups where every attempt scored the same carry no
        signal and are never stored.
        """
        g = self.cfg["trainer"].get("grpo", {})
        rows = self.store.rollouts(self.name, only_unused=True, limit=g.get("max_rollouts", 8192))
        if len(rows) < g.get("min_rollouts", 256):
            return {}
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["group"], []).append(r)
        data_rows = []
        for rs in groups.values():
            rewards = [r["reward"] for r in rs]
            sd = statistics.pstdev(rewards) if len(rs) > 1 else 0.0
            if sd < 1e-9:
                continue
            mu = statistics.fmean(rewards)
            data_rows += [{"prompt": r["prompt"], "completion": r["completion"],
                           "advantage": (r["reward"] - mu) / (sd + 1e-4)} for r in rs]
        if not data_rows:
            return {}
        out = self._train(round_, {"algorithm": "grpo", "init_adapter": self.adapter_path},
                          "\n".join(json.dumps(d) for d in data_rows),
                          f"GRPO step on {len(data_rows)} attempts in {len(groups)} groups")
        if out.get("trained"):
            self.store.mark_rollouts_used([r["id"] for r in rows])
            out.update({"train_examples": len(data_rows), "grpo_groups": len(groups)})
        return out

    def _train(self, round_: int, extra: dict, data: str, what: str) -> dict:
        """Train adapter v(N+1) in its own process (or as a Ray GPU task) and load it.

        The policy is unloaded first so the trainer gets the GPU. On any failure the
        previous adapter stays in use and the loop goes on.
        """
        c, pc = self.cfg, self.cfg["policy"]
        version = self.version + 1
        out_dir = os.path.abspath(os.path.join(self.run_dir, "adapters", f"v{version}"))
        os.makedirs(out_dir, exist_ok=True)
        job = {"model": pc["model"], "trainer": c["trainer"], "seed": self.full_cfg["run"]["seed"] + version,
               "out_dir": out_dir, **extra}
        t0 = time.time()
        try:
            if pc["backend"] == "mock":
                with open(os.path.join(out_dir, "MOCK_ADAPTER.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Simulated {job['algorithm']} step (mock policy, tests only).\n")
                with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
                    f.write(data)
            else:
                if not c["trainer"].get("keep_policy_loaded", False):
                    self.pool.unload()
                    self.pool = None
                if self.use_ray and self._ray_has_gpu():
                    import ray

                    from ..models.policy import _read_dir
                    from ..models.trainer import train_remote

                    rjob = {**job, "data_text": data}
                    if rjob.get("init_adapter"):      # the GPU may be on another machine
                        rjob["init_adapter_files"] = _read_dir(rjob.pop("init_adapter"))
                    files = ray.get(ray.remote(num_gpus=1)(train_remote).remote(rjob))
                    for name, blob in files.items():
                        with open(os.path.join(out_dir, name), "wb") as f:
                            f.write(blob)
                else:
                    data_path = os.path.join(out_dir, "train.jsonl")
                    with open(data_path, "w", encoding="utf-8") as f:
                        f.write(data)
                    job["data"] = data_path
                    job_path = os.path.join(out_dir, "job.json")
                    with open(job_path, "w", encoding="utf-8") as f:
                        json.dump(job, f, indent=2)
                    subprocess.run([sys.executable, "-m", "mathforge.models.trainer", job_path], check=True)
        except Exception as e:  # noqa: BLE001 - keep running on the previous adapter
            log.warning(f"[selfplay] training step v{version} failed, keeping v{self.version}: {e}")
            self._ensure_policy()
            return {"train_failed": 1}
        self.version, self.adapter_path = version, out_dir
        self.last_train = round_
        if self.pool is None:
            self._ensure_policy()
        else:
            self.pool.set_adapter(out_dir, version)
        log.info(f"[selfplay] {what}: adapter v{version} in {time.time() - t0:.0f}s")
        return {"trained": 1, "train_seconds": round(time.time() - t0, 1)}

    @staticmethod
    def _ray_has_gpu() -> bool:
        try:
            import ray

            return ray.is_initialized() and ray.cluster_resources().get("GPU", 0) > 0
        except ImportError:
            return False

    # =============================================================== checkpoint
    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update({"version": self.version, "adapter_path": self.adapter_path,
                  "last_train": self.last_train, "frontier": self.frontier[-300:],
                  "best_level": self.best_level, "eval_set": self.eval_set,
                  "fam_curr": {f: cur.state_dict() for f, cur in self.fam_curr.items()},
                  "theorems": self.theorems, "prop_offset": self._prop_offset})
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        if not d:
            return
        self.version = d.get("version", 0)
        self.adapter_path = d.get("adapter_path")
        self.last_train = d.get("last_train", 0)
        # checkpoints from before every family had a proposer stored integrands only
        self.frontier = [e if "family" in e else {"family": "integrate", "spec": e["integrand"], "rate": e["rate"]}
                         for e in d.get("frontier", [])]
        self._prop_offset = d.get("prop_offset", 0)
        self.best_level.update(d.get("best_level", {}))
        self.eval_set = d.get("eval_set")
        for f, s in d.get("fam_curr", {}).items():
            if f in self.fam_curr:
                self.fam_curr[f].load_state_dict(s)
        if d.get("theorems"):
            known = {leanv.normalize_statement(t["statement"]) for t in d["theorems"]}
            extra = [t for t in self.theorems if leanv.normalize_statement(t["statement"]) not in known]
            self.theorems = d["theorems"] + extra

    def close(self):
        if self.pool is not None:
            try:
                self.pool.unload()
            except Exception:  # noqa: BLE001
                pass

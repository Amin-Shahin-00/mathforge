# MathForge: handoff

*Last updated 2026-10-23 (second session: the roadmap R1 to R9). Written so Claude Code
(and you) can pick the project up cold.*

---

## 0. Start here

**What this is.** MathForge is a self-improving mathematical discovery system. It proposes
formulas, problems and theorems, checks every one with an exact verifier, keeps only what
passes, learns from what it kept, and repeats for as long as the machine stays on. The CPU
engines run on any laptop; the neural engine scales to cloud GPUs and multi-machine clusters.

**Where the code is.** This file sits at the project root as `HANDOFF.md`, next to
`CLAUDE.md` (which Claude Code loads automatically) and `README.md` (the user guide).

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]" matplotlib
pytest -q                                               # expect: 91 passed, 8 skipped (~25 s)
mathforge run -c configs/laptop.yaml --rounds 2 --set run.dir=runs/smoke
```

The 8 skipped tests are the slow ones (`MF_SLOW=1`, see section 7). Then open the folder in
Claude Code and start with, for example:

> Read HANDOFF.md, run the tests, then do roadmap item R10 (vLLM on a Linux GPU box).

If you have this file but not the code, sections 3 to 7 are a complete specification for
rebuilding the system, and section 8 lists the traps that were already hit.

### What the second session did (summary)

The first session built the scaffold; the second worked through the whole roadmap on a
Windows laptop (6 cores, 16 GB, RTX 2060 6 GB):

- **Windows support** (it was Linux-only in practice): hard verifier time limits without
  `SIGALRM`, UTF-8 Lean output, `lake` discovery, a Windows fake `lake` for tests,
  PowerShell twins of the shell scripts.
- **R4 fitted constants and R5 dimensional analysis** in the laws engine, measured with a
  new cold-start benchmark: 16/20 → 18/20 laws and 113 → 141 of 200 runs, beating an
  equal-compute control (130/200); Bernoulli solved in 9 of 10 runs; every accepted law
  proved symbolically. Building R4 exposed a soundness gap (junk terms riding on fitted
  constants passed the numeric test), closed by `check_claim` (section 5.2).
- **R6** convergence speed and irrationality-measure estimates for constants, equivalence
  classes in the report, and a structured Apéry-like search family (finds a known Catalan
  fraction in round 1; reproduces Apéry's ζ(2) and ζ(3) fractions).
- **R7** proposers for all three self-play families, the 4r(1−r) learnability reward, GRPO.
- **R8** Lean repair from error messages, a lemma library, miniF2F/PutnamBench loaders.
- **R9** `mathforge bench`, `mathforge ablate`, `mathforge dashboard`, W&B/MLflow tracking.
- **R1, R2, R3 run for real** (with this hardware's limits): a real 1.5B model trained and
  reloaded on the GPU; the Lean verifier against real Lean 4 + Mathlib (which found two
  bugs the fake compiler could not); Ray on a simulated multi-node cluster with a node
  killed mid-round.

---

## 1. The goal

What the user asked for:

- A system that learns mathematics by itself: it rediscovers the formulas we already know,
  goes on to find new ones, and eventually takes on problems that are hard for experts.
- It runs continuously and keeps improving from its own experience.
- Research-level quality ("make it the best").
- Compute is flexible: cloud GPUs, or several computers working together.

How that became a design: mathematics is one of the few fields where results can be
**checked mechanically**. That makes an AlphaZero-style loop possible: propose, attempt,
verify, learn from verified successes only, repeat. Without a verifier, a system that
trains on its own output drifts into confident nonsense; with one, it can only add true
things. Every engine is built around a verifier.

Expectations agreed with the user: on modest hardware the system really does rediscover
classical results and finds formulas in narrow areas; genuinely new results need more
compute and good targets; beating experts on hard open problems needs frontier-scale models.
The project is the complete scaffold, sized to start small and scale up.

In the second session the user handed over this file and said "complete"; that was taken
to mean: make it run here and work through the roadmap as far as the hardware allows.

---

## 2. Status

### What exists

- Three engines: `constants` (formulas for constants such as π, e, ζ(3)), `laws` (laws
  rediscovered from raw data, with fitted constants, units and library learning) and
  `selfplay` (a language model that invents and solves problems and trains on its verified
  successes by expert iteration and optionally GRPO, with SymPy or Lean 4 as the judge).
- Verifiers: exact integer arithmetic plus 50-digit numerics, symbolic proofs of
  equivalence, SymPy-based answer checks, and the Lean 4 kernel with anti-cheating checks.
- A forever loop with per-round checkpoints, exact resume and graceful Ctrl+C; a SQLite
  knowledge base; execution on one core, all cores, or a Ray cluster; optional W&B/MLflow.
- CLI (`run`, `status`, `discoveries`, `verify`, `fit`, `bench`, `ablate`, `dashboard`,
  `report`, `export`), configs for laptop / small GPU / GPU / Lean / cluster, Docker, systemd
  and AWS Ray deployment files, and reports (`discoveries.md`, `metrics.csv`,
  `learning_curves.png`).
- About 7,000 lines including tests: 91 fast tests plus 8 slow ones in 3 files.

### Laws benchmark (cold start, 10 seeds per law, idle 6-core laptop)

`mathforge bench laws --seeds 10`: every law searched from scratch (no library, no learned
grammar), judged by the engine's own head-side check; population 500, 60 generations.

| search | laws solved | runs solved | CPU time |
|---|---:|---:|---:|
| fixed constants only (the first session's search) | 16/20 | 113/200 | 800 s |
| same, 6× the generations (equal-compute control) | 16/20 | 130/200 | 4488 s |
| **fitted constants + units (default)** | **18/20** | **141/200** | **3488 s** |

Per law (of 10) for original / control / default: Bernoulli 0/0/9, Lorentz 1/2/5, Doppler
2/5/8, 3-D distance 0/0/1, damped oscillation 6/9/9, Pythagoras 6/9/8, thin lens 7/10/9,
Coulomb 2/3/1; the six tier-1/2 laws, escape velocity and Kepler are 10/10 everywhere;
relativistic energy and the Planck-like spectrum are 0 everywhere from a cold start (the
forever loop gets them through the learned library). Every accepted result of the default
configuration is proved symbolically.

How the default was chosen (5 seeds each, seeds 0–4, runs solved / CPU time): fitted
constants at 15% of new leaves and units penalty 2: 62/100, 2194 s; constants only
(no units): 70/100; units only (no constants): 59/100, 351 s; 15% and penalty 1: 64/100;
**8% and penalty 2: 72/100, 1618 s**; 8% and penalty 1: 69/100. The winner was then
confirmed on held-out seeds 5–9: 69/100 (1870 s) against 65/100 for the equal-compute
control (2182 s) and 55/100 for the original (393 s). Variance between seeds is large;
treat differences of a few runs as noise.

### Results of the 40-round runs

- First session (old code, Linux, 2 cores, `examples/demo_run/`): 19 of 20 laws (all but
  Bernoulli), 21 library primitives, 34 continued-fraction formulas (14 e, 9 π, 6 φ, 3 ln 2,
  2 √2), all known families or close relatives.
- Second session (Windows, 10 workers, 33 minutes sharing the machine with a GPU job,
  `examples/demo_run_v2/`; made just before `const_prob` was lowered from 15% to 8%):
  **all 20 laws, the last in round 26**, every one proved
  symbolically (Bernoulli round 7, relativistic energy round 11, Planck-like round 26);
  15 library primitives; 32 continued-fraction formulas (10 π, 9 e, 5 φ, 3 ln 2, 2 √2,
  2 Catalan, 1 ζ(3)). The structured family found **Apéry's ζ(3) fraction** by itself
  (`u=[5,17], s=-1`: 6/ζ(3), 3.062 digits/term, δ = 0.095) and a known Catalan fraction in
  round 1. 72 verified results, no errors.

### What has actually been run, and what hasn't

| Part | Status | Evidence |
|---|---|---|
| `constants` and `laws` engines | ✅ run for real, Linux and Windows | unit tests; benchmark; 40-round runs |
| Fitted constants (R4), units (R5), `check_claim` | ✅ | `tests/test_laws.py`; 5 benchmark variants |
| Constants convergence profile, classes, Apéry family (R6) | ✅ | `tests/test_constants.py`; demo run |
| SymPy verifiers, safe parsing, LaTeX answers, time limits | ✅ Linux + Windows | tests, including code injection and a runaway answer |
| Orchestrator, checkpoint/resume, Ctrl+C, reports, CLI, bench, ablate, dashboard | ✅ | tests and manual runs |
| W&B / MLflow tracking | ⚠️ logic only | tested with a stand-in `wandb` module; neither package installed |
| transformers policy, expert iteration, GRPO training | ✅ real GPU (RTX 2060, fp16) | `MF_SLOW=1 pytest tests/test_model_path.py` (tiny Llama, now on GPU incl. a GRPO step) |
| Real-size model on a GPU (R1) | ✅ with the `hf` backend | Qwen2.5-Math-1.5B, `configs/gpu_small.yaml` (16 problems × 4 samples), 2 rounds with `train_every=1`: adapters v1 (17 traces, 185 s) and v2 (28 traces, 74 s) trained and loaded; `eval_pass1` = 0.267 reported both rounds; an `nvidia-smi` log shows generation at 3.1–3.6 GB, 0 MB when the model process ends, training peaking at 5.9 of 6 GB, then the reloaded policy; ~16–19 min per round |
| vLLM backend, Ray GPU actors | ❌ never run | vLLM is Linux-only; no Linux GPU machine |
| Lean verifier against real Lean 4 + Mathlib (R2) | ✅ Windows | `MF_SLOW=1 pytest tests/test_lean_real.py`; found traps 18 and 19 |
| Lean self-play with a real model and real Lean | ✅ plumbing / ❌ proofs | one round, Qwen2.5-Math-1.5B (not a prover) + real Lean: 8 theorems attempted, 5 repair rounds triggered by real Lean errors, 2 conjectures proposed (rejected), 0 proved: the model writes no Lean or `sorry`. Needs a prover model (R11) |
| Multi-node Ray (R3) | ✅ simulated on one machine | `MF_SLOW=1 pytest tests/test_ray_cluster.py`: real separate Ray nodes, one killed mid-round |
| Real multi-machine Ray | ❌ | needs two machines |
| Dockerfile, systemd unit, AWS cluster YAML | ❌ never run | written against current docs |
| `setup_lean.ps1`, `run_forever.ps1` | ✅ / ⚠️ | setup script ran end to end; the restart loop was not exercised |
| `setup_lean.sh`, `run_forever.sh` | ❌ | no Linux machine in the second session |

Treat every ❌ row as "probably right, must be smoke-tested".

Dev environments: (1) Linux, Python 3.13, 2 CPU cores, 7 GB RAM (first session);
(2) Windows 11, Python 3.10, i7-10750H 6 cores / 12 threads, 16 GB RAM, RTX 2060 6 GB,
CUDA 12.6 wheels (second session). Versions: sympy 1.14, mpmath 1.3, numpy 2.2, scipy 1.15,
torch 2.14, transformers 5.20, peft 0.21, ray 2.59, Lean v4.36.0-rc1 with Mathlib.

---

## 3. Design rules (do not break these)

1. **Verify before you store, verify before you learn.** Discoveries and training traces
   are written only after a verifier passes them. Reinforcement-learning rewards are the
   verifier's verdicts (or quantities computed from them, such as 4r(1−r)), never the
   model's own claims.
2. **Proposers are untrusted; our verifier code is the only judge.** Workers return
   claims. For `constants` and `laws` the head process re-verifies every claim
   independently before storing it. Self-play answers are judged by our checker code,
   never by the model's own claim; for model-invented problems everything the checker
   relies on (roots, degrees) is computed by SymPy.
3. **Never execute model output.** Maths goes through a whitelist parser with empty
   builtins; Lean code is screened for escape hatches before it is compiled.
4. **Exact resume.** All learned state is JSON in the database, saved every round.
   Stopping and restarting loses at most the round in progress. Older checkpoints must
   keep loading (new fields get defaults; see `SelfPlayEngine.load_state_dict`).
5. **No single failure stops the loop.** Worker errors come back as `{"error": ...}`;
   an engine crash is logged and the other engines continue; a failed training step
   keeps the previous model; tracking failures only switch tracking off.
6. **Collapse disguised duplicates before counting.** Canonical forms (fraction shifts
   and scalings, simplified expressions, normalised theorem statements) are the
   deduplication keys.
7. **Measure honestly.** The self-play evaluation set is fixed and checkpointed; the
   mock policy is for tests only; search changes are judged by `mathforge bench` on fixed
   seeds against an equal-compute control.
8. **Free parameters make "numerically exact" cheap.** Anything with fitted constants is
   cleaned, normalised and snapped before it may be accepted; a constant without a short
   exact form needs a symbolic proof (section 5.2).

---

## 4. Architecture

### 4.1 The loop

```
Orchestrator.run()                                   mathforge/orchestrator.py
 for each round, for each enabled engine:
   tasks   = engine.make_tasks(round)                propose (head process)
   results = executor.map(engine.worker_fn, tasks)   attempt and verify (workers / cluster)
   stats   = engine.ingest(results, round)           re-verify, deduplicate, store
   stats  |= engine.learn(round)                     update the engine's own model
   store.log_metrics(round, engine.name, stats);  tracking.log_round(...)
 every round:    engine.state_dict() -> state["engine:<name>"], round counter saved
 every 5 rounds: discoveries.md, metrics.csv, learning_curves.png
```

`SelfPlayEngine` overrides `run_round`: generation happens on the GPU model, verification
fans out through the executor, and training happens between rounds.

### 4.2 Module map

| File | Lines | Role |
|---|---:|---|
| `mathforge/orchestrator.py` | 157 | forever loop, signals, checkpoints, logging (one log file per run, also across runs in one process) |
| `mathforge/distributed.py` | 95 | `Executor`: serial, local process pool (forkserver / spawn on Windows), or Ray |
| `mathforge/store.py` | 204 | SQLite knowledge base: discoveries, traces, rollouts, metrics, state |
| `mathforge/config.py` | 166 | `DEFAULTS`, YAML deep-merge, `--set a.b.c=value` overrides |
| `mathforge/curriculum.py` | 57 | adaptive difficulty (success-rate band, rehearsal mixture) |
| `mathforge/novelty.py` | 40 | `stable_hash`, `interestingness` score |
| `mathforge/units.py` | 121 | SI unit parsing and dimensional inference over expression trees |
| `mathforge/report.py` | 129 | discoveries.md (incl. constant equivalence classes), metrics.csv, plot |
| `mathforge/bench.py` | 111 | cold-start benchmark of the laws search |
| `mathforge/ablate.py` | 155 | config variants × seeds → results table (resumable) |
| `mathforge/dashboard.py` | 139 | live web dashboard (stdlib HTTP server, inline JS/SVG) |
| `mathforge/tracking.py` | 80 | optional W&B / MLflow logging |
| `mathforge/cli.py` | 322 | `mathforge` command |
| `mathforge/engines/base.py` | 69 | `Engine` interface |
| `mathforge/engines/constants.py` | 655 | continued fractions, MITM + PSLQ, Apéry-like family, convergence profiles |
| `mathforge/engines/laws.py` | 970 | symbolic regression, fitted constants, units, `check_claim`, library learning |
| `mathforge/engines/selfplay.py` | 735 | proposer/solver self-play, expert iteration + GRPO, eval, Lean repair + library |
| `mathforge/verify/safe_parse.py` | 127 | safe maths parsing, LaTeX normalisation, answer extraction |
| `mathforge/verify/sympy_tasks.py` | 365 | problem generators, proposal validators, exact checkers, time limits |
| `mathforge/verify/lean.py` | 349 | Lean 4 verifier, theorem loaders |
| `mathforge/models/policy.py` | 393 | vLLM / transformers / mock policies, dtype choice, process isolation, Ray pool |
| `mathforge/models/trainer.py` | 244 | LoRA expert iteration and GRPO, fp16 loss scaling |
| `mathforge/data/lean_starter.jsonl` | 22 | starter theorems for Lean mode |
| `configs/` | | `laptop.yaml`, `gpu_small.yaml`, `gpu.yaml`, `lean.yaml`, `cluster.yaml` |
| `deploy/`, `scripts/` | | Dockerfile, systemd unit, AWS Ray cluster; `setup_lean.{sh,ps1}`, `run_forever.{sh,ps1}` |
| `tests/` | 1248 | fast suite + `test_model_path.py`, `test_lean_real.py`, `test_ray_cluster.py` (slow, `MF_SLOW=1`) |
| `examples/demo_run/`, `examples/demo_run_v2/` | | outputs and databases of the two 40-round runs |

### 4.3 Engine contract

```python
class Engine:                                   # mathforge/engines/base.py
    name: str
    worker_fn: Callable[[dict], dict]           # top-level, picklable, verifies what it claims
    def make_tasks(self, round_) -> list[dict]
    def ingest(self, results, round_) -> dict   # returns metrics for this round
    def learn(self, round_) -> dict
    def state_dict(self) -> dict                # JSON-serialisable
    def load_state_dict(self, d) -> None
```

To add an engine: subclass `Engine`, register it in `ENGINES` in `orchestrator.py`, and add
an `engines.<name>` block with `enabled` to `DEFAULTS` in `config.py`.

### 4.4 Persistence

`runs/<name>/mathforge.db` (SQLite, WAL mode, written only by the head process):

- `discoveries(engine, key UNIQUE, kind, statement, payload JSON, score, round)`
- `traces(engine, problem, solution, meta, round, used)`: verified training pairs
- `rollouts(engine, round, grp, prompt, completion, reward, used)`: every attempt in a
  group with a signal (some right, some wrong), for GRPO; written only with
  `trainer.algorithm: grpo`
- `metrics(round, engine, name, value)`
- `state(key -> JSON)`: `engine:<name>`, `round:<sorted enabled engine names>`, and
  `tracking:wandb_id` / `tracking:mlflow_id` so resumed runs continue the same tracked run

The round counter is keyed on the set of enabled engines, so enabling another engine in an
existing run directory restarts the round count (engine states still load by name).

### 4.5 Configuration

`DEFAULTS` in `config.py`, deep-merged with the YAML passed by `-c`, then `--set a.b.c=value`
overrides (values are parsed as YAML, so `--set x='[1, 2]'` gives a list). The resolved
config is written to `runs/<name>/config.yaml`.

### 4.6 Execution backends (`distributed.py`)

- `serial`: in-process; for tests and debugging.
- `local`: multiprocessing pool using `forkserver` (fork after CUDA or vLLM initialisation
  is unsafe), `spawn` on Windows; workers ignore SIGINT so only the coordinator handles
  Ctrl+C; `maxtasksperchild=200`.
- `ray`: `ray.init(address=...)`, falling back to starting a local Ray instance when none is
  running. One remote task per item with `max_retries = compute.ray_max_retries` (3): a task
  lost with its machine is re-run elsewhere, then comes back as an error entry. Results are
  fetched one at a time so a vanished node turns into an error entry instead of an exception.

---

## 5. The engines

### 5.1 `constants`: formulas for fundamental constants

Object: a polynomial continued fraction (PCF)
`L = a(0) + b(1)/(a(1) + b(2)/(a(2) + ...))` with integer polynomials, stored as coefficient
lists from low to high degree (`a=[1, 2]` means `1 + 2n`). Example: `a=[1,2], b=[0,0,1]`
is Brouncker's fraction for 4/π.

Pipeline per candidate (`constants_worker`):

1. Exact integer recurrences for the convergents p_n/q_n (common factors divided out every
   64 steps). Reject if b(n) = 0 for some n ≤ depth (the fraction terminates: rational).
2. Convergence: compare the convergents at 0.8·depth and depth (depth 300); require at
   least `screen_digits` = 18 matching digits.
3. Reject rational limits (PSLQ on [1, L]) and limits that are 0.
4. Identify the limit as a Möbius transform (p + qK)/(r + sK) of a target constant K:
   - Fast path: a meet-in-the-middle table of all such values with |p|, |q|, |r|, |s| ≤
     `mitm_coeff` = 6 for every target (about 110k entries keyed by `round(1e8 * value)`,
     built in 0.1 s and cached per process). A table hit is confirmed at working precision.
   - Slow path: for a random `pslq_fraction` = 10% of unidentified limits (30% in
     structured tasks), full PSLQ on [1, K, −L, −KL] with coefficients up to 200.
5. Verify: recompute at `verify_depth` = 2000 and confirm the relation to `verify_digits` =
   50 digits. The head re-runs `verify_candidate` before storing.

Where the time goes (profiled in the second session): 74% PSLQ, 23% exact evaluation, so a
vectorised convergence pre-screen would buy little (84–93% of random candidates converge
anyway). gmpy2 (`pip install -e ".[fast]"`, used automatically by mpmath) gives about 10%.

Targets: π, e, ln 2, ζ(3), Catalan's constant, √2, φ, Euler's γ, ζ(5), π² (ln 3, ζ(2), √3
are also defined in `target_values`).

Deduplication: `canonical_pcf` maps every fraction to one representative of its family under
index shifts n → n+s (|s| ≤ 6), scaling a → ka, b → k²b, and sign. Candidates are
deduplicated on this form before testing, and it is the discovery key. Fractions with
constant a and b are skipped (they are periodic, so the limit is always a quadratic
irrational).

**Structured family (R6).** `apery_like(u, s, odd)`: a(n) = (2n+1)·P(n(n+1)) (odd) or
P(n(n+1)) (even), P(m) = u0 + u1·m + ..., b(n) = s·n^(2·deg a). It contains Apéry's
fractions: `u=[5,17], s=-1, odd` → 6/ζ(3) and `u=[3,11], s=1, even` → 5/ζ(2) (= 30/π²).
`structured_fraction` = 25% of tasks draw from it; degree unlocks with the level (3 at
level 1, 4 at level 3, 5 = the ζ(5) analogue of Apéry from level 5); u0..u(k-1) ∈ [−R, R],
the leading u ∈ [1, R], R = 4·(2 + level), s ∈ ±{1, 2, 4, 8, 16}. Hits become
`structured_seeds` whose integers are nudged (family search) and teach the u-slot weights.
Nearly every structured fraction converges, so these tasks run more PSLQ.

**Convergence profile (R6).** `convergence_profile(a, b)` on the head for every new
discovery (reference value: depth 2000 at 1200 digits): `digits_per_term` (slope of
−log10|p_n/q_n − L|; Apéry's ζ(3) fraction measures 3.062, theory 8·log10(1+√2) = 3.061)
and `delta`, the Ramanujan-Machine irrationality-measure estimate |L − p_n/q_n| =
q_n^−(1+δ) with q_n reduced; δ > 0 for all large n would prove L irrational (Apéry: ≈ 0.08
asymptotically; finite-n estimates come out higher). e and φ give δ ≈ 1, as their
irrationality measure 2 says they should. Both go into the payload.

**Equivalence classes (R6).** Every limit (p + qK)/(r + sK) is a rational Möbius transform of
K, so all formulas for one target are equivalent in that sense; `report.constant_classes`
groups them per target and reports the count, distinct values, the fastest member and the
best δ.

Learning (cross-entropy-method style): weights over (degree of a, degree of b) and over the
values of each coefficient slot are incremented by every new verified hit and decay by 0.5%
per round. Hits become seeds for mutation (nudge a coefficient, swap in a or b from another
hit, grow or shrink a degree). `explore_fraction` = 0.5 of candidates are fresh random draws.

Curriculum: the level is the size of the search space: coefficient range R = 2 + level,
degrees with deg a ≤ 3, deg b ≤ 2·deg a, and deg a + deg b ≤ 2 + level. The level goes up
when a round could fill less than half of its requested candidates (the space is exhausted)
or after 3 rounds without a new discovery.

Status of results: numerical conjectures verified to 50+ digits (the same status as the
Ramanujan Machine's published conjectures), not proofs.

### 5.2 `laws`: laws of nature from raw numbers

Benchmark (`LAWS` in `laws.py`, 20 laws in 6 tiers): spring energy, kinetic energy,
centripetal force, ideal gas, Newton's gravity, pendulum period, Pythagoras, escape velocity,
thin lens, Kepler's third law, Gaussian, Doppler shift, 3-D distance, projectile range,
Lorentz factor, relativistic energy, Coulomb superposition, damped oscillation, Bernoulli
pressure, a Planck-like spectrum. Each task samples 200 points from the law's ranges.
`LAW_UNITS` gives every variable's SI units.

Search (`laws_worker`), genetic programming over expression trees:

- Tree encoding (tuples): `("v", i)` variable, `("k", c)` fixed constant from {1, 2, 3, 0.5},
  `("c", value)` **fitted constant** (`("c", value, "981/50")` once snapped),
  `("u", op, x)`, `("b", op, x, y)`, `("m", idx, args...)` call of a learned library
  primitive, `("a", i)` argument inside a primitive's body.
- Operators: neg, sq, sqrt(|x|), inv, exp (clipped), log(|x|), sin, cos; add, sub, mul, div.
- Fitness: NMSE after the best linear rescaling y ≈ α·f + β (α, β in closed form), on a
  75/25 train/validation split. Score = log10(NMSE) + 0.02·size (+ `units_penalty` = 2 if
  dimensionally inconsistent); maximum size 40.
- Variation: crossover 55%, subtree mutation 20%, point mutation 12%, hoist 8%, unary wrap
  5%; duplicate elimination; 5% elitism; tournament size 5; everything except the elite is
  re-seeded after 10 generations without improvement; fitness cache.
- Operator sampling weight = a fixed physics prior × weights learned from verified solutions.
- Defaults: population 500, 60 generations, 8 tasks per round.

**Fitted constants (R4, `fit_constants: true`).** Terminals are variables 75%, fitted
constants `const_prob` = 8% (started at ±1, 2, 3, 1/2), fixed constants the rest (15%
fitted was both worse and slower in the benchmark). `fit_constants()` runs
Levenberg-Marquardt (`scipy.optimize.least_squares`) on a tree's constants with α and β
kept in closed form (variable projection), so it only searches the constants inside the
formula. Lamarckian: every generation, `tune()` fits the untuned trees with constants that
are (a) among the 10 best, (b) small (size ≤ `tune_small` = 12; a small tree with a guessed
constant looks hopeless until fitted: on `exp(-0.37x)` this took success from 1–3/8 to
8/8 seeds), or (c) 10 random ones, and writes the values back so offspring inherit them. A
cache by tree shape (constants blanked) reuses earlier fits: a third of all fits repeat a
shape (38% faster). The returned best tree gets a final 600-evaluation polish. Fitted
constants never enter library primitives. Constant fitting is still about 2/3 of the
search time on hard laws.

**Dimensional analysis (R5, `units: true`).** `mathforge/units.py`: units are exponent
vectors over (kg, m, s, K, mol, A) (derived N, J, W, Pa, Hz, C, V parse too). Inference is
bottom-up: fixed constants are dimensionless, a fitted constant is a wildcard (it can carry
units, like 2g in ρ(v² + 2g·h)), add/sub need equal units, exp/log/sin/cos need a
dimensionless argument; the output unit is free because α can carry units (G, R). An
inconsistent tree is penalised, not banned, so an exact formula still beats any
approximation. Benchmark laws bring units; datasets can declare them
(`datasets: [{path, target, units: {col: "m/s"}}]`, `mathforge fit --units m=kg,v=m/s`).

**Verification on the head (`check_claim`, shared with the benchmark).**

- Trees without fitted constants: `verify_law`, unchanged:
  1. 2,000 fresh points on ranges widened 2× (out of distribution): NMSE must be < 1e-10.
  2. Symbolic proof (`prove_equivalence`): f and the true law g are affinely equivalent iff
     ∂g/∂xᵢ = α·∂f/∂xᵢ with one constant α for every variable. Label `symbolic`.
  3. Protected operations leave `Abs` in f. `strip_abs_on_domain` replaces Abs(u) by u or
     −u when u has a single sign on all sampled points; then the proof is retried, and a
     gradient ratio SymPy can't reduce is simplified with powers combined
     (`powdenest`/`powsimp` with `force=True`), allowed only after
     `bases_positive_on_domain` confirms every fractional-power base is positive at all
     sampled points. Label `symbolic (on the physical domain)`.
  4. Otherwise accepted only if the out-of-distribution NMSE is < 1e-12. Label `numeric-ood`.
- Trees with fitted constants are cleaned first, at the SymPy level where constants merge
  (the search wrote Bernoulli as ρ/(c1/v)/(c2/v) + ρh with only c1·c2 = 19.62 identifiable):
  1. **drop negligible constants**: zero any float whose removal keeps the formula
     numerically exact OOD (kills `T*n/V + 4.8e-9*n`);
  2. **`normalize_affine`**: remove the overall factor and additive constant, and divide a
     fraction's numerator and denominator by one denominator coefficient
     (`d1*d2/(1.4e8*d1 + 1.4e8*d2)` → `d1*d2/(d1 + 0.99999999*d2)`);
  3. **`snap_value`**: shortest exact number within 1e-7 relative: a decimal with ≤ 6
     significant digits (9.81, 6.674e-11), a fraction with denominator ≤ 1000, or a
     fraction times π, at most 7 digits in total (no "free" long integers);
  4. a formula all of whose constants snapped may be accepted by proof or numerically; one
     with a constant that did not snap only by a symbolic proof.
- The truth is parsed with `rational=True` (9.81 means exactly 981/100), otherwise Float
  arithmetic blocks exact cancellation in the proof (trap 22).

Symbols are declared `positive` when the variable's range is positive and `real` otherwise.

Learning:

- Grammar: the operators in each law's best verified solution get extra weight.
- Library learning (`LawsEngine.learn`, in the spirit of DreamCoder): every subtree of size
  3 to 9 in a solved law, with its 1 to 3 variables abstracted into arguments, is a
  candidate primitive. Gain = (size − 1) per law that uses it. A candidate is accepted if
  two or more laws use it, or if its gain is at least 5 once 3 laws are solved. Rejected:
  single-operator wrappers, bodies that simplify away an argument (A0/A0), subtrees with
  fitted constants, and duplicates of an existing primitive. Maximum 40. New primitives
  become operators with initial weight 2.
- Curriculum: laws are drawn from tiers up to the current level, 80% from unsolved ones.

Your own data: `engines.laws.datasets: [{path, target, name, units}]` (CSV; every other
column is an input) is searched every round; the best formula is kept and stored as
`empirical` when its held-out NMSE is below 1e-6 (constants printed to 6 digits). One-off:
`mathforge fit data.csv --target y [--units ...] [--no-constants]`.

### 5.3 `selfplay`: a language model that teaches itself

Off by default (it needs a GPU). Two verifier modes.

**SymPy mode** (`verifier: sympy`):

- Problem families, generated backwards so the answer is checkable: `integrate`
  (differentiate a random composite F; complexity grows with level), `solve` (polynomial
  built from chosen roots; from level 6 also exp and shift substitutions), `sum` (closed
  form of Σ p(k); the degree of p grows with level).
- Per round: `problems_per_round` = 128, of which 25% are invented by the model itself,
  spread over the three families in turn (R7). The proposer sees 4 examples from that
  family's frontier and answers `PROBLEM: ...`; `validate_proposed` makes it checkable:
  integrands through the safe parser; equations must be polynomials of degree 1–4 with
  rational coefficients and at least one real root, and **SymPy computes the roots** (exact
  when short, else 30 digits); summands must be polynomials of degree 1–7. The solver
  samples k = 8 completions per problem at T = 0.8.
- Verification runs in executor workers (`sympy_verify_worker`).
- Training data: up to the 2 shortest distinct correct completions per problem; plus the
  proposer's completion when its problem turned out "learnable" (pass rate r strictly
  between 0 and 1), with reward 4r(1−r) recorded.
- Discoveries: solutions to model-invented problems of every family (payload has the
  problem, so `mathforge verify` re-checks it); "milestones" (first solve at a new level).
- One curriculum per family (target success 0.5 ± 0.15).
- Held-out evaluation: a fixed set (levels 1, 3, 5, 7, 9 × 3 families × 4 problems), greedy
  pass@1 every 5 rounds, reported as `eval_pass1` and `eval_pass1_<family>`. The set is
  checkpointed and never changes, so the curve is comparable across the whole run.

**Lean mode** (`verifier: lean`):

- Theorem pool: 22 starter theorems (`mathforge/data/lean_starter.jsonl`), plus
  `lean.theorem_file`: a `.jsonl` (`{name, statement}` or miniF2F's `formal_statement` +
  `split`, filtered by `lean.theorem_split`), a `.lean` file, or a directory of them
  (PutnamBench; statements that need a `def`/`abbrev` of their own file are skipped), plus
  conjectures the model invents.
- Conjectures: the proposer sees up to 4 proved statements. A proposed statement must
  type-check with a `sorry` proof (with `autoImplicit` off) and must not be closed by any
  one of `rfl | decide | norm_num | simp | omega | positivity | linarith | nlinarith`
  (each alternative must close the goal). Unproved conjectures are abandoned after 5 attempts.
- Per round: the least-attempted open theorems first, plus 10% rehearsal of solved ones;
  k samples each with a DeepSeek-Prover-style prompt; identical proofs are checked once;
  proofs are checked 16 per Lean process.
- **Lemma library (R8):** up to `library_size` = 3 proved theorems sharing the most
  identifiers with the target are shown in the prompt as citable lemmas and prepended to
  the checked file with their proofs (Lean re-checks them every time; `#print axioms`
  covers them transitively). Each proved theorem records `lib_used`; `_library_code`
  emits the dependency closure, each lemma once, in dependency order.
- **Repair (R8):** for every theorem whose attempts all failed with a real Lean error, the
  model sees one failed attempt plus Lean's message and samples `repair_samples` = 2 fixes.
  Successful repairs are stored and trained on as (repair prompt → fixed proof).

**Training** (`_maybe_train`, both modes; every `train_every` = 5 rounds):

- **Expert iteration** (default, and always the first step): if at least `min_new_traces` =
  64 unused verified traces exist, write the latest `max_traces` = 20,000 to JSONL; free the
  GPU (end the model process or kill the Ray actors) unless `trainer.keep_policy_loaded`;
  train a fresh LoRA adapter (r = 32, α = 64, lr 1e-4, cosine schedule with 5% warmup,
  1 epoch, loss on completion tokens only, gradient checkpointing) on the base model, in a
  subprocess (`python -m mathforge.models.trainer job.json`) or as a Ray GPU task; save to
  `runs/<name>/adapters/v<N>`; reload. On any failure, keep the previous adapter.
  Retraining from the base on all accumulated verified data, instead of stacking adapters,
  avoids compounding drift (ReST / expert-iteration style).
- **GRPO** (R7, `trainer.algorithm: grpo`, from the second step on): needs
  `grpo.min_rollouts` = 256 unused attempts in groups with a signal. Advantage =
  (reward − group mean)/(group std + 1e-4); groups are one prompt's k attempts, and all
  proposals of a round form one group (reward 4r(1−r), 0 if invalid). Loss per attempt:
  −advantage · mean token log-prob + β · mean token KL (β = 0.04; k3 estimator against the
  base model with the adapter switched off); lr 1e-5; continues from the current adapter
  (shipped as bytes to a Ray GPU task). One pass over fresh attempts, so the importance
  ratio is 1 and needs no clipping; attempts may be up to `train_every` rounds old, so use
  `train_every: 1` for strictly on-policy updates.
- dtype: bf16 where the GPU has it natively (compute capability ≥ 8), fp16 with a gradient
  scaler on older GPUs (bf16 is only emulated there), fp32 on CPU (`policy.model_dtype`).
  Token log-probs are computed in chunks so a 150k-token vocabulary fits in 6 GB.

**Policies** (`models/policy.py`):

- `vllm` (LoRA hot-swapped through `LoRARequest`), `hf` (transformers + peft), and `mock`
  (a scripted SymPy oracle whose accuracy rises with the adapter version; its failed Lean
  attempts use `bad_tactic`, which the fake `lake` rejects; tests only).
- Real backends run in a separate spawned process (`ProcessPolicy`). Ending that process is
  the only reliable way to give GPU memory back before training. Calls poll the pipe with a
  liveness check, so a crashed model process raises an error instead of hanging the loop.
- On a Ray cluster with GPUs, `PolicyPool` starts one `_PolicyActor` per `tensor_parallel`
  GPUs, shards prompts round-robin, and ships adapter files to the actors as bytes.

---

## 6. Verifiers

**`verify/safe_parse.py`**: whitelisted characters and identifiers (maths functions,
constants, declared variables), no `__`, at most 400 characters, empty `__builtins__`;
SymPy `parse_expr` with implicit multiplication and `^`. `latex_to_text` handles `\frac`,
`\sqrt[n]{}`, `^{}`, `\cdot`, `\ln|x|`, `\left`/`\right`, `\text{or}`, set braces and
"sin x". `extract_answer` takes the last `ANSWER:` line, or the last `\boxed{...}` with full
brace matching; `normalize_answer` strips `F(x) =`, `+ C`, `x =` and "or"/"and".

**`verify/sympy_tasks.py`**:

- `integrate`: compare d/dx(answer) with the integrand at up to 20 complex points with real
  part in (0.3, 2.7) and |imaginary part| ≤ 0.2; 6 agreements within 1e-7 and no
  disagreement are required. Answers without x, or containing an unevaluated `Integral`,
  are rejected. ln|u| is rewritten as ln(u) (same derivative, so exact); any other |u|
  switches the comparison to real points with a real symbol.
- `solve`: the answer's root set must equal the true root set within 1e-9.
- `sum`: the answer must be a polynomial of degree ≤ d + 1 matching the partial sums at
  n = 0..d+2. That is a proof, not a spot check.
- Every check runs under a 10 s time limit: `SIGALRM` where it exists; on Windows a timer
  thread raises SIGINT in the main thread (`_time_limit_interrupt`), which Python delivers
  between bytecodes just like SIGALRM; a genuine Ctrl+C is passed on to the previous
  handler, and an interrupt that arrives as the block finishes is absorbed. Only works in a
  process's main thread (pool workers qualify); not in Ray tasks run off the main thread.
- `validate_proposed`, `problem_text`, `problem_spec`: model-invented problems (5.3).

**`verify/lean.py`**:

1. Block comments are stripped (unbalanced ones are rejected), then statement, proof and
   helper lemmas are screened for escape hatches: `sorry`, `admit`, `axiom`,
   `native_decide`, `implemented_by`, `extern`, `unsafe`, `opaque`, `elab`, `macro`,
   `macro_rules`, `syntax`, `notation`, `infix`/`prefix`/`postfix`, `run_tac`, `run_cmd`,
   `run_elab`, `skipKernelTC`, `ofReduceBool`, `declare_syntax_cat`, `#exit`, `#eval`.
2. The proof is attached to **our** statement text under a unique name; the model can never
   restate the theorem. Helper lemmas are allowed only as `theorem`/`lemma`, renamed per
   item, so they cannot add definitions, instances or notation.
3. One file per batch: header (`set_option autoImplicit false`, trap 18), items, then
   `#print axioms` for every item. Run `lake env lean FILE` inside the project; `lake` is
   found on PATH (PATHEXT-aware) or in `$ELAN_HOME/bin` / `~/.elan/bin`; output is decoded
   as UTF-8.
4. An item is accepted only if it has no error in its line range, **and** Lean printed its
   axioms, **and** they are a subset of {propext, Classical.choice, Quot.sound}. Items that
   are error-free but have no axiom report are re-checked alone. Errors before the first
   item are reported as a workspace error.
5. Message formats, **confirmed on Lean v4.36.0-rc1** (`tests/test_lean_real.py`):
   `'name' depends on axioms: [propext, Classical.choice, Quot.sound]`,
   `'name' does not depend on any axioms`, and errors as `file:line:col: error: ...`.

---

## 7. Running and deploying

| Command | Purpose |
|---|---|
| `mathforge run -c CONFIG [--rounds N] [--only a,b] [--set k=v]` | start or resume |
| `mathforge status [--run-dir D]` | knowledge-base size, latest metrics, curriculum levels, adapter version |
| `mathforge discoveries [--engine E] [--top N] [--order score\|new\|old]` | list verified results |
| `mathforge verify ID [--depth 6000 --digits 100]` | independently re-verify one result |
| `mathforge fit FILE --target COL [--units c=u,...] [--no-constants]` | formula discovery on a CSV |
| `mathforge bench laws [--seeds 5] [--only a,b] [--out rows.json]` | cold-start benchmark of the laws search |
| `mathforge ablate -c CONFIG --vary k=a,b [--seeds 3 --rounds 10 --out-dir D]` | variants × seeds → `ablation.md`/`.csv` (resumable) |
| `mathforge dashboard [--run-dir D] [--port 8765]` | live web page (works during a run) |
| `mathforge report` / `mathforge export --out all.jsonl` | rewrite reports / dump everything |

Tests: `pytest -q` (fast, ~25 s). Slow, with `MF_SLOW=1` (all 8 together: ~6.5 min on the
second session's laptop): `tests/test_model_path.py` (tiny Llama: generation in its own
process, expert iteration, adapter reload, a GRPO step; uses the GPU if there is one; ~45 s
on the RTX 2060, several minutes on a busy CPU), `tests/test_lean_real.py` (real Lean;
~3 min, each Lean run loads Mathlib), `tests/test_ray_cluster.py` (needs `ray`; ~2 min).

Configs: `laptop.yaml` (CPU engines, all cores); `gpu_small.yaml` (one 6–8 GB GPU, any OS:
transformers backend, Qwen2.5-Math-1.5B, 32 problems × 4 samples, LoRA r = 16, batch 1 ×
accumulation 8); `gpu.yaml` (self-play with Qwen2.5-Math-1.5B-Instruct on vLLM, 256 problems
per round); `lean.yaml` (DeepSeek-Prover-V2-7B, 8 Lean workers because each needs 3 to 6 GB
of RAM); `cluster.yaml` (Ray, Qwen2.5-Math-7B-Instruct, 2,048 problems per round).

Deployment: `scripts/run_forever.sh` / `run_forever.ps1` (restart loop), `scripts/setup_lean.sh`
/ `setup_lean.ps1` (elan + Mathlib; the PowerShell one leaves PATH alone unless `-ModifyPath`),
`deploy/mathforge.service` (systemd; SIGINT stop, restart on failure), `deploy/Dockerfile`
(CUDA 12.4 image with every extra), `deploy/ray-cluster-aws.yaml` (head on demand, spot CPU
workers, L40S GPU workers; needs a Deep Learning AMI id filled in). Graceful stop: the first
Ctrl+C or SIGTERM finishes the current step and checkpoints; a second one aborts.
`mathforge ablate` stops the whole ablation on Ctrl+C; rerunning continues it.

---

## 8. Traps already hit and fixed (don't reintroduce them)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | constants workers crashed with `ValueError` | `mpmath.pslq` rejects zero entries (limit equal to 0) | zero check in `is_rational`, per-candidate try/except |
| 2 | 95% of constants time spent in PSLQ | PSLQ for every target and every limit | meet-in-the-middle table plus PSLQ on a 10% sample (about 10× faster) |
| 3 | hundreds of "discoveries" that were one fraction in disguise | index shifts, a→ka with b→k²b, sign flips; two mutation operators generated exactly these | `canonical_pcf` for test dedupe and as the discovery key; those mutations removed |
| 4 | endless "formulas" for √2 and φ | constant a and b give periodic fractions | skipped in `_novel_candidate` |
| 5 | constants curriculum stuck at level 1 or 2 | a success-rate curriculum is the wrong model for exhaustive search | level grows on exhaustion or after 3 dry rounds |
| 6 | law verification wrong when a range crosses 0 | SymPy `Symbol(positive=False)` means non-positive, not "unknown" | `real=True` for those variables |
| 7 | printed law differed from the evaluated one | the search uses sqrt(abs(x)) and log(abs(x)) but the symbolic translation used plain sqrt and log | `_sym_sqrt`/`_sym_log` emit `Abs`; `strip_abs_on_domain` for proofs |
| 8 | search stuck at a+b for √(a²+b²) | premature convergence | duplicate elimination, point mutation, restarts, operator prior, population 500 |
| 9 | library full of junk (A0/2, "1", single operators) | no filters | filters in `LawsEngine.learn` |
| 10 | Ctrl+C gave a traceback storm and lost state | SIGINT reached pool workers too | workers ignore SIGINT; `close()` terminates on a second interrupt |
| 11 | (latent) unsafe fork after CUDA or vLLM start-up | default `fork` start method | `forkserver` pool (`spawn` on Windows) |
| 12 | coordinator would hang forever if the model process died | blocking `Pipe.recv` | `ProcessPolicy._recv` polls with a liveness check |
| 13 | correct LaTeX answers marked wrong; `\frac` turned into a form feed | `re.sub` escape handling in replacement strings; a one-level brace regex for `\boxed` | literal replacement lambdas; brace-matching extractor |
| 14 | transformers deprecation and invalid-flag warnings | transformers ≥ 4.56 renamed `torch_dtype` to `dtype`; sampling flags passed with greedy decoding; `use_cache` with gradient checkpointing | `dtype_kwargs`; sampling flags only when sampling; `use_cache=False` |
| 15 | a test failed against a correct verifier | the test's formula was wrong (1/√(c²−v²) is not the Lorentz factor) | lesson: when a verifier rejects something, check the candidate first |
| 16 | the integral checker accepted \|x − 3\| as an antiderivative of 1 | every Abs(u) was replaced by u | only ln\|u\| → ln u is rewritten; any other Abs is compared on the real line |
| 17 | Lorentz factor and relativistic energy only reached `numeric-ood` | SymPy can't simplify √(c²−v²)·√(1/(c²−v²)) without knowing c > v | domain-checked power combination in `affine_equivalent_symbolic` |
| 18 | `theorem w : n ≤ undefined_thing` "type-checked" on real Lean | `autoImplicit` turns unknown names into variables, so a conjecture with a typo means something else | `set_option autoImplicit false` in `HEADER` (found only by `test_lean_real.py`) |
| 19 | the trivial-conjecture filter failed to close `a + b = b + a` | `first \| … \| norm_num \| simp \| …` stops at the first tactic that *succeeds*; norm_num/simp succeed when they merely make progress | every alternative is `(tac; done)` (found only on real Lean) |
| 20 | fitted-constant formulas such as `T*n/V + n/210280000`, `d1*d2*(d1 + 652377492)/(d1 + d2)` accepted as laws | junk terms riding on tiny or huge fitted constants pass the 1e-12 numeric test | `check_claim`: junk removal, `normalize_affine`, short snapping only, proof required for unsnapped constants |
| 21 | snapping "succeeded" for any huge constant | every number is close to some long integer | `snap_value(max_cost=7)` |
| 22 | correct Bernoulli formula only `numeric-ood` | truth literal 9.81 parsed as a Float; 9.81·50/981 ≠ 1/2 in floats | `law_truth` parses with `rational=True` |
| 23 | per-leaf snapping failed on correct formulas | constants identifiable only as a product (ρ/(c1/v)/(c2/v)) | snap at the SymPy level after simplification merges them |
| 24 | right-structured trees with guessed constants never got fitted (1/8 on `exp(-0.37x)`) | only the 10 best + 10 random trees were tuned; an unfitted small tree loses to any approximant | also tune every small tree (`tune_small` = 12): 8/8 |
| 25 | Ray tests "passed" with every task failing | pytest imports the test module under a name Ray workers can't import | `ray.cloudpickle.register_pickle_by_value`; assert the errors are node-loss errors |
| 26 | Windows: no `SIGALRM` (no verifier time limit), shebang fake `lake` not runnable, `lake` not found, Lean output garbled, tests writing ℕ crashed | platform differences | timer-interrupt time limit; `lake.cmd` wrapper; `lake_executable()`; `encoding="utf-8"` everywhere |
| 27 | later runs in one process logged into the first run's log file | `setup_logging` only attached a handler once | handlers follow the current run dir; `Orchestrator.close` releases the file |
| 28 | a real model's proposals were 0/4 valid (the mock's were always fine) | Qwen2.5-Math ignores "output one PROBLEM: line"; it states the problem in LaTeX and starts solving it | `extract_proposal`: PROBLEM line, else the first formula of the family's shape; LaTeX → text before the safe parser: 10/12 valid |
| 29 | every laws task failed after `--set engines.laws.success_nmse=1e-300` | PyYAML follows YAML 1.1: `1e-300` (no dot) is a *string*; `lr: 1e-4` in a config would break the same way | `config.yaml_load` with the YAML 1.2 float rule; failed tasks now log their first error message |

Operational notes: `pkill -f PATTERN` run from a shell whose own command line contains
PATTERN kills that shell too. Benchmarks are only comparable on an otherwise idle machine
(a concurrent GPU job doubled search times in the second session). Ray's memory monitor
kills tasks when RAM runs low, which looks like node loss.

---

## 9. Known limitations

- `constants`: the sets of already-tested candidates live only in memory, so after a
  restart some candidates are tested again (harmless; the store deduplicates).
- `constants`: δ is measured at finite n against a 1200-digit reference; it overestimates
  the asymptotic value and is a signal, not a proof.
- `laws`: constant fitting multiplies search cost about 5× (most of it LM fits); no noise
  robustness study yet (`noise` is 0 by default); relativistic energy, Coulomb superposition
  and the Planck-like spectrum are rarely solved from a cold start.
- `laws`: the units penalty makes fitted constants attractive as "unit adapters"; a
  dimensionally exact search (units on constants) would be stricter.
- `selfplay`: GRPO is single-pass (no PPO epochs, no clipping) and slightly off-policy when
  `train_every` > 1; no reference-model logprobs are stored with rollouts.
- `selfplay`: the transformers backend is slow (about 19 minutes per small round on an
  RTX 2060); serious runs need vLLM on Linux.
- Lean: every proof re-checks its library lemmas (cost grows with `library_size`); lemma
  relevance is identifier overlap only.
- The store has a single writer (the head process); that is fine for one coordinator.
- Time limits need the main thread: no hard limit inside Ray tasks run off it.

---

## 10. Roadmap

### Done in the second session

| Item | Result |
|---|---|
| R1 GPU smoke test | done with the `hf` backend on an RTX 2060 (vLLM impossible on Windows): two training steps, adapter reload, eval, memory released (section 2); found trap 28 (real proposers write LaTeX) |
| R2 real Lean | done: setup script, `test_lean_real.py` (4 tests), traps 18 and 19 found and fixed; `lean.yaml` itself not run (its 7B prover needs ~16 GB of GPU memory) |
| R3 multi-node Ray | done on a simulated cluster (head 1 CPU + worker node 4 CPUs, worker killed mid-round): without retries the lost tasks come back as `NodeDiedError` entries while the head's finish; with retries all complete on the head; a full orchestrator round completes with "4 of 10 tasks lost with the node, counted as errors" (and found trap 29 on the way) |
| R4 constant optimisation | done (5.2); Bernoulli 0/5 → 5/5 |
| R5 dimensional analysis | done (5.2) |
| R6 convergence, classes, ζ(5) push | done (5.1); profiling showed a vectorised evaluator would not pay, gmpy2 added as an option |
| R7 RL | done: GRPO, 4r(1−r), proposers for all families |
| R8 Lean repair and search | done: repair, lemma library, miniF2F/PutnamBench loaders |
| R9 research tooling | done: bench, ablate, dashboard, W&B/MLflow |

### Next, in priority order

**R10. vLLM on a Linux GPU box.** `pip install -e ".[llm,vllm]"`; `MF_SLOW=1 pytest
tests/test_model_path.py`; `mathforge run -c configs/gpu.yaml --rounds 2 --set
engines.selfplay.train_every=1 --set engines.selfplay.min_new_traces=8`. Confirm vLLM loads
adapters and memory is released before training. Then a long run with `eval_pass1` curves,
expert iteration vs GRPO (`mathforge ablate --vary engines.selfplay.trainer.algorithm=expert_iteration,grpo`).

**R11. Lean at scale.** A 24 GB+ GPU with DeepSeek-Prover-V2-7B (`configs/lean.yaml`) on
miniF2F-valid (`theorem_file` + `theorem_split: valid`); measure the effect of repair and
the lemma library with `ablate`.

**R12. Faster constant fitting.** Constant fitting is ~2/3 of laws search time: analytic
Jacobians (forward-mode over the tree), fewer LM evaluations for trees far from the front,
or a compiled evaluator; then raise population/generations at equal cost.

**R13. Laws robustness.** Noise (`noise` > 0) with an NMSE threshold relative to the noise
level, and units on fitted constants (strict dimensional search).

**R14. Constants: proofs.** For fractions with δ > 0 and a closed form, attempt a
symbolic proof of the limit (Zeilberger / creative telescoping), or export them for the
Ramanujan Machine's conjecture-proving pipeline.

**R15. Real multi-machine Ray and the deployment files** (R3 on two machines, Docker,
systemd, AWS YAML), and `run_forever.sh`/`setup_lean.sh` on Linux again.

---

## 11. Working conventions

- Python ≥ 3.10, `from __future__ import annotations`, type hints on public functions.
- Docstrings explain *why* (the maths or the failure being guarded against), not only what.
- Worker inputs and outputs are plain JSON-like dicts; trees are tuples (`to_tuple` converts
  JSON lists back).
- New metrics are just new keys in the dict returned by `ingest`/`learn`; add the important
  ones to `SUMMARY_KEYS` (orchestrator log line) and `CURVES` (report plot and dashboard).
- Measure search changes with `mathforge bench` (same seeds before/after, idle machine,
  plus an equal-compute control); run long jobs from a copy of the package (PYTHONPATH) so
  edits can't reach recycled workers.
- See `CLAUDE.md` for the short list of rules.

---

## 12. Glossary

- **PCF**: polynomial continued fraction.
- **PSLQ**: integer-relation algorithm; finds integers c with c·x ≈ 0 for a vector x of reals.
- **Meet-in-the-middle**: precompute one side of a relation in a table and look the other up.
- **δ (irrationality-measure estimate)**: |L − p/q| = q^−(1+δ) for a fraction's reduced
  convergents; δ > 0 for all large n proves L irrational.
- **GP**: genetic programming, evolving expression trees.
- **NMSE**: normalised mean squared error (MSE divided by the variance of y).
- **OOD**: out of distribution; test data outside the training range.
- **Variable projection**: fitting the nonlinear constants while the linear ones (α, β) are
  solved exactly for each trial.
- **Lamarckian**: optimised values are written back into individuals and inherited.
- **Expert iteration**: sample attempts, keep the verified successes, fine-tune on them, repeat.
- **GRPO**: group-relative policy optimisation: policy gradient with a baseline from other
  attempts at the same prompt, no value network.
- **LoRA**: a small trainable adapter on top of a frozen language model.
- **Lean / Mathlib**: a proof assistant and its mathematics library; Lean's kernel is the final arbiter.
- **Ray**: a framework for running Python across many machines.
- **vLLM**: a fast GPU inference library for language models (Linux only).

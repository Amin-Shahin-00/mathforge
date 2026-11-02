# MathForge

**A self-improving mathematical discovery system.** It invents candidate formulas,
theorems and problems, checks them with an exact verifier, keeps what is
provably or numerically certain, learns from its own verified successes, and
repeats, for as long as you leave the computer running.

```
   propose  ──►  attempt  ──►  VERIFY  ──►  knowledge base
      ▲                          │
      └────────  learn  ◄────────┘   (only verified results are ever learned from)
```

The one rule that makes self-learning work: **nothing enters the knowledge base
or the training data unless a verifier has checked it.** Verifiers are exact
integer arithmetic, 50+ digit numerics, SymPy-based checks, symbolic proofs and
the Lean 4 proof kernel. A model that trains on its own unchecked output drifts
into confident nonsense; a model that trains on verified output can only get
better.

---

## Three engines

| Engine | What it discovers | Verified by | How it learns from itself | Hardware |
|---|---|---|---|---|
| **constants** | New formulas for π, e, ζ(3), Catalan's constant, ln 2 … as polynomial continued fractions (the Ramanujan Machine's search space). Also hunts open targets such as ζ(5) and Euler's γ, including in the structured "Apéry-like" family that Apéry's ζ(2) and ζ(3) fractions belong to | Exact integer recurrences, then the identity is confirmed to 50+ digits at a much deeper level; each formula's convergence speed and irrationality-measure estimate δ are measured | A cross-entropy-method proposal model is fitted to its hits; successful formulas seed local "family" searches; trivially equivalent rewrites are collapsed by canonicalization | CPU |
| **laws** | Rediscovers laws of physics from raw numbers (Newton, Kepler, relativity, optics, Bernoulli, ...), or finds formulas in your own CSV data | Out-of-distribution data **and** a symbolic proof that the found formula equals the true law | Free constants fitted by least squares inside the search, dimensional analysis, a learned grammar prior, and **library learning**: sub-expressions that keep recurring become new primitives (e.g. `sqrt(A0/A1)`), so it invents its own vocabulary | CPU |
| **selfplay** | A language model plays proposer and solver: invents problems at the edge of its ability and solves them. SymPy mode: integrals, equations, sums. Lean mode: machine-checked proofs of theorems, including conjectures it invented | SymPy-based exact checks, or the Lean 4 kernel with anti-cheating safeguards | **Expert iteration** (LoRA fine-tuning only on its own verified solutions), optionally followed by **GRPO** with the verifier as reward; the proposer is rewarded for *learnable* problems (4r(1−r)); Lean mode repairs failed proofs from Lean's error messages and reuses proved theorems as lemmas; a held-out benchmark measures real improvement | GPU |

All engines share: adaptive curricula that keep the work at the frontier of
ability, novelty and "interestingness" scoring, a SQLite knowledge base,
checkpoint/resume, and a compute layer that scales from one laptop to a cluster.

---

## Quick start (any computer, no GPU)

```bash
cd mathforge
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]" matplotlib
pytest -q                                              # ~85 tests, under a minute
mathforge run --config configs/laptop.yaml
```

Linux, macOS and Windows all work (the Windows specifics, such as hard time
limits without `SIGALRM`, are handled). Optional: `pip install -e ".[fast]"` adds
gmpy2, which makes the constants engine about 10% faster.

It now runs forever. In a second terminal:

```bash
mathforge status                       # what it knows, how it's learning
mathforge discoveries --top 20         # best results so far
mathforge verify 42                    # independently re-check result #42 (deeper, more digits)
mathforge dashboard                    # live charts in the browser: http://127.0.0.1:8765
```

Open `runs/laptop/discoveries.md` (refreshed every 5 rounds) and
`runs/laptop/learning_curves.png`. Press **Ctrl+C** to stop: it finishes the
current step and saves. Run the same command again and it resumes exactly where
it left off, with everything it has learned.

### What a first run looks like

A real 40-round run of `configs/laptop.yaml` on **2 CPU cores (12 minutes)**,
included in `examples/demo_run/` (its database too, so `mathforge discoveries
--run-dir examples/demo_run` works straight away):

**Laws: 19 of the 20 benchmark laws rediscovered from raw numbers**, each
checked on out-of-distribution data and proved equivalent to the true law (17
symbolically during the run; the Lorentz factor and relativistic energy passed
the numeric test at the time, and the final verifier proves them too, on the
physical domain v < c). The round of first success shows the learning:

| Round | Rediscovered |
|---|---|
| 1 | spring energy, kinetic energy, Newton's gravity, pendulum period, ideal gas |
| 2–6 | centripetal force, escape velocity, thin lens, Kepler's third law, Gaussian, Pythagoras, projectile range `v²·sin(2θ)/g` |
| 9–19 | 3-D distance, Lorentz factor, Doppler shift, Coulomb superposition, damped oscillation `e^(−t/2)·cos(ωt)` |
| 29 | relativistic energy `m·c²/√(1 − v²/c²)` |
| 38 | Planck-like spectrum `x³/(eˣ − 1)` |

Along the way it invented 21 primitives of its own, such as `sqrt(|A0/A1|)`
(abstracted from both escape velocity and the pendulum), `sqrt(A0² + A1²)`
(from Pythagoras) and `A1²/(A0² − A1²)` (from the Lorentz factor), which later
searches can use as single building blocks. Bernoulli pressure, the one law it
missed, needs a fitted constant inside the formula (9.81). That run predates
free-constant fitting; the current search finds it (`ρ·(v² + 981/50·h)/2`,
proved symbolically) in every benchmark run, see *Benchmark* below.

**Constants: 34 distinct continued-fraction formulas**, each confirmed to 50+
digits (14 for e, 9 for π, 6 for the golden ratio, 3 for ln 2, 2 for √2), with
trivial rewrites already collapsed by canonicalization. A few:

```
a(n) = −2n − 1,  b(n) = n²            ->  −4/π         (Brouncker's 1655 fraction, sign-flipped)
a(n) = −6n − 3,  b(n) = −n²           ->  −2/ln 2
a(n) = 3n + 3,   b(n) = −2n²          ->  1/(ln 4 − 1)
a(n) = 3n + 3,   b(n) = −2n² − n + 1  ->  1 + π/2
a(n) = −4n − 2,  b(n) = 4n² − 1       ->  −1 − √2
a(n) = n² + 4n + 3,  b(n) = −n³ − n²  ->  e
```

These are known families or close relatives of them, as you would expect on
2 cores. The search space and the open targets (ζ(5), γ) are where more compute
goes.

**The same 40 rounds with this session's code** (`examples/demo_run_v2/`, 6-core
laptop, 33 minutes): all 20 laws, the last in round 26, every one proved symbolically,
including Bernoulli pressure in round 7. The constants engine's structured family
found Apéry's 1978 fraction for ζ(3) by itself:

```
a(n) = 34n³ + 51n² + 27n + 5,  b(n) = −n⁶  ->  6/ζ(3)     3.06 digits per term, δ = 0.095 > 0
a(n) = 3n² + 3n + 1,  b(n) = −2n⁴           ->  1/(2·Catalan)
```

δ > 0 is the property Apéry's irrationality proof of ζ(3) rests on; the report's
*Equivalence classes* table lists the fastest formula and best δ per constant.

### Benchmark: how good is the laws search on its own?

`mathforge bench laws --seeds 10` searches every benchmark law from a cold start
(no library, no learned grammar) ten times with fixed seeds, and judges each
result with the same check the engine uses before storing anything. On a
6-core laptop, idle:

| search | laws solved | runs solved | CPU time |
|---|---:|---:|---:|
| fixed constants only (the original search) | 16/20 | 113/200 | 800 s |
| the same with 6× the generations (equal-compute control) | 16/20 | 130/200 | 4488 s |
| **fitted constants + dimensional analysis (default)** | **18/20** | **141/200** | **3488 s** |

The default beats the control while using less compute, and the gains are where
they should be: Bernoulli pressure 9/10 (never solved otherwise; it needs a fitted
9.81 inside the formula), the Lorentz factor 5/10 (2/10), Doppler 8/10 (5/10),
3-D distance 1/10 (0/10). Every accepted result is proved symbolically. One knob
was tuned on seeds 0–4 (the share of fitted constants among new leaves) and
confirmed on held-out seeds 5–9 (69/100 runs vs 65/100 for the control). Switch
the parts off with `--set engines.laws.fit_constants=false` or
`--set engines.laws.units=false`.

---

## Scaling up

### One GPU machine (local or cloud)

Turns on the neural self-play engine. Any NVIDIA GPU with 24 GB+ works for the
default 1.5B model (cloud: AWS `g6e`/`g5`, GCP `g2`, Lambda, RunPod, ...).

```bash
pip install -e ".[llm,vllm]" matplotlib
mathforge run --config configs/gpu.yaml
```

Every 5 rounds it fine-tunes a LoRA adapter on its verified solutions
(`runs/gpu/adapters/v1`, `v2`, ...), reloads, and continues. Watch
`eval_pass1` in `mathforge status` or the learning curves: that is accuracy on a
fixed held-out problem set, the honest measure of whether it is improving.

**A consumer GPU (6–8 GB) or Windows:** vLLM is Linux-only, so use the
transformers backend with `configs/gpu_small.yaml` (Qwen2.5-Math-1.5B, one
sequence per training step with gradient accumulation). Install a CUDA build of
torch first, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`,
then `pip install -e ".[llm]"`. GPUs without native bf16 (RTX 20xx, T4) run in
fp16 with loss scaling automatically. This path has been run on an RTX 2060.

**Reinforcement learning:** `--set engines.selfplay.trainer.algorithm=grpo` adds
GRPO on top of expert iteration. The first step is the usual supervised one; after
that, each step compares the attempts at the same prompt, uses the verifier's
verdict as the reward (and 4r(1−r) for the proposer's inventions), and pushes
failures down as well as successes up, with a KL penalty towards the base model.

### Many computers

Join any machines into one [Ray](https://www.ray.io) cluster (same Python
version and `pip install -e ".[cluster]"` on each):

```bash
ray start --head --port=6379                 # on the main machine
ray start --address=<main-machine-ip>:6379   # on every other machine
mathforge run --config configs/cluster.yaml  # on the main machine
```

CPU work (formula search, verification) spreads across every core of every
machine, the model gets one copy per GPU, and prompts are sharded across them.
Tasks lost when a machine drops out are re-run elsewhere (`compute.ray_max_retries`,
default 3), after which they are counted as errors and the round goes on; this is
tested on a simulated multi-node cluster (`MF_SLOW=1 pytest tests/test_ray_cluster.py`).

### Cloud cluster that scales itself (AWS)

`deploy/ray-cluster-aws.yaml` starts a head node plus CPU workers (spot,
cheap) and GPU workers on demand:

```bash
ray up deploy/ray-cluster-aws.yaml
ray exec deploy/ray-cluster-aws.yaml "cd mathforge && mathforge run --config configs/cluster.yaml" --tmux
ray rsync-down deploy/ray-cluster-aws.yaml mathforge/runs ./runs     # fetch results
ray down deploy/ray-cluster-aws.yaml                                 # stop paying
```

Fill in a Deep Learning AMI id for your region first, and check instance
prices: a GPU cluster left running costs real money.

### Keep it running 24/7

* `bash scripts/run_forever.sh configs/laptop.yaml` (Windows:
  `powershell -ExecutionPolicy Bypass -File scripts\run_forever.ps1 configs\laptop.yaml`):
  restarts after any crash.
* `deploy/mathforge.service`: systemd unit (auto-start on boot, restart on failure, graceful stop).
* `deploy/Dockerfile`: GPU image with everything installed.

Because state is checkpointed every round, crashes, reboots and spot-instance
interruptions cost at most one round.

---

## Theorem proving in Lean 4

```bash
bash scripts/setup_lean.sh            # installs Lean + Mathlib (~7 GB), runs a smoke test
                                      # Windows: powershell -ExecutionPolicy Bypass -File scripts\setup_lean.ps1
pip install -e ".[llm,vllm]"
mathforge run --config configs/lean.yaml
```

Starts from `mathforge/data/lean_starter.jsonl` (22 theorems from easy to
olympiad-flavoured) plus your own targets (`engines.selfplay.lean.theorem_file`):
a `.jsonl` of `{"name", "statement"}` or the miniF2F layout with
`formal_statement` and `split` (pick one with `lean.theorem_split: test`), a
`.lean` file, or a directory of `.lean` files such as PutnamBench's. The model
also invents conjectures; those that type-check and are not dispatched by a
one-line automatic tactic join the pool. A failed proof gets a second chance:
the model is shown Lean's error message and asked to fix it. Proved theorems
become a growing lemma library that later proofs may cite (Lean re-checks them
in every file that uses them).

A proof is accepted only when:
1. it proves **our** statement text (the model can never restate the theorem),
2. it contains no escape hatches (`sorry`, `admit`, `axiom`, `native_decide`,
   `#exit`, macros, `implemented_by`, ...); helper lemmas are allowed but cannot
   define notation, instances or definitions that would change what the statement means,
3. Lean compiles it, **and** `#print axioms` confirms it uses only the standard
   axioms. This positive-evidence check means a crash or an unterminated comment
   that silently skips the rest of a file can never be mistaken for success.

Statements are compiled with `autoImplicit` off, as in Mathlib: otherwise an
unknown name in a statement silently becomes a variable, and a conjecture with a
typo would mean something else than it says. The checks are tested against a
real Lean 4 + Mathlib install (`MF_SLOW=1 pytest tests/test_lean_real.py`).

---

## Bring your own mathematics

```bash
mathforge fit mydata.csv --target y          # find a formula for your own data
```

Free numeric constants are fitted by least squares (`--no-constants` turns that
off), so formulas like `y = 3.71·x·e^(−0.42·t)` are within reach.

* **Your data, continuously:** list CSVs under `engines.laws.datasets` and the laws engine hunts for formulas every round. Add `units: {column: "m/s", ...}` to a dataset to switch on dimensional analysis for it (SI units: kg, m, s, K, mol, A and N, J, W, Pa, Hz, C, V).
* **Your theorems:** point `engines.selfplay.lean.theorem_file` at them.
* **Your constants:** add a target to `target_values()` and `SYMBOLS` in `mathforge/engines/constants.py`.
* **Your own engine:** subclass `Engine` (`mathforge/engines/base.py`): implement `make_tasks`, a worker function that verifies what it claims, `ingest`, and `learn`; register it in `orchestrator.py`.

Any setting can be overridden from the command line:

```bash
mathforge run -c configs/laptop.yaml --set engines.constants.targets='[zeta5, euler_gamma]' --set engines.constants.depth=600
mathforge run -c configs/laptop.yaml --only laws        # run a single engine
```

---

## Using it for research

* **Learning curves:** `runs/<name>/metrics.csv` has every metric for every round; `learning_curves.png` plots the key ones; `mathforge dashboard` shows them live in a browser.
* **Experiment tracking:** `--set run.loggers='[wandb]'` (or `mlflow`) sends every round's metrics there, if the package is installed; a resumed run continues the same tracked run.
* **Reproducibility:** each run saves its exact `config.yaml`; seeds are fixed; `mathforge verify ID` re-checks any result independently; `mathforge export --out all.jsonl` dumps everything.
* **Benchmarks:** `mathforge bench laws --seeds 5` measures the raw laws search from a cold start on fixed seeds; run it before and after a change.
* **Ablations** show which ingredient matters. `mathforge ablate` runs every combination of values times seeds and writes a mean ± spread table (`ablation.md`, `ablation.csv`); an interrupted ablation resumes:
  ```bash
  mathforge ablate -c configs/laptop.yaml --rounds 10 --seeds 3 \
      --vary engines.laws.max_library=0,40 --vary engines.laws.units=true,false
  ```
  Useful switches: constants exploitation `engines.constants.explore_fraction=1.0`,
  the structured family `engines.constants.structured_fraction=0`, library learning
  `engines.laws.max_library=0`, fitted constants `engines.laws.fit_constants=false`,
  the proposer `engines.selfplay.proposer_fraction=0`, training
  `engines.selfplay.train_every=1000000`, RL `engines.selfplay.trainer.algorithm=grpo`.
* **Interesting directions:**
  * Fast-converging continued fractions for ζ(5), Catalan's constant or γ. Apéry's irrationality proof of ζ(3) rests on exactly such a fraction; none is known for ζ(5).
  * Families of formulas (vary one coefficient of a hit and watch the limit move): the Ramanujan Machine line of work found deep structure this way.
  * Lean self-play on competition benchmarks (miniF2F, PutnamBench) with larger provers.
  * Library learning across physical domains: do the invented primitives transfer?

---

## Honest expectations

* On a laptop the CPU engines really do rediscover classical results (Brouncker's
  4/π fraction, Euler's e fractions, Newton, Kepler) and turn up continued-fraction
  identities for π, e and other constants quickly. Most are known or equivalent to
  known ones; genuinely new mathematics is rarer and needs more compute,
  better targets, and human judgement about what matters.
* Constant formulas are **numerical conjectures verified to 50+ digits**, not
  proofs. That is the same status as the Ramanujan Machine's published conjectures,
  many of which were later proved.
* Beating professional mathematicians on hard open problems takes frontier-scale
  models and large compute. This project is the complete scaffold (verifiers,
  self-play, expert iteration, curricula, distribution), sized so you can start
  small and scale.
* What was tested where (details in `HANDOFF.md`, section 2):
  * **Tested for real:** both CPU engines on Linux and Windows, all verifiers,
    orchestration, checkpoint/resume, graceful stop, the CLI, benchmark, ablation
    runner and dashboard; a real 1.5B model on a 6 GB RTX 2060 with the transformers
    backend (generation, two expert-iteration steps, adapter reload, held-out eval,
    GPU memory released before training); GRPO training steps; the Lean verifier
    against a real Lean 4 + Mathlib install (which found two bugs a simulated
    compiler could not); and the Ray backend on a simulated multi-node cluster with
    a node killed mid-round.
  * **Not run yet:** vLLM (Linux-only; no Linux GPU machine was available), the
    7B Lean prover config, a real multi-machine Ray cluster, and the Docker,
    systemd and AWS files. Start any of these small (`--rounds 2`).

## Safety

* Model output is never executed. Answers are parsed with a whitelist-only
  parser (no Python `eval` exposure), every check runs under a timeout, and Lean
  proofs are screened for escape hatches before compilation.
* Cloud clusters cost money while they run. `ray down` when you're done.

---

## Commands

| Command | |
|---|---|
| `mathforge run -c CONFIG [--rounds N] [--only a,b] [--set k=v]` | start or resume |
| `mathforge status` | knowledge base size, latest metrics, curriculum levels, model version |
| `mathforge discoveries [--engine E] [--top N] [--order score\|new\|old]` | list verified results |
| `mathforge verify ID` | independently re-verify a result |
| `mathforge fit FILE --target COL [--units c=u,...]` | formula discovery on a CSV |
| `mathforge bench laws [--seeds N]` | cold-start benchmark of the laws search |
| `mathforge ablate -c CONFIG --vary k=a,b [--seeds N --rounds R]` | variants × seeds → results table |
| `mathforge dashboard` | live charts and latest results in the browser |
| `mathforge report` | rewrite discoveries.md / metrics.csv / learning_curves.png |
| `mathforge export [--out FILE]` | dump all results as JSON Lines |

## Project layout

```
HANDOFF.md               full technical handoff: design, status, traps, roadmap
CLAUDE.md                rules and commands for Claude Code sessions
mathforge/
  orchestrator.py        the forever loop: rounds, checkpoints, graceful stop
  distributed.py         serial / all-cores / Ray-cluster execution
  store.py               SQLite knowledge base, training traces, metrics, state
  curriculum.py          keeps every engine at the edge of its ability
  novelty.py             deduplication keys, interestingness score
  units.py               dimensional analysis for expression trees
  report.py              discoveries.md, metrics.csv, learning curves
  bench.py / ablate.py   cold-start benchmark; variants x seeds tables
  dashboard.py           live web dashboard
  tracking.py            optional W&B / MLflow logging
  engines/
    constants.py         continued fractions, MITM + PSLQ, Apery-like family, convergence profiles
    laws.py              symbolic regression, fitted constants, units, verification, library learning
    selfplay.py          proposer/solver self-play, expert iteration + GRPO, Lean repair and lemmas
  verify/
    safe_parse.py        maths parsing that never executes model output
    sympy_tasks.py       problem generators, proposal validators, exact checkers, time limits
    lean.py              Lean 4 verifier with positive-evidence checking, theorem loaders
  models/
    policy.py            vLLM / transformers policies, one per GPU across a cluster
    trainer.py           LoRA expert iteration and GRPO
configs/                 laptop, gpu_small, gpu, lean, cluster
deploy/                  Dockerfile, systemd unit, AWS Ray cluster
scripts/                 setup_lean.sh/.ps1, run_forever.sh/.ps1
examples/demo_run*/      outputs and databases of two real 40-round runs
tests/                   fast suite incl. end-to-end loops; MF_SLOW=1 adds real-model,
                         real-Lean and multi-node Ray tests
```

## Background reading

The ideas combined here come from:

* Raayoni et al., *Generating conjectures on fundamental constants with the Ramanujan Machine*, Nature (2021)
* Ferguson, Bailey & Arno, *Analysis of PSLQ, an integer relation finding algorithm*, Math. Comp. (1999)
* Apéry, *Irrationalité de ζ(2) et ζ(3)*, Astérisque (1979)
* Silver et al., *Mastering the game of Go without human knowledge*, Nature (2017): self-play
* Udrescu & Tegmark, *AI Feynman: a physics-inspired method for symbolic regression*, Science Advances (2020)
* Cranmer, *Interpretable machine learning for science with PySR and SymbolicRegression.jl* (2023)
* Ellis et al., *DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning*, PLDI (2021)
* Zelikman et al., *STaR: bootstrapping reasoning with reasoning*, NeurIPS (2022)
* Gulcehre et al., *Reinforced Self-Training (ReST) for language modeling* (2023)
* Polu et al., *Formal mathematics statement curriculum learning* (2022): expert iteration in a proof assistant
* Romera-Paredes et al., *Mathematical discoveries from program search with large language models* (FunSearch), Nature (2023)
* Zhao et al., *Absolute Zero: reinforced self-play reasoning with zero data* (2025): learnability-rewarded proposers
* Ren et al., *DeepSeek-Prover-V2* (2025)

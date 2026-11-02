# MathForge: notes for Claude Code

A self-improving mathematical discovery system: propose → verify → store → learn, forever.
**Read `HANDOFF.md` before any non-trivial change.** It has the architecture, what has
and hasn't been tested, the traps already hit, and the prioritised roadmap.

## Commands

- Install: `pip install -e ".[dev]" matplotlib` (GPU work: add `.[llm]`, plus `vllm` on Linux; clusters: `.[cluster]`; `.[fast]` = gmpy2)
- Tests: `pytest -q` (under a minute). Slow tests, behind `MF_SLOW=1`:
  `tests/test_model_path.py` (real transformers model, LoRA + GRPO training),
  `tests/test_lean_real.py` (real Lean 4 + Mathlib; needs `scripts/setup_lean.*`),
  `tests/test_ray_cluster.py` (simulated multi-node Ray, a node killed mid-round)
- Smoke run: `mathforge run -c configs/laptop.yaml --rounds 2 --set run.dir=runs/smoke`
- Inspect a run: `mathforge status|discoveries|report|dashboard --run-dir runs/<name>`, `mathforge verify <id> --run-dir runs/<name>`
- Measure a search change: `mathforge bench laws --seeds 5` before and after (same seeds, cold start);
  whole-system comparisons: `mathforge ablate --vary key=a,b --seeds 3 --rounds 10`
- Windows: use `.venv\Scripts\...`; set env vars with `$env:MF_SLOW = "1"` in PowerShell

## Rules

1. Nothing enters the knowledge base or the training data without passing a verifier.
   Never loosen a verifier to make a test pass. When a check rejects something, first
   assume the candidate is wrong (it usually is). RL rewards come only from our verifiers.
2. Never execute model output. Parse maths only through `mathforge/verify/safe_parse.py`;
   send Lean only through `mathforge/verify/lean.py`.
3. Worker functions must be top-level and picklable (they run in other processes or on
   other machines). Failures come back as `{"error": ...}`; one task must never crash the loop.
4. Engine state must round-trip as JSON through `state_dict()` / `load_state_dict()`.
   Every new field goes in both, and loading an older checkpoint must still work.
5. Every new config key gets a default in `mathforge/config.py`; update `configs/*.yaml`
   when users need to see it.
6. Every change comes with tests; keep `pytest -q` green and fast (slow tests go behind `MF_SLOW=1`).
7. `MockPolicy` exists only for tests. Never use it to produce results.
8. Don't edit engine code while a run is in progress: pool workers are recycled and would
   load the new code mid-run. (Benchmarks/demo runs: run them from a copy via PYTHONPATH.)
9. On new hardware, do a 2-round smoke run before any long run.
10. Fitted constants make "numerically exact" cheap: a formula with fitted constants is
    accepted only after junk removal and snapping (`laws.check_claim`), never on raw floats.
11. Test fakes (fake `lake`, `MockPolicy`) accept too much: a verifier change that matters
    must also be checked against the real thing (`test_lean_real.py`, `test_model_path.py`).

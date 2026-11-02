# Example run

A real 40-round run of `configs/laptop.yaml` on 2 CPU cores (12 minutes, 76 verified
results, no errors), made with the first session's code (before fitted constants, units
and the structured constants family; `../demo_run_v2/` is the same run with today's code).
See the "What a first run looks like" section of the main README and section 2 of
`HANDOFF.md` for a summary.

| File | What it is |
|---|---|
| `discoveries.md` | the report: top results per engine and the learned library |
| `learning_curves.png` | formulas found, laws solved and library size by round |
| `metrics.csv` | every metric for every round |
| `mathforge.log` | the run's log |
| `config.yaml` | the exact configuration used |
| `mathforge.db` | the knowledge base itself |

Because the database is included, the CLI works on it directly:

```bash
mathforge status      --run-dir examples/demo_run
mathforge discoveries --run-dir examples/demo_run --engine laws
mathforge verify 3    --run-dir examples/demo_run
```

Two labels in the database say `numeric-ood` (Lorentz factor, relativistic energy): that is
what the verifier could do during the run. The final verifier proves both on the physical
domain, as `mathforge verify` shows.

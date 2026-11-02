# Example run (second session's code)

A real 40-round run of `configs/laptop.yaml` with 10 worker processes on a 6-core Windows
laptop (33 minutes while a GPU job shared the machine; 72 verified results, no errors),
made with the code after roadmap items R1 to R9. Compare with `../demo_run/`, the same
40 rounds with the first session's code on 2 cores.

| | first session (`demo_run`) | this run |
|---|---|---|
| benchmark laws rediscovered | 19 of 20, the last in round 38 | **20 of 20, the last in round 26** |
| Bernoulli pressure (needs a fitted 9.81) | never | round 7: `rho*(981*h + 50*v**2)/50` |
| relativistic energy, Planck-like spectrum | rounds 29 and 38 | rounds 11 and 26 |
| proofs | 17 symbolic, 2 numeric-only | **all 20 symbolic** (2 of them on the physical domain v < c) |
| continued-fraction formulas | 34 (e, π, φ, ln 2, √2) | 32 (π, e, φ, ln 2, √2, Catalan, ζ(3)) |

The constants engine's structured "Apéry-like" family found, by itself, **Apéry's 1978
fraction for ζ(3)**, `a(n) = 34n³ + 51n² + 27n + 5, b(n) = −n⁶ → 6/ζ(3)`, gaining 3.06
digits per term with an irrationality-measure estimate δ = 0.095 > 0 (the property Apéry's
irrationality proof rests on), and a known fraction for Catalan's constant,
`a(n) = 3n² + 3n + 1, b(n) = −2n⁴ → 1/(2G)`, in round 1. The *Equivalence classes* table in
`discoveries.md` lists the fastest formula per constant. None of this is new mathematics;
it shows the search reaching the right structures on its own.

| File | What it is |
|---|---|
| `discoveries.md` | the report: top results per engine, equivalence classes, the learned library |
| `learning_curves.png` | formulas found, laws solved and library size by round |
| `metrics.csv` | every metric for every round |
| `mathforge.log` | the run's log |
| `config.yaml` | the exact configuration used |
| `mathforge.db` | the knowledge base itself |

```bash
mathforge status      --run-dir examples/demo_run_v2
mathforge discoveries --run-dir examples/demo_run_v2 --engine constants
mathforge dashboard   --run-dir examples/demo_run_v2
```

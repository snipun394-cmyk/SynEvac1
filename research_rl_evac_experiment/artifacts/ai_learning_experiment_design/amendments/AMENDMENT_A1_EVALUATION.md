# Amendment A1: evaluation record

**Result: DATASET ACCEPTED UNDER AMENDMENT A1.** This was evaluated once, exactly as written in `amendment_A1_acceptance_criteria.json`. The script is `scripts/evaluate_amendment_A1.py` and the full numbers are in `amendment_A1_evaluation.json`.

## Provenance

- **The original Spec v2 status stays FULL DATASET NO-GO.** The original G1, X5 and X7 results stay FAIL. Both are permanent.
- **A1 is a post-hoc acceptance-criteria amendment.** It was created after the frozen dataset had been inspected, and before any model training. It does not retroactively change the original result.
- **Nothing changed during evaluation.** The dataset, Spec v2 files and A1 record were SHA-256 checked before and after; all are byte-identical.
- **No model results informed the execution.** No model exists yet.
- **X7 multiplicity.** The two X7-A1 endpoints (layout × timing and layout × compliance) are treated as separately prespecified acceptance tests, each at α = 0.05, with no family-wise-error correction.

## Results

| Test | Statistic | p-value | Result |
|---|---:|---:|---|
| G1-A1 | mean distinct share at ticks ≥ 10: 0.5344 (5–95%: 0.4953–0.5714) vs pilot 0.5158 | 0.7831 | **PASS** |
| X5-A1 | mean overall distinct share: 0.2385 (5–95%: 0.2224–0.2540) vs pilot 0.2299 | 0.8066 | **PASS** |
| X7-A1 timing | T = Σ within-role χ² = 0.0 | 1.0000 | **PASS** |
| X7-A1 compliance | T = Σ within-role χ² = 0.0 | 1.0000 | **PASS** |

The p-values are one-sided: for G1-A1 and X5-A1, p = (1 + #{resampled ≥ pilot}) / 2001. Monte Carlo SE of p: 0.0092 (G1-A1) and 0.0088 (X5-A1).

**Retained clauses:**

| Criterion | Clause | Measured | Result |
|---|---|---|---|
| G1-A1 | ≥ 10 distinct tick ≥ 10 inputs per layout | minimum 118 | pass |
| X5-A1 | ≥ 110 distinct inputs | 1,389 | pass |
| X5-A1 | ≥ 1,400 distinct (input, candidate) pairs | 413,922 | pass |
| X7-A1 | seeds unique | 5,544 of 5,544 | pass |
| X7-A1 | perturbation rate within 0.5 ± 0.1 | 0.5145 | pass |
| X7-A1 | no censored T100 | 0 censored rows | pass |

Within-role χ² and Cramér's V are 0 in all 7 split roles, for both tests (reported, not gated).

## Execution

- **G1-A1 and X5-A1:** 2,000 resamples, seed 5544240. Each resample takes 240 whole trajectories, 2 per pilot cell. Main layouts draw from pools of 20 TRAIN trajectories per cell; zero-shot layouts from pools of 3.
- **X7-A1:** 10,000 permutations, seed 5544007, with layout shuffled within each of the 7 split roles.
- **Invariant failures:** none.
- **Implementation notes (disclosed):**
  - The permutation comparison T_b ≥ T_obs uses a floating-point tolerance of 1e-9.
  - The perturbation rate is computed as in the original full audit.

## Status

- **The acceptance criteria are all met under A1.** G1-A1, X5-A1 and X7-A1 pass. The 14 unchanged criteria (G2–G7, X1–X4, X6, X8–X10) keep their original PASS results.
- **The original Spec v2 NO-GO remains on record.**
- **No training has started, and no selector has been built.**
- **Preserved limitations:**
  - A: 60% of decision states are fixed by the occupant layout (20 distinct inputs across 11,088 states).
  - B: G3 has never been tested beyond 4 non-empty zones.

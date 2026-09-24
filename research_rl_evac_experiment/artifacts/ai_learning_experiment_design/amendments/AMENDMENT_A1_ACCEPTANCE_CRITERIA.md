# Amendment A1 to Spec v2: acceptance criteria G1, X5, X7

**Status: PROPOSED, awaiting approval. Not evaluated.** Machine-readable record: `amendment_A1_acceptance_criteria.json`.

## Provenance

| Item | Value |
|---|---|
| Amendment | A1 |
| Created (UTC) | 2026-09-24T18:27:20Z |
| Parent | Spec v2; `spec_version_record.json` SHA-256 `b0e761df...97c1e2`; `diversity_requirements.json` `a0bd5786...2d9052` |
| Frozen dataset | `full_ai_learning_rows.csv.gz` SHA-256 `36aa757f...967179` |
| Original result | `full_go_no_go.json` SHA-256 `b16d6b96...349998` (git `bdfb54e`) |

- Written **before any model training**.
- **No dataset row** was changed or regenerated.
- **No model result** exists, so none informed this amendment.
- **Not blind to the data.** You and I had already seen the exploratory results: the G1/X5 pilot-shaped resampling (seed 20260924, 300 draws) and the per-role X7 association values (0 in every role). Each rule below is derived from the criterion's stated purpose and the project's existing α = 0.05 / 95% convention, not from those numbers. The seed is new and the resample count differs from the exploratory run.

## Original results (preserved, permanent)

| Criterion | Original text | Measured | Result |
|---|---|---|---|
| G1 | ticks ≥ 10: distinct inputs ≥ 25% of those states AND ≥ 10 distinct per template | share 0.1838; min per template 118 | **FAIL** |
| X5 | distinct inputs ≥ 110 AND distinct share ≥ 0.1222 AND distinct (input, candidate) pairs ≥ 1400 | 1,389; 0.0749; 413,922 | **FAIL** |
| X7 | every template has every (t_h, p) level (association 0); seeds unique; perturbation 0.5 ± 0.1; no censored T100 | V = 0.0399 (t_h), 0.0477 (p); seeds unique; 0.5145; 0 censored | **FAIL** |
| Overall | Spec v2 | | **FULL DATASET NO-GO** |

## Why the original clauses are invalid

**G1 and X5 share clauses.** The distinct share can only fall as the sample grows.
- For a finite reachable state space, E[D(n)] = Σᵢ (1 − (1 − pᵢ)ⁿ).
- The chance that the next draw is new, πₙ = Σᵢ pᵢ(1 − pᵢ)ⁿ, is non-increasing, and E[D(n)] ≥ n·πₙ.
- So E[D(n+1)]/(n+1) ≤ E[D(n)]/n, and the share → 0.
- A fixed share threshold therefore fails at a large enough n regardless of data quality.
- Observed in the frozen data (tick ≥ 10): 0.746 at ~200 states, 0.542 at ~800, 0.286 at ~6,400, 0.184 at 18,538.
- X5 also compares a share at n = 18,538 with 5× a share measured at n = 450.

**X7 association clause.** It measures the sampling design, not a confound.
- The all-trajectory Cramér's V is fixed by the allocation table alone. Computed from `dataset_design.json` before any data existed, it gives exactly 0.0399 / 0.0477.
- It is 0 within every split role, within the six main layouts, and within the two zero-shot layouts.
- The non-zero value exists only because zero-shot layouts get a flat 3 trajectories per cell while main layouts are role-weighted.
- V is also biased upward: random layout assignment with the same margins gives about 0.035–0.038, never 0.

## Amended criteria

### G1-A1: pilot-matched diversity (ticks ≥ 10)

**Purpose:** the full dataset must contain at least pilot-level observable-state diversity when evaluated at the pilot's sampling scale.

- **Reference:** pilot share at ticks ≥ 10 = **0.5158** (163 / 316).
- **Sampling unit:** whole trajectories; rows are never sampled individually.
- **Stratification:** by the 120 pilot cells: 8 layouts × t_h {none, 3, 5, 7, 9} × p {0.5, 0.7, 0.9}.
- **Draw:** exactly **2 distinct trajectories per cell** without replacement, giving 240 trajectories per resample (the pilot's structure).
  - Main layouts draw from TRAIN-role trajectories of that cell (20 available).
  - Zero-shot layouts draw from DIAG_TEMPLATE_ZERO_SHOT trajectories of that cell (3 available).
- **Resamples:** **2,000**, using `numpy.random.default_rng(5544240)`.
  - Cells are visited in the fixed order layout → t_h → p.
  - Each cell's pool is sorted by `trajectory_id`, then `rng.choice(len(pool), 2, replace=False)`.
  - X5-A1 uses the same 2,000 resamples.
- **Statistic:** sᵣ = distinct 26-feature vectors among the resample's tick ≥ 10 states ÷ number of those states.
- **PASS** when **both** clauses hold:
  1. One-sided p = (1 + #{sᵣ ≥ 0.5158}) / 2001 is ≥ 0.05, meaning the pilot is not significantly more diverse than the full dataset at the pilot's scale.
  2. Retained unchanged: minimum distinct tick ≥ 10 inputs per layout in the full dataset ≥ 10.
- **Confidence procedure:** percentile method. Report the mean, median, 2.5/5/95/97.5 percentiles, and the Monte Carlo SE of p.
- **Rationale:** the pilot value is a single noisy realization, so the question is one-sided non-inferiority at the project's α = 0.05. Two alternatives were rejected:
  - "Resampled median ≥ pilot" treats one realization as exact, and would fail about half the time for two equally diverse processes.
  - A two-sided equivalence test would penalize *more* diversity.

### X5-A1: pilot-matched diversity (all ticks) plus retained absolute clauses

- **Reference:** pilot overall share = **0.2299** (183 / 796).
- **Resampling:** identical to G1-A1 (same 2,000 resamples, seed and draw order).
- **Statistic:** uᵣ = distinct vectors among all of the resample's decision states ÷ number of those states.
- **PASS** when **all three** clauses hold:
  1. p = (1 + #{uᵣ ≥ 0.2299}) / 2001 is ≥ 0.05.
  2. Retained unchanged: distinct inputs in the full dataset ≥ 110.
  3. Retained unchanged: distinct (input, candidate) pairs ≥ 1400.
- **Removed:** the distinct share ≥ 0.1222 clause.
- **Confidence procedure:** as in G1-A1.
- **Rationale:** the same scale argument as G1-A1. The original "materially exceeds the old dataset" purpose stays with the two unchanged absolute 10× clauses.

### X7-A1: stratified independence test

**Purpose:** within the allocation structure, layout must not be systematically associated with hazard timing or compliance probability.

- **Unit:** one record per trajectory present in the frozen dataset. Layout, role, t_h and p are read from the dataset rows' metadata, not from the design file.
- **Strata:** the 7 split roles.
  - Main layouts appear only in the first six roles; zero-shot layouts only in DIAG_TEMPLATE_ZERO_SHOT. So the two groups are never compared with each other.
  - Their different allocation is design-intended and is reported descriptively, never gated.
- **Tests:** layout × t_h, and layout × p.
- **Statistic:** T = Σₛ X²ₛ, the sum of within-stratum Pearson chi-squares over the levels present in each stratum. A stratum with a single layout or level contributes 0.
- **Null distribution:** layouts are exchangeable within each stratum.
  - Layout labels are permuted independently within each stratum, preserving all within-stratum margins.
  - **B = 10,000** permutations, `numpy.random.default_rng(5544007)`, strata in the listed order, labels ordered by `trajectory_id`.
  - The same permutations are used for both tests.
- **p-value:** p = (1 + #{T_b ≥ T_obs}) / 10001.
- **PASS** when **all four** clauses hold:
  1. p ≥ 0.05 for layout × t_h **and** p ≥ 0.05 for layout × p. There is no multiplicity correction, because correcting would make acceptance easier.
  2. Retained unchanged: seeds unique per trajectory.
  3. Retained unchanged: realized perturbation rate within 0.5 ± 0.1.
  4. Retained unchanged: no censored T100.
- **Reported, not gated:** per-stratum X², p and V; pooled T for the main-layout strata and for the zero-shot stratum; the main vs zero-shot marginals.
- **Rationale:** this is a Cochran–Mantel–Haenszel-style stratified independence test with an exact permutation null.
  - It asks the intended question, and it avoids the upward bias of V and any impossible "exactly 0" requirement.
  - A balanced realized allocation gives T_obs = 0 and p = 1.
  - It fails only if the realized dataset shows layout-dependent timing or compliance inside a role, for example from missing, duplicated or mislabeled trajectories.

## Unchanged

G2, G3, G4, G5, G6, G7, X1, X2, X3, X4, X6, X8, X9 and X10 keep their definitions and their recorded results (all PASS).

Acceptance under A1 = those results AND G1-A1 AND X5-A1 AND X7-A1, evaluated once, after approval. The original Spec v2 NO-GO stays on record.

## Preserved limitations (not weakened)

- **A.** 60% of decision states occur at ticks 0 and 5 and are fixed by the occupant layout: **20 distinct inputs across 11,088 states**. Outcomes there are learnable only as conditional distributions.
- **B.** G3 candidate sufficiency has **never been tested on states with more than 4 non-empty eligible zones**.

## Not done

No amended criterion evaluated. No model trained. No selector. No dataset change. No Spec v2 file changed.

# Selector Repair Diagnostic: R1 and R6 (VALIDATION only, offline)

- **Numbers:** `selector_r1_r6_diagnostic.json` (same folder)
- **Script:** `scripts/diagnose_selector_r1_r6.py`
- **Follows:** `SELECTOR_SPECIFICATION_AUDIT.md` (commit 6a3453d)

**Status:**
- This is a diagnostic, not a selector and not an amendment.
- The frozen selector code, A2, C1, C2, C3, models, quantiles, tolerances and thresholds are all unchanged.
- PRIMARY TEST and every other held-out set remain sealed.

**Method:**
- R1 and R6 were simulated offline on the same hash-verified C3 predictions for the 1,845 VALIDATION states.
- Harm and capture use the frozen A2 definitions.
- Confidence intervals use the frozen trajectory-clustered bootstrap (`cluster_boot`, 2,000 resamples) with the frozen seeds: 5544102 capture, 5544103 harm, 5544105 descriptive deltas.
- Variant A reproduces the frozen result exactly (0 switches). The script asserts this.

## Headline

**R1 fixes the self-incompatibility, but it does not restore a defensible decision mechanism.**
- With R1 the selector switches in 540 states and captures 102 of 753 improvable states (13.5%).
- 198 of those switches are harmful: 10.7% of all states (95% CI 9.5 to 12.0%), and 36.7% of switches.
- The frozen safety endpoint requires the upper 95% CI of the harm rate to be at most 1%, so R1 misses it by about ten times on VALIDATION.
- The R1 floor does not reject a single one of the switches the selector actually makes. The selector under R1 behaves exactly as if there were no floor.

## Step 1. R1 formalized

**R1 safety floor (replaces step 2):** keep candidate `a` iff `q9_a,k ≤ q9_d,k + δ_k` for every criterion k in K. `d` is kept.

**Nothing else changes:**
- the step 1 T90 skip;
- the step-3 tolerances;
- the step-4 switch rule;
- OOD τ = 0.9354 and the width threshold 1.7897;
- δ = (0.5, 0.5, 4.7079, 4.7651).

**Notation:**
- `I_k = m_d,k − m_a,k` is the predicted improvement.
- `u_a,k = q9_a,k − m_a,k` and `u_d,k = q9_d,k − m_d,k` are the upper widths.
- `Δu_k = u_a,k − u_d,k` is the width difference.

**The floor rewrites as `I_k ≥ Δu_k − δ_k`.** Under the frozen floor it was `I_k ≥ u_a,k − δ_k`.

**Self-incompatibility is removed.** The default gives `Δu = 0` and `I = 0`, so it passes (`0 ≥ −δ_k`) in every state by construction; this was verified in all 1,845 states. Any candidate predicted identical to the default also passes.

**R1 feasibility of a switch on criterion k\*:**

| Criteria | Condition |
|---|---|
| Before k* | `Δu_k − δ_k ≤ I_k ≤ δ_k`, possible iff `Δu_k ≤ 2δ_k` (a width *difference*, no longer an absolute width) |
| k* | `I_k* > max(δ_k*, Δu_k* − δ_k*)` |
| After k* | `I_k ≥ Δu_k − δ_k` |
| Abstention | not OOD; `u_a*,T100 ≤ 1.7897` |

**Empirically:**

| | T90 | T100 | U | H |
|---|---|---|---|---|
| Median `Δu` | −0.04 | −0.01 | 0.01 | 0.01 |
| Pairs with `Δu ≤ 2δ` | 99.7% | 100% | 99.2% | 89.7% |
| Pairs where the floor demands a predicted improvement, R1 | 1.0% | 4.9% | 11.1% | 15.8% |
| Same, frozen floor | 76.5% | 74.0% | 62.7% | 60.8% |

## Step 2. Attrition, R1 applied offline

Counts are over 547,965 non-default candidate/state pairs.

| Stage | A (frozen) | B (R1) |
|---|---|---|
| Pass floor: pairs / states | 43,609 / 240 | 314,765 / 1,845 |
| Survive step 3: non-default pairs / states | 203 / 203 | 1,855 / 1,845 |
| `a*` ≠ default (states) | 203 | 1,845 |
| … of which no criterion differs by more than δ | 203 | 1,305 |
| Pass switch rule (states) | 0 | 540 |
| OOD abstentions | 0 | 0 |
| Width abstentions | 0 | 0 |
| **Final switches** | **0** | **540** |

All 540 R1 switches are decided on hazard-edge unmet. On T90, T100 and U the switched candidate is predicted to be within δ of the default.

## Step 3. Safety of the 540 R1 switches

**Headline counts:**
- **Switches:** 540.
- **Harmful switches:** 198 (harm rate among switches 36.7%, CI 32.6 to 40.6%).
- **Harmful states:** 10.7% of all states (CI 9.5 to 12.0%).
- **Frozen hazard sub-criterion:** met; no switch raises hazard-edge unmet at all (maximum increase 0).

**Which criterion makes a switch harmful** (the first criterion on which the switch is worse than the default):

| Criterion | T90 | T100 | Unmet | Hazard |
|---|---|---|---|---|
| Harmful switches | 126 | 39 | 33 | 0 |

**Paired outcome deltas among switches** (selected minus default; negative = better):

| Target | Mean | 95% CI | Better / equal / worse |
|---|---|---|---|
| T90 (ticks) | −0.50 | −0.59 to −0.40 | 250 / 164 / 126 |
| T100 (ticks) | −0.34 | −0.41 to −0.26 | 278 / 153 / 109 |
| Unmet | −1.66 | −2.37 to −0.90 | 295 / 12 / 233 |
| Hazard-edge unmet | −13.22 | −13.91 to −12.49 | 430 / 110 / 0 |

**Predicted versus actual** (medians over the 540 switches):

| Target | Predicted mean, candidate | Predicted mean, default | Predicted q0.9, candidate | Actual, candidate | Actual, default | q0.9 covers actual |
|---|---|---|---|---|---|---|
| T90 | 12.57 | 12.33 | 14.00 | 12 | 13 | 91% |
| T100 | 13.32 | 13.30 | 14.97 | 13 | 14 | 95% |
| Unmet | 69.1 | 65.7 | 90.9 | 65 | 74.5 | 91% |
| Hazard | 2.9 | 10.8 | 16.2 | 1 | 20 | 95% |

**Reading:**
- On average the switches improve all four outcomes, and hazard strongly.
- Harm under the frozen lexicographic definition comes from T90 first. The model treats T90 differences below 0.5 tick as ties (its T90 MAE is 0.57), but the realized T90 is an integer and is 1 tick or more worse in 126 switches.
- The harm is therefore created by step 4 treating earlier criteria as ties within δ, not by the floor.

## Step 4. R6 (step-3 consistency), evaluated separately

**R6 as simulated:**
- Step 3 keeps candidates with `m_H ≤ min H + δ_H`.
- `a*` = default if the default survives; otherwise the candidate with minimal H mean, ties to the lowest id.
- The tie-break is a diagnostic choice, because the request did not specify a non-default tie-break.

**Results:**
- **Without R1 (variant C):**
  - 43,609 non-default pairs survive step 3 (up from 203) in 240 states;
  - the default wins every one of those states, so the 203 winners that could never switch disappear;
  - 0 switches, 0 harm.
  - R6 alone changes no recommendation.
- **With R1 (variant D):**
  - 245,783 non-default pairs survive step 3;
  - `a*` ≠ default in 540 states, all of which switch;
  - the aggregate results are identical to B: the same counts, capture, harm and deltas.
- **Tie-break sensitivity:** with lowest id inside the band instead of minimal H, D gives 180 switches, 47 harmful and 39 captured. C is unchanged. The R6 tie-break is therefore not neutral when combined with R1 and would have to be specified in any amendment.

## Step 5. Crossed diagnostics (none is an official selector)

| | A frozen | B R1 | C R6 | D R1 + R6 |
|---|---|---|---|---|
| Non-default recommendations (= switches) | 0 | 540 | 0 | 540 |
| Capture, improvable (of 753) | 0 (0%) | 102 (13.5%, CI 11.3 to 15.8%) | 0 | 102 (13.5%, CI 11.3 to 15.8%) |
| Harmful states | 0 | 198 | 0 | 198 |
| Harm rate, all states (CI) | 0 (0 to 0) | 10.7% (9.5 to 12.0%) | 0 | 10.7% (9.5 to 12.0%) |
| Frozen safety endpoint (≤ 1% upper CI) | met | **not met** | met | **not met** |
| Paired Δ T90 / T100, all states | 0 / 0 | −0.146 / −0.099 | 0 / 0 | −0.146 / −0.099 |
| Paired Δ unmet / hazard, all states | 0 / 0 | −0.49 / −3.87 | 0 / 0 | −0.49 / −3.87 |
| Abstention rate | 0 | 0 | 0 | 0 |
| OOD rate | 0.22% (4 states) | 0.22% | 0.22% | 0.22% |
| Agreement with default | 100% | 70.7% | 100% | 70.7% |

Paired Δ is selected minus default; negative is better. All-state CIs are in the JSON; every non-zero delta excludes 0.

## Step 6. Is the R1 floor informative?

**On the switches the selector would actually make** (the 540 no-floor switches):

| | Frozen floor | R1 |
|---|---|---|
| Harmful switches rejected | 100% | **0%** |
| Non-harmful switches rejected | 100% | **0%** |
| AUC, floor margin vs harm | 0.52 | 0.48 |

**Over all 547,965 non-default candidate/state pairs:**

| | Frozen floor | R1 |
|---|---|---|
| Harmful pairs rejected | 100% | 82.9% |
| Non-harmful pairs rejected | 85.9% | 11.6% |
| AUC, floor margin vs harm | 0.82 | 0.88 |
| Passing pairs: better / equal / worse than default | 0 / 43,609 / 0 | 47,630 / 226,463 / 40,672 |

**Interpretation:**
- **The frozen floor is safe but inert.** Every pair it admits has a realized outcome exactly equal to the default's, so it can never admit an improvement. This sharpens the earlier audit: its safety is bought by admitting nothing that changes the outcome.
- **R1 is a real screen over the candidate set.** It removes 83% of harmful pairs while keeping 88% of non-harmful ones, and AUC rises to 0.88.
- **But R1 does not bind at the decision point.** Step 3 already prefers low predicted means, and with similar widths those candidates pass R1 automatically. The residual harm comes from within-δ ties on T90/T100/U that R1 cannot see, because on those criteria the candidate's q0.9 sits within δ of the default's.

**Answer to the key question:** no. On VALIDATION, R1 does not preserve a meaningful safety screen while letting the predictor act. It lets the predictor act exactly as if there were no floor, and the result misses the frozen harm endpoint by about ten times. R1 on its own is not supported by this evidence.

## Step 7. Proposed amendment specification (NOT created, NOT adopted)

**Type:** post-hoc VALIDATION-motivated amendment. It was motivated by VALIDATION results (audit 6a3453d and this diagnostic), and must be recorded as a separate file linked by hash to A2, C1, C2 and C3, none of which is modified.

**Precondition for adoption:** on the evidence above, R1 as specified fails the frozen safety endpoint on VALIDATION. Adopting it unchanged would mean knowingly entering PRIMARY TEST with a selector that failed safety on VALIDATION. Any amendment would need either an additional change that addresses the within-δ tie harm, or an explicit decision to accept that risk. This diagnostic does not choose either.

| Item | Proposal |
|---|---|
| Exact rule change | Step 2: `keep a iff q9_a,k ≤ q9_d,k + δ_k for all k in K; d kept`. If R6 is included as a separate numbered item: step 3 final H filter `m_a,H ≤ min + δ_H`; `a* = d` if `d` survives, else minimal H mean, ties to the lowest id (the tie-break must be stated, see Step 4). |
| Reason | The frozen floor compares a candidate's q0.9 with the default's mean. The default fails its own floor in 64 to 85% of states, and the floor admits only outcome-identical candidates. |
| Validation evidence required | (1) This diagnostic, disclosed in full, including the failed harm endpoint. (2) The same diagnostic for the exact-lookup baseline under the same change (not yet produced). (3) Because the rule was chosen after seeing VALIDATION, a confirmatory run on a fresh VALIDATION-2 batch before any PRIMARY TEST access, with the frozen harm endpoint as a gate. |
| Affected tolerances | None. δ_T90 = δ_T100 = 0.5 and δ_U, δ_H come from the C2 MAE rule, which does not depend on the selector. |
| OOD / width thresholds | No re-derivation needed. τ is state-space only. The width threshold is the 95th percentile of `q9 − m` for T100 over all VALIDATION rows, independent of which candidate wins. |
| Exact-lookup baseline | Must receive the identical change (A2 Issue 2 fairness), evaluated and reported alongside. |
| P2/P5 reference | Stays valid. Per its record it is model-free and computed from labels, it does not use the step-2 floor, and the dataset is unchanged. |
| What remains frozen | Dataset (SHA-256 `36aa757f…`), splits, candidate set, targets, features, C3 models and quantile handling, δ values, τ, width threshold, step 1, step-3 T90/T100/U tolerances, step 4, step 5, harm and capture definitions, success criteria, bootstrap method, PRIMARY TEST seeds and bootstrap seeds 5544101 to 5544105. |
| New validation seeds (proposed) | VALIDATION-2: 540 trajectories, one seed each, 170000 to 170539 (range 170000 to 170999 is unused in every artifact and script), same VALIDATION cells and generator as Spec v2. |
| New bootstrap seeds (proposed) | For VALIDATION-2 only: 5544301 recommendation, 5544302 safety, 5544303 prediction, 5544304 P4 usefulness, 5544305 descriptive Δ. No `55443xx` seed is currently used. |

## Integrity

- Selector code unchanged; no amendment written.
- No retraining, no recalibration.
- All 33 protected file hashes were identical before and after the run.
- PRIMARY TEST not accessed; other held-out sets were read only as `split_role`.

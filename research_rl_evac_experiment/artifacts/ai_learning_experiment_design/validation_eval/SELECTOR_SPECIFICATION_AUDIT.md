# Selector Specification Audit (VALIDATION, read-only)

- **Numbers:** `selector_specification_audit.json` (same folder)
- **Script:** `scripts/audit_selector_specification.py`
- **Commit of the results:** 6a3453d

**Question:** is the zero-switch result of the frozen selector (A) intended conservatism or (B) unintentional over-constraint from incompatible quantities or scales?

**Answer:** the evidence points to **(B)**. The safety floor compares a candidate's upper quantile (q0.9) against the default's *mean*. That demands a predicted improvement on every criterion at once, and the default itself fails its own floor in 64 to 85% of states. Whether the text as written was the intended standard is a decision for the experiment owner; this audit reports what the arithmetic does.

**Notation:**
- `m` is a mean prediction and `q9` a q0.9 prediction.
- `d` is the label-free default and `a` a candidate.
- `I_k = m_d,k − m_a,k` is the candidate's predicted improvement on criterion k.
- `u_k = q9_a,k − m_a,k` is the candidate's upper width.
- K is the set of active criteria (T90, T100, U, H). T90 is skipped when `normalized_remaining ≤ 4/40`.

## 1. Exact frozen inequalities

- **Safety floor (step 2):** keep `a` only if `q9_a,k ≤ m_d,k + δ_k` for every k in K. The default is always kept.
- **Tolerance ranking (step 3):**
  - keep candidates whose T90 mean is within δ_T90 of the best;
  - then T100 within δ_T100;
  - then U within δ_U;
  - then only the exact minimum H mean.
  - Ties go to `d`, otherwise the lowest id. The winner is `a*`.
- **Switch (step 4):** find the first criterion where `|m_a*,k − m_d,k| > δ_k`. Switch to `a*` only if it is better there by more than δ_k; otherwise stay with `d`.
- **Abstention (step 5):** fall back to `d` if either:
  - the state is OOD (distance to TRAIN > τ = 0.9354); or
  - `q9_a*,T100 − m_a*,T100 > 1.7897`.
- **Learned tolerances:** δ_T90 = 0.5, δ_T100 = 0.5, δ_U = 4.7079, δ_H = 4.7651.

## 2. Feasibility condition

The floor rewrites as `I_k ≥ u_k − δ_k` for all k in K. A non-default switch on criterion k* therefore needs:

| Criteria | Condition |
|---|---|
| Before k* | `u_k − δ_k ≤ I_k ≤ δ_k`, which is only possible if **`u_k ≤ 2δ_k`** (upper width at most 1.0 tick for T90 and T100, at most 9.4 for U) |
| The switch criterion k* | `I_k* > max(δ_k*, u_k* − δ_k*)` |
| After k* | `I_k ≥ u_k − δ_k` |
| Abstention | not OOD, and `u_T100 ≤ 1.79` |

A candidate predicted identical to the default passes the floor only if `u_d,k ≤ δ_k` on every criterion. The default fails that on:

| Criterion | T90 | T100 | U | H |
|---|---|---|---|---|
| States where the default fails its own floor | 78% | 85% | 64% | 64% |

So in practice the floor tests "predicted strictly better by about one interval width", not "no worse".

For comparison, typical sizes:

| Quantity | T90 | T100 | U | H |
|---|---|---|---|---|
| Median candidate upper width `u` (approx.) | 1.0 | 1.05 | 18.2 | 7.4 |
| 2δ | 1.0 | 1.0 | 9.4 | 9.5 |
| Pairs where the floor demands a predicted improvement | 77% | 74% | 63% | 61% |
| Best realized improvement over default per state, median / 75th percentile | 0 / 1 | 0 / 1 | 0 / 4 | 0 / 12 |

## 3. Attrition by selector stage

VALIDATION has 1,845 states and 547,965 non-default candidate/state pairs.

| Stage | Pairs | States |
|---|---|---|
| Non-default pairs | 547,965 | 1,845 |
| Pass safety floor (all criteria) | 43,609 | 240 |
| Survive step-3 tolerances | 203 | 203 |
| Win step 3 (`a*`) | 203 | 203 |
| Pass switch rule | **0** | 0 |
| Not OOD | 0 | 0 |
| Pass width abstention | 0 | 0 |

Failures per condition, each counted independently:

| Failure reason | Pairs |
|---|---|
| Mean not better enough (switch rule) | 523,125 |
| Floor failure on T90 or T100 | 503,731 |
| Floor failure on unmet | 372,575 |
| Floor failure on hazard | 305,993 |
| Any floor failure | 504,356 |
| Removed by step-3 tolerances after the floor | 43,406 |
| OOD | 1,188 (4 states) |
| Width abstention | 27,089 |

- **The two rules never overlap:** 24,840 pairs pass the switch rule alone and 0 pass both the switch rule and the floor.
- **All 203 step-3 winners are indistinguishable from the default:** 0 of 203 differ from it by more than δ on any criterion.
  - Step 3's exact minimum on H removes `d` over hazard-mean differences of about 0.05, against δ_H = 4.77.
  - This is a separate inconsistency between steps 3 and 4. It causes no harm, but it produces winners that can never switch.

## 4. The 540 no-floor diagnostic switches

These come from the same selector with step 2 skipped. This is a description of the mechanism only, not a protocol.

- **Every one of the 540 switches is decided on hazard;** T90, T100 and U are within δ.
- **The hazard signal is real:** predicted improvement averages 7.8 and realized improvement 13.2.
- **Harm is 198 of 540 (37%).** Harm here means the realized lexicographic key is worse than the default's. A breakdown of harm by criterion was not computed.
- **The floor does not identify harmful candidates:**
  - it rejects 100% of harmful and 100% of non-harmful switches;
  - the floor margin predicts harm with AUC 0.52, which is chance.
- **The intervals are not the cause:** on these candidates q0.9 covers the actual outcome 91 to 95% of the time.
- **Conclusion:** the rejection comes from the structure of the comparison, neither from detecting harm nor from miscalibration.

| Target | Candidate predicted mean (median) | Candidate q0.9 (median) | Candidate actual (median) | Default actual (median) | q0.9 covers actual |
|---|---|---|---|---|---|
| T90 | 12.57 | 14.00 | 12 | 13 | 0.91 |
| T100 | 13.32 | 14.97 | 13 | 14 | 0.95 |
| Unmet | 69.1 | 90.9 | 65 | 74.5 | 0.91 |
| Hazard | 2.9 | 16.2 | 1 | 20 | 0.95 |

## 5. Quantile calibration

No recalibration was done. Values below are after sorting the quantiles, as the selector consumes them.

| Target | P(y ≤ q0.1), nominal 0.10 | P(y ≤ q0.5), nominal 0.50 | P(y ≤ q0.9), nominal 0.90 | Central 80% coverage | Diagnosis |
|---|---|---|---|---|---|
| T90 | 0.155 | 0.40 | 0.75 | 0.60 | Miscalibrated; upper quantile under-covers |
| T100 | 0.109 | 0.42 | 0.83 | 0.72 | Mildly under-conservative at the top |
| Unmet | 0.253 | 0.46 | 0.84 | 0.84 | Under-covers; q0.1 is stuck at 0 on zero-inflated data |
| Hazard | 0.407 | 0.41 | 0.76 | 0.76 | Under-covers; q0.1 and q0.5 are stuck at 0 on a point mass |

- **The q0.9 errs low, not too wide,** so the floor is not rejecting because the intervals are over-conservative.
- **Part of the T90/T100 shortfall looks like an integer-rounding effect.**
  - Predictions such as 13.9999998 miss an actual value of 14 by a hair.
  - When q0.9 is exceeded, the mean miss is only 0.26 ticks for T90 and 0.50 for T100, compared with 11.2 for unmet and 13.1 for hazard.

## 6. Minimal repair categories (not ranked, none chosen)

**Rules that apply to every repair:**
- Any selector change made after seeing VALIDATION needs a new linked amendment (A3, or C4 under A2) that says it was motivated by VALIDATION.
- Tolerances and the width threshold would be re-derived under the new rule.
- The exact-lookup baseline should get the same change, so the comparison stays fair.
- The dataset is not invalidated by any of these. Its records describe the P2/P5 reference as model-free, so it does not depend on step 2.

| # | Repair | Rule it changes | Why | Safety property kept | New evidence needed | Retrain? |
|---|---|---|---|---|---|---|
| R1 | Compare like with like: `q9_a ≤ q9_d + δ` | Step 2 | The default then passes its own floor by construction | Upper tail no worse than the default's, plus δ | Attrition and harm on VALIDATION | No |
| R2 | Model the paired difference: require `q9(y_a − y_d) ≤ δ` | Step 2 and C3 (new estimators) | Directly bounds paired harm | Paired non-inferiority | Coverage of the difference model on VALIDATION | Yes |
| R3 | Widen the floor tolerance to cover interval width | C2 tolerance rule | Scales δ to the width it is compared against | Bounded loss, but weaker than now | Attrition and harm on VALIDATION | No |
| R4 | Calibrate quantiles (for example, conformal) | Adds a calibration step | Under-coverage of 0.75 to 0.84 against 0.90 | Honest q0.9 meaning | A calibration split, which would use up VALIDATION | No, but new data |
| R5 | Handle integer values (compare on whole ticks) | Steps 2 and 4 for T90/T100 | The 13.9999 effect | Unchanged | Minor | No |
| R6 | Make steps 3 and 4 consistent (use δ_H in step 3, prefer `d`) | Step 3 | The 203 winners that cannot switch | Unchanged | Minor; produces no switches by itself | No |
| R7 | Accept as intended; no repair | None | Zero switches taken as the correct conservative outcome | Unchanged | None | No |

## 7. Integrity

- No selector, threshold, tolerance, model, amendment or record was modified.
- Nothing was retrained or recalibrated.
- All 33 protected file hashes were verified identical before and after the run.
- VALIDATION predictions were regenerated from the hash-verified models on the same rows, because they had not been stored row by row. This is deterministic.
- PRIMARY TEST and every other held-out set remain sealed; the script read only their `split_role`.

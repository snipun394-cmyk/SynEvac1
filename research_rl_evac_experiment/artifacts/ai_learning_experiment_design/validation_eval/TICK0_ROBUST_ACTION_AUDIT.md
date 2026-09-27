# Tick-0 Robust-Action Audit: IN-SAMPLE ROBUST-ACTION FEASIBILITY

- **Numbers:** `tick0_robust_action_audit.json` (same folder)
- **Script:** `scripts/audit_tick0_robust_actions.py`
- **Follows:** `DECISION_TICK_OBSERVABILITY_DIAGNOSTIC.md` (commit e35c7a7)

**Label:** every result here is **in-sample robust-action feasibility** on VALIDATION labels. None of it is generalization.

**Status:**
- Read-only.
- No classifier, no lookup policy, no hard-coded classes.
- No selector, tolerance or protocol change, no amendment, no retraining.
- PRIMARY TEST and every other held-out set remain sealed.

**Hidden labels:** `t_h`, `p`, `compliance_key`, `template` and `seed` are diagnostic labels only. The models received exactly the frozen 54 columns.

## Definitions used (fixed before results)

- **Worse:** the frozen A2 harm comparison, `key(a) > key(d)`, using the exact lexicographic key (T90, T100, unmet, hazard). Equal keys are not harmful. The selector tolerances δ are not part of the harm definition and are not used.
- **Weak robust:** a non-default action that is never worse than the default for any VALIDATION member of the class.
- **Strict robust:** weak robust, and strictly better for at least one member.
- **Best fixed action under full VALIDATION knowledge:** fewest members harmed, then most strictly improved, then most captured, then lowest id.

## Headline

- **Four of the six tick-0 classes have strict-robust actions; two have none.**
- **The robust actions are modest improvers.**
  - They improve 52 to 67 of 90 members.
  - Average hazard gain is 2.3 to 4.9.
  - They capture the oracle-best outcome in only 0 to 8 members per class.
  - Their worst case is never worse than the default's.
- **Within VALIDATION they are stable across hidden regimes:**
  - leaving any one of the 9 regimes out never produces an action that harms the held-out regime;
  - they improve outcomes only when the hazard starts at t_h 3, 5, 6 or 7;
  - with t_h = 9 or no hazard they merely tie the default.
- **The existing predictor does not recognize robustness.**
  - Within a class it ranks robust actions well on predicted T90/unmet (AUC 0.91 to 0.97).
  - But in class 5, which has no robust action, 11 actions have predictions that look just like the robust actions of the other classes, and every one of them harms 22 to 32 of 90 members.
  - The selector itself never picks a robust action: its step-3 hazard minimum prefers the aggressive `BASE=B` actions, and every robust action's predicted hazard gain (at most 4.52) is below δ_H = 4.77.

## 1. The six tick-0 observable classes

**What every class has in common:**
- All non-zero observable features are waiting counts in R1 to R4, plus `residual_cap_H_CE1_normalized` = 1 and `normalized_remaining` = 1.
- Hidden `t_h` ∈ {3, 5, 6, 7, 9, none}; `p` ∈ {0.5, 0.7, 0.8, 0.9}.
- 9 hidden (t_h, p) regimes with 10 members each.
- One compliance key per class, because the key is set by the template.
- 298 candidates evaluated.
- Default = `BASE=P` (id 100).

| Class (template) | Observable signature (waiting R1 / R2 / R3 / R4) | States | Improvable | Best fixed action (full VALIDATION knowledge) | Never worse for every realization? |
|---|---|---|---|---|---|
| 0 D0_sym | 10 / 10 / 10 / 10 | 90 | 70 | 102 `BASE=P\|R1=B`: 67 improved, 0 harmed | yes |
| 1 D1_east_heavy | 5 / 5 / 15 / 15 | 90 | 70 | 104 `BASE=P\|R2=B`: 52 improved, 0 harmed | yes |
| 2 D2_west_heavy | 15 / 15 / 5 / 5 | 90 | 70 | 110 `BASE=P\|R4=B`: 59 improved, 0 harmed | yes |
| 3 D3_first_room_heavy | 15 / 5 / 15 / 5 | 90 | 70 | 150 `BASE=P\|R2=B\|R4=B`: 66 improved, 0 harmed | yes |
| 4 D4_second_room_heavy | 5 / 15 / 5 / 15 | 90 | 70 | none better; the best is outcome-identical to the default (0 improved) | trivially |
| 5 D5_R1_concentrated | 25 / 5 / 5 / 5 | 90 | 80 | none better; the best is outcome-identical to the default (0 improved) | trivially |

## 2 and 3. Robust action sets

| Class | Actions outcome-identical to default | Weak robust (non-default) | Weak robust, excluding identical | Strict robust | Distinct outcome profiles among strict robust | Profile that dominates the others for every member |
|---|---|---|---|---|---|---|
| 0 | 19 | 33 | 14 | 14 | 2 | none |
| 1 | 33 | 43 | 10 | 10 | 1 | the single profile |
| 2 | 33 | 43 | 10 | 10 | 1 | the single profile |
| 3 | 19 | 35 | 16 | 16 | 3 | {150, 225} |
| 4 | 53 | 53 | 0 | 0 | 0 | – |
| 5 | 33 | 33 | 0 | 0 | 0 | – |

**Strict-robust outcome profiles** (mean improvement over the default; positive = better):

| Class | Profile (representative actions) | Members improved | Captured | Δ T90 | Δ T100 | Δ unmet | Δ hazard |
|---|---|---|---|---|---|---|---|
| 0 | R1=B family (102, 121, 122, 133, 134, 137, 138) | 67 | 5 | 0.24 | 0.17 | 1.99 | 4.70 |
| 0 | R3=B family (108, 160, 162, 181, 182, 185, 186) | 65 | 8 | 0.26 | 0.18 | 1.97 | 4.64 |
| 1 | R2=B family (104, 116, 118, 141, 142, 153, 154, 157, 158, 277) | 52 | 6 | 0.00 | 0.09 | 1.40 | 2.28 |
| 2 | R4=B family (110, 164, 166, 176, 178, 189, 190, 193, 194, 217) | 59 | 3 | 0.00 | 0.06 | 1.56 | 2.57 |
| 3 | {150 `P\|R2=B\|R4=B`, 225 `B\|R1=P\|R3=P`} | 66 | 0 | 0.33 | 0.11 | 2.20 | 4.92 |
| 3 | R2=B family (104, 141, 142, 153, 154, 157, 158) | 60 | 0 | 0.06 | 0.02 | 1.19 | 2.60 |
| 3 | R4=B family (110, 164, 166, 189, 190, 193, 194) | 53 | 0 | 0.06 | 0.02 | 1.03 | 2.33 |

- **Multiple robust actions are mostly behaviourally equivalent.** They add harmless overrides, for example on C1, that don't change the outcome.
- **Class 0 has real ambiguity:** two robust profiles, neither dominating, which benefit different members.

**Stability across hidden regimes:**
- In classes 0 to 3, every class-level robust action strictly improves the members with t_h ∈ {3, 5, 6 (all p), 7}, and ties the default for t_h ∈ {9, none}.
- The per-regime robust sets are much larger than the class set. For example, class 0 has 21 to 121 per regime but 14 for the class. The class set is set by the tightest regimes: t_h 3, 5, 9 and none.
- **Leave-one-regime-out, within VALIDATION only:** in every class and every held-out regime, the actions robust on the other 8 regimes stay non-harmful in the held-out one (0 failures). The robust set is also unchanged by dropping any one regime.

## 3 (cont.). Worst, best and mean outcomes, and comparison with the selectors

Key = (T90, T100, unmet, hazard); worst and best are the lexicographic max and min across the 90 members.

| Class | Action | Worst-case key | Best-case key | Mean (T90, T100, U, H) | Harmed | Improved |
|---|---|---|---|---|---|---|
| 0 | Default = frozen (100) | (14, 16, 78, 36) | (11, 12, 48, 0) | 12.90, 13.72, 69.2, 20.1 | 0 | 0 |
| 0 | R1 winner (237 `B\|R1=P\|H=P`) | (14, 16, 76, 33) | (12, 12, 49, 0) | 12.37, 13.48, 67.2, 6.1 | **35** | 55 |
| 0 | Best robust (102) | (14, 16, 78, 36) | (11, 12, 48, 0) | 12.66, 13.56, 67.2, 15.4 | 0 | 67 |
| 1 | Default = frozen | (14, 16, 79, 30) | (12, 12, 50, 0) | 13.11, 13.72, 69.8, 16.0 | 0 | 0 |
| 1 | R1 winner (178) | (14, 15, 99, 0) | (12, 13, 58, 0) | 12.47, 13.44, 69.6, 3.3 | **41** | 49 |
| 1 | Best robust (104) | (14, 15, 77, 28) | (12, 12, 50, 0) | 13.11, 13.63, 68.4, 13.7 | 0 | 52 |
| 2 | Default = frozen | (14, 15, 77, 28) | (12, 12, 50, 0) | 13.11, 13.72, 70.1, 16.1 | 0 | 0 |
| 2 | R1 winner (293) | (14, 15, 103, 0) | (12, 12, 54, 0) | 12.54, 13.44, 70.0, 3.0 | **40** | 50 |
| 2 | Best robust (110) | (14, 15, 77, 28) | (12, 12, 50, 0) | 13.11, 13.67, 68.6, 13.5 | 0 | 59 |
| 3 | Default = frozen | (14, 15, 76, 32) | (11, 12, 48, 0) | 12.97, 13.69, 69.4, 20.2 | 0 | 0 |
| 3 | R1 winner (249) | (14, 15, 104, 0) | (12, 12, 50, 0) | 12.63, 13.33, 67.9, 4.0 | **35** | 55 |
| 3 | Best robust (150) | (14, 15, 75, 31) | (11, 12, 48, 0) | 12.63, 13.58, 67.2, 15.3 | 0 | 66 |
| 4 | Default = frozen | (14, 16, 78, 36) | (11, 12, 48, 0) | 12.93, 13.67, 69.6, 21.4 | 0 | 0 |
| 4 | R1 winner (257) | (14, 15, 74, 30) | (12, 12, 49, 0) | 12.34, 13.48, 65.6, 9.0 | **20** | 70 |
| 4 | Robust set | none | | | | |
| 5 | Default = frozen | (14, 16, 83, 24) | (13, 13, 59, 0) | 13.46, 14.06, 74.4, 12.3 | 0 | 0 |
| 5 | R1 winner (283) | (14, 15, 111, 0) | (13, 13, 61, 0) | 13.13, 13.38, 72.2, 1.3 | **27** | 63 |
| 5 | Robust set | none | | | | |

- The R1 winners cut mean hazard far more (to 1.3 to 9.0) and improve more members, but harm 20 to 41 of 90 in every class.
- The robust actions keep the default's worst case, with no harm, and give a smaller average gain.

## 4. Does the existing predictor recognize the robust actions?

**Predicted vs realized improvement for the robust actions:**

| Class | Predicted T90 / T100 / unmet / hazard | Realized mean T90 / T100 / unmet / hazard |
|---|---|---|
| 0 | −0.01 to −0.09 / −0.02 to 0.07 / −0.76 to 0.13 / 2.66 to 3.16 | 0.24 / 0.17 / 1.99 / 4.70 |
| 1 | about 0 / about 0 / −0.44 to 0.12 / 1.41 to 2.41 | 0.00 / 0.09 / 1.40 / 2.28 |
| 2 | about 0 / about 0 / −0.72 to 0.05 / 1.12 to 2.48 | 0.00 / 0.06 / 1.56 / 2.57 |
| 3 | about 0 / about 0 / −0.38 to 0.24 / 2.10 to 4.52 | up to 0.33 / 0.11 / 2.20 / 4.92 |

- Predicted hazard gains are directionally right and about the right size.
- In class 3 the dominant profile {150, 225} is predicted highest among the robust set (4.19 / 4.52).

**Ranking within a class** (classes 0 to 3; robust vs the other 296 non-default actions):

| Score | AUC |
|---|---|
| Predicted T90 improvement | 0.91 to 0.96 |
| Predicted unmet improvement | 0.91 to 0.97 |
| Predicted T100 improvement | 0.74 to 0.85 |
| Predicted hazard improvement | 0.61 to 0.64 |
| Frozen floor margin | 0.88 to 0.93 |
| R1 floor margin | 0.76 to 0.79 |

**What the selectors do with them:**
- All strict-robust actions pass the R1 floor and survive R1's step-3 T90/T100/U tolerance filter. None passes the frozen floor.
- Ranked by predicted hazard among R1-admitted actions, the best robust action is only 99th to 117th of 163 to 183, so step 3's hazard minimum never picks it.
- Even if it were chosen, its predicted hazard gain (≤ 4.52) is below δ_H = 4.77, so step 4 would not switch.

**Prediction check across classes:** take the per-target range of predicted improvements spanned by all strict-robust actions. This range is defined from those actions, so it is circular for classes 0 to 3. Within it:

| Class | Actions inside | Of which robust | Members harmed by inside actions |
|---|---|---|---|
| 0 | 14 | 14 | 0 |
| 1 | 16 | 10 | up to 29 |
| 2 | 14 | 10 | up to 27 |
| 3 | 18 | 16 | up to 21 |
| 4 | 3 | 0 | 0 (outcome-identical to default) |
| **5** | **11** | **0** | **22 to 32 each (min 22, median 28)** |

**Conclusion:** the predictor ranks robust actions well *within* a class that has them. It cannot certify robustness. In class 5, actions with predictions like the robust ones harm 24 to 36% of members, and nothing in the predictions tells class 5 apart.

## 6. Is robustness observable?

- **Classes 0 to 3:** there is a common strict-robust action across all members.
  - Classes 1 and 2 effectively have one (a single outcome profile).
  - Class 3 has a dominating profile.
  - Class 0 has two non-dominating profiles, which is real action-selection ambiguity.
  - These classes support potentially observable robust control, in-sample.
- **Classes 4 and 5:** no common strict-robust action. Early information is not enough for guaranteed-safe improvement.
- **Do identical observations need different actions?**
  - No action strictly improves every member of any class.
  - Pairs of members with identical observations that are both improvable, yet share no improving action: class 5 has 161, class 1 has 29, class 2 has 1, and classes 0, 3 and 4 have 0.
  - In class 5 in particular, identical observations genuinely require different actions to improve.

## 7. Is robust tick-0 control structurally feasible?

**In-sample, partially:**
- A zero-harm, strictly-improving action exists for 4 of 6 observable classes (360 of 540 states).
- It improves 244 of those 360 states, but captures the oracle-best in only 14.
- Within VALIDATION it is stable under leaving out any one regime.

**With the existing predictor and selector, no:**
- The predictor does not separate robust from harmful actions across classes (class 5).
- The frozen/R1 selection logic prefers maximum predicted hazard reduction, which is the wrong objective for robustness.

## What evidence would be needed before any robust action becomes a policy

1. **An out-of-sample robustness test without PRIMARY TEST.**
   - Robust sets defined on TRAIN (regimes t_h ∈ {none, 3, 5, 7, 9} × p ∈ {0.5, 0.7, 0.9}) and checked on VALIDATION (t_h = 6 and p = 0.8, which TRAIN never sees).
   - This asks whether TRAIN-defined robust actions remain non-harmful on unseen regimes.
2. **A pre-specified, label-free way to reach the robust action at decision time.**
   - For example, a model-based robustness criterion fixed before evaluation, not a lookup of these four classes.
   - It must also *abstain* in classes like 5, which the current predictions cannot do.
3. **A pre-specified tie-break among robust actions.** Class 0 has two non-dominating profiles.
4. **A power and safety analysis.** Only 10 members per regime per class here; the frozen ≤ 1% harm endpoint needs a far larger number of decisions before zero observed harm means anything.
5. **Coverage of the PRIMARY regimes** (t_h 4/8, p 0.6/0.85). They are inside the TRAIN/VALIDATION ranges but unobserved. Robustness to them can be claimed only after a single sealed evaluation under a pre-registered amendment.
6. **A decision on the objective.** Robust actions trade most of R1's hazard reduction (mean hazard about 13 to 16 vs 1 to 9) for zero observed harm. Which trade the experiment targets is the user's decision.

## Integrity

- All 33 protected file hashes were identical before and after the run. All 16 model hashes were verified.
- Predictions are identical within each class (asserted).
- No selector, model, tolerance, dataset or protocol change; no policy created.
- PRIMARY TEST not accessed; other held-out sets were read only as `split_role`.

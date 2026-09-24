# Amendment A2: pre-training evaluation protocol (Issues 1–6)

**Status: PROPOSED, awaiting approval. Nothing run or evaluated.** Machine-readable record: `amendment_A2_evaluation_protocol.json`.

## H. Provenance

- **A2 is post-hoc.** It was created **before the P2/P5 reference run and before any model training**, on 2026-09-24T21:07:08Z. No model results informed it.
- **Nothing is rewritten.** The dataset (`36aa757f…`), the Spec v2 files and their original wording, and the A1 criteria and evaluation are all unchanged. A2 supersedes the listed clauses for evaluation only.
- **Parent hashes:**

| File | SHA-256 (prefix) |
|---|---|
| `preregistered_success_criteria.json` | `3ec6fdb6…` |
| `closed_loop_evaluation.json` | `a81de9f7…` |
| `selector_design.json` | `8a41c777…` |
| `final_feature_manifest.json` | `67f54855…` |
| `candidate_action_set.json` | `78dee3c2…` |
| `amendment_A1_acceptance_criteria.json` | `cde59177…` |
| `amendment_A1_evaluation.json` | `51575d64…` |

- **Parent git commit:** `74de2ac`.
- **Recorded preregistration gap:** Spec v2 committed to set the closed-loop usefulness threshold "from the Phase-1 pilot (spread between P2 and P5) BEFORE Phase 3". No P2/P5 run exists. The wording also specifies no metric, no mapping, no statistic, and no rule for a small spread.

**Conventions used throughout:**
- **Ordering key:** key(a, s) = (future_T90, future_T100, future_unmet, future_hazard_unmet), compared as exact integers in the frozen order.
- **Default:** d(s) = `BASE=l(s)`, where l is the label-free default.
- **Bootstrap:** two-sided 95% percentile interval, 2,000 resamples of whole trajectories (for closed-loop runs, whole paired episodes).
- **Seeds:** 5544101–5544105, fixed before any run.

## Issue 1: closed-loop usefulness (decision M-C)

**A. Original.** "Usefulness: NO threshold is justified yet -- the minimum meaningful gain will be set from the Phase-1 pilot (spread between P2 and P5) BEFORE Phase 3 and appended here with its hash." The problem: no P2/P5 result exists, and no metric or mapping was ever defined.

**B. Amended.**

*Reference run (model-free):*
- Configurations: the 240 discarded pilot configurations exactly as frozen (8 layouts × t_h {none, 3, 5, 7, 9} × p {0.5, 0.7, 0.9} × 2), seeds 190000–190239, keys `AILX_v1|PILOT|<layout>`. They sit outside every split.
- Full episodes, no ε perturbation, identical configurations and compliance vectors for P2 and P5.
- **P2:** the label-free default at every decision.
- **P5:** at every decision, the BASE2_298 candidate with the best true one-step key (frozen rollout, label-free continuation). Ties go to d(s), otherwise to the lowest candidate id.
- **Reference gap:** Δ = mean(T100_P2 − T100_P5), point estimate. Its bootstrap CI is reported descriptively only.
- The run's outputs and their hash are appended to A2, completing the original commitment.

*Feasibility:*
- If Δ ≥ 0.5 tick, usefulness is **assessable**.
- If Δ < 0.5 tick, usefulness is **not assessable**: it is reported descriptively, neither PASS nor FAIL.

*P4 endpoint:* gain = mean(T100_P2 − T100_P4) over the 960 TEST_PRIMARY_JOINT configurations (closed loop, no ε, same configurations for both).

**C. Threshold.** If assessable: **PASS iff the lower bound of the 95% bootstrap CI of the gain is ≥ 0.5 tick** (seed 5544101). There is no point-estimate rule.
- *Where 0.5 comes from:* it is the frozen `selector_design.json` tolerance δ_T100, justified there as "half a tick separates distinct values".
- *What is not required:* P4 need not capture any fixed fraction of the P2→P5 gap.
- *Safety is unchanged and independent:* P4 must be strictly worse than P2 in ≤ 1% of paired episodes (upper CI bound).

**D. Unit.** Paired episode (one per configuration).

**E. Denominator.** 240 reference episodes for Δ; 960 primary-test episodes for P4.

**Caveat:** Δ is measured on pilot configurations (8 layouts, training-level levels) and applied to the primary test (6 main layouts, held-out levels). This population shift is inherent to Option B.

## Issue 2: recommendation (state-level oracle-behavior capture)

**A. Original.** "Outcome-equivalence rate greater than d(s)'s own, CI excluding 0, on states where some candidate strictly beats d(s)." The problem: d(s)'s rate on those states is 0 by construction, so the criterion is degenerate.

**B. Amended.**
- **Improvable state:** min over a of key(a, s) < key(d(s), s).
- **Success on a state:** key(selected, s) = min over a of key(a, s).
- **Primary metric:** oracle-behavior capture rate = successes ÷ improvable states, per test family (primary: TEST_PRIMARY_JOINT).
- **Semantics boundary:**
  - *Evaluation-side outcome equivalence* (equal realized target vectors) is used only to score recommendations after the fact. It is never a selector input.
  - *Observable candidate semantics:* two candidates are the same when their assignments agree on every eligible zone whose admissible waiting count is > 0. This is used only for a secondary diagnostic.
  - Hidden compliance, Q_A/Q_B, exit labels and `equivalence_class` are never used.
- **Reference rates reported:**
  - The default's rate (= 0 by construction; stated, not a competitor).
  - The expected rate of a random candidate, computed from the labels.
- **Secondary diagnostics:** capture over all states, T100/unmet regret, observable-semantic match, exact oracle-candidate match.

**C. Threshold, FLAGGED.** **An absolute numeric capture threshold is not defensible without an arbitrary choice.** None exists in the preregistration, and its only anchor is 0 by construction. The least arbitrary proposal, marked post-hoc pre-training, **requires your approval**:
- The learned selector's capture rate must exceed that of the **same frozen selector driven by the best preregistered non-learning baseline**, with the lower bound of the paired state-level difference's 95% bootstrap CI > 0 (seed 5544102).
- *Choosing that baseline:* take the one with the lowest VALIDATION primary prediction loss among mean-by-candidate, mean-by-candidate × tick, exact-input lookup (falling back to candidate × tick for unseen inputs) and 1-NN. Its quantiles are the empirical TRAIN quantiles of its own grouping.
- *Excluded:* default-outcome-as-prediction, because it always selects d(s).
- *Rationale:* this mirrors the frozen prediction criterion ("lower than the best baseline, paired CI excluding 0").

**D. Unit.** Decision state. **E. Denominator.** Improvable states in the family.

## Issue 3: prediction-loss weighting

**A. Original.** The pinball-loss criterion on future_T100 and future_unmet has no weighting convention, and 91.7% of rows are behavioral duplicates.

**B. Amended.**
- **Every primary prediction metric uses equal weight per candidate row:** pinball loss at q = 0.1/0.5/0.9 on future_T100 and future_unmet, and quantile coverage, for the model and every baseline.
- **This is equal weight per decision state:** every state has exactly 298 rows (5,524,324 = 18,538 × 298), so every state carries the same total weight.
- **Paired CI:** 95% trajectory-clustered bootstrap (seed 5544104).
- **Training objective:** also uses equal row weights. Weighting by `equivalence_class`, hidden compliance or candidate frequency is prohibited.
- **Behavior-level metrics:** secondary diagnostics only.

**D. Unit.** Candidate row. **E. Denominator.** All rows of the family.

## Issue 4: safety denominator

**A. Original.** "≤ 1% of decisions", with a derivation that counts "≥ 300 independent deviating decisions". The denominator is ambiguous.

**B. Amended.**
- **Unit:** the decision state, with exactly one recommendation per state. The 298 candidate rows never count as separate decisions.
- **Eligible state:** all 298 labels present and finite (all 18,538 states qualify). Any failing state is masked and reported. States where T90 was already reached stay eligible.
- **Harm:** key(selected, s) > key(d(s), s).
- **Abstention, fallback or no switch:** the recommendation is d(s). It counts in the denominator and is never harmful. A different candidate with the same key is not harmful either.
- **Harm rate:** harmful eligible states ÷ eligible states.

**C. Threshold.** Upper bound of the 95% bootstrap CI ≤ 1% in every test family (seed 5544103). **"≥ 300" is the minimum number of eligible states per family, not the denominator.**
- Current eligible counts: 3,256 primary; 1,841 held-out timing; 2,989 held-out compliance; 1,442 extrapolation; 1,180 zero-shot.
- Secondary: the share of deviating recommendations, and the harm rate among deviating states.

**D. Unit.** Decision state. **E. Denominator.** Eligible states in the family.

## Issue 5 (G): selector semantics

**A. Original.** Step 1 says "drop behaviorally duplicate candidates". That needs hidden labels.

**B. Amended step 1:**
- Skip T90 when it has been reached. This is decided observably: `normalized_remaining ≤ 4/40`, i.e. at least 36 of 40 evacuated.
- **No candidate is removed.**

**How candidates are scored and ties broken:**
- All 298 candidates are scored independently from (26 features, descriptor).
- Removing candidates via hidden compliance, hidden exits, `equivalence_class` or realized outcomes is prohibited.
- Steps 2–5 of `selector_design.json` (tolerances, safety floor, switch rule, abstention) are unchanged.
- **Final tie-break:** d(s) if it is in the remaining set, otherwise the lowest candidate id.
- Observable grouping is not used. Any later optimization would need separate approval and could use only observable semantics.

## Issue 6 (F): leakage boundary, the exact model input

**Model input: exactly 54 columns, in this order.**
1. **The 26 admissible features:** `waiting_R1, waiting_R2, waiting_C1, waiting_R3, waiting_R4, waiting_C2, waiting_CE1, waiting_CE2, in_transit_R1_C1, in_transit_R2_C1, in_transit_C1_H_1tick, in_transit_C1_H_2tick, in_transit_R3_C2, in_transit_R4_C2, in_transit_C2_H_1tick, in_transit_C2_H_2tick, in_transit_H_CE1_1tick, in_transit_H_CE1_2tick, in_transit_H_CE2_1tick, in_transit_H_CE2_2tick, in_transit_CE1_EXITA, in_transit_CE2_EXITB, residual_cap_H_CE1_normalized, normalized_time, normalized_remaining, waiting_H_total`.
2. **The 28 descriptor columns:** `cand_<zone>_<value>` for zone ∈ (R1, R2, C1, R3, R4, C2, H) and value ∈ (UNTOUCHED, FAVOR_A, PARITY, FAVOR_B), derived only from `candidate_id` via `candidate_action_set.json`.

**Runtime assertion, before every fit and every predict:** `assert list(input_columns) == WHITELIST`. Any extra, missing or reordered column aborts.

**Explicitly excluded:**
- *Trajectory and scenario metadata:* t_h, p, seed, compliance_key, cell, template, distribution, replicate, split_role, trajectory_id, decision_tick, epsilon, pre_decision_*, state_fingerprint, n_nonempty_eligible_zones, hazard_started, seed_rule.
- *Candidate bookkeeping:* equivalence_class, is_default_candidate, default_action, raw candidate_id, candidate_name.
- *Hidden state:* hidden compliance, Q_A/Q_B.
- *Labels and targets:* oracle labels, all targets and T90/T100 fields.
- *Provenance fields:* continuation_*, spec_*.

## Preserved limitations

- **A:** 60% of decision states are fixed by the occupant layout.
- **B:** G3 has never been tested beyond 4 non-empty zones.
- **C:** diagnostic family A needs the deferred Test A batch.

## Not done

No P2/P5 run. No training. No baseline fitted. No selector. No dataset, Spec v2 or A1 change.

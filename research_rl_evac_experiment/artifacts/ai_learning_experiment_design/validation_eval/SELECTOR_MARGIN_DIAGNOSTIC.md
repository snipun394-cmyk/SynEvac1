# Selector Margin Diagnostic (VALIDATION only, offline)

- **Numbers:** `selector_margin_diagnostic.json` (same folder)
- **Script:** `scripts/diagnose_selector_margins.py`
- **Follows:** `SELECTOR_R1_R6_DIAGNOSTIC.md` (commit 799450a)

**Question:** is the R1 safety failure a decision-margin problem, and does the existing C3 predictor contain enough signal to support a selector that meets the frozen 1% harm limit?

**Status:**
- This is a diagnostic only. It is not a selector, not a threshold choice and not an amendment.
- Selector code, A2, C1, C2, C3, models, quantiles, tolerances and dataset are unchanged.
- PRIMARY TEST and every other held-out set remain sealed.

**Base:**
- The R1 selector: step 2 is `q9_a,k ≤ q9_d,k + δ_k`, and every other frozen step is unchanged.
- It reproduces the 540 switches and the 126 T90-harm cases; the script asserts both.

**Fixed before any result was inspected:**
- **Margin grid:** g ∈ {0, 0.5, 1, 1.5, 2, 3, 4} × δ_k. For T90/T100 (δ = 0.5) this is exactly 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 ticks. For unmet/hazard it scales to their own δ (4.71 / 4.77), because a raw 0 to 2 "people" margin would be negligible there.
- **Integer rule:** T90/T100 predicted means rounded half up, and q0.9 read as `ceil(q9 − 1e-6)`.

**Statistics:** frozen trajectory-clustered bootstrap (2,000 resamples) with seeds 5544102 capture, 5544103 harm and 5544105 descriptive deltas.

Sign conventions:
- *Improvement* = default minus candidate; positive = candidate better.
- *Difference* = selected minus default; positive = worse.

## Headline

**No.** Within VALIDATION, no deterministic rule on the existing predictor's outputs can bring R1 to the ≤ 1% harm limit while still switching. This is structural, not a threshold problem:

- **All 540 R1 switches happen at tick 0,** one per VALIDATION trajectory (540 distinct trajectories). No switch happens at any later tick.
- **At tick 0 the predictor produces only 6 distinct predictions,** one per starting template. Each is shared by 90 states, with the same winning candidate and the same default (id 100).
- **The model can't see the cause of the difference:** the hidden hazard time and compliance level that decide the outcome are not visible to the 26 admissible features at tick 0.
- **So any rule that uses only the predictions switches all 90 states of a group or none of them.**
- **Even the least harmful group fails:** its winner is harmful in 20 of 90 states. Switching that group alone gives 20 / 1,845 = 1.08% harm (95% CI 0.65 to 1.59%), which already misses the limit. Every other group is worse (27 to 41 harmful of 90).

| Group (winner id) | States | Harmful | Captured |
|---|---|---|---|
| 257 | 90 | 20 | 31 |
| 283 | 90 | 27 | 21 |
| 237 | 90 | 35 | 10 |
| 249 | 90 | 35 | 8 |
| 293 | 90 | 40 | 12 |
| 178 | 90 | 41 | 20 |

## 1. Predicted versus realized margins (the 540 R1 switches)

**Medians over the switches:**

| Target | Predicted improvement | Interval width (q0.9 − q0.1) | Realized improvement, mean |
|---|---|---|---|
| T90 | −0.17 (all between −0.41 and −0.085) | 2.37 | +0.50 |
| T100 | −0.01 | 2.97 | +0.34 |
| Unmet | −3.59 | 90.9 | +1.66 |
| Hazard | +7.80 | 16.2 | +13.22 |

- **Every switch is decided on hazard.**
- **Harmful and harmless switches get identical predictions but different outcomes.**
  - Mean realized improvement, harmful vs harmless: T90 −0.68 vs +1.18; unmet −7.9 vs +7.2; hazard +6.1 vs +17.3.
  - The predicted distributions are the same for both groups.
- **Which criterion makes a switch harmful** (the first one worse than the default): T90 126, T100 39, unmet 33, hazard 0.

## 2. The 0.5-tick issue

**Switches by predicted T90 improvement:** all 540 fall in the **≤ 0** bucket.

- The model predicts every one of these candidates to be slightly *worse* on T90, by 0.085 to 0.41 ticks, which is always inside the 0.5 tolerance.
- Realized T90 differences are integers: −2 in 154, −1 in 96, 0 in 164, +1 in 117, +2 in 9.
- So 70% of switches differ on T90 by at least one whole tick while the prediction says "tie".

**Switches by predicted T100 improvement:**

| Bucket (ticks) | Switches | Harmful | Harm rate (95% CI) | Actual T90 diff (CI) | Actual T100 diff (CI) | Actual unmet diff (CI) |
|---|---|---|---|---|---|---|
| ≤ 0 | 270 | 116 | 43.0% (37.1 to 48.9) | −0.52 (−0.66 to −0.36) | −0.30 (−0.41 to −0.20) | −0.59 (−1.73 to 0.52) |
| (0, 0.25] | 180 | 55 | 30.6% (23.9 to 37.8) | −0.56 (−0.74 to −0.38) | −0.22 (−0.33 to −0.11) | −2.99 (−3.87 to −2.03) |
| (0.25, 0.5] | 90 | 27 | 30.0% (21.0 to 39.6) | −0.32 (−0.46 to −0.18) | −0.68 (−0.81 to −0.54) | −2.23 (−4.23 to −0.16) |
| (0.5, 0.75] | 0 | | | | | |
| > 0.75 | 0 | | | | | |

**The 126 T90-harmful switches:**
- All are in the ≤ 0 T90 bucket. Predicted T90 difference ranges from +0.085 to +0.41 ticks (median +0.21).
- Realized T90 is exactly 1 tick worse in 117 and 2 ticks worse in 9.
- By predicted T100 bucket: 65 in ≤ 0, 49 in (0, 0.25], 12 in (0.25, 0.5].
- Actual T90 difference by bucket: +1.14, +1.00 and +1.00 ticks.

**Is the 0.5-tick tolerance masking a one-tick integer outcome?** Yes. But the predicted direction has no information about the realized direction: the AUC for predicting harm from predicted T90 improvement is 0.499. The model says "slightly worse" for all 540, while 250 are actually better and 126 actually worse. A tighter tolerance cannot fix this. It could only stop switching altogether, which is what the per-target T90 rule in §5 does.

## 3. Margin versus uncertainty: separation of harm

**Among the 540 switches, harmful vs harmless:**

| Margin | AUC range across the 4 targets | \|Cohen's d\| |
|---|---|---|
| Predicted improvement | 0.43 to 0.50 | ≤ 0.26 |
| q0.9 candidate − q0.9 default | 0.46 to 0.51 | ≤ 0.15 |
| Interval width | 0.46 to 0.51 | ≤ 0.18 |
| Improvement / width | 0.43 to 0.52 | ≤ 0.27 |
| Default mean − candidate q0.9 | 0.43 to 0.52 | ≤ 0.16 |

- **Strongest single margin:** predicted unmet improvement, with oriented AUC 0.573. It is weak, and it is limited by the 6-group structure above.
- **Captured vs improvable-but-not-captured** shows larger effect sizes (|d| up to 1.0), but it compares different sets of states: the uncaptured group includes improvable states where nothing switches. It is not evidence of within-switch separation.
- **No simple observable margin separates harmful from harmless switches.**

## 4. Fixed grid: extra margin on the switch criterion (hazard)

Rule: the switch requires `I_H > δ_H + g·δ_H`.

| g | Switches | Captured (of 753) | Harmful | Harm rate (95% CI) | Harm among switches | Paired Δ T90 / T100 / unmet / hazard |
|---|---|---|---|---|---|---|
| 0 | 540 | 102 | 198 | 10.7% (9.5 to 12.0) | 36.7% | −0.146 / −0.099 / −0.49 / −3.87 |
| 0.5 | 360 | 61 | 130 | 7.0% (6.0 to 8.1) | 36.1% | −0.099 / −0.052 / −0.37 / −2.71 |
| 1.0 | 90 | 8 | 35 | 1.9% (1.3 to 2.6) | 38.9% | −0.016 / −0.017 / −0.07 / −0.79 |
| 1.5 to 4 | 0 | 0 | 0 | 0 | – | 0 |

A larger margin reduces the *number* of switches but not the *share* that is harmful (36 to 39% at every g).

**Also run:** requiring predicted non-inferiority `I_k ≥ g·δ_k` on every criterion except the switching one gives **0 switches at every g**, because all 540 are predicted slightly worse on T90.

## 5. Per-target non-inferiority (`I_k ≥ g·δ_k` on one target at a time)

| Target | g | Switches | Captured | Harmful | Harm rate (95% CI) | Harm among switches |
|---|---|---|---|---|---|---|
| T90 | any | 0 | 0 | 0 | 0 | – |
| T100 | 0 | 270 | 62 | 82 | 4.4% (3.6 to 5.4) | 30.4% |
| T100 | 0.5 | 90 | 21 | 27 | 1.5% (0.97 to 2.0) | 30.0% |
| T100 | ≥ 1 | 0 | 0 | 0 | 0 | – |
| Unmet | 0 | 90 | 31 | 20 | 1.08% (0.65 to 1.59) | 22.2% |
| Unmet | ≥ 0.5 | 0 | 0 | 0 | 0 | – |
| Hazard | 0 to 1 | 540 | 102 | 198 | 10.7% | 36.7% |
| Hazard | 1.5 / 2 | 360 / 90 | 61 / 8 | 130 / 35 | 7.0% / 1.9% | 36.1% / 38.9% |

**Is one target driving the false-safe decisions?** Yes, T90: 126 of the 198 harmful switches, all predicted within the tolerance.
- **Prediction uncertainty:** the T90 interval is about 2.4 ticks wide.
- **Selector tolerance:** 0.5 tick.
- **Outcome resolution:** 1 tick.

The model's uncertainty is about five times the tolerance, and the outcome moves in whole ticks. So "within tolerance" on T90 carries no information about the realized ordering.

## 6. Integer-tick diagnostic (not an official change)

**Edge cases like 13.9999998:**
- 163,984 of 549,810 VALIDATION T90 q0.9 predictions (30%) sit within 1e-4 below an integer.
- In 76,001 of those rows the actual outcome equals that integer, so the outcome is "uncovered" only by floating-point.
- Reading q0.9 as `ceil(q9 − 1e-6)` raises coverage:

| Target | Raw coverage | Integer-read coverage | Nominal |
|---|---|---|---|
| T90 | 0.746 | **0.959** | 0.90 |
| T100 | 0.832 | **0.969** | 0.90 |

The T90/T100 under-coverage reported in the earlier audit is mostly this integer effect; read on integers, both quantiles over-cover slightly.

**Integer rules:**

| Rule | Switches | Same winner as R1 | Captured | Harmful | Harm rate (95% CI) | T90-harm cases (of 126): stop switching / switch but not harmful / still harmful |
|---|---|---|---|---|---|---|
| Integer in steps 2 to 4 | 540 | 180 | 83 | 173 | 9.4% (8.2 to 10.6) | 0 / 17 / 109 |
| Integer in step 4 only | 270 | 270 | 62 | 82 | 4.4% (3.6 to 5.4) | 65 / 0 / 61 |

- Integer handling removes some harm, but it doesn't change the picture: 30 to 32% of switches are still harmful, and both rules miss the limit by more than four times.
- Captured improvements drop from 102 to 83 or 62.

## 7. Can the 1% limit be reached from the existing predictor?

**No, quantitatively.**

- **Across every fixed rule tested** (§4, the "also run" rule, §5 and §6), no rule with at least one switch meets the ≤ 1% upper-CI requirement.
  - The closest is unmet non-inferiority at g = 0: 90 switches, 31 captured, 20 harmful, 1.08% (CI 0.65 to 1.59%).
- **This is not just a limit of the grid.** Every R1 switch is at tick 0, where the predictor gives 6 distinct predictions shared by 90 states each. Any deterministic rule on these outputs, with any threshold, switches whole groups, and the least harmful group already gives 1.08%.
- **The harmful share among switches never goes below 22%.** Rules approach 1% only by switching less, never by choosing safer switches.
- **One boundary I have not tested:** choosing a candidate other than the R1 winner within a group. That would need a separate, label-free rule to justify it, and nothing in this diagnostic supports one.

**Implication:** reaching 1% while still switching would need information the tick-0 features don't carry. That means a new model, new features or data (for example, decisions after the hazard becomes observable), or a different decision point. A different margin on the existing predictions won't do it.

## Integrity

- All 33 protected file hashes were identical before and after the run. `selector_design.json` is `8a41c777…`; the full list is in the JSON.
- All 16 C3 model hashes were verified against `training_manifest.json`.
- No selector code change, retraining, recalibration or amendment.
- PRIMARY TEST not accessed; other held-out sets were read only as `split_role`.

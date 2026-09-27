# Decision-Tick Observability Diagnostic (VALIDATION only)

- **Numbers:** `decision_tick_observability_diagnostic.json` (same folder)
- **Script:** `scripts/diagnose_decision_tick_observability.py`
- **Follows:** `SELECTOR_MARGIN_DIAGNOSTIC.md` (commit ce30dd3)

**Question:** is the selector failure a tick-0 observability limit, or a failure of the outcome-prediction approach itself? And is a later decision point a defensible next experiment?

**Status:**
- Read-only.
- No retraining, no new feature, no selector, tolerance or threshold change, no new data.
- PRIMARY TEST and every other held-out set remain sealed.

**Hidden labels:**
- `t_h`, `p`, `compliance_key`, `template` and `hazard_started` were read for VALIDATION rows as diagnostic labels only. The models received exactly the frozen 54 columns.
- **Hidden regime** means (`t_h`, `p`): 9 combinations in VALIDATION. The finer version adds `compliance_key`: 54 combinations.

**Decision ticks present in VALIDATION:** 0, 5, 10, 15, 20.

## Headline

**Neither framing fully holds. The limit is timing: the information arrives after the opportunity has passed.**

- **Actionable opportunity is concentrated at ticks 0 and 5.** Improvable states: 430 → 314 → **8** → 1 → 0.
- **Observability and prediction accuracy arrive at tick 10:** 76% of regime uncertainty is resolved and R² is about 0.97 on all four targets.
- **By tick 10 the default is already best** in 532 of 540 states.
- **The model can predict when the state carries the information.** Its poor accuracy at ticks 0/5 comes from missing information, not model capacity.
- **Tick 0 is not purely information-limited for *safe* action.**
  - In-sample, 4 of the 6 observable classes have a fixed action that is never worse than the default for any of their 90 members.
  - Those actions strictly improve 244 states.
  - The R1 floor admits all of them, but the switch rule never picks them, because their predicted hazard gain (≤ 4.0) is below δ_H = 4.77.
- **Tick 5 is the worst decision point:** no observable class has a zero-harm improving action at all.

## 1. Observable-state diversity by tick

| Tick | States | Unique 26-feature inputs | Unique 54-column inputs | Distinct prediction matrices | Hidden regimes (t_h, p) / with compliance | Hazard started |
|---|---|---|---|---|---|---|
| 0 | 540 | **6** | 1,788 | **6** | 9 / 54 | 0 |
| 5 | 540 | 10 | 2,980 | 10 | 9 / 54 | 120 |
| 10 | 540 | 251 | 74,798 | 251 | 9 / 54 | 480 |
| 15 | 221 | 52 | 15,496 | 50 | 8 / 46 | 221 |
| 20 | 4 | 4 | 1,192 | 4 | 4 / 4 | 4 |

- 298 candidate descriptors are all distinct, so unique 54-column inputs = unique states × 298.
- The collapse to 6 predictions ends at tick 10.

## 2. Hidden-regime ambiguity by tick

| Tick | States in classes with several regimes | Classes with several regimes | States whose regime is identified | Regime uncertainty resolved | States in classes whose true outcomes differ | States in classes with no common best action |
|---|---|---|---|---|---|---|
| 0 | 100% | 6 of 6 | 0% | 0% | 100% | 100% |
| 5 | 100% | 10 of 10 | 0% | 24% | 100% | 100% |
| 10 | 47% | 43 of 251 | 53% | **76%** | 37% | **0.7%** |
| 15 | 89% | 29 of 52 | 11% | 38% | 4% | 0% |
| 20 | 0% | 0 of 4 | 100% | 100% | 0% | 0% |

- "Regime uncertainty resolved" is 1 − H(regime | observation) / H(regime).
- **Tick 15 is not a regression.** Most tick-15 classes mix regimes whose outcomes are already identical, because the evacuation has finished.

**Does the next observation separate states that share a class but differ in regime?** From tick 0 to tick 5, 39% of those pairs are separated. From tick 5 to tick 10, 93.5% are.

**Answer to the key question:** the observable state begins to separate the tick-0 regimes at **tick 10**, after the hazard becomes visible (480 of 540 states have started by then). Tick 5 separates only a quarter of the uncertainty.

## 3. Prediction diversity, error and separability by tick

**R² (all candidate rows):**

| Tick | T90 | T100 | Unmet | Hazard |
|---|---|---|---|---|
| 0 | 0.11 | 0.11 | 0.18 | 0.28 |
| 5 | −0.23 | −0.23 | 0.14 | 0.10 |
| 10 | 0.98 | 0.97 | 0.98 | 0.98 |
| 15 | 1.00 | 0.99 | 0.97 | 0.97 |

**Mean absolute error:**

| Tick | T90 | T100 | Unmet | Hazard |
|---|---|---|---|---|
| 0 | 0.83 | 0.85 | 13.6 | 12.7 |
| 5 | 1.00 | 1.05 | 17.7 | 18.9 |
| 10 | 0.11 | 0.19 | 0.82 | 0.90 |
| 15 | 0.01 | 0.03 | 0.07 | 0.07 |

**Ranking of candidates within a state** (correlation of predicted vs actual deviation from the state mean):

| Tick | T90 | T100 | Unmet | Hazard |
|---|---|---|---|---|
| 0 | 0.64 | 0.71 | 0.78 | 0.86 |
| 5 | 0.60 | 0.67 | 0.79 | 0.85 |
| 10 | 0.96 | 0.95 | 0.98 | 0.98 |

**Direction AUC** (predicted improvement vs realized better-than-default, non-default pairs):

| Tick | T90 | T100 | Unmet | Hazard |
|---|---|---|---|---|
| 0 | 0.89 | 0.93 | 0.83 | 0.995 |
| 5 | 0.76 | 0.75 | 0.76 | 0.999 |
| 10 | 0.93 | 1.00 | 0.75 | no pair is better |

**Harm separability** (AUC of the R1 floor margin vs pair harm): 0.84, 0.87, **0.99** and **1.00** at ticks 0, 5, 10 and 15.

- Harmful and harmless candidates become almost perfectly separable at tick 10.
- At ticks 0 and 5, members of a class have identical inputs and therefore identical predictions, while their true outcomes differ. At tick 0 that is 100% of states. No model on these features can resolve it.

## 4. Frozen and R1 selectors by tick

| Tick | Improvable | Frozen: switches / harm | R1: switches | R1: captured | R1: harmful | R1: harm among switches (CI) | OOD | Abstentions | Agreement with default (R1) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 430 | 0 / 0 | 540 | 102 | 198 | 36.7% (32.6 to 40.6) | 0 | 0 | 0% |
| 5 | 314 | 0 / 0 | 0 | 0 | 0 | – | 0 | 0 | 100% |
| 10 | 8 | 0 / 0 | 0 | 0 | 0 | – | 4 | 0 | 100% |
| 15 | 1 | 0 / 0 | 0 | 0 | 0 | – | 0 | 0 | 100% |
| 20 | 0 | 0 / 0 | 0 | 0 | 0 | – | 0 | 0 | 100% |

**Why R1 does not switch at tick 5:**
- Its winner's predicted hazard improvement is small: median 0.16 and 95th percentile 1.2, far below δ_H = 4.77.
- The realized best improvement on hazard is real: median 2, 75th percentile 10, 95th percentile 16.
- The model compresses candidate differences at tick 5: the predicted spread of hazard across candidates within a state is 11.2, against 15.9 actual.

**Realized best improvement over the default** (95th percentile, all targets): at ticks 10 and above it is **0**.

## 5. Counterfactual information test

**Within classes of identical observations that mix hidden regimes:**

| Tick | Ambiguous classes | Share with a common best action | Share with a fixed action never worse *and* better for some member | Share where the R1 winner is safe for every member |
|---|---|---|---|---|
| 0 | 6 | 0% | 67% | 0% |
| 5 | 10 | 0% | **0%** | 70% (it never switches there) |
| 10 | 43 | 98% | 0% | 100% |
| 15 | 29 | 100% | 0% | 100% |

- Members of a class share one prediction.
- At tick 5, none of the observable classes offers any action that improves someone without harming someone else in the same class.

**Zero-harm upper bound** (label-based, in-sample, not a rule). For each class, take the single fixed non-default action that is never worse than the default for any member and improves the most members:

| Tick | Classes with such an action | States strictly improved | States captured | Does R1 admit it? | Its predicted hazard improvement |
|---|---|---|---|---|---|
| 0 | 4 of 6 | **244** of 540 | 14 | yes, 4 of 4 | median 2.5, max about 4.0 (< δ_H 4.77) |
| 5 | 0 of 10 | 0 | 0 | – | – |
| 10 | 6 of 251 | 6 | 6 | 0 of 6 | predicted worse (median −7.8) |
| 15 | 1 of 52 | 1 | 1 | 0 of 1 | predicted worse |

**Caveats:**
- The bound uses VALIDATION labels and is in-sample. It says what an observation-based deterministic rule *could* achieve with zero harm on these 540 trajectories. It does not show that such a rule would generalize.
- The PRIMARY regimes (t_h 4/8, p 0.6/0.85) are not represented.
- "Never worse" follows the frozen lexicographic key. A safe action may trade hazard for an earlier-criterion gain.

## 6. Earliest potentially viable decision tick

- **Tick 0:**
  - *Information:* not separated.
  - *Accuracy:* low cross-regime (R²).
  - *Opportunity:* high (430 improvable).
  - *Outcome:* R1 gives 36.7% harm among switches. A zero-harm improvement exists in-sample (244 improved), but the selector's δ_H switch rule never picks it.
- **Tick 5:**
  - *Information:* partly separated (24%).
  - *Accuracy:* lowest (negative R² on T90/T100).
  - *Opportunity:* moderate (314 improvable).
  - *Outcome:* there is no zero-harm improving action in any class, so no observation-based deterministic rule can improve anyone at tick 5 without harming someone in the same class.
- **Tick 10:**
  - *Information and accuracy:* both conditions are met (76% resolved, R² about 0.97, harm AUC 0.99).
  - *Opportunity:* essentially gone (8 improvable of 540; realized best gain 0 at the 95th percentile). The zero-harm bound is 6 states, and the model predicts those 6 actions as worse.
- **Ticks 15 and 20:** nothing left to improve.

**Result:** tick 10 is the earliest tick where both conditions hold, but it isn't viable, because there is almost nothing left to decide. No tick in the current dataset offers both information and opportunity.

## 7. Temporal availability versus model capacity

- **Model capacity is not the bottleneck.**
  - On the same frozen models, R² rises from 0.11–0.28 (tick 0) to 0.97–0.98 (tick 10).
  - Within-state ranking correlation rises from 0.64–0.86 to 0.95–0.98.
  - When the state contains the information, the model uses it.
- **At ticks 0 and 5 the state does not contain the information.** 100% of states sit in classes whose identical inputs have different true outcomes. That error cannot be removed with these features, by this model or any other.
- **The deciding constraint is timing.** In this simulator the information that separates the regimes (hazard onset and its capacity effect) becomes visible after the period in which a different action changes the outcome.

## Is later decision-making promising enough to justify a new experiment?

**Not as "the same experiment at a later tick".** With the current decision interval, a later-tick experiment would have almost nothing to improve: 8 improvable states at tick 10, versus 430 at tick 0.

**What the data does show, for the user's decision (none is chosen or recommended here):**
- **(a) Robust choice at tick 0.** In-sample, a zero-harm improving action exists for 4 of 6 tick-0 observation classes. The frozen/R1 switch rule prefers the action with the largest predicted hazard gain, which picks the riskier actions instead. Whether a label-free, pre-specified robustness criterion could find those actions, and generalize to unseen regimes, is untested.
- **(b) Finer decision timing.** The interval is 5 ticks. Between ticks 5 and 10, observability jumps from 24% to 76% while opportunity collapses from 314 to 8. Whether any intermediate tick has both is unknown, because the dataset has no decision states there. Answering it would need new data.
- **(c) Richer observations at ticks 0 and 5.** Features that carry hazard or compliance information earlier. That would mean new features and a new dataset, which is outside the current frozen specification.

## Integrity

- All 33 protected file hashes were identical before and after the run. All 16 model hashes were verified.
- Selector code unchanged; no retraining, no new features, no new data, no amendment.
- PRIMARY TEST not accessed; other held-out sets were read only as `split_role`.

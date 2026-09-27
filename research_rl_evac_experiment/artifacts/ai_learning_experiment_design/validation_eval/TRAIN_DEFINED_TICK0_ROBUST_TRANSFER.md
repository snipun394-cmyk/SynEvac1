# TRAIN-Defined Tick-0 Robust-Set Transfer to VALIDATION

- **Numbers:** `train_defined_tick0_robust_transfer.json` (same folder)
- **Script:** `scripts/audit_train_defined_tick0_robust_transfer.py`
- **Tests:** `tests/test_train_defined_tick0_robust_transfer.py`
- **Follows:** `TICK0_ROBUST_ACTION_AUDIT.md` (commit 0ada5ec)

**Status:**
- Read-only.
- No model loaded or trained, no selector, tolerance, dataset or protocol change, no policy, no amendment.
- PRIMARY TEST labels were not read: its 970,288 rows were seen only as `split_role`.

## Classification

**A. TRAIN-defined robust actions transfer to VALIDATION, including t = 6 and p = 0.8.**
- Every frozen TRAIN-robust action has **zero harm on every VALIDATION tick-0 state it applies to**: 360 states in 4 classes, across all 9 VALIDATION regimes.
- None of those 9 regimes appears in TRAIN.

**Is this enough to proceed toward a policy? No.**
- The evidence supports robustness *transfer*. It does not support a policy or a safety claim.
- Further read-only experiments are needed first; see "What this does not establish".

## Method

1. **Phase 1, TRAIN only.**
   - The dataset is streamed, keeping only TRAIN rows at decision tick 0: 1,800 states, 300 per class.
   - Per tick-0 observable class, the frozen label-free default is `BASE=P` (id 100) in every class.
   - The **TRAIN-robust set** contains the non-default actions whose realized key is never worse than the default for *any* TRAIN member of the class, and strictly better for at least one. It uses the exact frozen lexicographic key (T90 → T100 → future unmet → hazard-edge unmet), with no δ.
   - The sets were frozen and hashed (`24c7a7c5c55167dcd6903f076607ba47fbcccba9344b8e3111178a58902a4f55`) **before any VALIDATION row was read**.
2. **Phase 2, VALIDATION.**
   - A second stream keeps only VALIDATION tick-0 rows (540 states).
   - Every frozen action is evaluated unchanged. VALIDATION outcomes never define, rank, select or alter an action.
   - Because no tie-break is pre-specified, every action in a set is reported. The conservative set-level figure counts a state as harmed if *any* action in the set would harm it.

**TRAIN regimes:** t_h ∈ {none, 3, 5, 7, 9} × p ∈ {0.5, 0.7, 0.9}, 20 states per regime per class.

**VALIDATION regimes:** t_h = 6 × p ∈ {0.5, 0.7, 0.8, 0.9}, plus p = 0.8 × t_h ∈ {none, 3, 5, 7, 9}, 10 states per regime per class.
- **None of the 9 VALIDATION regimes is seen in TRAIN.**
- The "all other VALIDATION regimes" group is therefore **empty**. That is the design of the split, not an omission.

## 1. TRAIN-derived robust sets (frozen)

| Class (template) | TRAIN states | Weak robust (non-default) | Of which outcome-identical to default | TRAIN-robust improving | Distinct TRAIN outcome patterns | Tie-break needed? | Tie-break changes TRAIN outcomes? |
|---|---|---|---|---|---|---|---|
| D0_sym | 300 | 33 | 19 | **14** | 2 | yes | yes |
| D1_east_heavy | 300 | 43 | 33 | **10** | 1 | yes | no |
| D2_west_heavy | 300 | 43 | 33 | **10** | 1 | yes | no |
| D3_first_room_heavy | 300 | 35 | 19 | **16** | 3 | yes | yes |
| D4_second_room_heavy | 300 | 53 | 53 | **0** | – | – | – |
| D5_R1_concentrated | 300 | 33 | 33 | **0** | – | – | – |

**Exact frozen sets** (candidate ids, grouped by identical TRAIN outcome pattern):

| Class | Pattern | Actions | TRAIN members improved (of 300) |
|---|---|---|---|
| D0 | R1=B family | 102, 121, 122, 133, 134, 137, 138 | 157 |
| D0 | R3=B family | 108, 160, 162, 181, 182, 185, 186 | 159 |
| D1 | R2=B family | 104, 116, 118, 141, 142, 153, 154, 157, 158, 277 | 126 |
| D2 | R4=B family | 110, 164, 166, 176, 178, 189, 190, 193, 194, 217 | 127 |
| D3 | R2=B family | 104, 141, 142, 153, 154, 157, 158 | 123 |
| D3 | R4=B family | 110, 164, 166, 189, 190, 193, 194 | 126 |
| D3 | {150 `P\|R2=B\|R4=B`, 225 `B\|R1=P\|R3=P`} | 150, 225 | 157 |

- **A common robust action exists** in D0 to D3.
- **No TRAIN-robust improving action exists** in D4 or D5. Their weakly robust actions all behave exactly like the default.
- **Matches the earlier audit:** the TRAIN sets are identical to the VALIDATION in-sample sets found in 0ada5ec (full overlap: 14, 10, 10, 16). That comparison is descriptive and was not used to build anything.

## 2. VALIDATION results for the frozen sets

**Per class, all VALIDATION tick-0 states** (per action; Δ = selected − default, negative = better):

| Class | States | Improved (range over actions) | Harmed | Tied | Mean Δ T90 | Mean Δ T100 | Mean Δ unmet | Mean Δ hazard | Worst observed key (default's worst) | Zero harm survives |
|---|---|---|---|---|---|---|---|---|---|---|
| D0 | 90 | 65 to 67 | **0** | 23 to 25 | −0.24 to −0.26 | −0.17 to −0.18 | −1.97 to −1.99 | −4.64 to −4.70 | (14, 16, 78, 36) (same) | **yes, every action** |
| D1 | 90 | 52 | **0** | 38 | 0.00 | −0.09 | −1.40 | −2.28 | (14, 15, 77, 28) (default (14, 16, 79, 30)) | **yes, every action** |
| D2 | 90 | 59 | **0** | 31 | 0.00 | −0.06 | −1.56 | −2.57 | (14, 15, 77, 28) (same) | **yes, every action** |
| D3 | 90 | 53 to 66 | **0** | 24 to 37 | −0.06 to −0.33 | −0.02 to −0.11 | −1.03 to −2.20 | −2.33 to −4.92 | (14, 15, 75, 31) (default (14, 15, 76, 32)) | **yes, every action** |
| D4 | 90 | – | – | – | – | – | – | – | default retained | n/a |
| D5 | 90 | – | – | – | – | – | – | – | default retained | n/a |

The worst case under every robust action is never worse than the default's, and in D1 and D3 it is strictly better.

**By regime group** (states under a TRAIN-robust action; harmed = by *any* action in the set):

| Group | Unseen in TRAIN | Acted states | Harmed | Improved by every action | Harm upper bound, one-sided 95% |
|---|---|---|---|---|---|
| All VALIDATION tick-0 | all | 360 | **0** | 221 | 0.83% |
| t_h = 6 (all p) | timing | 160 | **0** | 117 | 1.85% |
| p = 0.8 (all t_h) | compliance | 240 | **0** | 140 | 1.24% |
| t_h = 6 and p = 0.8 | both | 40 | **0** | 36 | 7.2% |
| t_h = 6, p ∈ {0.5, 0.7, 0.9} | timing only | 120 | **0** | 81 | 2.5% |
| p = 0.8, t_h ∈ {none, 3, 5, 7, 9} | compliance only | 200 | **0** | 104 | 1.5% |
| Neither unseen | – | 0 | – | – | – |

**Per regime and class:**
- 10 states each, harmed = 0 in every one of the 36 acted cells. The upper bound for 0 of 10 is 25.9%.
- The robust actions improve outcomes only when the hazard starts at t_h ∈ {3, 5, 6, 7}. For t_h = 9 and no hazard they tie the default in every state.

## 3. The unseen conditions (item 9)

- **Hazard at t = 6:** 160 acted states, 0 harmed, for every action in every class set.
- **Compliance p = 0.8:** 240 acted states, 0 harmed.
- **Both at once:** 40 acted states, 0 harmed.
- **Conclusion:** the TRAIN-derived robust actions **remain robust** on VALIDATION's unseen hazard time and compliance level.

## 4. Coverage (item 10)

| | Share of the 540 VALIDATION tick-0 states |
|---|---|
| At least one TRAIN-robust improving action | 66.7% (360; classes D0 to D3) |
| No such action | 33.3% (180; D4 and D5) |
| Multiple TRAIN-robust actions | 66.7% (every covered state) |
| Multiple distinct TRAIN outcome patterns (the tie-break changes the outcome) | 33.3% (D0 and D3) |

## 5. Finite-sample limitation

- **These bounds are descriptive only.** They use the exact one-sided 95% Clopper–Pearson upper bound and treat states as independent draws. **This is not a population-level safety claim.**
- **VALIDATION is a fixed design**, not a random sample: 9 chosen regimes × 10 seeds × 4 templates. The effective number of independent conditions is far smaller than 360, so the true uncertainty is larger than the bounds suggest.
- **The frozen ≤ 1% endpoint** would need at least 299 independent zero-harm decisions in *each* test family. Only the pooled 360 reaches that. No single unseen condition does: t_h = 6 has 160 and t_h = 6 with p = 0.8 has 40.

## What this does not establish

- **Extrapolation.** Both unseen VALIDATION values lie inside the TRAIN range (6 between 5 and 7; 0.8 between 0.7 and 0.9). Timing or compliance outside that range (the extrapolation split: t_h = 10, p = 0.95) was not tested.
- **New templates or perturbed starts.** The tick-0 classes are exact starting templates. A new template (the zero-shot split) or any perturbation of the start has no class, so no TRAIN set applies. This finding covers exactly these six starting states.
- **A decision rule.** Robust sets exist for only 4 of 6 classes. There is no pre-specified tie-break among robust actions, and it matters for D0 and D3. There is also no label-free mechanism, other than a template lookup that was not requested, for reaching the robust set at decision time.
- **Magnitude.** The gains are modest: mean hazard 2.3 to 4.9, unmet 1.0 to 2.2, T90 at most 0.33. The oracle-best outcome is captured in few states. R1's much larger hazard reductions came with 36.7% harm.

**Possible next read-only experiments, not chosen here:**
- A TRAIN-defined robust-set check on the extrapolation split. It is a diagnostic role, not PRIMARY, and would need your explicit authorization, since it is a held-out split.
- A pre-registered tie-break rule defined on TRAIN only.
- A read-only check of whether a label-free *model-based* robustness criterion reproduces these TRAIN sets. The earlier audit showed the current predictor cannot certify robustness in D5.

## Integrity

**Before running:**
- Dataset SHA-256 `36aa757f…` verified.
- All 33 protected protocol/audit file hashes verified against the evaluation record.
- All 16 C3 model hashes verified against `training_manifest.json`.
- The tick-0 audit artifacts from 0ada5ec are unchanged; they are included in the artifact-tree hash.

**Role separation:**
- TRAIN and VALIDATION trajectory ids and seeds are disjoint (asserted).
- The robust sets were built from TRAIN rows only, and hashed before the VALIDATION stream was opened.
- The loader never parses fields of other roles; a test proves it on a malformed PRIMARY row.

**After running:**
- All 100 files under `artifacts/ai_learning_experiment_design/` were hashed before and after (the before-hashes are in the JSON).
- 0 files changed, apart from this audit's own outputs.

**Tests:** 142 passed: the 14 new tests plus the existing pre-training, specification, label-free-default, pilot and full-dataset suites.

# Cross-Topology Generalization Investigation

Status: **INVESTIGATION ONLY.** No Model V4 trained/exported, nothing wired
into `LiveRuntime`, Recommendation, Guidance, Advisory, Voice, Dynamic
Signage, or Decision Policy, no new Designer asset, no GNN, no
hyperparameter search. Target V2 (`v2-persistent-demand-service-imbalance`)
and Dataset V3 (`data/predictive_dataset_campaign_v3`) are read-only
inputs, never modified. Every experiment below reuses the SAME XGBoost
configuration (`predictive_model.tree_models.build_tree_models`) every
prior milestone selected — this is a causal-comparison investigation, not
a leaderboard.

Follows [[predictive_dataset_campaign_v3_milestone]] (commit `ecf3f9b`,
verdict B): structural diversity (16 variants across 4 families)
dramatically improved unseen-*variant* generalization but did **not**
improve unseen-*family* generalization, which regressed 6-17% relative to
[[localized_predictive_model_v3_1_robustness_milestone]]'s own Dataset-V2
family-holdout numbers. This milestone investigates **why**, distinguishing
four candidate causes the charter names explicitly: (A) insufficient
family diversity, (B) absolute-scale feature shift, (C) concept shift
between families, (D) missing graph context — rather than assuming "more
data" is the answer.

New code: `predictive_dataset/experimental_features_v4.py` (Phase 6
normalized-ratio + Phase 8 graph-context experimental extractors, additive
only, never touches `predictive_dataset/schema.py`) and
`scripts/model_v4_cross_topology_investigation.py` (all computational
phases, one pass over Dataset V3, one JSON report at
`data/cross_topology_generalization_investigation/cross_topology_investigation_report.json`,
gitignored/regenerable). Total run: **1,220.9s (~20.3 min)** for ~54
XGBoost/LogisticRegression/HistGradientBoosting fits, peak memory
available never dropped below 411MB (this milestone's own tight ~7.9GB-RAM
environment).

## 1. Phase 1 — family-holdout reproduction

Reproduces Dataset V3's own Phase 18B (`predictive_model/topology_holdout.py`,
unmodified), with full metrics added:

| Held-out family | n test | pos. rate | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | FNR | Det. baseline PR-AUC | Relative lift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multi_exit_wide | 1,224,187 | 1.02% | 0.2540 | 0.8798 | 0.089 | 0.629 | 0.156 | 6.6% | 37.1% | 0.1082 | 2.35× |
| single_exit_lowrise | 138,476 | 4.74% | 0.5126 | 0.9198 | 0.237 | 0.789 | 0.364 | 12.7% | 21.1% | 0.1442 | 3.56× |
| twin_stair_highrise | 365,148 | 4.63% | 0.2858 | 0.6906 | 0.106 | 0.474 | 0.173 | 19.5% | 52.6% | 0.1168 | 2.45× |
| v1_topology_fixed | 229,860 | 3.07% | 0.4969 | 0.9265 | 0.161 | 0.873 | 0.272 | 14.4% | 12.7% | 0.1802 | 2.76× |

Matches Dataset V3's own committed numbers (0.254 / 0.513 / 0.286 / 0.497)
almost exactly (float32-vs-float64 and thread-scheduling noise only) —
**the failure is confirmed reproduced, not an artifact of this
milestone's own pipeline.** ML beats the deterministic baseline by
2.3×-3.6× in every holdout, even at its worst. `twin_stair_highrise` is
the qualitatively different failure: its ROC-AUC (0.691) is far below the
other three (0.88-0.93), and its FNR (52.6%) is 2-4× the others' — it
isn't just "lower PR-AUC", its ranking quality itself degrades, unlike
the other three families which mostly over-alarm (moderate FPR, tolerable
FNR).

## 2. Phase 2 — feature distribution shift (covariate shift) + identifiability

Standardized mean difference (SMD), max across any family vs. the rest:

| Feature | max SMD |
|---|---|
| candidate_walking_distance | 1.202 |
| total_active_occupant_count | 1.091 |
| candidate_adjacent_zone_occupancy | 1.064 |
| candidate_alternative_route_count | 0.715 |
| candidate_recent_flow_rate | 0.472 |
| **candidate_queue_length** | **0.059** |

Same ranking shape V3.1 found on Dataset V2 (`candidate_queue_length`
nearly invariant; structural/scale features shift hardest) — **still true
at Dataset V3's broader structural diversity.** `candidate_type`
distribution also shifts hard by design: `multi_exit_wide` has **zero**
Stair candidates (0%/53.1% Exit/46.9% Door split); the other three all
carry a real Stair share (6.5%-32.2%).

**Family classifier accuracy: 98.4%** (macro, scenario-level held-out
test) vs. a 62.9% majority baseline. Slightly *higher* than V3.1's Dataset
V2 number was on the OLD 4-fixed-graph dataset in relative terms (V3.1:
100.0% on 4 fixed graphs) is not directly comparable since V3 spans 16
structural shapes per family — but the headline finding is unchanged:
**topology family remains almost perfectly recoverable from the
canonical feature vector**, even after Dataset V3's structural
diversification.

## 3. Phase 3 — conditional target-rate analysis: covariate shift vs. concept shift

This is the milestone's most important, previously-uninvestigated finding.
Conditional positive rate `P(target=1 | feature bin)`, restricted to bins
with ≥30 rows in ≥2 families:

| Feature | Max spread across families (same bin) | Covariate shift (Phase 2 SMD) |
|---|---|---|
| candidate_queue_length | 0.075 | 0.059 (low) |
| candidate_walking_distance | 0.063 | 1.202 (high) |
| total_active_occupant_count | **0.345** | 1.091 (high) |
| candidate_recent_flow_rate | **0.608** | 0.472 (moderate) |

Two genuinely different regimes, not one:

- **`candidate_walking_distance`: covariate shift WITHOUT concept shift.**
  Its marginal distribution differs enormously by family (SMD 1.20), but
  at matched walking-distance bins the conditional congestion rate is
  nearly identical across families (spread 0.063). A pure rescaling
  problem — exactly what Phase 6's normalization experiment targets.

- **`total_active_occupant_count` and `candidate_recent_flow_rate`: BOTH
  covariate shift AND concept shift.** At `total_active_occupant_count`
  HIGH (16-30), positive rate is 0.6% in `multi_exit_wide` vs. 19.6% in
  `single_exit_lowrise` — a **35× difference at the same occupancy
  level.** At `candidate_recent_flow_rate` >10, `twin_stair_highrise`
  hits 62.7% positive vs. 1.9%-10.1% in the other three — the SAME raw
  flow-rate reading means a qualitatively different thing structurally in
  a 3-floor twin-stair building than in a 1-floor multi-exit hub. **No
  amount of rescaling removes this — the underlying relationship between
  the feature and the outcome genuinely differs by family.** This is real
  concept shift, confirmed quantitatively, not asserted.

`candidate_queue_length` is stable on BOTH axes (low covariate shift, low
concept shift, near-saturating rate once queue≥1 in every family) — it is
the one feature the model can trust unconditionally across topology.

## 4. Phase 4 — failure-region analysis

Every held-out family's false positives concentrate the same way: **RISING/UNKNOWN
trend** (FP rate 0.6-0.9 vs. 0.06-0.2 under STABLE), **multi-bottleneck
ticks** (FP rate 0.69-0.90 vs. 0.06-0.19 single-bottleneck), and **high
occupancy** (FP rate climbs monotonically with occupancy band in every
family, reaching 0.70-1.0 at VHIGH). False negatives concentrate
oppositely: **STABLE/FALLING trend**, **LOW occupancy**, and — the
sharpest, most family-specific signal — **high flow rate**:
`twin_stair_highrise`'s FNR climbs to 94.4% at `candidate_recent_flow_rate`
>10 (vs. 62.5%-69.6% for the other three at the same band), directly
matching §3's finding that this exact feature/family combination is where
the learned relationship is most wrong. `multi_exit_wide` additionally
shows a severe LOW-occupancy false-negative blind spot (70.2% FNR at
occupancy ≤5) not shared as strongly by the other families.

## 5. Phase 5 — candidate-type interactions (diagnostic only)

Door/Exit-only models roughly match the unified model's own per-type
breakdown (within a few points, sometimes slightly better, sometimes
slightly worse). **Stair-only models are dramatically WORSE than the
unified model's Stair slice in every family that has Stairs:**
`single_exit_lowrise` 0.523 (specific) vs. 0.637 (unified), `twin_stair_highrise`
0.321 vs. 0.447, `v1_topology_fixed` 0.125 vs. 0.339 — Stair alone has too
little positive-class data to train a competitive specialized model; the
unified architecture's shared learning across types is load-bearing for
the sparsest type, not a compromise. **`multi_exit_wide` — which has ZERO
Stair candidates at all — is still the single worst-performing family
overall**, driven by weak Exit-type prediction (PR-AUC 0.144, its own
worst type) in that family's specific wide-hub structure. **Conclusion:
the family-level failure is not reducible to "it's just the Stair
problem" — it is a genuine family-structural effect, present even in a
family that has no Stairs at all.** No case for replacing the unified
architecture; the evidence points the other way.

## 6 & 8. Normalization and graph-context experiment (A/B/C/D)

Controlled comparison at every family holdout, same XGBoost config, same
splits — A (canonical) is Phase 1's own fit, not refit:

| Held-out family | A: canonical | B: +normalized | C: +graph-context | D: +both |
|---|---|---|---|---|
| multi_exit_wide | 0.2540 | 0.2863 (+12.7%) | 0.2823 (+11.1%) | 0.2884 (+13.5%) |
| single_exit_lowrise | 0.5126 | 0.4973 (−3.0%) | 0.5163 (+0.7%) | 0.4900 (−4.4%) |
| twin_stair_highrise | 0.2858 | 0.3089 (+8.1%) | 0.2985 (+4.4%) | 0.3056 (+6.9%) |
| v1_topology_fixed | 0.4969 | 0.4659 (−6.2%) | 0.5038 (+1.4%) | 0.4962 (−0.1%) |
| **Average** | **0.3873** | **0.3896 (+0.6%)** | **0.4002 (+3.3%)** | **0.3951 (+2.0%)** |

Two disclosed, different results, not one blended verdict:

- **Normalization (B)** is a mixed bag — exactly what V3.1's own relational-feature
  experiment found on Dataset V2 (a small net-neutral effect masking real wins on
  some families and real losses on others). Helps the two HARD families
  (`multi_exit_wide` +12.7%, `twin_stair_highrise` +8.1%) but actively hurts the
  two easier ones (`single_exit_lowrise` −3.0%, `v1_topology_fixed` −6.2%). Given §3's
  finding that `total_active_occupant_count`/`candidate_recent_flow_rate` carry
  real CONCEPT shift, this is expected: rescaling a feature whose relationship to
  the target itself differs by family cannot fully fix it, and can even blur signal
  the raw scale carried for families where the raw scale WAS informative.
- **Graph context (C) is the only variant that is non-negative in every single
  family** (+11.1%, +0.7%, +4.4%, +1.4%) and has the best average (+3.3%). Modest,
  not transformative, but the cleanest, most consistent positive signal this
  milestone found. D (both combined) is worse than C alone on 2 of 4 families —
  the normalized features' family-specific harm partially cancels graph context's
  benefit when combined, so they should not be naively stacked.

## 9. Phase 9 — family-ID diagnostic (DIAGNOSTIC ONLY, never for production)

**In-distribution** (ordinary scenario split, family always seen at train
AND test): without family-ID 0.6365, with family-ID 0.6401 (+0.6%) —
**negligible.** Consistent with §2: canonical features already encode
family at 98.4% accuracy, so an explicit family-ID one-hot is almost
entirely redundant information.

**Family holdout** (family-ID present at train, held-out family's
indicator column is a never-before-seen all-zero vector at test):
`multi_exit_wide` 0.2524 vs. 0.2540 (−0.6%), `single_exit_lowrise` 0.5245
vs. 0.5126 (+2.3%), `twin_stair_highrise` 0.2771 vs. 0.2858 (−3.0%),
`v1_topology_fixed` 0.5279 vs. 0.4969 (+6.2%) — **mixed, no reliable
direction.** Explicit family-ID neither reliably helps nor catastrophically
hurts unseen-family transfer, because the model was already implicitly
conditioning on family through ordinary features (§2); making that
explicit changes little. **Confirms V3.1's own hypothesis (the model
implicitly classifies topology) without over-claiming a fix — the
bottleneck is the learned per-family CALIBRATION (§3's concept shift),
not a lack of family-awareness the model could simply be handed.**

## 10. Phase 10 — leave-multiple-families-out scaling test

| Training family count | Avg. held-out PR-AUC |
|---|---|
| 1 (train on 1, test on other 3) | 0.2830 |
| 2 (3 complementary pairs, test on other 2) | 0.2996 |
| 3 (Phase 1's own family holdout) | 0.3873 |

**Monotonically increasing** as more distinct families enter training —
`monotonic_improvement: true`. Not perfectly apples-to-apples (the test
set's family composition changes at each training size, a disclosed
limitation, not hidden), but the direction is consistent and the jump from
2→3 families is much larger than 1→2, suggesting returns are not yet
saturating. **This is the milestone's strongest evidence for cause (A):
family diversity in training genuinely, monotonically helps**, independent
of §6/8's representation-level findings.

## 11. Phase 11 — data-vs-representation decision

Weighing §1-10 together against the charter's A-G menu:

- **(A) Insufficient family diversity — SUPPORTED.** §10's monotonic
  training-family-count trend is direct, quantitative evidence.
- **(B) Absolute-scale feature shift — PRESENT but not, by itself, the
  dominant fixable cause.** §2 confirms real covariate shift; §6 shows
  correcting it (normalization) is a mixed bag, not a reliable win —
  consistent with `candidate_walking_distance` being pure covariate shift
  (fixable in principle) while the two features that matter most for
  outcome quality also carry concept shift (not fixable by rescaling
  alone).
- **(C) Concept shift between families — SUPPORTED, and the most novel
  finding of this milestone.** §3 shows `total_active_occupant_count` and
  `candidate_recent_flow_rate` carry a genuinely different
  feature→outcome relationship per family (35× and 33× conditional-rate
  differences at matched feature values). This is why normalization alone
  cannot fully close the gap — some of the problem is not about feature
  SCALE, it is about feature MEANING differing structurally by family.
- **(D) Missing graph context — PARTIALLY SUPPORTED.** §8's graph-context
  features are the one variant that never hurts and average +3.3%,
  concentrated exactly on the two hardest families. Real, modest, worth
  pursuing further — not transformative on its own.
- **(E) Model-specific generalization failure — NOT the dominant
  explanation.** §12 (below) shows all three algorithms fail on
  `multi_exit_wide` similarly (0.25-0.29); no algorithm clears the gap.
- **(F) Target V2 instability — NOT SUPPORTED.** §13 (below) finds Target
  V2 stable and in-range across all four families.
- **(G) Other — a genuine, additional finding.** §5 shows the failure is
  not reducible to "the Stair type is hard" (a family with zero Stairs is
  still the single worst performer); §9 shows the model already
  implicitly encodes family, so the bottleneck is the CALIBRATION it
  learned per family, not a lack of family awareness.

**Multiple causes, not one — (A) and (C) are the dominant, best-evidenced
causes; (D) is a validated-but-modest secondary lever; (B) is real but not
independently sufficient; (E) and (F) are ruled out.**

## 12. Phase 12 — model robustness

| Held-out family | XGBoost | HistGradientBoosting | LogisticRegression | Best generalizer |
|---|---|---|---|---|
| multi_exit_wide | 0.2540 | 0.2360 | **0.2931** | LogisticRegression |
| single_exit_lowrise | **0.5126** | 0.3717 | 0.4246 | XGBoost |
| twin_stair_highrise | 0.2858 | 0.3273 | **0.3966** | LogisticRegression |
| v1_topology_fixed | **0.4969** | 0.3646 | 0.4686 | XGBoost |

Same pattern V3.1 found on Dataset V2: **no single algorithm wins every
holdout.** LogisticRegression — the SIMPLEST model tested — generalizes
best on both of the two HARD families (`multi_exit_wide`,
`twin_stair_highrise`), plausibly because it cannot overfit to
family-specific duplicate-feature-vector memorization (V3.1's own §4
finding) the way a 300-tree/depth-6 XGBoost can; XGBoost still wins on the
two easier, more in-distribution-like families. HistGradientBoosting is
never the best generalizer for any family. **The failure is data/representation-driven,
not XGBoost-specific — switching algorithms is not a fix, but the
consistent LR-wins-on-hard-families pattern is itself informative evidence
for concept shift/overfitting, not noise.**

## 13. Phase 13 — Target V2 stability across families

| Family | Positive rate | Median lead time (s) | n positive |
|---|---|---|---|
| multi_exit_wide | 1.02% | 8.83 | 12,454 |
| single_exit_lowrise | 4.74% | 8.92 | 6,570 |
| twin_stair_highrise | 4.63% | **6.13** | 16,893 |
| v1_topology_fixed | 3.07% | 8.29 | 7,048 |

All four rates fall within Target V2's established real-congestion range
(consistent with Dataset V3's own campaign-level 0.9%-3.0% Door/Exit/Stair
figures scaled to family level). No anomaly, no zero-duration artifact.
**One real, disclosed asymmetry**: `twin_stair_highrise`'s median lead
time (6.13s) is meaningfully shorter than the other three (8.3-8.9s) —
its congestion episodes develop faster, giving the model less runway to
react within the fixed 20s horizon. This is a genuine, structural
contributor to `twin_stair_highrise`'s uniquely poor ROC-AUC (§1), separate
from and additional to the covariate/concept-shift findings above — not a
Target V2 defect, a real property of vertical-stair congestion dynamics.

## 14. Phase 14 — live parity of experimental features

| Feature | Simulation source | Live source | Parity | Failure semantics | Cost |
|---|---|---|---|---|---|
| rel_queue_to_capacity, rel_flow_to_capacity, rel_adjacent_occupancy_to_building_occupancy | Ratios of already-EXACT/ESTIMABLE raw columns (`candidate_queue_length`, `candidate_recent_flow_rate`, `candidate_adjacent_zone_occupancy`, `total_active_occupant_count`) | Identical ratio of the SAME live-estimable raw columns | Inherits component parity (EXACT×STRUCTURAL / PARTIAL for Door/Stair flow) | Divide-by-zero guarded (floor at 1.0), never fabricates a value where the numerator is missing | O(1) per row |
| rel_walking_distance_to_building_mean | `candidate_walking_distance` (EXACT, NavigationGraph geometry) / mean walking distance over the variant's structural signature | IDENTICAL — `topology_signature.compute_structural_signature` reused verbatim, and it is pure Building/NavigationGraph geometry, computable live from the SAME Building object | EXACT | Guarded against zero-mean degenerate buildings | O(1) per row, mean computed once per building |
| rel_alt_route_share | `candidate_alternative_route_count` (STRUCTURAL) / (variant candidate_count − 1) | IDENTICAL — candidate_count is a static NavigationGraph property | STRUCTURAL | Guarded against single-candidate buildings | O(1) per row |
| graph_edge_betweenness_centrality, graph_is_bridge, graph_upstream_catchment_count | `networkx` computation over the Building's NavigationGraph (zones + OUTSIDE, edges = candidates, weight = walking_distance) | IDENTICAL call over the SAME live NavigationGraph structure LiveRuntime already builds — no simulation-only input is read | STRUCTURAL, EXACT parity in principle | Static per structural shape — recomputed only when the building's graph itself changes (a design-time event, not a runtime one); never a function of occupant/fire state, so it cannot go stale mid-incident | O(candidates² × log) once per building, then O(1) lookup per row — cheap at this campaign's 2-11-candidate building scale |

**Every experimental feature has an honest live-parity story.** None
reads simulation-only truth. `topology_family`/`structural_variant_id`
(§9's diagnostic) are explicitly EXCLUDED from this table — they are
dataset/campaign bookkeeping (which generator built a scenario), never a
live-observable property of a real, unlabeled building, and per the
charter must never enter a production schema regardless of any diagnostic
result.

## 15. Phase 15/16 — next dataset design and structural coverage map

All 4 existing families are single-floor-or-low-rise, orthogonal
corridor/hub/chain layouts with conventional Door/Exit/Stair mixes. What
Dataset V3's 16 variants do NOT cover, based on §1-12's evidence (families
that would specifically stress the concept-shift and graph-context
findings above, not cosmetic variety):

| Structural dimension | V3's current range (16 variants) | Gap |
|---|---|---|
| Floors | 1-4 | No high-rise (8+) tested |
| Route topology | linear, branching, hub/star, chained-serial-stairs | No **ring/circular corridor** (creates 2 genuinely equal shortest paths, unlike hub/star's asymmetric spokes) tested |
| Building shape | single-block per family | No **multi-wing** (2+ largely-independent sub-buildings sharing few connectors) tested — would stress `graph_upstream_catchment_count`/bridge-detection directly |
| Vertical connectivity | atrium/multi-floor NOT present | No **atrium-connected multi-floor** (a single large zone spanning floors) tested |
| Corridor structure | mostly through-routes | No **dead-end corridor** (a zone with only one edge, zero alternative routes, at HIGH occupancy) isolated as its own family |
| Occupant density pattern | roughly uniform per family | No **compartmentalized classroom/office** (many small, densely-partitioned zones behind a single corridor) tested |

Per this milestone's own instruction, **these are candidate directions,
not commitments** — see §17 for whether to act on them now.

## 16. Phase 16 — production-readiness consequence

**C — BOTH MORE FAMILIES AND REPRESENTATION CHANGES ARE NEEDED.**

Not A (current features sufficient, just need more families): §8 shows
graph-context features give a real, non-negative improvement current
canonical features cannot provide — the representation itself has room to
improve. Not B (representation gap only, fix schema before more data):
§10's monotonic family-count trend is direct evidence that family
diversity independently matters, not just feature engineering. Not D
(fundamentally unsuitable): ML beats the deterministic baseline by
2.3×-3.6× in every single holdout tested across this milestone (§1, and
every controlled variant in §6/8/9/10/12) — the approach works, it is
under-informed, not broken.

## 17. Testing

`tests/test_predictive_dataset_experimental_features_v4.py` (15 tests):
hand-built-graph correctness for all three graph-context descriptors
(pure-chain bridge/catchment values verified by inspection, not just
"doesn't crash"), parallel-edge collapse behavior, empty-building
degenerate case, normalized-feature divide-by-zero guards,
variant/candidate join correctness including an unknown-pair graceful
default, and three explicit isolation guards (no `LiveRuntime` import, no
mutation of `predictive_dataset.schema`'s canonical schema, family/variant
identity never present in the reusable experimental feature-name tuples).
**Full suite: 4,730/4,730 passing** (4,715 baseline + 15 new).

---

## Final Report

1. **Were V3 family-holdout failures reproduced?** Yes, near-exactly (§1: 0.254/0.513/0.286/0.497 vs. V3's own 0.254/0.513/0.286/0.497).
2. **Can topology family still be inferred from canonical features?** Yes, 98.4% accuracy vs. 62.9% majority baseline (§2).
3. **Which features reveal family most strongly?** `candidate_walking_distance` (SMD 1.20), `total_active_occupant_count` (1.09), `candidate_adjacent_zone_occupancy` (1.06); `candidate_queue_length` is nearly invariant (0.06) (§2).
4. **Is covariate shift present?** Yes, strongly, for 5 of 6 audited features (§2).
5. **Is concept shift present?** Yes — `total_active_occupant_count` (35× conditional-rate spread) and `candidate_recent_flow_rate` (33×) at matched feature values differ by family; `candidate_walking_distance`/`candidate_queue_length` do not (§3). This is the milestone's most novel finding.
6. **Which candidate type drives the largest failure?** None exclusively — `multi_exit_wide` (zero Stairs) is the single worst family, driven by weak Exit prediction; Stair is hardest to isolate (specialized Stair-only models collapse relative to the unified model) but is not the sole driver of family-level failure (§5).
7. **Where do false positives concentrate?** RISING/UNKNOWN trend, multi-bottleneck ticks, high occupancy — consistent across all four families (§4).
8. **Where do false negatives concentrate?** STABLE/FALLING trend, low occupancy, and (sharply, family-specifically) high flow rate in `twin_stair_highrise` (94.4% FNR at flow>10) (§4).
9. **Did normalization improve unseen-family transfer?** Mixed — helped the two hard families (+8-13%), hurt the two easier ones (−3 to −6%), net average +0.6% (§6).
10. **Which normalized features helped?** Not isolated individually this milestone (evaluated as one bundle); the bundle's benefit concentrates on the two families with the strongest concept-shift features (§6, cf. §3).
11. **What graph context is missing from the current schema?** Structural centrality, cut-edge/bridge status, and upstream demand catchment — none present in the canonical 12-field schema (§7 audit reasoning embedded in §14's parity table; implemented in §8).
12. **Did graph-context features improve transfer?** Yes, modestly and consistently — average +3.3%, non-negative in all 4 families, the cleanest single positive signal this milestone found (§8).
13. **Which graph-context features helped?** Evaluated as one 3-feature bundle (betweenness, is-bridge, catchment), not isolated individually — a natural follow-up ablation for a future milestone.
14. **Did combined normalization + graph context improve transfer?** Worse than graph-context alone on 2 of 4 families (+2.0% avg vs. +3.3%) — normalization's family-specific harm partially cancels graph context's benefit; do not naively stack them (§6/8).
15. **What happened in the explicit-family-ID diagnostic?** Negligible in-distribution effect (+0.6%), mixed/unreliable at family holdout (−3.0% to +6.2%) — confirms the model already implicitly encodes family and isn't missing an "am I in a known family" signal it could simply be handed (§9).
16. **What happened when training-family diversity was reduced?** PR-AUC dropped monotonically: 0.387 (3 families) → 0.300 (2) → 0.283 (1) (§10).
17. **Do LR/HGB/XGBoost fail similarly?** All three struggle on the same hard families; LogisticRegression (simplest) actually generalizes best on both hard families, XGBoost best on the two easier ones — a genuinely model-dependent but not model-FIXABLE pattern (§12).
18. **Is Target V2 stable across families?** Yes — positive rates and lead times all in-range; one real, disclosed asymmetry (`twin_stair_highrise`'s shorter 6.13s median lead time) is a structural property of vertical-stair dynamics, not a Target defect (§13).
19. **Which experimental features have genuine live parity?** All of them — every normalized and graph-context feature is a pure function of already-audited live-estimable raw columns or static Building/NavigationGraph geometry (§14).
20. **Is the dominant problem data diversity, representation, model, or target?** Both data diversity (A) and representation (C dominant, D secondary) — not model (E ruled out) or target (F ruled out) (§11).
21. **What structural regions are missing from Dataset V3?** High-rise (8+ floors), ring/circular corridors, multi-wing buildings, atrium-connected multi-floor, isolated dead-end corridors, and compartmentalized classroom/office layouts (§15).
22. **Which new topology families are scientifically justified?** Candidates only, not commitments: multi-wing (stresses catchment/bridge detection directly) and ring/circular corridor (stresses the alternative-route-count feature's current hub/star-only redundancy pattern) are the two most directly justified by this milestone's own evidence (§15).
23. **Should Dataset V4 be generated?** Not this milestone — per the charter's explicit stop condition. §10/§15 together justify it as the next milestone's likely direction.
24. **Should the canonical feature schema change first?** Not yet — §8's graph-context gain (+3.3% avg) is real but modest; validate on a larger, purpose-built family set (once it exists) before promoting any experimental feature into `predictive_dataset/schema.py`.
25. **Final category: C — BOTH MORE FAMILIES AND REPRESENTATION CHANGES ARE NEEDED** (§16).
26. **Full-suite result:** 4,730/4,730 passing.
27. **Commit hash:** see `git log` — this milestone's commit immediately follows `ecf3f9b` (Predictive Dataset V3) on `main`.

**A. Is generating more rows from the existing 16 variants useful?** No — V3.1 already showed this doesn't help (unique-state experiments, Dataset V2), and this milestone adds no new evidence to revisit that.

**B. Is generating more variants inside the same four families enough?** No — §10/§21 show the unsolved gap is at the FAMILY level, and V3 already scaled variants-per-family from 1→4 without closing it.

**C. Do we need entirely new building families?** Yes, per §10's monotonic training-family-count evidence — though only 2-3 well-chosen new families (§15/§22), not an unbounded expansion.

**D. Does the flat candidate representation lack important topology context?** Yes, partially — §8's non-negative, +3.3%-average graph-context result is real evidence of a genuine, fixable representation gap, though not the whole story (§3's concept-shift finding is not a representation-completeness problem, it is a genuine cross-family relationship difference no additional feature trivially resolves).

**E. Can any proposed new features be computed honestly in live operation?** Yes — every feature tested this milestone has full live parity (§14); nothing simulation-exclusive was proposed or adopted.

**F. Is Model V4 justified immediately after this milestone?** No — the charter's explicit stop condition, and substantively: family-level generalization is still the unsolved half of the problem this milestone diagnosed but did not yet fix.

**G. Should predictive AI influence Recommendation ranking yet?** No — unchanged from every prior milestone's verdict; this remains offline research.

**H. What EXACT milestone should come next?** A targeted **Dataset V4** generating 2-3 NEW topology families chosen from §15's coverage gaps (multi-wing and ring/circular corridor are the two most directly justified), specifically designed to test whether family-count diversity keeps improving monotonically past 4 (§10's open question) — combined with promoting the validated graph-context features (§8/§14) into that new campaign's extraction pipeline as a first-class (still experimental, not yet canonical) feature set, so the next generalization experiment tests A and D together rather than separately.

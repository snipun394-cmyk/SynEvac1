# Predictive Dataset V4 — Cross-Family Structural Generalization

Status: **dataset/feature-schema/data-science milestone. No Model V4 trained/exported, no predictive inference wired into LiveRuntime, Recommendation, Guidance, Advisory, Voice, Dynamic Signage, or Decision Policy, no new Designer asset, no GNN.**

Follows [[cross_topology_generalization_investigation]] (commit `e8d728a`, verdict C — both more topology families AND representation changes are needed) and [[predictive_dataset_campaign_v3_milestone]] (commit `ecf3f9b`, verdict B). Two coordinated changes, both directly evidenced by e8d728a's own findings:

- **(A) Promote the 3 experimentally-validated graph-context descriptors** (edge betweenness centrality, bridge/cut-edge status, upstream demand catchment — the only representation change that improved every one of e8d728a's 4 family holdouts, average +3.3% PR-AUC) into a new, versioned canonical feature schema.
- **(B) Add genuinely new topology families** occupying structural regions absent from Dataset V3 (e8d728a's leave-multiple-families-out test showed held-out PR-AUC rising monotonically with training-family count: 0.283 → 0.300 → 0.387 as family count went 1→2→3).

## 1. Graph-context promotion (Phase 1)

Promoted **exactly** the 3 fields e8d728a validated — no other experimental field from that investigation (normalized/ratio features showed a mixed, net-neutral effect there and were NOT promoted):

| Field | Mathematical definition | Physical meaning | Sim/live source | Nullable |
|---|---|---|---|---|
| `candidate_betweenness_centrality` | Edge betweenness centrality (Brandes' algorithm, walking-distance-weighted, normalized to [0,1]) | Fraction of all-pairs shortest paths passing through this candidate — structural centrality | IDENTICAL: `graph_context_v4.compute_graph_context_for_building`, same `NavigationGraphGenerator` LiveRuntime already uses | No |
| `candidate_is_bridge` | Standard cut-edge test (`networkx.bridges`) | Whether this candidate is a structural single point of failure | IDENTICAL, same function | No |
| `candidate_upstream_catchment_count` | Count of zones whose shortest path to OUTSIDE uses this edge | How much building-wide demand structurally depends on this candidate under normal routing | IDENTICAL, same function | No |

**A real bug was found and fixed during this promotion audit**: `networkx.edge_betweenness_centrality`'s returned dict keys use its own internal `(u, v)` order, not necessarily matching this codebase's `tuple(sorted(...))` lookup key — silently defaulting 30 of 92 candidate edges (32.6%) across all 16 Dataset V3 structural variants to a wrong `0.0` in e8d728a's own investigation code. Fixed in `predictive_dataset/graph_context_v4.py` (normalize the returned dict's keys to sorted tuples immediately) before promotion — e8d728a's own headline finding (graph context helps, +3.3% avg) survived the fix; the corrected values are, if anything, a stronger signal (verified: only 1/92 candidates now reads a genuine, structurally-correct `0.0`).

**Why promotion is exact, not approximate**: `live_runtime/factory.py:220` builds LiveRuntime's own shared `navigation_graph` via `NavigationGraphGenerator().build(building)` — the exact same class `predictive_dataset/candidate.py` already used for simulation extraction. LiveRuntime's `building` is, per the Designer→`LiveRuntimeController`→`live_runtime_launcher.session` construction chain, literally `self.canvas.scene_obj.project.building` — this repo's own "Building Designer IS the Digital Twin" rule holding by construction, not convention. There is therefore no second, independently-computed live graph representation that could ever silently drift from simulation's.

`predictive_dataset/graph_context_v4.py` is the ONE shared implementation; `predictive_dataset/simulation_extractor_v4.py` and `predictive_dataset/live_extractor_v4.py` both call it (never reimplement it — mechanically guarded by `tests/test_predictive_dataset_graph_context_v4.py`'s `SimLiveEquivalenceTests`). `predictive_dataset/experimental_features_v4.py` (e8d728a's own investigation module) was refactored to delegate to the same shared function too, rather than duplicating the algorithm — its own public API and 15 pre-existing tests are unchanged.

## 2. Feature schema versioning (Phase 2)

`predictive_dataset/schema.py` (`SCHEMA_VERSION = "1.0"`, 9-field `CANDIDATE_FEATURE_SCHEMA`) is **imported, never mutated** — Dataset V1/V2/V3 remain fully reproducible/loadable against their own original schema. `predictive_dataset/schema_v4.py` is a new, additive module: `SCHEMA_VERSION_V4 = "4.0"`, `CANDIDATE_FEATURE_SCHEMA_V4` = the frozen 9 + 3 already-operational-but-never-formally-schematized V2.1 fields (`candidate_recent_flow_rate`, `candidate_congestion_trend`, `candidate_alternative_route_count`) + the 3 newly-promoted graph-context fields = **15 fields total**. This is the first milestone in this codebase's history to actually bump `SCHEMA_VERSION`/extend `CANDIDATE_FEATURE_SCHEMA` — every earlier experimental tier (V2.1, V3, e8d728a) stayed additive-external by design (confirmed via research before implementing, not assumed). `tests/test_predictive_dataset_schema_v4.py` mechanically guards: old schema byte-unchanged (9 fields, version "1.0"), V4 schema is a strict ordered superset, all 15 names unique, every promoted field documented, all 3 graph-context fields non-nullable (unlike `candidate_capacity`/`candidate_walking_distance`), and `topology_family`/`structural_variant_id` (dataset bookkeeping) never enter the schema.

## 3. Live parity implementation (Phase 3)

`predictive_dataset/live_extractor_v4.py` — NOT wired into LiveRuntime this milestone. `tests/test_predictive_dataset_graph_context_v4.py::SimLiveEquivalenceTests` proves `extract_v4_candidate_features` (sim) and `extract_live_v4_candidate_features` (live) return byte-identical graph-context values for the same `Building`, and a static-source guard confirms neither module reimplements `edge_betweenness_centrality`/`bridges`/`shortest_path` — both must delegate to `graph_context_v4.compute_graph_context_for_building`.

## 4. Graph-context correctness (Phase 4)

`tests/test_predictive_dataset_graph_context_v4.py`, 6 hand-built fixtures with hand-computed expected values (not just "networkx ran"):

- **Linear chain** (3 zones in series + 1 exit): every edge a bridge; betweenness hand-computed exactly (0.5/0.667/0.5); catchment increases toward OUTSIDE (1/2/3).
- **Branching star** (hub + 3 symmetric spokes + 1 exit): every edge a bridge; catchment 1 per spoke, 4 at the exit; a genuinely SYMMETRIC construction gives EQUAL betweenness (0.4) to every spoke AND the exit — confirmed by hand computation, not assumed.
- **Ring** (3-zone cycle + 1 exit): ring edges are NOT bridges (redundant), the sole exit edge IS a bridge — the fundamental redundancy structure V3 almost entirely lacked.
- **Two independently-exited zones + 1 connecting door**: forms a TRIANGLE through the single shared OUTSIDE node — a genuinely surprising, disclosed finding (a naive "each zone's own exit is its only way out" intuition wrongly predicts the connecting door is a bridge; it is not, since OUTSIDE gives an alternate path).
- **Multi-exit catchment** (2 zones, each its own exit): each exit's catchment is exactly its own zone (1), the connecting door's catchment is a genuine, honest 0 (neither zone's shortest path uses it).
- **Stair-connected multi-floor**: caught a real fixture-authoring bug during test development — a `Staircase` must be registered in its OWN `from_floor`'s `stairs` list (not just anywhere), or `navigation/graph_builder.py`'s `_add_stair_edges` silently drops the edge (`from_zone` fails to resolve against the wrong floor). Fixed in the test fixture, documented so it isn't rediscovered.

**27 tests total** (`tests/test_predictive_dataset_graph_context_v4.py` + `tests/test_predictive_dataset_schema_v4.py`), all passing.

## 5. Structural coverage gap analysis (Phase 5)

New `predictive_dataset/topology_signature_v4.py` extends V3's structural signature with bridge-edge prevalence, betweenness/catchment distribution stats, branching factor (zone degree), cyclomatic number, corridor depth, and exit-catchment asymmetry — run against all 16 Dataset V3 variants BEFORE designing any new family:

**Decisive finding**: a naive graph-cycle check (`cyclomatic_number` on the full graph including the shared OUTSIDE node) shows 12/16 V3 variants "have a cycle" — but this is an ARTIFACT: any building with 2+ exits (or 2+ independent paths to OUTSIDE) trivially closes a loop through the single shared OUTSIDE node, regardless of whether any REAL physical corridor connects two zones without going outside and back. Recomputing cyclomatic number on the **zone-only subgraph** (OUTSIDE excluded) — the honest measure of genuine physical route redundancy between zones — shows **15 of 16 V3 variants have ZERO genuine zone-only cycles**; the one exception (`v1_fixed_dual_stair`) has exactly one, as a side effect of two stairs converging on the same zone, not a designed ring.

The second gap: V3's closest hub-and-spoke analog, `multi_exit_wide`, has shallow (single-zone) spokes each with its OWN direct exit — symmetric, every candidate a bridge, `max_upstream_catchment` capped at 5 across all 16 variants. No V3 variant combines multi-zone-DEEP wings with a SHARED connector multiple wings funnel through.

**Decision**: 2 new families, chosen from measured evidence, not preference — `ring_corridor` (targets the zone-only-cycle gap) and `multi_wing` (targets the wing-depth + funneling-connector gap). A third family was investigated (Phase 8) and **not added**: no other dimension `topology_signature_v4` measures showed a comparably stark gap (e.g. 15/16 == 0); per the charter's own "do not add a family merely to increase the count" instruction.

## 6-7. New families (Phases 6-7)

**`multi_wing`** (4 variants, `predictive_dataset/topologies_v4.py`): core + wings 2 zones deep, funneling through shared connector(s).

| Variant | Structure | What's different |
|---|---|---|
| `multi_wing` (base) | Core + 3 wings (2 zones deep), 1 shared exit | — |
| `multi_wing_asymmetric_exits` | 2-zone-deep core, 2 exits, wings attached asymmetrically (1 on core-a, 2 on core-b) | Real exit-catchment imbalance |
| `multi_wing_four_wings` | 4 wings instead of 3 | Higher core branching factor |
| `multi_wing_vertical` | 2 ground wings + 1 wing reached via Stair to floor 2 | Combines wing-depth/funneling with vertical connectivity — no V3 variant has this combination |

**`ring_corridor`** (4 variants): genuine zone-to-zone cycles.

| Variant | Structure | What's different |
|---|---|---|
| `ring_corridor` (base) | 4-zone ring, 1 exit | Genuine zone-only cycle (near-absent from V3) |
| `ring_corridor_dual_exit` | Same ring, 2 exits at OPPOSITE ring positions | Clockwise/counterclockwise travel favors different exits — a route-CHOICE structure no V3 variant has |
| `ring_corridor_large` | 6-zone ring | Deeper corridor depth around the ring |
| `ring_corridor_partial_chord` | 4-zone ring + 1 chord door (A↔C) | 2 independent cycles — higher cyclomatic number than a simple ring |

**Measured confirmation the new families closed exactly the targeted gaps** (not just "different"):

| Metric | V3 (16 variants) | V4 new families | Result |
|---|---|---|---|
| Genuine zone-only cycles | 1/16 | 5/24 (the 4 new `ring_corridor` variants + `v1_fixed_dual_stair`) | Gap materially closed |
| `max_upstream_catchment` | 5 (cap across all 16) | 9 (`multi_wing_four_wings`) | New high-load-bearing region opened |

**Diversity gate**: all 24 variants (16 reused unmodified + 8 new) mechanically confirmed genuinely structurally distinct — `structural_diversity_report`: 24 requested, 24 distinct signatures, 0 duplicate groups.

## 9. Structural signature V4 (Phase 9)

`predictive_dataset/topology_signature_v4.py`'s new fields (bridge prevalence, betweenness/catchment distribution, branching factor, cyclomatic number, corridor depth, exit asymmetry) are **dataset/campaign metadata only** — deliberately NOT added to `predictive_dataset/schema_v4.py`'s trainable `CANDIDATE_FEATURE_SCHEMA_V4` (only the 3 already-validated per-candidate descriptors were promoted; this milestone does not additionally expose whole-building aggregate structural stats as ML features).

## 10. Target V2 freeze (Phase 10)

Unchanged: `v2-persistent-demand-service-imbalance`, `MIN_PERSISTENCE_SECONDS = 3.0`. `tests/test_predictive_dataset_v4_architecture_guards.py` mechanically reconfirms: the persistence floor and version string are unchanged constants; `campaign_runner_v4.py` imports `target_generator_v2` (the designated caller role, same as `campaign_runner_v3.py`); none of the 4 new extraction-side modules (`graph_context_v4`, `simulation_extractor_v4`, `live_extractor_v4`, `schema_v4`) import `target_generator_v2` or `target_semantics_analysis`; `graph_context_v4.py` never references `movement_result`/`occupancy_snapshot`/`crowd_snapshot`/`evacuation_snapshot` (static-analysis proxy for its own "zero occupancy/fire/time dependence" leakage claim).

## 11. Pilot campaign (Phase 11)

`scripts/run_predictive_dataset_campaign_v4_pilot.py`: 25 scenarios × 24 variants = 600 scenarios, `master_seed=20260729`.

**Result: 600/600 accepted, 0 failed, 770,358 rows, 88.3s (8,725 rows/sec).** Every variant AND every family contributed rows. Zero zero-walking-distance candidates. Target V2 positive rate 3.09% (in-range). Structural diversity: all 24 signatures genuinely distinct. Graph-context distribution already reaching the new high-catchment region (max 9, mean 1.70). Pilot quality gate: **PASS**.

## 12. Graph-context distribution audit (Phase 12)

`scripts/model_v4_pilot_diagnostics.py`, run on the pilot:

| Field | Old families (4) | New families (2) | Assessment |
|---|---|---|---|
| `candidate_betweenness_centrality` | 0.0–0.667, mean 0.277 | 0.1–0.5, mean 0.293 | Mean shifts; max does NOT extend upward at pilot scale (larger, more-redundant new-family graphs dilute the shortest-path-fraction metric — an expected, disclosed property, not a bug) |
| `candidate_is_bridge` | mean 0.31 | mean 0.67 | Substantial mean shift (new families' candidates are structurally more often bridges on average) |
| `candidate_upstream_catchment_count` | 0–5, mean 1.46 | 0–9, mean 2.00 | Range genuinely extends upward |

**Not identical distributions** by any of the 3 measures — the charter's explicit stop condition ("if new families produce essentially the same distributions, stop and redesign") does not trigger. Proceeded to full-scale campaign.

## 13. Family identifiability (Phase 13)

Pilot-scale, 6-way (not V3's 4-way) family classifier: **(A) old canonical (12-field) schema: 97.44% accuracy** (majority baseline 36.5%). **(B) new V4 (15-field, +graph context) schema: 100.0% accuracy.** Expected and disclosed, not concerning: graph-context descriptors are near-deterministic per structural variant, so adding them makes family trivially recoverable — consistent with every prior milestone's finding that topology is inherently, heavily encoded in honestly-available structural features.

## 14. Controlled generalization pilot (Phase 14)

Pilot-scale leave-one-family-out, old vs. new schema, same XGBoost config:

| Held-out family | Old schema PR-AUC | New V4 schema PR-AUC | Relative Δ |
|---|---|---|---|
| `multi_wing` (NEW) | 0.298 | 0.319 | **+7.0%** |
| `ring_corridor` (NEW) | 0.432 | 0.471 | **+9.0%** |
| `multi_exit_wide` | 0.237 | 0.211 | −10.7% |
| `single_exit_lowrise` | 0.464 | 0.441 | −5.0% |
| `twin_stair_highrise` | 0.581 | 0.549 | −5.5% |
| `v1_topology_fixed` | 0.481 | 0.422 | −12.2% |

**Both NEW families' transfer IMPROVED with graph context; the 4 OLD families showed a mild-to-moderate REGRESSION at pilot scale.** The charter's explicit stop condition ("if it catastrophically worsens NEW-family transfer, investigate before scaling") does not trigger — new-family transfer improved, not worsened. The old-family softening is real, disclosed, and flagged for direct re-examination at full scale (§21-24 below) rather than dismissed as pilot noise. Proceeded to full-scale campaign.

## 15-16. Coverage targets and campaign scale (Phases 15-16)

12 measurable targets defined BEFORE the full campaign ran (`predictive_dataset/campaign_config_v4.py`): 7 mirroring V3's own targets (scaled to 24 variants/6 families) + 5 new (`multi_wing_family_rows`, `ring_corridor_family_rows`, `genuine_zone_cycle_scenarios`, `non_bridge_candidate_rows`, `high_catchment_candidate_rows`, `high_betweenness_candidate_rows`). Scale: 125 scenarios/variant × 24 variants = **3,000 scenarios** — inside the charter's suggested 2,500-3,500 range, chosen for balanced per-variant density close to V3's own 150/variant (not row-count maximization).

## 17-18. Full campaign (Phases 17-18)

`scripts/run_predictive_dataset_campaign_v4_fullscale.py`, `master_seed=20260729`.

**3,000/3,000 accepted, 0 failed. 3,945,171 rows. Wall time 469.8s (8,397 rows/sec).** Memory: available dropped from 893MB to 653MB at the tightest point — never breached the 300MB abort floor (this milestone's own environment runs persistently tight, ~7.9GB total RAM with other applications open — the same discipline every prior campaign's streaming/never-accumulate-in-memory design already established held here too).

**All 12 coverage targets pass:**

| Target | Minimum | Actual |
|---|---|---|
| Every structural variant represented | 24 | 24 |
| Every family represented | 6 | 6 |
| High-occupancy scenarios | 200 | 2,030 |
| Multi-bottleneck rows | 2,000 | 8,144 |
| Stair rows with real demand | 1,000 | 82,899 |
| Total-lockout scenarios with rows | 3 | 231 |
| Multi-floor scenarios | 300 | 1,250 |
| `multi_wing` family rows | 20,000 | 1,265,712 |
| `ring_corridor` family rows | 20,000 | 450,154 |
| Genuine zone-cycle scenarios | 300 | 2,000 |
| Non-bridge candidate rows | 50,000 | 2,109,456 |
| High-catchment candidate rows | 20,000 | 232,715 |
| High-betweenness candidate rows | 50,000 | 251,199 |

No zero-walking-distance regression.

## 19. V3 vs V4 structural comparison (Phase 19)

`scripts/predictive_dataset_v4_structural_comparison.py`, run over the FULL 24-variant registry (all structural facts, not campaign rows):

**Genuine zone-only cycles: 1/16 (V3) → 5/24 (V4)** — the headline, directly-targeted improvement. **`max_upstream_catchment`: 5 → 9.** `door_count` range broadens (1-6 → 1-8, 5→8 distinct), `zone_count` broadens (2-8 → 2-9), `mean_zone_degree` broadens (max 2.5 → 2.75), `mean_candidate_walking_distance` broadens (max 55.8 → 79.0). Several dimensions do NOT broaden in raw min/max range (`floor_count`, `exit_count`, `stair_count`, `candidate_count`, `cyclomatic_number` on the full graph, `corridor_depth_hops`) — honestly reported: this milestone deliberately targeted the zone-cycle and catchment/funneling gaps specifically, not every dimension simultaneously; V3 already had adequate range on floor/exit/stair counts. Distinct-value counts increase on MORE dimensions than raw range does (e.g. `mean_betweenness` distinct values 15→19, `bridge_edge_fraction` 9→10) even where the min/max didn't move — finer-grained coverage within the existing range, not just edge-extension.

## 20. Concept-shift reanalysis (Phase 20)

`scripts/model_v4_generalization_evaluation.py`, full-scale (3,945,171 rows). e8d728a found `total_active_occupant_count`/`candidate_recent_flow_rate` carried genuine concept shift (33-35× conditional-rate spread at matched states across families). Repeating that measurement, then adding `candidate_upstream_catchment_count` as a third conditioning variable:

| | Max spread | Mean spread | Bins compared |
|---|---|---|---|
| P(target \| occupancy, flow) | 0.644 | 0.228 | 19 |
| P(target \| occupancy, flow, catchment) | 0.690 | 0.200 | 35 |

**Mixed, not a clean win — reported honestly.** MAX spread did NOT shrink (it grew slightly, 0.644→0.690) — with finer binning (35 vs 19 well-supported bins), the single worst-case bin combination gets slightly worse, an expected side effect of subdividing the data more finely. MEAN spread DID shrink (0.228→0.200, −12%) — conditioning on catchment makes the AVERAGE cross-family disagreement smaller, even though it doesn't eliminate the single worst case. **Conclusion: graph context provides a partial, not complete, explanation for the concept shift e8d728a found — consistent with, not contradicting, §21's finding that graph-context features help some families and not others.**

## 21-24. Family-holdout evaluation, new-family transfer, old-family regression (Phases 21-24)

Full-scale (3,945,171 rows) leave-one-family-out, all 6 families, 3 controlled variants (A/B/C) isolating the diversity benefit from the representation benefit:

| Held out | A: V3-pop, old schema | B: V4-pop, old schema | C: V4-pop, new schema | Diversity (A→B) | Representation (B→C) | vs e8d728a (B) | vs e8d728a (C) |
|---|---|---|---|---|---|---|---|
| `single_exit_lowrise` | 0.4834 | 0.4858 | 0.4008 | +0.5% | **−17.5%** | −5.2% | −21.8% |
| `twin_stair_highrise` | 0.3539 | **0.6402** | 0.5903 | **+80.9%** | −7.8% | **+124.0%** | +106.5% |
| `multi_exit_wide` | 0.2508 | 0.2844 | 0.2766 | +13.4% | −2.7% | +11.9% | +8.9% |
| `v1_topology_fixed` | 0.5340 | 0.5427 | 0.5058 | +1.6% | −6.8% | +9.2% | +1.8% |
| `multi_wing` (NEW) | n/a | 0.3995 | 0.3914 | n/a | −2.0% | n/a | n/a |
| `ring_corridor` (NEW) | n/a | 0.4543 | **0.4719** | n/a | **+3.9%** | n/a | n/a |

ML beats the deterministic-current-state baseline in **every single one of the 16 fits** (relative lift 2.4×–5.3×) — never falls below, at either schema, on any family.

**The single most important finding of this milestone**: `twin_stair_highrise` — the qualitatively worst-behaved family in e8d728a (ROC-AUC 0.691, FNR 52.6%, the one family whose ranking quality itself degraded, not just its PR-AUC) — improves from ROC-AUC 0.775 (A, this run's own V3-style reproduction) to **ROC-AUC 0.961 (B)**, with recall rising from 0.572 to **0.945** and FNR collapsing from 0.428 to **0.055**, purely from adding 2 new topology families to the training population — no new features, same old 12-column schema. This is a dramatic, direct, full-scale confirmation of e8d728a's own Phase 10 finding (PR-AUC rises monotonically with training-family count) and the single clearest piece of evidence this milestone produced.

**Diversity (A→B) helped 3 of 4 old families** (dramatically for `twin_stair_highrise`, meaningfully for `multi_exit_wide`, marginally for `v1_topology_fixed`/`single_exit_lowrise`) — never hurt any of them.

**Representation (B→C, adding graph-context features on top of the now-diverse population) did NOT reliably help at full scale** — it helped only `ring_corridor` (+3.9%) and hurt every other family, including the OTHER new family, `multi_wing` (−2.0%, reversing the pilot's own +7.0% finding — see §14/§27's discrepancy note). This is the one place this milestone's own evidence complicates, rather than confirms, e8d728a's finding: e8d728a measured the representation benefit against a training population that did NOT yet contain the 2 new families; once that diversity is already present (as it is in every B/C comparison here), the graph-context features' MARGINAL value shrinks or reverses for most families — consistent with §20's "partial, not complete" concept-shift explanation, and suggesting some real overlap between what training-family diversity and explicit graph-context features each contribute.

**New-family transfer** (Phase 23, entirely-unseen-family holdout, the cleanest test of the original problem): both new families are FAR from being "unlearnable" — `multi_wing` PR-AUC 0.40/0.39, `ring_corridor` PR-AUC 0.45/0.47, both with ROC-AUC 0.92-0.93 and relative lift over deterministic 3.2×-3.5×. `multi_wing`'s C variant trades a small PR-AUC dip for a much better precision/false-alarm profile (precision 0.190 vs 0.132, FPR 0.145 vs 0.262) — headline PR-AUC alone understates a real operational improvement in that specific case.

**Old-family regression (Phase 24)**: no family regressed catastrophically (>50% relative, the charter's own bar) at either B or C. `single_exit_lowrise`'s C variant (−21.8% vs e8d728a) is the one real, non-trivial regression worth flagging on its own — driven by §25's ablation showing this family's schema-C combination is not uniformly beneficial (not investigated further at the individual-family level this milestone; a natural next-step ablation).

## 25. Graph-context ablation (Phase 25)

4 families (2 new + the 2 historically hardest old ones), each graph-context feature added ALONE on top of the old schema vs. all three combined:

| Held out | betweenness only | is_bridge only | catchment only | all three |
|---|---|---|---|---|
| `multi_wing` | 0.411 | 0.379 | **0.412** | 0.391 |
| `ring_corridor` | 0.462 | 0.450 | 0.454 | **0.472** |
| `multi_exit_wide` | 0.270 | **0.291** | 0.279 | 0.277 |
| `twin_stair_highrise` | 0.557 | 0.615 | **0.621** | 0.590 |

**Only `ring_corridor` shows the expected "combining all three beats any single feature" pattern.** In the other 3 families, some SINGLE feature outperforms the full 3-feature combination — real, measured evidence of feature interaction/redundancy, not a clean monotonic "more graph context is always better" story. **`candidate_betweenness_centrality` alone is the WEAKEST individual feature in 3 of 4 families** (`multi_wing`, `multi_exit_wide`, `twin_stair_highrise`) — `candidate_is_bridge` and `candidate_upstream_catchment_count` carry more marginal predictive signal individually. This is a genuine, actionable finding for any future schema refinement: if a smaller feature set were ever needed, `is_bridge`/`catchment` are the stronger candidates to keep.

## 27. Dataset readiness gate (Phase 27)

**B — V4 IMPROVED TRANSFER BUT CROSS-FAMILY GENERALIZATION REMAINS TOO WEAK FOR MODEL V4.**

Not A (failed): family diversity is unambiguously, dramatically validated (§21, `twin_stair_highrise` +80.9%/+124.0%) — this is real, substantial, measured improvement, not a null result. Not C (ready): the charter's own C-gate requires "graph-context schema validated" — §21/§25 show the representation half of this milestone's hypothesis did NOT reliably replicate at full scale once training-family diversity was already present (helped only 1 of 6 families; the pilot's own encouraging `multi_wing` result reversed at full scale, §14 vs §21); `single_exit_lowrise`'s new-schema regression (−21.8% vs e8d728a) is real, not noise. Absolute PR-AUC for several families (`multi_exit_wide` ~0.28, `multi_wing` ~0.39-0.40) remains modest in absolute terms even after this milestone's improvements. Every other individual C-criterion is met (genuinely new families ✅, Target V2 valid ✅, no leakage ✅, all coverage targets pass ✅, new-family holdout beats deterministic ✅, no catastrophic >50% regression ✅, honest live parity ✅) — but the gate requires ALL of them, and the graph-context-schema criterion is not cleanly met, so B, not C, is the correct, conservative verdict.

## 21-24. Family-holdout evaluation, new-family transfer, old-family regression (Phases 21-24)

<!-- FILLED IN AFTER scripts/model_v4_generalization_evaluation.py COMPLETES -->

## 25. Graph-context ablation (Phase 25)

<!-- FILLED IN AFTER scripts/model_v4_generalization_evaluation.py COMPLETES -->

## 26. Live parity final check (Phase 26)

Every promoted V4 field's live source is documented in `predictive_dataset/schema_v4.py`'s `AIFeatureField.source` (§1's table above). No field reads simulation-only truth, scenario metadata, a topology-family identifier, or future state — mechanically guarded by `tests/test_predictive_dataset_schema_v4.py::test_topology_family_and_variant_id_never_enter_the_v4_schema` and `test_graph_context_fields_are_never_nullable`. Computational cost at this codebase's realistic building scale (2-11 candidates in V3, up to 11 in V4): betweenness centrality O(V·E) (Brandes'), bridges O(V+E), catchment O(V·(E + V log V)) — all computed ONCE per building shape (at Designer-edit time in live operation, once per structural variant in simulation), never per row/tick.

## 27. Dataset readiness gate (Phase 27)

<!-- FILLED IN AFTER §20-25 -->

## 28. Versioning

| Field | Value |
|---|---|
| `campaign_version` | `predictive_dataset_campaign_v4` |
| `schema_version` | `4.0` (new — `predictive_dataset/schema_v4.py`, `predictive_dataset/schema.py`'s own `1.0` unchanged) |
| `structural_variant_version` | `v4-cross-family-24-variants` |
| `target_version` | `v2-persistent-demand-service-imbalance` (unchanged, frozen) |
| `master_seed` | `20260729` |
| Pilot scenarios/variant | 25 (600 total) |
| Full-scale scenarios/variant | 125 (3,000 total) |

`data/predictive_dataset_campaign_v4_pilot/` and `data/predictive_dataset_campaign_v4/` are gitignored (per this project's established per-milestone convention). Dataset V1/V2/V3 artifacts are untouched.

## 29. Testing

New tests (54): `tests/test_predictive_dataset_graph_context_v4.py` (14, hand-built-graph correctness + sim/live equivalence), `tests/test_predictive_dataset_schema_v4.py` (13, versioning/ordering guards), `tests/test_predictive_dataset_topologies_v4.py` (10, registry/diversity/stair-regression/zone-cycle-coverage), `tests/test_predictive_dataset_campaign_config_v4.py` (10, config/coverage-target construction), `tests/test_predictive_dataset_v4_architecture_guards.py` (7, Target V2 freeze + leakage-boundary guards). `tests/test_predictive_dataset_experimental_features_v4.py` (e8d728a's own 15 tests) re-verified passing unmodified after the `graph_context_v4` delegation refactor.

**Full suite: 4,784/4,784 passing** (4,730 baseline + 54 new), 380.4s.

---

## Final Report

1. **Final topology-family count?** 6 (4 old, reused unmodified: `single_exit_lowrise`, `twin_stair_highrise`, `multi_exit_wide`, `v1_topology_fixed`; 2 new: `multi_wing`, `ring_corridor`).
2. **Which new families were added and why?** `multi_wing` (wing-depth + shared-funneling-connector structure, §5's second measured gap) and `ring_corridor` (genuine zone-only route redundancy, §5's decisive gap: 15/16 V3 variants had zero). A third family was investigated and NOT added — no comparably stark gap found elsewhere (§5, §8).
3. **Structural variant count?** 24 (16 reused unmodified + 8 new: 4 `multi_wing` + 4 `ring_corridor`), all mechanically confirmed genuinely distinct (§6-7).
4. **Scenario count?** 3,000 full-scale (125/variant × 24), 600 pilot (25/variant × 24).
5. **Candidate-time rows?** 3,945,171 full-scale; 770,358 pilot.
6. **Failure count?** 0 at both pilot and full scale.
7. **New feature schema/version?** `predictive_dataset/schema_v4.py`, `SCHEMA_VERSION_V4 = "4.0"`, 15 fields (9 frozen + 3 promoted V2.1 + 3 promoted graph-context). `predictive_dataset/schema.py`'s own `1.0`/9-field schema is byte-unchanged (§2).
8. **Exact graph-context features promoted?** `candidate_betweenness_centrality`, `candidate_is_bridge`, `candidate_upstream_catchment_count` — exactly the 3 e8d728a validated, no others (§1).
9. **Do simulation/live values match?** Yes, by construction — both extractors delegate to the identical `graph_context_v4.compute_graph_context_for_building`, which uses the exact same `NavigationGraphGenerator` LiveRuntime already builds against the exact same `Building` object (§3, mechanically proven in `SimLiveEquivalenceTests`).
10. **Target V2 still unchanged and valid?** Yes — version string and 3.0s persistence floor mechanically reconfirmed unchanged; no extraction-side V4 module imports it (§10).
11. **Coverage targets passed?** All 12, comfortably (§17-18).
12. **Graph-context distributions broadened?** Partially, honestly disclosed — `upstream_catchment_count`'s range genuinely extends (5→9) and `is_bridge`'s mean shifts substantially (0.31→0.67 at pilot), but `betweenness_centrality`'s max did not extend upward at pilot scale (§12).
13. **Family-identifiability result?** 97.4% (old schema) → 100.0% (new schema), 6-way, vs. 36.5% majority baseline — expected, not concerning (§13).
14. **Did pilot transfer improve?** Mixed — both NEW families improved (+7.0%/+9.0%), all 4 OLD families softened slightly (−5% to −12%) (§14) — a pattern that did NOT fully replicate at full scale (§21).
15. **V3 vs V4 structural diversity?** Genuine zone-only cycles 1/16→5/24; max upstream catchment 5→9; several other dimensions broadened, several did not (honestly itemized, §19).
16. **Did matched-state concept shift shrink after conditioning on graph context?** Partially — mean spread shrank 12% (0.228→0.200), max spread did not shrink (0.644→0.690) (§20).
17. **Dataset V4 + old-schema family-holdout result (vs V3-population reproduction)?** Diversity alone (training on all 5 other V4 families vs. just the 3 other old families) improved 3 of 4 old families, dramatically for `twin_stair_highrise` (+80.9%) (§21).
18. **Dataset V4 + new-schema family-holdout result?** Helped only `ring_corridor` (+3.9% over the diverse-population baseline); hurt the other 5 families, including `multi_wing` (§21).
19. **How much improvement came from data diversity?** The dominant lever — up to +124% vs. e8d728a for `twin_stair_highrise`, positive for 3 of 4 old families, never negative for any old family (§21, §24).
20. **How much came from graph context?** Small and inconsistent once diversity is already present — positive for 1 of 6 families, negative for the other 5 (§21, §25).
21. **New-family holdout results?** `multi_wing` PR-AUC 0.40/0.39 (old/new schema), `ring_corridor` 0.45/0.47 — both comfortably beat deterministic (3.2×-3.5× lift), neither "unlearnable" (§23).
22. **Original-family regression results?** No family regressed >50% (the charter's own catastrophic-regression bar); `single_exit_lowrise`'s new-schema variant (−21.8% vs e8d728a) is the one real, flagged, non-catastrophic regression (§24).
23. **Does ML beat deterministic intelligence in every family?** Yes, unambiguously — every one of the 16 family-holdout fits, both schemas, 2.4×-5.3× lift (§21).
24. **Betweenness ablation?** The WEAKEST individual graph-context feature in 3 of 4 ablated families (§25).
25. **Bridge/cut-edge ablation?** Individually competitive, best single feature for `multi_exit_wide` (§25).
26. **Upstream-catchment ablation?** Individually strong, best single feature for `multi_wing` and `twin_stair_highrise` (§25).
27. **Peak memory/runtime?** Full campaign: 469.8s / 653MB available at tightest (never breached 300MB floor). Generalization evaluation: 1,385.2s for 28 XGBoost fits over ~3.9M rows, memory recovered to 1,600-2,700MB available between fits (this milestone's persistently tight ~7.9GB-RAM environment, same discipline as every prior campaign).
28. **Remaining scientific weaknesses?** The representation (graph-context) benefit is real but small/inconsistent once family diversity is present (§21) — not yet well enough understood to promote further without a dedicated ablation-focused follow-up; `single_exit_lowrise`'s specific new-schema regression is unexplained (§24); pilot-vs-full-scale results disagreed for `multi_wing` (§14 vs §21), a reminder that pilot-scale exploratory signals can reverse at scale.
29. **Dataset-readiness verdict?** **B — improved transfer, but not yet sufficient for Model V4** (§27).
30. **Full-suite result?** 4,784/4,784 passing (380.4s).
31. **Commit hash?** See `git log` — this milestone's commit immediately follows `e8d728a` (Cross-Topology Generalization Investigation) on `main`.

**A. Did adding entirely new families solve a problem that more within-family variants could not?** Yes, decisively — V3 already tried more within-family variants (16 vs. V2's 4) and family-level transfer did NOT improve (e8d728a); adding 2 genuinely new families improved it dramatically for the hardest case (`twin_stair_highrise` +124%).

**B. Did graph context reduce topology-specific concept shift?** Partially — mean spread shrank 12%, max spread did not (§20); and once training diversity already includes the new families, the graph-context FEATURES' own marginal contribution is small/mixed (§21) — the two levers overlap rather than being purely additive.

**C. Are all promoted features genuinely available live?** Yes — every promoted field has an EXACT (not approximate) live source, since sim and live share the identical NavigationGraph-building code path (§3, §26).

**D. Is Target V2 still defensible?** Yes — unchanged, mechanically reconfirmed (§10).

**E. Did new-family generalization materially improve?** Yes — both new families transfer well above the deterministic baseline and are not qualitatively different in difficulty from the old families (§23).

**F. Are the original four families still healthy?** Yes — no catastrophic regression; 3 of 4 improved with the diverse-population old schema; only 1 (`single_exit_lowrise`) shows a real, flagged regression with the new schema specifically (§24).

**G. Is Dataset V4 genuinely ready for Model V4 training?** No — verdict B, not C (§27). The diversity lever is proven; the representation lever needs more targeted investigation (which specific families/conditions it helps or hurts, and why) before being trusted at production scale.

**H. Should predictive AI influence Recommendation ranking now?** No — unchanged from every prior milestone's verdict; this remains offline dataset/data-science research, explicitly required by the charter to stay that way regardless of the verdict reached.

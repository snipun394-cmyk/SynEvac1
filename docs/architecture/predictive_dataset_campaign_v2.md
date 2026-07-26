# Predictive Dataset Campaign V2 — Topology Diversity, Stair Repair & Multi-Bottleneck Coverage

Status: **data-generation and validation milestone. No model trained, no scoring code changed, no live runtime touched.** Builds on `docs/architecture/localized_predictive_model_v1.md` (commit `7ee61e9`, 4471/4471 tests passing), which trained the first localized congestion model and found it "PROMISING BUT NEEDS MORE DATA" — specifically flagging Stair prediction as effectively non-functional, multi-bottleneck false-negative rates 11x worse than single-bottleneck, no single-exit coverage, and zero total-lockout rows. This milestone does not train Model V2; it fixes the *dataset* those findings pointed at.

## 1. V1 Stair root cause (Phase 1)

Traced mechanically, not guessed: `ai_registry/training_scenario.py`'s `make_training_building()` constructs its one `Staircase(...)` without ever setting `from_floor_id`. `Staircase.vertical_height()` requires **both** `from_floor_id` and `to_floor_id` to resolve real `Floor` objects (`building.get_floor("")` returns `None` for the never-set default) — with one endpoint unresolved, `vertical_height()` silently returns `0.0`. That zero propagates: `travel_distance()` → `Edge.walking_distance` → simulated stair traversal duration, collapsing the entire stair-crossing to an instantaneous, zero-duration event. With `start_time == end_time` for every stair traversal, `candidate_queue_length` and `candidate_approaching_count` structurally could never register real demand — V1's own apparent ~2.1% Stair positive rate was largely a zero-duration boundary/timestamp artifact, not genuine stair congestion.

`ai_registry/training_scenario.py` was **not modified** — V1 remains byte-for-byte reproducible (`tests/test_predictive_dataset_campaign_v2_pipeline.py`'s `V1BackwardCompatibilityTests` proves `make_training_building()`'s stair still resolves `from_floor_id == ""` and `vertical_height() == 0.0`, unchanged).

## 2. V2 topology architecture (Phase 3)

`predictive_dataset/topologies_v2.py` defines four genuinely different `Building`/`ScenarioDefinition` pairs, reusing the existing `models.building`/`scenario_definition` abstractions (no new graph representation):

| Family | Shape | Purpose | Scenario count |
|---|---|---|---|
| `single_exit_lowrise` | 1 floor, 1 exit, 1 door | No alternative route exists at all — the highest-criticality case V1 had zero coverage of | 500 |
| `twin_stair_highrise` | 3 floors, 2 independent dedicated stairs, high upper-floor occupancy | Genuinely load-bearing stairs against a capacity-limited chokepoint; Door+Stair simultaneous-bottleneck potential | 800 |
| `multi_exit_wide` | 1 floor, hub-and-spoke, 3 exits, 4 doors | Door+Door / Door+Exit simultaneous-bottleneck potential without any floor crossing | 700 |
| `v1_topology_fixed` | Same shape as V1 (2 floors/2 doors/2 exits/1 stair) | Isolates the bug-fix effect alone from the other three families' added diversity | 500 |

Every V2 `Staircase` sets **both** `from_floor_id` and `to_floor_id` explicitly (`tests/test_predictive_dataset_topologies_v2.py`'s `StaircaseVerticalHeightRegressionTests` mechanically guards every V2 stair candidate has `walking_distance > 0.0`, and specifically confirms `v1_topology_fixed`'s stair resolves the exact `vertical_height() == 3.0` V1 should have had).

## 3. Coverage targets (Phase 2)

`predictive_dataset/campaign_config_v2.py`'s `COVERAGE_TARGETS` defines six **measurable** targets before generation, checked mechanically after (not asserted blindly):

| Target | Minimum | Rationale |
|---|---|---|
| Single-exit scenarios | 300 | V1 had zero |
| Multi-floor scenarios | 900 | Stair contention needs floor crossing |
| Stair rows with real demand (`queue_length>0` or `approaching_count>0`) | 1,000 | V1 had zero across 501,696 stair rows |
| High-occupancy scenarios (≥30 occupants) | 300 | V1's FN rate rose with occupancy |
| Multiple-simultaneous-bottleneck rows | 5,000 | V1's single worst operational case |
| Total-lockout scenarios that still contribute rows | 5 | V1 had zero — only zero-row lockouts |

## 4. Full campaign results (Phase 13)

`scripts/run_predictive_dataset_campaign_v2.py`, `master_seed=20270115`, run against the current (500/800/700/500 = 2500-scenario) `topologies_v2.py` configuration.

**Scenario campaign**: 2,500 requested, **2,500 accepted, 0 failed**. Generation+simulation+extraction: 391.5s (31,038 rows/s extraction throughput).

**Row counts**:

| Metric | Value |
|---|---|
| Candidate-time rows | **9,620,196** |
| Distinct contributing scenarios | 2,500 (100%) |
| Rows per scenario (mean) | 3,848.08 |
| Door rows | 4,978,056 |
| Exit rows | 3,711,704 |
| Stair rows | 930,436 |
| Trainable rows (target not `None`) | 9,316,792 |
| Excluded (currently-congested-at-`t`) | 303,404 (3.2%) |
| Overall positive rate | 15.9% |

**Topology family distribution**: `single_exit_lowrise` 500, `twin_stair_highrise` 800, `multi_exit_wide` 700, `v1_topology_fixed` 500 — matches the requested config exactly (no generator rejection/attrition).

**Structural distribution**: floor_count {1: 1200, 2: 500, 3: 800}; exit_count {1: 500, 2: 1300, 3: 700}; stair_count {0: 1200, 1: 500, 2: 800}; door_count {1: 500, 2: 500, 3: 800, 4: 700}.

**Occupancy strata**: LOW 209 (8.4%), MEDIUM 800 (32.0%), HIGH 1,491 (59.6%) — occupant count ranges 3–111, mean 45.9, stddev 28.8, 107 distinct values across the campaign.

**Positive rate by candidate type**: Door 22.8%, Exit 4.3%, Stair 21.5% (spread 18.6 points).
**Positive rate by horizon**: 10s 7.5%, 20s 12.9%, 30s 17.8%, 60s 25.5% — recommended first horizon unchanged at **20s**.

**Blocked-route distribution**: 550 scenarios with a blocked exit, 1,277 with a blocked door, 357 with an unavailable stair, 829 fully open; `fraction_scenarios_with_any_blocked_route`: 66.8%.

## 5. V1 vs V2 comparison

| Metric | V1 | V2 |
|---|---|---|
| Scenarios | 2,000 | 2,500 |
| Candidate-time rows | 2,508,480 | 9,620,196 |
| Topology families | 1 | 4 |
| Single-exit scenarios | 0 | 500 |
| Stair positive rate | 2.1% | **21.5%** |
| Stair `queue_length`/`approaching_count` nonzero rows | 0 / 501,696 | 207,384 / 380,924 (of 930,436) |
| Stair `candidate_walking_distance` | constant `0.0` | genuinely varying, nonzero |
| Multi-bottleneck rows | not separately tallied | 1,024,108 |
| Total-lockout scenarios with rows | 0 | 88 |

## 6. Stair repair evidence (Phase 14) — direct comparison

`predictive_dataset/topology_analysis_v2.py`'s `stair_feature_repair_report`, run against the full 930,436 V2 Stair rows:

| Field | V1 (all 501,696 stair rows) | V2 (all 930,436 stair rows) |
|---|---|---|
| `candidate_walking_distance` | constant `0.0` (every row) | 2 distinct nonzero values; `walking_distance_is_zero_for_all_rows`: **False** |
| `candidate_queue_length` nonzero | 0 rows (0.0%) | 207,384 rows (22.3%), max 31 |
| `candidate_approaching_count` nonzero | 0 rows (0.0%) | 380,924 rows (40.9%), max 20 |
| `currently_congested == True` | not applicable (see below) | 0 rows |
| Stair positive rate | 2.1% | **21.5%** |
| `zero_walking_distance_candidates` flagged | n/a (check didn't exist yet) | **0 of 21** candidates flagged |

`currently_congested_true_count: 0` for Stair is expected, not a gap — V1's own §5/§10 documented the same `[join_time, start_time)` interval-membership definition, and V2 reuses it unchanged (no new congestion-detection logic was introduced by this milestone).

**A more nuanced finding, not hidden**: candidate-level utilization (`predictive_dataset.diversity.candidate_utilization_report`) shows the bug fix alone is not sufficient by itself:

| Candidate | Family | Active fraction |
|---|---|---|
| `v1f-stair-1` | `v1_topology_fixed` (bug fix only, same shape/occupancy as V1) | **1.2%** |
| `tsh-stair-2` | `twin_stair_highrise` (bug fix + high-occupancy dedicated-stair design) | 47.6% |
| `tsh-stair-3` | `twin_stair_highrise` | 53.1% |

Fixing `from_floor_id` alone moves `v1_topology_fixed`'s stair from **0.0% → 1.2%** active — a real, mechanically-proven improvement (from literally impossible to observe, to observable), but still low in absolute terms, consistent with V1's own §10 finding that a low-traffic, always-first-hop stair candidate has genuinely little real queueing to detect in that specific occupancy configuration. The overwhelming majority of the 21.5% campaign-wide Stair positive rate comes from `twin_stair_highrise`'s deliberate topology/occupancy design (high upper-floor demand against a capacity-limited dedicated stair), not the bug fix in isolation. **Both were necessary**: the bug fix makes stair demand *representable* at all; the topology diversity makes it *actually occur* at meaningful volume.

## 7. Single-exit coverage (Phase 4)

500 `single_exit_lowrise` scenarios (target: 300) — a topology with exactly one exit and no alternative route, engineered so the sole exit can itself draw `CLOSED` (the genuine total-lockout case for this family, §11). V1 had zero scenarios of this shape.

## 8. Multi-floor coverage (Phase 5)

1,300 scenarios drawn from a 2+-floor family (`twin_stair_highrise`: 800, `v1_topology_fixed`: 500) against a target of 900.

## 9. High-occupancy coverage (Phase 8)

1,491 scenarios with ≥30 occupants (target: 300) — occupant count ranges 3–111 campaign-wide, mean 45.9. `twin_stair_highrise`'s upper-floor zones were deliberately drawn from wide, high ranges (10–35) specifically to stress-test stair capacity.

## 10. Multi-bottleneck coverage (Phase 15) — V1's weakest operational case

1,024,108 rows belong to a (scenario, time, horizon) bucket with 2+ simultaneously-positive candidates — 205x the 5,000-row target. Bucket-level breakdown: 460,159 no-or-single-bottleneck buckets vs **369,791 multiple-bottleneck buckets**. Candidate-type-pair breakdown across those multi-bottleneck buckets:

| Combination | Count |
|---|---|
| Door+Door | 165,877 |
| Door+Stair | 120,788 |
| Door+Exit | 114,395 |
| Exit+Stair | 53,953 |
| Stair+Stair | 3,669 |
| Exit+Exit | 571 |

Every combination V1's error analysis called out by name (Door+Door, Door+Exit, Door+Stair, Stair+Exit) is now represented at real volume. Same-type double-bottlenecks (Stair+Stair, Exit+Exit) are comparatively thin — expected, since most topologies here have only 1–2 stairs/exits structurally available to co-congest.

## 11. Total-lockout semantics (Phase 9)

**Decision (Option B)**: every V2 scenario's `SimulationRuntime` receives an explicit `minimum_end_time_seconds = 30.0` floor (`runtime.clock.end_time = max(runtime.clock.end_time, config.minimum_end_time_seconds)`), applied **uniformly to every scenario**, not as a special case detected after the fact. `resolve_default_end_time()`'s own natural formula already exceeds this floor for any scenario with real movement or scheduled events — the floor only ever changes behavior for the genuine "nobody can ever move" case, where the natural formula would otherwise resolve to `0.0` (V1's exact §2 failure mode: 34/2000 scenarios contributed zero rows this way).

**Why rows, not zero rows**: a total-lockout scenario is not absent of information — occupants exist, candidates exist, and "nobody moved because every route is blocked" is itself a real, observable state a predictive model should be able to see (e.g. `candidate_traversable=False`, zero queue activity). Fabricating movement would be dishonest; but reporting zero rows for a state the campaign nonetheless simulated is also a loss of real information. Giving the clock a small nonzero floor lets the simulator's own already-legitimate current-state extraction machinery (occupancy snapshots, structural features) produce real, non-fabricated rows for these ticks, without inventing any occupant movement that didn't happen.

**Result**: 88 total-lockout scenarios (every exit blocked/closed at t=0), **all 88 contributing rows** (`total_lockout_zero_rows: 0`) — against V1's 34 zero-row lockout scenarios and a 5-scenario target.

## 12. Sim/live feature parity (Phase 12)

**V2 introduces zero new predictive features.** `predictive_dataset/schema.py`, `simulation_extractor.py`, and `live_extractor.py` are untouched by this milestone (`git diff --stat` confirms only `campaign_config_v2.py`, `topologies_v2.py`, `topology_analysis_v2.py`, `quality_checks.py`, and `versioning.py` changed). The same 8-field `CANDIDATE_FEATURE_SCHEMA` and live-parity matrix documented in `docs/architecture/localized_predictive_ai_dataset.md` §7 apply unchanged:

| Feature | Live-observable? | Live-estimable? | Missing-data semantics |
|---|---|---|---|
| `total_active_occupant_count` | No | Yes | `None` when no occupancy facts supplied |
| `candidate_type` | Yes | — | Never missing |
| `candidate_capacity` | Yes (identical function call) | — | `None` if capacity model can't derive one |
| `candidate_walking_distance` | Yes (shared Navigation Graph) | — | `None` if an endpoint has no geometry |
| `candidate_traversable` | Yes | — | Never missing (known limitation: doesn't incorporate mid-scenario overrides, unchanged from V1) |
| `candidate_adjacent_zone_occupancy` | No | Yes | `None` when no occupancy facts / no zone |
| `candidate_queue_length` | No | Yes (geometric proxy) | Live: `None` when position unavailable |
| `candidate_approaching_count` | No | Yes (geometric proxy) | Live: `None` when position unavailable |
| `candidate_congestion_level` | No | Yes | `None` when capacity/demand missing |

`tests/test_predictive_dataset_campaign_v2_pipeline.py`'s `LeakageReAuditV2Tests` re-runs the identical parity/purity check against a real V2 topology's Stair candidate specifically (the repaired type this milestone centers on) — same result, unchanged from V1.

## 13. Leakage audit (Phase 10/16)

Re-confirmed, not re-litigated: feature extraction at time `t` may only use information available at or before `t`; only target generation may inspect `(t, t+horizon]`. `tests/test_predictive_dataset_leakage_guards.py` (unchanged) and `tests/test_predictive_dataset_architecture_guards.py` (unchanged, import-boundary guards: `live_extractor`/`simulation_extractor`/`schema` never import `target_generator`) both still pass. `predictive_dataset_campaign_v2_pipeline.py` adds one more leakage proof specific to V2's repaired Stair candidate (`test_stair_feature_extraction_is_still_blind_to_the_future_on_a_v2_topology`) — identical feature rows given identical up-to-`t` state, regardless of what happens after `t`. **No feature exceeds the 0.9 leakage-review threshold** (`flagged_for_leakage_review: []`); strongest correlation is `candidate_queue_length` at r=0.49 (current queueing correlating with near-future congestion — expected, not suspicious).

## 14. Data-quality results (Phase 16)

All checks run against the full 9,620,196 rows:

| Check | Result |
|---|---|
| Duplicate exact rows | 0 |
| Duplicate identity keys | 0 |
| Invalid candidate IDs | 0 |
| Missing `target` key | 0 |
| NaN values | 0 |
| Invalid ranges | none (`{}`) |
| Invalid candidate types | 0 |
| Invalid congestion levels | 0 |
| `currently_congested`/`target` inconsistencies | 0 |
| Zero-walking-distance candidates flagged | **0 of 21** |
| Duplicate scenario IDs | 0 |
| Redundant feature pairs (r≥0.95) | none (`[]`) |

**A genuine bug was found and fixed during this milestone, not hidden**: `run_quality_checks`'s exact-duplicate-row detector originally keyed a dict by the full sorted tuple of every row's ~19 fields. At V1's ~2.5M-row scale this was survivable; at V2's 9.6M rows, retaining millions of large tuple objects as dict keys drove the analysis process to **8.4GB private memory on a 7.3GB-RAM development machine** (0.6GB free), and the step was still running after 80+ minutes when it was killed. Fixed by hashing the fingerprint (`hash(tuple(sorted(row.items())))`) instead of retaining the tuple itself as the dict key — collision probability across 9.6M rows is ~2.5×10⁻⁶ by the birthday bound, and even a collision would only ever merge two coincidentally-identical-hash duplicate counts, never affect the underlying dataset rows. Re-run: the same check completed in **65.6s** with memory holding flat (~5.6GB, no growth), and all 14 pre-existing unit tests pass unchanged, confirming identical semantics at both small and full scale (0 duplicates found either way, as expected for a correctly-generated campaign).

A second, independent bug was also found and fixed: `operational_coverage_report`'s `building_topology_note` was a hardcoded string from V1 ("every scenario uses the SAME fixed Building... single-exit topologies are NOT represented") that became factually false the moment it ran against V2's multi-topology data. Fixed to derive the note from the actual `scenario_metadata` passed in (3 new tests added: single-family fallback, multi-family accurate reporting, missing-key graceful fallback for V1 compatibility) — patched into the already-generated report without re-running the full campaign, since the fix only affects a derived string, not any counted statistic.

## 15. Dataset version metadata

```json
{
  "schema_version": "1.0",
  "campaign_version": "predictive_dataset_campaign_v2",
  "feature_version": "1.0",
  "target_version": "v1-congestion-threshold-2-horizon-window",
  "prediction_horizon_seconds": 20.0
}
```

`predictive_dataset.versioning.dataset_version()` now accepts an optional `campaign_version` parameter (defaults to V1's own `CAMPAIGN_VERSION` for full backward compatibility — every existing V1 call site is unaffected). `tests/test_predictive_dataset_campaign_v2_pipeline.py`'s `DatasetVersionMixingGuardTests` proves `predictive_model.dataset_loader` mechanically rejects a V1-tagged dataset against a V2-expecting requirement (and vice versa) — V1 and V2 datasets can never be silently mixed by a future training script.

## 16. Remaining limitations

- **Exit congestion is now comparatively rare (4.3% positive rate)** across the diversified topology set — structurally, exits in these building shapes bottleneck far less often than doors or stairs. Model V2 evaluation should watch Exit-specific recall separately, the same way this milestone watched Stair.
- **Occupancy strata skew toward HIGH** (59.6% of scenarios) by design (`twin_stair_highrise`'s stair-stress requirement demanded high upper-floor occupancy) — LOW-occupancy coverage (8.4%) is comparatively thin relative to V1's more even spread.
- **Same-type multi-bottleneck combinations are thin** (Stair+Stair 3,669 rows, Exit+Exit 571 rows) — an artifact of most topologies here having only 1–2 stairs/exits, not something this milestone attempted to force.
- **`v1_topology_fixed`'s stair utilization (1.2%) remains low in absolute terms** even after the bug fix — consistent with V1's own §10 finding that a low-traffic, always-first-hop candidate has genuinely little real queueing in that specific building/occupancy shape. This is a property of that particular topology, not a remaining defect in the fix itself (proven by `twin_stair_highrise`'s 47–53% utilization under a stair-stressing topology).
- `candidate_traversable` still does not incorporate mid-scenario `ScenarioEvent` door/exit/stair overrides — a disclosed limitation carried over unchanged from V1.
- This milestone did not train Model V2, did not touch `LiveRuntime`, recommendation scoring, exit ranking, guidance, or dynamic signage — all deliberately out of scope, per the milestone's own instructions.

## Final report

- **A.** Yes — Stair blindness is genuinely fixed at the DATA level: positive rate 2.1%→21.5%, `queue_length`/`approaching_count` went from constant-zero across all 501,696 V1 rows to real, varying, nonzero demand on hundreds of thousands of V2 rows, and `walking_distance` is no longer degenerate for any of the 21 checked candidates.
- **B.** Yes — 1,024,108 multi-bottleneck rows (205x the 5,000-row target), covering every candidate-type combination V1's error analysis named.
- **C.** Yes — 4 structurally distinct topology families (1/2/3-floor shapes; 1/2/3-exit shapes; 0/1/2-stair shapes), not a single reused Building.
- **D.** Yes — 500 single-exit scenarios, absent entirely from V1.
- **E.** Yes — 1,491 scenarios with ≥30 occupants (target 300), occupant counts up to 111.
- **F.** Yes — every V2 predictive feature is the exact same 8-field schema already audited for live parity in V1; V2 added zero new features, so no new parity gap was introduced.
- **G.** No — leakage guards, architecture import-boundary guards, and correlation-threshold checks all pass unchanged; no feature exceeds the 0.9 leakage-review threshold.
- **H.** Yes, materially better, not merely larger: V2 is 3.8x the rows (9.6M vs 2.5M) but the meaningful improvement is qualitative — a previously non-functional candidate type (Stair) is now genuinely learnable, the single worst operational failure mode (multi-bottleneck) now has real volume, and two topology gaps (single-exit, total-lockout-with-rows) that had literally zero representation in V1 are now covered.
- **I.** Yes, with the limitations in §16 disclosed and carried forward into Model V2's own evaluation design (particularly: evaluate Exit and Stair recall separately, and treat `v1_topology_fixed`'s low-traffic stair as a known low-utilization case rather than a modeling failure if it underperforms).

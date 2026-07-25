# Predictive Dataset Campaign v1 — Large-Scale Generation & Validation

Status: **data-quality validation milestone. No model trained, no scoring code changed, no live runtime touched.** Builds on `docs/architecture/localized_predictive_ai_dataset.md` (commit `1581915`, 4384/4384 tests passing), which established the `SCENARIO × TIMESTEP × CANDIDATE` data foundation at validation scale (40 scenarios). This milestone scales that up to production scale and asks whether the result is actually *good* data, not just *more* data.

This campaign found and fixed **two real bugs** and **one root-caused, disclosed structural limitation** — all only visible at real campaign scale, none anticipated by hand-built unit fixtures. That is the central finding of this milestone: **the validation-scale (40-scenario) campaign in the prior milestone could not have caught any of these.**

## 1. Campaign configuration (Phase 1)

`predictive_dataset.campaign_config.build_campaign_config()` — a documented, versionable, JSON-serializable configuration (`CampaignConfig`), reusing `ai_registry.training_scenario`'s already-validated Building/ScenarioDefinition fixture (2 floors, 4 zones, 2 doors, 2 exits, 1 stair) rather than authoring a new one. Every distribution the campaign actually draws from is captured verbatim (each `scenario_definition.Distribution`'s own `to_dict()`, not re-described): occupant ranges per zone (`UniformRange`, 0–32 people depending on zone), fire ignition-zone preference and growth-parameter range (80–420s), allowed fire profiles, door/exit/stair/camera engineering-state distributions (the actual blocked/locked/closed probabilities), firefighter deployment. `master_seed=20260726`, deterministic — the same seed reproduces the same 2000 scenarios byte-for-byte (`tests/test_predictive_dataset_campaign_config.py` proves this for `build_campaign_config()` itself).

## 2. Scenario campaign scale (Phase 2)

**2000 scenarios requested, 2000 accepted, 0 failed** (`scripts/run_predictive_dataset_campaign_v1.py`, ~50s wall time total). 1000 was this milestone's own documented practical default (see that script's module docstring); 2000 was run instead once the validation-scale timing (40 scenarios in under a second) showed the larger number was comfortably practical.

There is no separate discarded/invalid category to report: `scenario_generator.generator`'s own docstring is explicit that it is a pure sampler, not a validator ("no accept/reject branch exists anywhere in this module... every attempt is attempt 0, always accepted") — no `scenario_validator/` package exists yet in this codebase. Every one of the 2000 requested scenarios generated and simulated without an exception.

**34 of the 2000 accepted scenarios (1.7%) contributed ZERO candidate-time rows** despite running without error — a real finding, not swept under "accepted". Root cause, fully diagnosed (§9 has the full mechanism): `EngineeringConstraints.min_open_exits=1` is Definition *metadata* that a not-yet-built Scenario Validator would enforce — `scenario_generator.generator` never consults it, so a scenario can legitimately draw **both exits closed simultaneously**. With nobody able to ever arrive and no scheduled events, `simulation_runtime.clock.resolve_default_end_time()`'s own formula (`max(last_arrival or 0.0, last_event_time or 0.0)`) resolves to `0.0`, so `SimulationRuntime.run()` produces zero ticks. Confirmed 34/34: every zero-row scenario has `blocked_exit_count == 2`. This is an upstream `scenario_generator`/`simulation_runtime` edge case, not a `predictive_dataset` bug — not fixed in this milestone (see §12).

## 3/15. Dataset statistics (Phase 3)

| Metric | Value |
|---|---|
| Candidate-time rows | **2,508,480** |
| Distinct scenarios contributing rows | 1,966 (of 2000 accepted — see §2) |
| Rows per scenario (mean) | 1,254.24 |
| Trainable rows (target not `None`) | 2,447,904 |
| Excluded (currently-congested-at-`t`) | 60,576 (2.4%) |
| Positive | 363,215 |
| Negative | 2,084,689 |
| Overall positive rate | 14.8% |

By candidate type (rows): Door 1,003,392, Exit 1,003,392, Stair 501,696 — proportional to candidate count (2 doors, 2 exits, 1 stair), as expected since every candidate gets one row per tick per horizon.

## 4. Scenario diversity (Phase 4)

Scenarios are genuinely different, not just numerous:

| Dimension | Spread |
|---|---|
| Occupant count | min 5, max 32, mean 17.9, stddev 4.7, 27 distinct values |
| Ignition zone | all 4 zones represented (292–632 scenarios each) |
| Fire growth time | min 80.7s, max 419.9s, 2000 distinct values (continuous) |
| Blocked doors | 0–2, mean 0.58 |
| Blocked exits | 0–2, mean 0.27 |
| Unavailable stairs | 0–1, mean 0.15 |
| Evacuation duration | min 16.1s, max 1833.2s (1966 scenarios that produced a duration), 1966 distinct values |
| Scenarios with ≥1 blocked route | 68.4% |

**Candidate utilization** (`predictive_dataset.diversity.candidate_utilization_report`, fraction of a candidate's own rows showing any recorded queue/approach demand): door-1 60.0%, door-2 38.8%, exit-1 63.8%, exit-2 28.5%, **stair-1 0.0%** — the last one is a real, root-caused finding, not noise (§9).

## 5. Feature distributions (Phase 5)

Full per-feature min/max/mean/median/stddev/missing/constant/near-constant/outlier report: `predictive_dataset.feature_statistics.feature_distribution_report()`, run over all 2,508,480 rows.

| Feature | Missing % | Constant? | Notes |
|---|---|---|---|
| `total_active_occupant_count` | 0% | No | 0–32, mean 8.7 |
| `candidate_type` | 0% | No | Door 1,003,392 / Exit 1,003,392 / Stair 501,696 |
| `candidate_capacity` | 0% | No | 1–50, mean 20.6 |
| `candidate_walking_distance` | 0% | No | 0–60.6m, mean 27.1m |
| `candidate_traversable` | 0% | **No — fixed** (§8) | 90.5% True / 9.5% False |
| `candidate_adjacent_zone_occupancy` | 57.1% | No | Honest `None` when the approach-side node has no zone reading this tick, never fabricated as 0 |
| `candidate_queue_length` | 0% | No | 0–14, mean 0.62, 3.1% outliers (>3σ) |
| `candidate_approaching_count` | 0% | **No — fixed** (§9) | 0–20, mean 1.03 |
| `candidate_congestion_level` | 0% | No | LOW 2,046,428 / HIGH 109,384 / CRITICAL 352,668 |

No feature is missing entirely; the one genuinely high missing-rate field (`candidate_adjacent_zone_occupancy`) is honestly `None`, not fabricated, exactly per this schema's own missing-data policy (`docs/architecture/localized_predictive_ai_dataset.md` §7). **Two constant-value findings during this analysis were the two real bugs this milestone found and fixed** — see §8/§9.

## 6. Label analysis (Phase 6)

By candidate type — a genuine, expected asymmetry, not a data-quality defect:

| Type | Positive rate |
|---|---|
| Door | 30.1% |
| Exit | 5.4% |
| Stair | 2.1% |

Doors, as the narrower internal chokepoints between zones in this building, congest far more often than exits or the stair. By building occupancy level: LOW 10.2%, MEDIUM 14.1%, HIGH 16.0% (occupancy driving congestion is an expected causal relationship, not bias). By fire severity (fire-growth-time proxy — see §10's disclosed caveat): FAST 15.2%, MODERATE 15.0%, SLOW 14.5% — essentially flat, meaning congestion in this dataset is driven by crowd/geometry dynamics far more than by fire severity, which is itself an honest, useful finding (fire severity is not a meaningful predictor by itself; a future model should not expect it to carry much signal alone).

## 7. Temporal coverage (Phase 7)

Positive rate by evacuation phase (each scenario's own EARLY/MID/LATE third, not an absolute time bucket — see `predictive_dataset.label_analysis.temporal_coverage_report`):

| Phase | Positive rate |
|---|---|
| EARLY | 25.0% |
| MID | 14.3% |
| LATE | 6.4% |

Congestion prediction is most valuable early in an evacuation (when the most people are simultaneously moving) and least useful late (most occupants have already cleared). This is an honest property of evacuation dynamics, not a dataset flaw — a future model's practical value is concentrated in the early-to-mid phase, which should inform how it's evaluated (e.g., don't expect uniform recall across the whole evacuation timeline).

## 8. First bug found and fixed: `candidate_traversable` wiring (Phase 5/9)

**Finding:** `candidate_traversable` reported a **constant `True` across all 2,508,480 rows** in the first full-scale run, despite 1,798 of 2000 scenarios (68.4% minus the fully-open 31.6%) drawing at least one blocked door/exit/stair.

**Root cause:** `scripts/run_predictive_dataset_campaign_v1.py` passed the shared, pristine template `Building` into `TimelineRun(building=...)` instead of `context.building` — the scenario-initialized *copy* `scenario_runner.building_initializer.build_initialized_building()` actually mutates with this scenario's resolved door/exit blocked/locked state. `predictive_dataset.simulation_extractor`'s `candidate_traversable` reads `Edge.traversable` straight off `Edge.reference` (the Door/Exit object) — against the wrong object, it always saw the template's own unblocked defaults. The actual simulated movement was **never affected** (`context.simulation`/`context.graph` were already built from the correct copy inside `scenario_runner.run()`) — only this one derived feature was reading the wrong source.

**Fix:** both campaign scripts (`scripts/generate_predictive_dataset_campaign.py` and this milestone's own) now pass `context.building`. `predictive_dataset/simulation_extractor.py`'s docstring now states the caller contract explicitly. Regression tests added: `tests/test_predictive_dataset_extractors.py::BlockedCandidateTraversabilityTests`.

**After the fix:** 90.5% True / 9.5% False — real, expected variance.

## 9. Second bug found and fixed: `candidate_approaching_count` (Phase 4/5/8)

**Finding, round 1:** `predictive_dataset.diversity.candidate_utilization_report` showed `stair-1: 0% active` across all 501,696 of its rows, in a campaign where the building diversity check (§4) confirms the stair is drawn AVAILABLE ~85% of the time. `door-2`/`exit-2` were also visibly lower-utilized than their siblings.

**Root cause, round 1:** the v1 field definition (from the prior milestone) used `route.edges[-1]` (the occupant's route's *final* edge). Every complete evacuation Route necessarily ends at an Exit (the only way to reach `Node.OUTSIDE_NODE_ID`) — so this made `candidate_approaching_count` **structurally, permanently zero for every Door and Stair candidate**, not a rare gap.

**Fix, round 1:** redefine as the occupant's immediate next not-yet-started route edge (generalizes to any position in the route, not just the final one).

**Finding, round 2:** after that fix, `predictive_dataset.feature_statistics.feature_distribution_report` showed `candidate_approaching_count` was **still constant `0.0` across all 2,508,480 rows** — the fix hadn't actually produced any variance at all.

**Root cause, round 2:** verified directly against real `OccupantTimelineStep` data from an actual simulation run — every recorded step has `join_time == depart_time` (first step) or `join_time == the previous step's end_time` (later steps), with no exception found. This simulator's occupant movement model (`simulator/coordinator.py`) has **no observable gap** between "just departed / just finished an edge" and "joined the next edge's queue" — the "immediate next hop, not yet joined" window this round-1 fix relied on never exists in practice.

**Fix, round 2 (final):** redefine once more as *any* not-yet-reached edge in the occupant's remaining route (not just the immediate next one) — `predictive_dataset.simulation_extractor._not_yet_reached_edge_ids()`. This has no degenerate window: it only requires "this candidate is still ahead of me in my plan," true from the instant of departure.

**After the fix:** `candidate_approaching_count` ranges 0–20, mean 1.03, stddev 2.34, genuinely non-constant. `predictive_dataset/simulation_extractor.py`'s own "BUG HISTORY" comment documents all three versions for future readers. Regression tests: `tests/test_predictive_dataset_extractors.py::MultiHopApproachingCountTests`.

## 10. Remaining, disclosed limitation: `stair-1` demand-blindness (not fixed in this milestone)

Even after the round-2 fix, **`stair-1` still shows 0% utilization** (`candidate_queue_length` and `candidate_approaching_count` both constant `0` across all 501,696 stair rows) — verified this is *not* a third bug: a direct diagnostic run confirmed real occupants genuinely do traverse `stair-1` in this campaign (87 occupants in a 50-scenario sample alone).

**Root cause, fully diagnosed:** only zone-upper occupants ever use `stair-1`, and for every one of them it is their **first** route hop. `candidate_queue_length` only registers demand during a genuine `[join_time, start_time)` queueing interval, which is empty whenever `queue_wait_time == 0` (stair capacity is never actually contended by the small zone-upper population in this building/definition). `candidate_approaching_count`'s "not yet reached" definition (§9) excludes an edge the instant its `join_time <= time` — and for a first hop, `join_time == depart_time` always (the same zero-gap finding from §9), so the edge is marked "reached" in the very same instant the occupant becomes observable at all. **A first-hop candidate with no real queueing is invisible to both of this schema's demand-signal features simultaneously** — not a rare fluke, a structural blind spot shared by both definitions whenever those two conditions coincide, which they always do for `stair-1` in this building.

**Not fixed in this milestone** — a third redefinition of `candidate_approaching_count` risks the same fate as the first two (an untested guess about simulator mechanics), and this milestone's job is to validate and document the dataset, not indefinitely re-engineer features. `candidate_adjacent_zone_occupancy` and `candidate_type`/`candidate_capacity`/`candidate_walking_distance`/`candidate_traversable` remain honest and correctly varying for stair rows — only the two demand-signal fields are blind here. **Recommended improvement** (§14): a future feature version could add a broader "route membership regardless of reached/unreached status" signal specifically to close this gap, at the disclosed cost of blurring "already passed" from "still relevant."

## 11. Correlation checks (Phase 8)

`predictive_dataset.correlation.feature_target_correlations()` — Pearson's r between every numeric feature and the binary target, trainable rows only:

| Feature | r with target |
|---|---|
| `candidate_queue_length` | **0.547** |
| `total_active_occupant_count` | 0.234 |
| `candidate_capacity` | -0.211 |
| `candidate_walking_distance` | 0.160 |
| `candidate_approaching_count` | 0.049 |
| `candidate_adjacent_zone_occupancy` | 0.050 |

**No feature exceeds the 0.9 leakage-review threshold** (`flagged_for_leakage_review: []`) — the strongest predictor (`candidate_queue_length`, current queueing) has a moderate, entirely expected correlation with *future* congestion, not a suspicious near-1.0 one. **No redundant feature pairs** were found either (`redundant_feature_pairs: []`, 0.95 threshold) — the six numeric features are not collinear duplicates of each other. `candidate_type`'s categorical association with target has a 28.1-point positive-rate spread (Door 30.1% vs Stair 2.1%) and `candidate_congestion_level`'s has a 60.9-point spread (LOW 3.6% vs CRITICAL 64.4%) — both real, expected, useful signal, not broken features.

## 12. Data quality checks (Phase 9)

`predictive_dataset.quality_checks.run_quality_checks()`, all 2,508,480 rows: **zero duplicate exact rows, zero duplicate identity keys (scenario/time/candidate/horizon), zero invalid candidate ids, zero missing target keys, zero NaN values, zero invalid feature ranges, zero invalid candidate types or congestion levels, zero `currently_congested`/`target` inconsistencies.** `predictive_dataset.quality_checks.duplicate_scenario_ids()`: zero duplicate scenario ids across all 2000 generated scenarios.

**Leakage re-check at scale:** no new leakage mechanism found beyond what `docs/architecture/localized_predictive_ai_dataset.md` §10/§16 already mechanically enforces (import-boundary guard + invariance test) — §11's correlation results are consistent with that (no near-1.0 correlations that would suggest an undetected leak).

## 13. Operational coverage (Phase 10)

| Situation | Representation |
|---|---|
| Low occupancy scenarios (≤10 people) | 118 (5.9%) |
| Medium occupancy (11–19) | 1,127 (56.4%) |
| High occupancy (≥20) | 755 (37.8%) |
| Blocked exit scenarios | 510 (25.5%) |
| Blocked door scenarios | 991 (49.6%) |
| Blocked stair scenarios | 297 (14.9%) |
| Fully open (no blocked route at all) | 633 (31.6%) |
| No bottleneck at all | 101 (5.1%, includes the 34 zero-row total-lockout scenarios) |
| Single bottleneck candidate | 978 (48.9%) |
| Multiple bottleneck candidates | 921 (46.1%) |
| Fast fire growth (more severe, proxy) | 398 (19.9%) |
| Slow fire growth (less severe, proxy) | 767 (38.4%) |

**Fire/smoke severity is a proxy only** (fire growth time) — direct hazard/smoke-score coverage cannot be checked, since hazard is deliberately excluded from the deployable candidate feature schema (`SIMULATION_ONLY`, see `docs/architecture/localized_predictive_ai_dataset.md` §7).

**Missing operational cases (real gaps, disclosed):**

- **Single-exit buildings are not represented at all.** Every scenario in this campaign uses the same fixed Building (2 exits, 2 doors, 1 stair, per `ai_registry.training_scenario.make_training_building()`). A single-exit topology — arguably the highest-criticality case for a bottleneck predictor — has zero coverage.
- **Building topology diversity in general is zero** — one building, always. Candidate count, floor count, and zone adjacency never vary.
- **Total-lockout scenarios (both exits blocked) are present in the SCENARIO metadata (34 of them) but contribute ZERO candidate-time rows** (§2) — the single most severe operational situation this campaign could produce is invisible to the actual dataset, an important and disclosed gap.
- **Stair demand is present in scenario terms (candidate rows exist, real occupants use it) but blind on two of three feature dimensions** (§10).

## 14. Dataset versioning (Phase 11)

`predictive_dataset.versioning.dataset_version()`:

| Field | Value |
|---|---|
| `schema_version` | `1.0` |
| `campaign_version` | `predictive_dataset_campaign_v1` |
| `feature_version` | `1.0` |
| `target_version` | `v1-congestion-threshold-2-horizon-window` |
| `prediction_horizon_seconds` | `20.0` (this campaign's own recommendation — see below) |

This is the version identifier a future training experiment should cite as its baseline. **Recommended improvements for a `v2` feature/campaign version** (not implemented here, out of this milestone's scope):

1. Add building-topology variation (at minimum: a single-exit building) to close the single-exit coverage gap (§13).
2. Give `SimulationRuntime`/campaign tooling an explicit minimum `end_time` floor so total-lockout scenarios (§2) produce at least one tick of "everyone stuck, nobody moving" data instead of zero rows — this is plausibly a legitimate, informative negative example currently missing entirely.
3. Investigate a route-membership-based (not reached/unreached-based) demand signal to close the `stair-1`-style blind spot (§10), OR accept it as a documented, permanent characteristic of low-traffic first-hop candidates and account for it in future model evaluation (e.g., report metrics separately by candidate type, which this milestone's own label analysis already supports doing, §6).
4. Consider whether `min_open_exits` (and other `EngineeringConstraints` fields not currently enforced by `scenario_generator.generator`) should be enforced by a future `scenario_validator/` package, independent of this dataset's own needs.

Recomputing `recommended_first_horizon_seconds` at 2000-scenario scale reproduced the prior milestone's 40-scenario finding exactly: **20 seconds**, the shortest horizon clearing both the genuine-advance-warning floor (≥20s) and the statistical-usability floor (`predictive_dataset.analysis.recommend_first_horizon()`), corroborating rather than merely repeating the smaller campaign's result.

## Extraction performance (Phase 18 continuation)

| Metric | Value |
|---|---|
| Simulation execution (2000 scenarios) | 16.5s |
| Dataset extraction (2,508,480 rows) | 32.6s |
| Extraction throughput | ~76,900 rows/second |
| Total campaign wall time | ~50s |

Consistent with the prior milestone's validation-scale numbers (~85,000 rows/second) — extraction performance scales linearly and remains comfortably fast; a future 10,000+-scenario campaign remains practical on a single machine.

## Known limitations (summary)

- 34/2000 scenarios (both-exits-blocked total lockouts) contribute zero rows (§2/§13) — an upstream `scenario_generator`/`simulation_runtime` interaction, not fixed here.
- `stair-1`-style low-traffic, always-first-hop candidates are structurally blind to both demand-signal features (§10) — root-caused, not fixed here.
- Single building topology only; no single-exit scenario representation (§13).
- Fire/smoke severity coverage is a growth-time proxy, not a direct hazard measurement (§13), consistent with hazard's deliberate exclusion from the deployable schema.
- `candidate_traversable` reflects each scenario's *initial* resolved door/exit/stair state correctly (§8) but still does not incorporate mid-scenario `ScenarioEvent` overrides — a disclosed limitation already carried over from the prior milestone's own documentation, unchanged by this one.

## Full-suite result (Phase 13)

Baseline entering this milestone: 4384/4384 (prior milestone). This milestone added regression tests for both fixes (`BlockedCandidateTraversabilityTests`, `MultiHopApproachingCountTests`) plus dedicated unit tests for every new analysis module (`campaign_config`, `versioning`, `feature_statistics`, `quality_checks`, `diversity`, `label_analysis`, `correlation`, `operational_coverage`). See the commit for the exact final count — zero regressions against the 4384 baseline.

## Final report

See the commit message / conversation record for the full point-by-point report and the explicit A–H answers (this document's sections above already answer each of them: A — yes, 2000 scenarios / 2.4M rows is a large, diverse initial dataset, though topology diversity (§13) should improve before a production model; B — yes, single-exit buildings and total-lockout scenarios (§13); C — yes, 14.8% overall, no horizon or type is degenerate (§3/§6); D — no feature is fully constant after the two fixes, though `stair-1`'s demand features are effectively dead for that one candidate (§10); E — no leakage evidence (§11/§12); F — no features should be removed, but `stair-1` rows' demand fields should be interpreted with the §10 caveat in mind; G — 20 seconds (§14); H — the dataset is frozen and versioned (§14) and ready for a **first exploratory** training experiment, with the disclosed limitations above factored into how that experiment is scoped and evaluated).

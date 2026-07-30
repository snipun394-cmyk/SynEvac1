# Stair Predictive-Feature Live Parity

Status: implemented (feature-parity and minimal wiring only — no model training, no model inference, no
Recommendation/Guidance change, no LiveRuntime ML). Builds on `docs/architecture/stair_flow_intelligence.md`
(Live Stair Flow & Movement Direction Intelligence) and the frozen `predictive_dataset/schema.py`
(SCHEMA_VERSION "1.0") / `predictive_dataset/schema_v4.py` (SCHEMA_VERSION_V4 "4.0") feature definitions,
neither of which this milestone modifies.

---

## Phase 1: the simulation feature, traced exactly

`predictive_dataset.simulation_extractor_v2_1._recent_flow_rate(movement_result, candidate_id, time,
window=FLOW_WINDOW_SECONDS)`:

```python
window_start = time - window
count = 0
for timeline in movement_result.occupants.values():
    for step in timeline.steps:
        if step.edge.id != candidate_id:
            continue
        if window_start < step.end_time <= time:
            count += 1
return count
```

- **Numerator/what it counts**: `OccupantTimelineStep`s whose `edge.id` matches this candidate, filtered
  by `step.end_time` — i.e., **completed edge crossings**, not started ones. For a Stair, `navigation.
  graph_builder._add_stair_edges()` creates exactly ONE `Edge(id=stair.id, ...)` per `Staircase`
  connecting `from_zone`↔`to_zone` — the simulator has no intermediate "on the stair" sub-state; one
  `OccupantTimelineStep` represents the WHOLE Zone→Zone traversal, `start_time` = departure, `end_time`
  = arrival.
- **Direction**: not filtered by direction at all — a step crossing the edge in either direction (Zone A
  → Zone B or Zone B → Zone A) counts identically. **Bidirectional total.**
- **Denominator/window**: a trailing `(time − 60, time]` window (`FLOW_WINDOW_SECONDS = 60.0`), strictly
  exclusive at the start, inclusive at the end.
- **Partially completed traversals**: never counted — only `step.end_time` (completion) matters;
  `step.start_time` is not read by this function at all. An occupant who has departed but not yet
  arrived contributes nothing.
- **Simultaneous events**: no deduplication — every qualifying step counts independently, even if
  several share the exact same `end_time`.
- **Candidate identity**: `candidate_id == edge.id == Staircase.id` (verified: `Edge(id=stair.id, ...)`
  in `navigation/graph_builder.py`) — no translation needed anywhere in this milestone's wiring.
- **Units**: a plain integer count over the window (not a per-minute rate) — `dtype="int"` in the schema.
- **Never fabricated**: simulation ground truth is always exactly computable — `nullable=True` per the
  schema, but the SIM side of this feature is documented as "never None."

## Phase 2: the live feature, traced exactly

`stair_flow.models.StairFlowMetrics` (see `docs/architecture/stair_flow_intelligence.md` for the full
derivation) exposes both `entries` and `exits` for a stair over a trailing window:

- **`entries`**: count of `StairTransitionRecord`s where `to_stair_id` is this stair — the moment
  `current_stair_id` becomes non-None. This is the analog of a traversal's **start** (`step.start_time`).
- **`exits`**: count of `StairTransitionRecord`s where `from_stair_id` is this stair — the moment
  `current_stair_id` becomes None/other. This is the analog of a traversal's **completion**
  (`step.end_time`).
- Both are windowed `(time − window_seconds, time]`, gated OBSERVED/UNKNOWN (see Phase 6 below),
  multi-camera-dedup-proof (inherited, not reimplemented — see the Stair Flow doc's own "Transition
  identity" section), and direction (`upward_count`/`downward_count`) is derived only for `entries`.

## Phase 3: semantic parity comparison

| Property | Simulation (`_recent_flow_rate`) | Live (`StairFlowMetrics`) | Equivalent? | Difference |
|---|---|---|---|---|
| Event counted | Completed edge crossing (`step.end_time`) | `.exits` = `StairTransitionRecord` with `from_stair_id` set (physical departure/completion) | **Yes**, when `.exits` is used | `.entries` (traversal start) is explicitly the WRONG counterpart — proved via the "incomplete traversal" test |
| Window length | 60.0s (`FLOW_WINDOW_SECONDS`) | Configurable; **60.0s used explicitly** for this feature (`build_stair_flow_snapshot_for_prediction()`), independent of whatever `CrowdIntelligenceEngine.stair_flow_window_seconds` an operator configures | Yes, by deliberate construction | None when built via the recommended helper |
| Window boundary | `(time − window, time]`, strict-start/inclusive-end | Identical (`record.timestamp <= window_start` excluded, `> window_end` excluded) | Yes | None — proved with exact-boundary tests |
| Units | Raw count (int) | Raw count (int, `.exits`) | Yes | `entry_rate_per_minute`/`exit_rate_per_minute` also exist live but are NOT what this feature uses (feature is a raw count, not a rate) |
| Direction | Not filtered — bidirectional total | `.exits` likewise not direction-filtered (counts departures either way) | Yes | None |
| Entry vs. exit | Only completion counted | Only `.exits` used; `.entries` deliberately unused for this feature | Yes | — |
| Observation completeness | Always complete (simulation ground truth) | Gated OBSERVED/UNKNOWN — see Phase 6 | **No** — genuine, disclosed epistemic difference | SIM is omniscient; LIVE can honestly be UNKNOWN |
| Identity handling | One step per completed traversal, per occupant timeline | One `StairTransitionRecord` per genuine value change, already deduplicated across cameras upstream | Yes | Multi-camera dedup proved in Phase 8 |
| Simultaneous events | All counted independently | All counted independently (no timestamp dedup) | Yes | Proved with a 3-simultaneous-completion test |
| Missing evidence | N/A (never missing) | `None`, never a fabricated 0 (Phase 6) | Disclosed difference | See Phase 6 |
| Candidate localization | `edge.id` | `Staircase.id` | Yes, identical id | `CandidateIdentity.candidate_id == edge.id == stair.id` |
| Temporal alignment | `time` is the simulation observation instant | `time` is the live extraction instant, same trailing-window convention | Yes | — |

## Phase 4: canonical live source

**Answer: none of A/B/D as literally worded — the correct mapping is the STAIR FLOW "exits" (completion)
quantity**, which is closest in spirit to option **B ("exit rate")** but must not be read as "building
exit" — it means *stair-traversal completions*, i.e. `StairFlowMetrics.exits`, a raw count (not a
per-minute rate) over the same 60s window. Ruled out explicitly:

- **A (entry rate / `.entries`)** — wrong: measures traversal START, not completion; the frozen feature
  only ever reads `step.end_time`. Proved wrong by the "incomplete traversal" controlled test (Phase 7):
  an occupant who entered but has not yet exited contributes 0 to both the frozen SIM feature and the
  correct LIVE counterpart, but WOULD contribute 1 if `.entries` were (incorrectly) used.
- **C (total bidirectional throughput, i.e. `entries + exits`)** — wrong: would double-count a single
  completed traversal (once as an entry, once as an exit) relative to the simulation's single
  per-traversal step count.
- **D (net flow, `entries − exits`)** — wrong: can be negative or zero even with heavy real traffic,
  nothing like the simulation's non-negative completion count.
- **E (another quantity)** — none matched better than `.exits`.

## Phase 5: window alignment

`stair_flow.compute.DEFAULT_WINDOW_SECONDS` and `predictive_dataset.simulation_extractor_v2_1.
FLOW_WINDOW_SECONDS`/`live_extractor_v2_1.FLOW_WINDOW_SECONDS` were ALL already 60.0 before this
milestone (the Stair Flow milestone deliberately chose 60.0 specifically because it matched this
existing convention — see that milestone's own Phase 5). No numeric mismatch exists today. **The
operational `CrowdIntelligenceEngine.stair_flow_window_seconds` default was NOT changed** (per this
milestone's own explicit instruction) — instead, `predictive_dataset.live_extractor_v2_1.
build_stair_flow_snapshot_for_prediction(stairs, occupants, building, time)` independently computes its
OWN `StairFlowSnapshot` via `stair_flow.compute.compute_stair_flow_snapshot(..., window_seconds=
FLOW_WINDOW_SECONDS)`, querying the SAME `occupant.history.stair_transitions` evidence with the window
the predictive feature specifically requires, entirely decoupled from whatever window a live deployment's
Crowd Intelligence panel happens to be configured with. If an operator ever changes
`stair_flow_window_seconds` for operational reasons, this predictive feature is completely unaffected.

## Phase 6: observability gating

| Scenario | `StairFlowMetrics.exits` | `candidate_recent_flow_rate` (Stair) |
|---|---|---|
| OBSERVED, zero flow (camera confirms nobody crossed) | `0` | `0` |
| UNKNOWN (no evidence, no confirmed observation) | `None` | `None` |
| No calibrated camera at all | `None` (status UNKNOWN) | `None` |
| No camera coverage (geometry) | `None` | `None` |
| Temporary camera failure, evidence already recorded before the failure | Real count (evidence-based gate fires even if current status is UNKNOWN) | Real count |
| Valid camera but temporary tracking loss | Unaffected — a transition already recorded stands; a occupant currently `TEMPORARILY_LOST` still contributes recorded evidence | Same |
| Occupant first appearing already on Stair | Excluded from `entries`, and therefore never spuriously produces a paired "exit" either unless a genuine later departure is observed | Consistent |
| Expired identity | Evidence genuinely lost (disclosed limitation, inherited from Stair Flow) | Same |

Follows `ai_features.feature_schema`'s own established `nullable=True` / "None means genuinely
unavailable, never a fabricated zero" convention exactly — the same one already governing every other
`LIVE_ESTIMABLE` field in this schema (`candidate_queue_length`, `candidate_approaching_count`, Exit's
own `candidate_recent_flow_rate`). **`0` and `None` (UNKNOWN) are never conflated.**

## Phase 7: controlled equivalence results (`tests/test_stair_predictive_feature_live_parity.py`)

All driven through the REAL, actually-wired public functions on both sides (`extract_experimental_
candidate_features()` / `extract_live_experimental_candidate_features()`), never a private helper in
isolation:

| Scenario | SIM result | LIVE result | Match |
|---|---|---|---|
| Zero flow, stair confirmed OBSERVED | 0 | 0 | ✓ |
| One completed crossing | 1 | 1 | ✓ |
| Multiple completed crossings (the milestone's own 3-occupant example, 20s window) | 3 | 3 | ✓ |
| Entry + exit, same occupant, one completion | 1 | 1 | ✓ |
| Three simultaneous completions (identical timestamp) | 3 | 3 | ✓ |
| Boundary: completion exactly at `time − window` | 0 (excluded) | 0 (excluded) | ✓ |
| Boundary: completion exactly at `time` | 1 (included) | 1 (included) | ✓ |
| **Incomplete traversal** (entered, not yet exited) | 0 (no completed step exists) | 0 (`.exits` correctly excludes it, even though `.entries == 1`) | ✓ |
| No observation available at all, zero occupants | 0 (sim always omniscient) | `None` (honest UNKNOWN) | **Disclosed difference, not forced to equality** |
| No `stair_flow_snapshot` supplied at all (backward compatibility) | n/a | `None` (falls back to the pre-milestone Door/Stair zone-transition proxy path, itself returning `None` with no occupants) | Unchanged pre-milestone behavior |

## Phase 8: multi-camera parity (`MultiCameraPredictiveFeatureParityTests`)

Three occupants (entering at t=5, 10, 15; each a 3s traversal), each traversal observed by 1, then 2,
then 3 simultaneous cameras (`CAM-A`/`CAM-B`/`CAM-C`, all calling `LiveOccupantManager.update()` for the
same occupant in the same cycle — the exact `live_camera_pipeline.pipeline.run_cycle()` pattern). Result:
**`candidate_recent_flow_rate` is exactly `3` in all three cases** — camera count never inflates the
predictive feature, inherited directly from `LiveOccupantManager`'s own idempotent same-cycle update
behavior (Stair Flow milestone's own "Transition identity" proof), never re-verified or re-implemented
here.

## Phase 9: `candidate_congestion_trend` audit — no change

`candidate_congestion_trend`'s FROZEN simulation definition (`simulation_extractor_v2_1._congestion_
trend()`) compares a **demand proxy** (`queue_length + approaching_count`) at `time` against the same
proxy at `time − 30s` — RISING/STABLE/FALLING/UNKNOWN. Its existing live source
(`crowd_intelligence.models.AssetApproachMetrics.trend`, via `crowd_intelligence.trends.TrendTracker`)
already tracks that exact same demand-proxy quantity. The new Stair flow evidence (`entries`/`exits`
throughput) answers a **different semantic question** — rate of people crossing, not backlog/demand
trend — and using it here would create a genuine mismatch with the frozen feature's own definition, not
an improvement. **No change made.** (`candidate_congestion_trend` is unaffected by this milestone in
every respect — same source, same code path, same tests.)

## Phase 10: complete Stair feature parity matrix

| Feature | Schema availability | Parity classification | Basis |
|---|---|---|---|
| `total_active_occupant_count` | LIVE_ESTIMABLE | APPROXIMATE_PARITY | Same definition; simulated ground truth vs. real camera/tracking-derived estimate — never provably equal even in principle |
| `candidate_type` | LIVE_OBSERVABLE | EXACT_PARITY | Shared `Edge.edge_type` |
| `candidate_capacity` | LIVE_OBSERVABLE | EXACT_PARITY | Identical shared `crowd_intelligence.capacity` call |
| `candidate_walking_distance` | LIVE_OBSERVABLE | EXACT_PARITY | Identical shared `Edge.walking_distance` |
| `candidate_traversable` | LIVE_OBSERVABLE | EXACT_PARITY (disclosed pre-existing v1 simplification: mid-scenario `ScenarioEvent` overrides not yet incorporated — unrelated to this milestone) | Identical shared `Edge.traversable` |
| `candidate_adjacent_zone_occupancy` | LIVE_ESTIMABLE | APPROXIMATE_PARITY | Same definition, different data-generating process (sim ground truth vs. live tracking) |
| `candidate_queue_length` | LIVE_ESTIMABLE | APPROXIMATE_PARITY | Disclosed geometric/behavioral proxy (`AssetApproachMetrics.queue_candidate_count`), not the same underlying mechanism as discrete-event queue bookkeeping |
| `candidate_approaching_count` | LIVE_ESTIMABLE | APPROXIMATE_PARITY | Disclosed different mechanism (short-range geometric proximity+heading vs. route-membership) |
| `candidate_congestion_level` | LIVE_ESTIMABLE | APPROXIMATE_PARITY | Shared classification formula, fed by the approximate queue/approaching inputs above |
| **`candidate_recent_flow_rate` (Stair)** | LIVE_ESTIMABLE | **STRUCTURAL_PARITY (upgraded by this milestone; was APPROXIMATE_PARITY)** | Genuinely different mechanism (`StairTransitionRecord` vs. `OccupantTimelineStep`), PROVEN numerically equivalent under matching evidence (Phase 7/8) |
| `candidate_recent_flow_rate` (Exit) | LIVE_ESTIMABLE | STRUCTURAL_PARITY (pre-existing, unchanged) | `evacuation_progress.models.ExitFlow.recent_flow_per_minute`, same 60s window, same units once converted |
| `candidate_recent_flow_rate` (Door) | LIVE_ESTIMABLE | APPROXIMATE_PARITY (pre-existing, unchanged) | Zone-transition proxy; no per-asset transition evidence analogous to Stair's `current_stair_id` exists for Door |
| `candidate_congestion_trend` | LIVE_ESTIMABLE | APPROXIMATE_PARITY (audited Phase 9, unchanged) | Live `TrendTracker` over the demand proxy; same semantic definition, not proven numerically identical to sim's own delta computation |
| `candidate_alternative_route_count` | LIVE_OBSERVABLE | EXACT_PARITY | Identical shared function, zero occupancy/time dependence |
| `candidate_betweenness_centrality` / `candidate_is_bridge` / `candidate_upstream_catchment_count` | LIVE_OBSERVABLE | EXACT_PARITY | Identical shared `graph_context_v4` function chain |

`NO_LIVE_SOURCE`: none present in the current schema for Stair. The one family investigated and
deliberately excluded from the schema entirely (all candidate types, not Stair-specific) is per-candidate
hazard exposure — `predictive_dataset/schema.py`'s own pre-existing, unrelated-to-this-milestone
docstring already documents this as SIMULATION_ONLY.

## Phase 11: minimal live wiring performed

`predictive_dataset/live_extractor_v2_1.py`:

- New optional parameter `stair_flow_snapshot: Optional[StairFlowSnapshot] = None` on
  `extract_live_experimental_candidate_features()` and `_live_recent_flow_rate()`.
- New `_stair_flow_rate(candidate, stair_flow_snapshot)` — returns `stair_flow_snapshot.for_stair(
  candidate.candidate_id).exits`.
- `_live_recent_flow_rate()` now branches: `Exit` → unchanged `_exit_flow_rate()`; `Stair` **when a
  `stair_flow_snapshot` is supplied** → the new `_stair_flow_rate()`; everything else (Door always, or
  Stair with no snapshot supplied) → the original, completely unchanged `_door_or_stair_flow_rate()`
  zone-transition proxy. **Door behavior is untouched in every code path.**
- New `build_stair_flow_snapshot_for_prediction(stairs, occupants, building, time, observable_assets=None,
  camera_coverage=None)` convenience builder, mirroring `build_alternative_route_counts()`'s own
  "compute once per tick, pass in" convention — explicitly at `FLOW_WINDOW_SECONDS` (60.0), independent
  of Crowd Intelligence's own operational window.

`predictive_dataset/live_extractor_v4.py`: the same optional `stair_flow_snapshot` parameter threaded
straight through to `extract_live_experimental_candidate_features()` — no other change.

`predictive_dataset/schema_v4.py`: only the `candidate_recent_flow_rate` field's `source=` documentation
string updated to describe the new Stair source and its fallback — no schema field added, removed,
renamed, or reordered; `SCHEMA_VERSION_V4` unchanged.

**Not touched by this milestone**: `simulation_extractor_v2_1.py`, `Target V2`, `Recommendation`,
`Guidance`, any `ai_registry`/`ai_inference`/model-training code, `NavigationGraph`, `simulator/`,
`live_system/`, `main.py`. None of `live_extractor_v2_1.py`/`live_extractor_v4.py` are called from
`live_system`/`live_runtime`/`main.py` today (verified — both remain research/offline-dataset tooling
only, called from tests and from each other), so this wiring cannot enable any production ML inference
by construction.

## Phase 12: no-camera / offline behavior — proven unaffected

`tests/test_stair_predictive_feature_no_camera_offline.py`: a Designer-only extraction call that omits
`stair_flow_snapshot` entirely; Door/Exit candidates unaffected; a no-camera LiveRuntime scenario
(`stair_flow_snapshot=None` explicitly); a legacy Staircase with no authored `from_observable_region`/
`to_observable_region` (both the raw `StairFlowSnapshot` and the full feature dict correctly report
`None`, never a fabricated flow); missing calibration (no `ObservableAssetSnapshot` at all) reported as
UNKNOWN, not zero; a Stair without observable regions remains a perfectly valid predictive candidate
(enumeration and extraction never raise).

## Phase 13: performance

`tests/test_stair_predictive_feature_performance.py`: 20 cameras, 20 stairs, 100 occupants (each a full
entry+exit, half observed by two simultaneous cameras) — building the `StairFlowSnapshot` AND extracting
`candidate_recent_flow_rate` for all 20 Stair candidates completed in **~1.2 ms**. No YOLO/ML inference
anywhere in this path (mechanically guaranteed — see Phase 14's architecture guards); this cost is
reported entirely separately from any perception/detection pipeline cost.

## Phase 14: architecture guards

`tests/test_stair_predictive_feature_live_parity_architecture_guards.py` mechanically proves: `stair_flow/`
never imports `predictive_dataset`/`predictive_model`/any AI/Recommendation/Guidance package (perception
evidence cannot know feature extraction exists); `predictive_dataset.live_extractor_v2_1` genuinely does
import `stair_flow` (the one allowed direction, positively confirmed); neither live extractor file imports
any model-inference, Recommendation, Guidance, Voice, Signage, Building Control, or FACP package, or
calls a `.predict(`/`.infer(`/action-execution verb; `simulation_extractor_v2_1.py` (frozen SIM
semantics) never imports `stair_flow` or any live-only package. Dependency direction proved:
**Perception/Crowd Intelligence → Feature extraction**, never the reverse.

## Remaining gaps / implications for future predictive inference

- `candidate_recent_flow_rate` (Stair) inherits every genuine limitation already disclosed in
  `docs/architecture/stair_flow_intelligence.md`'s own "Known limitations" section (disappearance
  mid-stairwell, first-observation-already-on-stair, expired-identity history loss, `OccupantHistory.
  max_length` interaction) — none newly introduced here, all still apply to `.exits` exactly as they do
  to `.entries`.
- `candidate_recent_flow_rate` (Door) remains APPROXIMATE_PARITY — Door has no `current_door_id`/
  transition-record analog to Stair's `current_stair_id`; extending this genuine parity to Door would
  require an Observable-Asset-style perception upgrade for Door first (out of scope here, and out of
  scope for the Observable Asset Perception Framework as of its own milestone too).
- `candidate_congestion_trend`, `candidate_queue_length`, `candidate_approaching_count`,
  `candidate_adjacent_zone_occupancy`, `total_active_occupant_count` all remain APPROXIMATE_PARITY —
  none were in this milestone's scope to improve, and none were touched.
- This milestone proves the Stair candidate feature VECTOR can now be honestly assembled from majority-
  EXACT/STRUCTURAL-parity live sources, with the remaining fields' approximate character fully disclosed
  and pre-existing (not introduced here). This is a genuine, meaningful strengthening of the live-data
  foundation — but it remains data-availability work only. No model has been trained or run against this
  data; a future "shadow-mode predictive inference" milestone would need to separately address model
  deployment, inference scheduling, and Recommendation-integration policy, none of which this milestone
  touches.

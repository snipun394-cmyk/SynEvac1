# Live Stair Flow & Movement Direction Intelligence

Status: implemented (perception/intelligence evidence only — no new ML, no NavigationGraph/Staircase/
WorldProjector/calibration redesign, no Recommendation/Guidance/Simulation change, no new Designer
asset). Builds directly on `docs/architecture/live_stair_perception.md` (Observable Stair Perception),
`docs/architecture/observable_asset_perception.md` (the generic asset framework), and
`docs/architecture/camera_coverage.md` (Camera Coverage Intelligence).

---

## Motivation

Prior milestones let SynEvac truthfully answer: *which cameras cover STAIR-1*, *is STAIR-1 currently
observed*, *which tracked occupants are on STAIR-1 right now*, and *how many unique occupants*. None of
that is a **flow** signal — entries, exits, rate, or direction. `live_occupants.history.OccupantHistory.
stair_transitions` (added by Observable Stair Perception, one record per genuine `current_stair_id`
change) was recorded specifically to make this derivable later, but nothing ever read it for this
purpose until now. This milestone is that read — a new `stair_flow/` package computing throughput facts
purely from evidence the system already collects.

## Phase 1 audit: what evidence already exists

Traced exactly what happens to `LiveOccupant`/`OccupantHistory` for `Zone A → Stair S1 → Stair S1 →
Zone B` and the reverse:

- `live_occupants.manager.LiveOccupantManager.update()` is the ONE place `current_stair_id` and
  `history.stair_transitions` change. On every call where `stair_id != existing.current_stair_id`, it
  appends exactly one `StairTransitionRecord(timestamp, from_stair_id, to_stair_id)` — never one per
  cycle regardless of change.
- `current_floor_id` and `world_position` are resolved by the SAME `camera_calibration.projection.
  WorldProjector.project()` call that resolves `stair_id` (`live_camera_pipeline/pipeline.py`
  `_process_camera_cycle()`), and `history.with_position_sample(timestamp, world_position, floor_id)` is
  written in the SAME `update()` invocation, at the SAME `timestamp` — so a `PositionSample` sharing a
  `StairTransitionRecord`'s exact timestamp genuinely co-times which floor the occupant was confirmed on
  at that instant. This is the one piece of evidence Phase 7's direction derivation needed, and it
  already existed.
- `sweep_missing()` (the TEMPORARILY_LOST/EXITED/EXPIRED path) **never calls `update()`** — it never
  touches `current_stair_id` or writes a stair transition. A confirmed genuine information gap (see
  "Known limitations" below).
- `LiveOccupantManager.update()` being called once per (camera, resolved-detection) pair — not once per
  occupant per cycle — was traced concretely in `live_camera_pipeline/pipeline.py::run_cycle()`. Two
  cameras seeing the same fused `occupant_id` this cycle DO both call `update()`. Multi-camera
  double-counting is nonetheless structurally impossible: the second call's `existing = self._occupants.
  get(occupant_id)` reads back what the FIRST call already stored this same cycle, so `stair_id !=
  existing.current_stair_id` is false on the second call whenever both cameras agree (the healthy-
  coverage case) — see "Transition identity" below for the full traced proof and its edge case.
- **Conclusion**: entry/exit derivation from `stair_transitions` alone is defensible. Direction
  derivation additionally requires cross-referencing `position_samples` by exact timestamp match, which
  is genuine (not fabricated) co-timed evidence, not an assumption.

## Event semantics (Phase 2)

Two event types, `stair_flow.models.StairFlowEventType`:

- **`ENTERED_STAIR`** — a `StairTransitionRecord` where `to_stair_id` is this stair's id.
- **`EXITED_STAIR`** — a `StairTransitionRecord` where `from_stair_id` is this stair's id.

A single record with BOTH set and different (a direct stair-to-stair landing, e.g. a shared mezzanine)
honestly produces BOTH one `EXITED_STAIR` for the old stair and one `ENTERED_STAIR` for the new one —
never collapsed, never dropped (`tests/test_stair_flow_events.py::test_5_direct_stair_to_stair_
transition_produces_exit_and_entry`).

**One deliberate exclusion**: `LiveOccupantManager.update()` unconditionally writes `with_stair_
transition(timestamp, None, stair_id)` the very first time ANY `occupant_id` is ever seen, even when
that occupant is already on a stair at first sighting. This record is indistinguishable in shape from a
genuine entry (`from_stair_id=None, to_stair_id=<id>`), but there is no honest entry EVIDENCE — tracking
began mid-traversal, and the real entry may have happened long before any camera saw them.
`stair_flow.events.extract_stair_flow_events()` detects this via `record.timestamp == occupant.
first_seen` (a later, genuine transition can never share this occupant's `first_seen` timestamp, since
timestamps only advance) and excludes it from `entries` entirely — the occupant's presence is still
honestly captured via `observed_occupant_count` (a separate, point-in-time measurement), just not as a
flow event.

UP/DOWN direction (Phase 7) is derived **once, at ENTRY**, never re-derived for the paired EXIT — see
its own section below.

## Occupancy vs. flow — a genuinely different measurement

`StairFlowMetrics.observed_occupant_count` is the SAME number `crowd_intelligence.models.
AssetApproachMetrics.observed_occupant_count` already reports for a stair (reused via
`compute_stair_flow_snapshot()`'s `observed_occupant_counts` parameter, never recomputed independently)
— a point-in-time occupancy fact. `entries`/`exits`/rates are a WINDOWED throughput fact over the
trailing `window_seconds`. Both can disagree completely and both be correct: a stair can show
`observed_occupant_count=3, entries=0, exits=0` (three people standing still on a landing, nobody has
crossed in or out this window) or `observed_occupant_count=0, entries=4, exits=4` (brisk through-traffic,
nobody currently mid-stair at this instant). Never conflated.

## Rate calculation (Phase 5)

`window_seconds` defaults to **60.0** — not arbitrary. It is the SAME `FLOW_WINDOW_SECONDS`
`predictive_dataset/simulation_extractor_v2_1.py` and `predictive_dataset/live_extractor_v2_1.py`
already establish ("matches evacuation_progress's own recent-flow window"), and the same
`evacuation_progress.models.EvacuationProgressConfig.flow_window_seconds` default. Rates are computed
with the exact same formula `evacuation_progress.engine.EvacuationProgressEngine._compute_exit_flow()`
already uses: `rate_per_minute = count / (window_seconds / 60.0)`. `CrowdIntelligenceEngine.
__init__(..., stair_flow_window_seconds=60.0)` exposes it as a configurable constructor parameter,
mirroring `approach_region_depth`'s own convention.

## Multi-camera deduplication / transition identity (Phase 3, proved in Phase 8)

This package performs **zero** identity resolution of its own. It reads `occupant.history.
stair_transitions`, which is already keyed by the canonical, fused `occupant_id` `cross_camera_identity`
+ `live_camera_pipeline.identity_resolver.MappingIdentityResolver` already resolved upstream. The
traced proof (`live_camera_pipeline/pipeline.py::run_cycle()`):

1. Every camera's own `RawHumanDetection` this cycle is collected into one list and resolved ONCE via
   `identity_resolver.resolve(raw_detections, time)` — returning one `Detection` per raw detection, in
   order, each carrying whatever `occupant_id` global identity resolution assigned it.
2. `for detection, pending in zip(resolved, pending_occupant_updates): live_occupant_manager.update(
   detection.occupant_id, ...)` — called once per (camera, detection) pair. If CAM-A and CAM-B both saw
   the SAME physical person this cycle and both resolved to the SAME `occupant_id`, `update()` genuinely
   IS called twice for that `occupant_id` within one cycle.
3. `update()`'s very first line is `existing = self._occupants.get(occupant_id)`. Because `_store()` is
   synchronous and the loop runs sequentially, the SECOND call's `existing` is exactly what the FIRST
   call just wrote THIS SAME CYCLE. When both cameras agree on `stair_id` (the expected healthy-coverage
   case — see `tests/test_stair_flow_multi_camera_e2e.py`), `stair_id != existing.current_stair_id` is
   false on the second call, so **no second `StairTransitionRecord` is appended**. One entry, one exit,
   one occupancy count — proved for 2 and 3 simultaneous cameras.

**Edge case, disclosed, not silently patched over**: if two cameras *disagree* within the same cycle
(e.g. a momentarily ambiguous localization on one camera differs from a clean read on another — an
already-rare case, since `WorldProjection.stair_id` is `None` whenever `asset_localization_ambiguous`
is `True`, never an arbitrary pick), the LAST `update()` call processed in the for-loop wins for that
cycle, exactly like every other field `update()` already unconditionally overwrites (`current_zone_id`,
`current_floor_id`, `world_position`). This is not a new risk this milestone introduces — it is the
same last-write-wins convention every existing per-cycle field already has, applied consistently.

## Direction derivation (Phase 7)

Grounded in building/floor **topology**, never screen-space movement or image-Y position (explicitly
prohibited). `models.building.Building.floor_elevation()` already derives a genuine vertical ordering —
accumulated `Floor.height`, walked via `ordered_floors()`'s own `display_order` sequence — the SAME
function `models.staircase.Staircase.vertical_height()` already reuses for an unrelated purpose (travel
distance), confirming it as the established source of truth. **`Staircase.from_floor_id` is never
assumed to mean "bottom"** (verified: `tests/test_stair_flow_direction.py::test_3_from_floor_id_does_
not_mean_bottom` constructs a Staircase with `from_floor_id` on the HIGHER floor and proves direction
still resolves correctly from real elevation).

`stair_flow.direction.derive_direction(building, staircase, entered_floor_id)`:

1. `entered_floor_id` is the floor the occupant was confirmed on at the SAME timestamp as the
   `ENTERED_STAIR` transition (the co-timed `PositionSample` cross-reference from Phase 1's audit).
2. Whichever of `staircase.from_floor_id`/`to_floor_id` it matches, the OTHER end is where they are
   headed (a Staircase connects exactly two floors — there is no third option).
3. Compare `Building.floor_elevation()` of both ends: heading toward the higher one is `UP`, the lower
   one is `DOWN`.
4. `UNKNOWN` whenever `entered_floor_id` is `None` (no co-timed position sample), matches neither end, a
   floor cannot be resolved in the `Building`, or (defensively) both ends report equal elevation.

Direction is derived **once, at entry** — the paired `EXITED_STAIR` event never carries a redundant
(and differently-derived) direction value; `upward_count + downward_count + unknown_direction_count`
always equals `entries` (proved in `tests/test_stair_flow_direction.py::test_11`).

## UNKNOWN semantics / failure & degradation behavior (Phase 9)

`StairFlowMetrics` gates every flow field (`entries`, `exits`, both rates, `net_flow`, direction counts)
with the exact convention `evacuation_progress.models.ExitFlow.recent_flow_per_minute` already
establishes: report real numbers (including an honest `0`) whenever **either** genuine window evidence
exists (`entries_count > 0 or exits_count > 0` — a nonzero count is itself proof observation happened
when it was recorded, regardless of this cycle's own camera status) **or** the stair is confirmed
`ObservationStatus.OBSERVED` *right now* (a `0` is then a confirmed current reading, not a coverage
gap). Otherwise every flow field stays honestly `None`. `observed_occupant_count` reuses `crowd_
intelligence`'s own `_observed_occupant_count()` gate exactly (`None` unless `ObservationStatus.
OBSERVED`).

Tested explicitly (`tests/test_stair_flow_observation_failure_modes.py`): no camera coverage at all; an
`ObservableAssetSnapshot` reporting `UNKNOWN`; a genuinely observed `0` (distinct from `UNKNOWN`);
recent evidence still reported even when the CURRENT cycle is unobserved (a camera that just failed
does not erase evidence it already produced); a `TEMPORARILY_LOST` occupant's recent transition still
counted; an `EXPIRED` (removed) occupant's evidence genuinely lost (a disclosed limitation, not patched
around); an ambiguous localization never producing a transition at all; one camera failing while another
covering the same stair remains healthy.

`camera_coverage.models.CameraCoverageSnapshot` (Camera Coverage Intelligence milestone) is consulted
OPTIONALLY, purely to enrich `provenance` text with which camera(s) currently cover a stair — it is
never a second, competing OBSERVED/UNKNOWN determination; `observable_assets.models.ObservableAssetSnapshot`
remains the single gate for that, so there is exactly one source of truth for "is this stair observed,"
never two disagreeing ones.

## Crowd Intelligence integration (Phase 10)

`stair_flow.compute.compute_stair_flow_snapshot()` is a **pure function**, not a second engine.
`crowd_intelligence.engine.CrowdIntelligenceEngine.compute()` calls it directly (reusing `all_occupants`
and each stair's own just-computed `observed_occupant_count`, never a second query) and threads the
result into a new `CrowdIntelligenceSnapshot.stair_flow_metrics: Mapping[str, StairFlowMetrics]` field —
a sibling to `door_metrics`/`exit_metrics`/`stair_metrics`, deliberately its OWN mapping rather than
grafted onto `AssetApproachMetrics` (which is explicitly shared across Door/Exit/Stair; entry/exit/
direction semantics do not apply to Door/Exit today). `CrowdIntelligenceSnapshot.stair_flow(stair_id)`
mirrors `.stair(stair_id)`'s own accessor convention. Because `CrowdIntelligenceSnapshot` already flows
through to `live_system.state_manager.LiveBuildingSnapshot.crowd_intelligence` unchanged (a whole-object
field, never copied field-by-field), `stair_flow_metrics` reaches that existing wiring with **zero**
changes to `live_system/` or `command_center/` — see "Command Center readiness" below.

## Trajectory Intelligence relationship (Phase 11)

Audited `trajectory_intelligence/trajectory.py`: it stays deliberately floor-blind/stair-blind at the
geometric layer by design (`_confirmed_floor_change()` zeroes out cross-floor distance contributions
specifically so a Zone-A → Stair → Zone-B crossing is never treated as a physically meaningless
same-plane Euclidean jump — it does not represent *that a stair was used* at all). `route_progress.py`'s
`_has_direct_stair_edge()` already recognizes a legitimate Stair-mediated route at the NAVIGATION-GRAPH/
route-deviation layer — a different question ("is this path legitimate") from flow ("how many people,
at what rate"). **No duplication, no changes made**: `stair_flow/` is the sole owner of throughput/rate
measurement; Trajectory Intelligence remains the sole owner of per-occupant geometric movement facts and
route-deviation reasoning. Clean, pre-existing separation — this milestone did not need to touch either
boundary.

## Evacuation Progress (Phase 12) — deliberately NOT wired

Audited whether stair flow should influence `evacuation_progress`. **Left unchanged.** Reasoning: a
Stair is an intermediate conduit, not a terminal safety state. `ZoneClearance`/`ExitFlow` are anchored to
CONFIRMED zone-clearing or CONFIRMED building-exit events — strong, (relatively) unambiguous progress
signals. A `StairFlowEvent.ENTERED_STAIR` alone proves neither: someone entering a stair could be
heading toward an exit (progress) or away from one (e.g. moving to a refuge floor, or backtracking) —
`UP` is not inherently "away from safety" (a refuge floor can be UP) and `DOWN` is not inherently
"toward safety" (a basement stair is DOWN too). Without additional context this package does not have
(confirmed final destination, building-specific egress topology), stair flow is honest OPERATIONAL FLOW
EVIDENCE, not a PROGRESS signal. `evacuation_progress/` was not modified.

## Predictive-AI parity audit (Phase 13, analysis only — `predictive_dataset`/`predictive_model` NOT modified)

| Predictive feature | Simulation source | Current live source (pre-milestone) | Live source available now | Parity status | Remaining limitation |
|---|---|---|---|---|---|
| `candidate_recent_flow_rate` (Stair) | `predictive_dataset.simulation_extractor_v2_1._recent_flow_rate()` — completed-crossing count over `FLOW_WINDOW_SECONDS=60.0`, from `movement_result` (ground truth) | `predictive_dataset.live_extractor_v2_1._door_or_stair_flow_rate()` — counts `ZoneTransitionRecord`s whose `(from_zone_id, to_zone_id)` matches the candidate's own two zones, **not observation-gated at all** (always attempts a count, never distinguishes UNKNOWN from a real 0) | `StairFlowMetrics.entries`/`exits`/`entry_rate_per_minute` — real `current_stair_id` evidence, observation-gated (UNKNOWN vs. observed-0 distinguished) | **Improved, not yet adopted** | `predictive_dataset`/`predictive_model` were explicitly out of scope for this milestone; the zone-transition proxy remains the live feature extractor's actual live source today. Adopting `stair_flow` here is a natural, disclosed next step, not done by this milestone. |
| `candidate_congestion_trend` | simulation trend tracker | `crowd_intelligence.models.AssetApproachMetrics.trend` (unchanged) | unchanged | Already at parity | None from this milestone (out of scope, untouched) |
| `candidate_queue_length` / demand | simulation queue model | `crowd_intelligence.models.AssetApproachMetrics.queue_candidate_count`/`estimated_queue_length` (unchanged) | unchanged | Already at parity | None from this milestone |
| Observed occupancy | simulation ground truth | `observable_assets.models.AssetObservation.occupant_count` (unchanged) | unchanged, now ALSO cross-referenced by `StairFlowMetrics.observed_occupant_count` (same number, no new source) | Already at parity | None from this milestone |
| Direction (UP/DOWN) | not a V2.1/V4 feature at all today | none | `StairFlowMetrics.upward_count`/`downward_count`/rates | **New capability, no prior live OR simulation counterpart in the audited feature set** | Would need its own simulation-side feature definition before any model could consume it; not attempted here |

## Command Center readiness (Phase 14) — data only, no UI built

No panel was implemented. `CrowdIntelligenceSnapshot.stair_flow(stair_id)` already returns everything a
future panel needs to render exactly the example in this milestone's own brief:

```
STAIR-1
Observed occupancy: 8   Entries (recent): 6   Exits (recent): 2
Net flow: +4            Up: 5   Down: 1        Observation: OBSERVED
```

`snapshot.by_stair` (building-wide) and `snapshot.events`/`events_for_stair()` (raw, timestamped,
per-occupant audit trail) are both already structured for a future table/timeline view. No existing
Command Center panel was extended, since none currently renders `crowd_intelligence`'s own `stair_
metrics` either — extending one is a legitimate follow-up, not attempted here per this milestone's own
"data architecture is the priority" instruction.

## Performance (Phase 15)

`tests/test_stair_flow_performance.py`: 20 cameras, 20 stairs, 100 occupants (each with a full entry +
exit, half observed by two simultaneous cameras) — `compute_stair_flow_snapshot()` completed in
**~2.1 ms**, asserted under a 2-second ceiling. Cost is `O(occupants × bounded_history_length)`, no
external I/O, no quadratic building-wide scan.

## Architecture guards (Phase 16)

`tests/test_stair_flow_architecture_guards.py` mechanically proves `stair_flow/` never imports
`navigation.graph`, `evacuation_recommendation`, `evacuation_guidance`, any `ai_*`/`predictive_*`
package, `advisory_system`, `command_center`, `voice_evacuation`/`speaker_manager`, `dynamic_signage`/
`sign_manager`, `building_control`, `facp`, `simulator`/`simulation_runtime`/`ground_truth`, or any raw
camera/ML backend (`cv2`, `torch`, `ultralytics`, `onvif`, YOLO/RTSP backends) — and never calls a
FACP/Voice/Building-Control action-execution verb, and never mutates a passed-in `Staircase`/`Building`
(no `.add_stair(`, `.add_floor(`, or direct attribute assignment onto either).

## Known limitations (genuine information gaps, disclosed per Phase 1/9/17)

1. **Disappearance while on a stair produces no exit event.** `LiveOccupantManager.sweep_missing()`
   (the `TEMPORARILY_LOST`/`EXITED`/`EXPIRED` path) never calls `update()`, so an occupant who loses
   camera coverage mid-stairwell — the single most physically common real scenario for a stairwell often
   sparsely covered between floors — leaves no "stair_id → None" evidence at all. `entries`/`exits`
   genuinely undercount in this case; `observed_occupant_count` (separately sourced) is unaffected by
   this specific gap.
2. **First-ever-observation-already-on-a-stair produces no entry event**, by design (Phase 6) — there is
   no honest entry-instant evidence for someone tracking begins mid-traversal on.
3. **Expired (fully removed) occupant history is genuinely lost.** If an occupant's global identity
   expires (`LiveOccupantManager.expire_after_seconds`) before a rate window elapses, their transition
   records disappear with them and can no longer contribute, even if the transition happened within
   the configured window.
4. **`OccupantHistory.max_length` (default 30) bounds retention independently of `window_seconds`.** At
   a high update cadence, `stair_transitions`/`position_samples` can be evicted before a long
   `window_seconds` elapses, silently narrowing the effective window below what was configured. Not
   fabricated around — simply a real capacity/window interaction to keep in mind when configuring both.
5. **Direction is derived only from the Staircase's own two connected floors** — a stairwell with more
   than two accessible landings, or genuinely non-monotonic elevation, is out of scope (matches
   `models.staircase.Staircase`'s own "one physical connector spanning exactly two floors" design,
   unchanged by this milestone).
6. **Multi-camera dedup inherits whatever accuracy `cross_camera_identity`/`MappingIdentityResolver`
   already have.** This package performs no identity resolution of its own (explicitly out of scope) —
   if upstream global identity resolution ever assigns two different `occupant_id`s to the same physical
   person, this package would count that as two people, exactly like every other consumer of
   `LiveOccupantManager` today. Not a new risk this milestone introduces.

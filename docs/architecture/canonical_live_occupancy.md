# Canonical Live Occupancy Source of Truth

Status: **implemented, tested.** Closes architecture-audit risk #2 (`docs/architecture/synevac_end_to_end_architecture_review.md` §10, finding 1): *"Zone occupant count — a genuine, unreconciled duplication."* Baseline before this milestone: 4303/4303 tests passing, commit `c62c2ce`.

## 1. Phase 1 — what was actually duplicated

Four independent code paths each filtered `LiveOccupantManager.active_occupants()` (occupants with `OccupantStatus` `NEW` or `ACTIVE`) and grouped them by `current_zone_id`, by hand, separately:

| # | Consumer | Produced | Filter/group logic lived in |
|---|---|---|---|
| 1 | `BuildingState.zone_occupancy` | per-zone `OccupancyObservation.occupant_count` | `live_perception/providers.py::LiveOccupantObservationProvider.collect()` |
| 2 | `CrowdIntelligenceSnapshot.zone_metrics[zone].occupant_count` | per-zone count + density | `crowd_intelligence/density.py::compute_zone_metrics()` |
| 3 | `EvacuationProgressSnapshot.zones[zone].current_active_count` | per-zone count | `evacuation_progress/engine.py::EvacuationProgressEngine.compute()` |
| 4 | `EmergencyResponseSnapshot.zones[zone].known_occupant_count` | per-zone count | `emergency_response/engine.py::EmergencyResponseIntelligenceEngine.compute()` |

**These agreed numerically in production, but only by coincidence of independent, parallel construction** — every one of the four was written against the identical `NEW`/`ACTIVE` + `current_zone_id is not None` rule, but nothing mechanically tied them together. A future edit to any one of the four (a changed lifecycle-inclusion rule, a second `OCCUPANCY` observation source added to `SensorFusionEngine`) could silently break the tacit agreement without any test catching it, since no code asserted the relationship — it only ever happened to hold.

Two further findings, deliberately **not** collapsed into the fix:

- **AI feature `total_occupant_count`** (`ai_features/building_state_extractor.py`) is sourced from `BuildingState.occupant_tracks` — `MultiCameraFusionEngine`'s own per-cycle `FusedTrack` count, i.e. **identity truth**, not **current occupancy truth** (see §5 below). It happens to numerically agree with canonical occupancy in the common case (both ultimately derive from "who was detected this cycle"), but is a structurally separate concept by design, and per this milestone's own explicit "do not alter feature schemas" instruction, it was **left unchanged** — verified (not modified) to still agree in the canonical worked example (§4).
- **Command Center displays** perform no independent computation of their own — every occupancy number they show is a direct read of one of the four snapshots above (`live_system/live_command_center_gateway.py::frame_from_building_state()` reads `BuildingState.zone_occupancy` directly; `LiveEvacuationProgressPanel`/`LiveEmergencyResponsePanel` read their respective snapshot fields directly). Fixing the four sources at the root automatically fixes every Command Center display; no Command Center code changed.
- `live_system/integration.py` also independently computes a `zone_occupancy` dict, but is **confirmed dead code** (zero production callers, per the prior end-to-end audit's Phase 14 finding) — left untouched, per this milestone's own explicit "do not remove `live_system/integration.py` yet" instruction.

## 2. Phase 2 — canonical semantics (derived, not invented)

"Current live occupancy" = every `LiveOccupant` whose `OccupantStatus` is `NEW` or `ACTIVE` — **exactly** `LiveOccupantManager.active_occupants()`'s own pre-existing filter, unchanged. This was not a new decision; it is the one rule all four duplicated implementations already independently arrived at, now centralized.

| Status | Counts? | Why |
|---|---|---|
| `NEW` | Yes | First-ever sighting this cycle — genuinely observed. |
| `ACTIVE` | Yes | Currently being observed. |
| `TEMPORARILY_LOST` | No | Not observed this cycle; recoverable, but not a current sighting. Still tracked separately (Crowd Intelligence's own `temporarily_lost_count`, untouched). |
| `EXITED` | No | An honest geometric guess ("likely left"), recoverable. Not a current sighting. |
| `EXPIRED` | N/A | Terminal — the occupant is removed from `LiveOccupantManager`'s store entirely; cannot appear in any query. |

Edge cases:

- **No `current_zone_id`**: the occupant still counts in `total_observed_occupant_ids` (never dropped), but appears in `unlocalized_occupant_ids`, never fabricated into a zone (Phase 13, tested).
- **Zone change**: `current_zone_id` is a single current value on `LiveOccupant` — an occupant is a member of exactly one zone's grouping (or unlocalized) per cycle, by construction. A same-cycle transition therefore mechanically decreases the old zone by exactly one and increases the new zone by exactly one (Phase 12, tested) — not a design choice this milestone added, a structural guarantee it preserved.
- **Seen by two cameras**: already resolved to one global `occupant_id` *before* reaching `LiveOccupantManager` (by `CrossCameraIdentityResolver`, upstream). Canonical occupancy trusts that identity space is already deduplicated — this is exactly what the multi-camera 4→3 worked example proves (§4).
- **Temporary disappearance**: `TEMPORARILY_LOST`, excluded from current occupancy, but the occupant record is retained (not deleted) for a later `ACTIVE` recovery (Phase 11, tested).
- **Missing calibration**: affects `world_position`/`world_position_provenance` only, a *different* concept from zone localization (`current_zone_id`) — an occupant can be zone-localized without a calibrated world position, or vice versa. Not conflated.
- **`BuildingState.occupant_tracks` vs. `LiveOccupant` lifecycle disagreement**: a known, pre-existing, deliberate distinction (§5) — untouched by this milestone.

## 3. Phase 3 — why `LiveOccupantManager` is the owner

Compared against three alternatives, all disqualified:

- **`BuildingStateEstimator`**: doesn't hold a `live_occupant_manager` reference at all — it receives an already-fused `OccupancySnapshot` as an input parameter. Making it the owner would require new coupling in the wrong direction.
- **`LivePerceptionFusionCoordinator`**: sits one layer *above* `LiveOccupantManager`, and is only constructed when a camera pipeline exists (`frame_sources` supplied). `CrowdIntelligenceEngine`/`EvacuationProgressEngine`/`EmergencyResponseIntelligenceEngine` don't hold a reference to it (`live_runtime/factory.py` never wires it to them) — choosing it as owner would mean re-plumbing three engines' constructors for no architectural benefit.
- **`CrowdIntelligenceEngine`**: a peer consumer, not upstream of the others. `LiveOrchestrator.run_cycle()`'s own fixed stage order is `BuildingState → Crowd Intelligence → Evacuation Progress → ... → Emergency Response` — `BuildingState` is computed *before* Crowd Intelligence every cycle, so `BuildingState` cannot depend on `CrowdIntelligenceEngine` without breaking that order. Architecturally disqualified.
- **`LiveOccupantManager`** (chosen): already the single shared instance per `LiveRuntime`; already the direct constructor dependency of all four duplicating consumers (`LiveOccupantObservationProvider(live_occupant_manager)`, `CrowdIntelligenceEngine(building, live_occupant_manager)`, `EvacuationProgressEngine(building, live_occupant_manager, event_bus)`, `EmergencyResponseIntelligenceEngine(building, live_occupant_manager)`); already owns the NEW/ACTIVE/TEMPORARILY_LOST/EXITED/EXPIRED lifecycle that decides who counts. Adding one query method requires **zero new cross-package coupling** — every consumer already holds the reference.

No new `CanonicalOccupancyManager` (or any other parallel manager) was created.

## 4. Phase 4 — `OccupancyFacts`

New, minimal, immutable value object: `live_occupants/occupancy.py::OccupancyFacts`, built by the pure function `compute_occupancy_facts(occupants, timestamp)`.

```python
@dataclass(frozen=True)
class OccupancyFacts:
    timestamp: float
    occupant_ids_by_zone: Mapping[str, Tuple[str, ...]]
    occupant_ids_by_floor: Mapping[str, Tuple[str, ...]]
    unlocalized_occupant_ids: Tuple[str, ...]
    total_observed_occupant_ids: Tuple[str, ...]
    # + total_observed_count / unlocalized_count / zone_count(id) / floor_count(id)
```

Deliberately narrow (no density, capacity, confidence, trend, or world position — every one of those stays a consumer-specific concern, per this milestone's own "observed occupancy only" instruction). Carries occupant **IDs**, not just counts, on every grouping — Phase 4's own explicit auditability/double-count-debugging requirement; any caller can resolve an ID back to the full `LiveOccupant` via `LiveOccupantManager.get(occupant_id)`.

`LiveOccupantManager.canonical_occupancy(timestamp) -> OccupancyFacts` is the one production entry point, memoized per `(timestamp, internal mutation version)` — every consumer calling it with the same `time` value within one orchestrator cycle gets back the identical `OccupancyFacts` instance, computed once (mirrors `LivePerceptionFusionCoordinator.collect()`'s own pre-existing per-timestamp memoization pattern one layer up).

## 5. What stayed deliberately separate

Four distinct truths this milestone does **not** conflate:

- **Identity truth** (`BuildingState.occupant_tracks`, `FusedTrack`) — `MultiCameraFusionEngine`'s own per-cycle, freshly-recomputed view of who a camera detected. No cross-cycle memory, no lifecycle awareness. AI's `total_occupant_count` reads this, unchanged.
- **Current occupancy truth** (`OccupancyFacts`, this milestone) — `LiveOccupantManager`'s lifecycle-aware, cross-cycle-persistent view of who currently counts. This is what `BuildingState.zone_occupancy`/Crowd/Progress/Emergency Response now all share.
- **Position truth** (`LiveOccupant.world_position`/`world_position_provenance`) — calibration-dependent, independent of zone localization.
- **Historical evacuation progress** (`EvacuationProgressEngine`'s own `EvacuationLedger`) — durable, event-driven, survives an occupant's full removal (`EXPIRED`) from `LiveOccupantManager`'s live store entirely (§9).

## 6-9. Consumer changes (Phases 5-8)

- **`live_perception/providers.py::LiveOccupantObservationProvider.collect()`** — per-zone `OCCUPANCY` observation count now reads `canonical_occupancy(time).occupant_ids_by_zone`, not a second, independently-incremented counter. `BuildingState.zone_occupancy`'s public shape (`OccupancySnapshot`, sparse — no entry for a zero-occupancy zone) is **unchanged** (Phase 5's own "preserve the public API" instruction — no compelling reason to change it found).
- **`crowd_intelligence/density.py::compute_zone_metrics()`** — gained one new leading parameter, `occupant_ids_by_zone` (from canonical facts); zone membership is now resolved from it, never independently decided. `occupant_count` is read directly from `len(occupant_ids_by_zone[zone_id])`. Per-occupant breakdown (moving/stationary/running/mean_speed/position coverage) is unchanged — genuinely Crowd Intelligence's own concern, explicitly kept (Phase 6). Only call site: `crowd_intelligence/engine.py::CrowdIntelligenceEngine.compute()`.
- **`evacuation_progress/engine.py::EvacuationProgressEngine.compute()`** — `active_by_zone`/`known_active` now read directly from canonical facts; the engine's own independent grouping loop was deleted outright (no longer needs a per-occupant pass for this at all). The durable ledger (`known_exited_occupants`, observed-exit semantics) is **completely untouched** (Phase 7).
- **`emergency_response/engine.py::EmergencyResponseIntelligenceEngine.compute()`** — zone *membership* now comes from canonical facts; the engine still resolves IDs back to full `LiveOccupant` objects (via a lookup built from `active_occupants()`) because it genuinely needs them for assistance-signal/classification evidence (Phase 8's own "human-state evidence must still come from `LiveOccupant` records" instruction) — canonical occupancy is a membership answer, not a replacement for occupant evidence.
- **AI features (Phase 9)** — `ai_features/building_state_extractor.py` is **unmodified**; schema unchanged; verified (§4's worked example) that `total_occupant_count` (identity truth) still agrees with canonical occupancy in the common multi-camera case.

## 10-14. Worked examples (`tests/test_canonical_live_occupancy.py`)

- **Multi-camera 4→3** (`MultiCameraFourToThreeAllSubsystemsTests`) — Camera A sees `{OCC-ONLY-A, OCC-SHARED}`, Camera B sees `{OCC-SHARED, OCC-ONLY-B}`; 4 raw detections, 3 global occupants. Proven, in one test, from one cycle: `canonical_occupancy()` = 3, `BuildingState.zone_occupancy` = 3.0, Crowd Intelligence = 3, Evacuation Progress = 3, Emergency Response = 3, AI `total_occupant_count` = 3, Command Center's `frame_from_building_state()` = 3.0 — all with the identical `{OCC-ONLY-A, OCC-SHARED, OCC-ONLY-B}` ID set.
- **Temporarily lost** (`TemporarilyLostOccupantTests`) — cycle 1 ACTIVE, cycle 2 missing (far from any exit → `TEMPORARILY_LOST`), cycle 3 reappears with the same global identity. Every subsystem (Crowd/Progress/Emergency Response) reads `1 → 0 → 1`, in lockstep, every cycle.
- **Zone transition** (`ZoneTransitionTests`) — `OCC-1`: Z1 → Z1 → Z2. At the transition cycle, Z1 drops from 1 to 0 and Z2 rises from 0 to 1, exactly once each, `total_observed_count` stays 1 throughout (never double-counted across both zones).
- **Unlocalized** (`UnlocalizedOccupantTests`) — one localized + one unlocalized occupant: `total_observed_count` = 2, `unlocalized_occupant_ids` = the unlocalized one, no zone's grouping ever contains it, Crowd Intelligence's zone density = 1 while its building-wide total = 2.
- **EXITED/EXPIRED** (`ExitedExpiredSemanticsTests`) — occupant near the one modeled exit goes missing → `EXITED` (current occupancy drops to 0 immediately, `EvacuationProgressSnapshot.known_exited_occupants` = 1 the same cycle, durably recorded by the event-driven ledger); later, once past `expire_after_seconds`, the occupant is fully removed from `LiveOccupantManager` (`EXPIRED`) — canonical occupancy stays honestly 0, and the durable ledger **still** reports `known_exited_occupants = 1`, proving historical evacuation progress survives an occupant's complete removal from the live store.

## 15. Command Center consistency

No Command Center code changed. `LiveEvacuationProgressPanel` already labels `known_active_occupants` as *"Observed active occupants"* and `known_total_observed_occupants` as part of a distinct *"...tracked/observed occupant(s) cleared"* line — these are **intentionally different metrics** (current vs. cumulative-ever-seen), already correctly distinguished in the UI's own existing labels, not merged or hidden. `building_view.py`/`occupancy_panel.py` both read `IncidentFrame.zone_occupancy`, which — in Live mode — is `frame_from_building_state()`'s direct pass-through of the now-canonical `BuildingState.zone_occupancy`.

## 16. Application-level E2E

`ApplicationLevelOccupancyConsistencyE2ETests` (`tests/test_canonical_live_occupancy.py`) exercises the real application entry path from the prior Application Live Runtime Launcher milestone — a real `designer.windows.main_window.MainWindow`, its real `live_runtime_panel`/`live_runtime_controller`, Offline Demo mode, **never** `build_live_runtime()` called directly. Two occupants driven into `runtime.live_occupant_manager` (perception bypassed, same precedent `test_live_dynamic_signage_operator_workflow.py::ProductionWiringOfflineE2ETests` already established), one `run_cycle()`, then: `CommandCenterSnapshot.building_state.zone_occupancy`, `.evacuation_progress.known_active_occupants`, `.emergency_response.zone(...).known_occupant_count`, and a direct `crowd_intelligence_engine.compute()` call all agree on 2 — read through the **same** `LiveCommandCenterDataSource` a real, application-opened Command Center window is shown to be looking at (`command_center_window.live_data_source is runtime.command_center_data_source`).

## 17. Performance

`scripts/benchmark_canonical_occupancy.py` (20 cameras, 100 occupants, 50 zones, 200 cycles — Phase 17's own required shape; YOLO inference excluded, matching `scripts/benchmark_live_camera_pipeline.py`'s own established convention of benchmarking only the seam a milestone actually built):

```
BEFORE (4 independent grouping loops/cycle):  10.13 ms total, 0.0506 ms/cycle
AFTER  (1 canonical grouping, memoized):       7.53 ms total, 0.0376 ms/cycle
Grouping-only speedup: 1.35x

Full real-consumer stage cost (provider + crowd + progress + emergency,
canonical, current production code): 277.51 ms total, 1.3875 ms/cycle
```

The BEFORE loops are reproduced inline in the benchmark script (labeled, not restored to production) since they no longer exist as production code after this milestone. Canonicalization removes real, measurable duplicate work at the grouping step (~1.35x), as intended — the much larger "full stage cost" figure is dominated by each consumer's own genuinely distinct per-occupant processing (density/queue/congestion, ledger bookkeeping, assistance-signal scoring), which this milestone explicitly does not touch.

## 18. Architecture guards

`tests/test_canonical_live_occupancy.py::NoDuplicatedOccupancyGroupingGuardTests` mechanically asserts `live_perception/providers.py`, `crowd_intelligence/engine.py`, `evacuation_progress/engine.py`, and `emergency_response/engine.py` each contain a call to `canonical_occupancy(` — a future edit that reintroduces an independent zone-grouping loop in any of the four is guaranteed to leave this literal call site removed or unreachable, which the guard would need to be (visibly) weakened to hide. Deliberately scoped to the **grouping** call site only — legitimate per-occupant processing (Crowd Intelligence's own density/queue math, Emergency Response's own assistance-signal scoring) is untouched and remains fully possible.

## 19. Regression

Baseline 4303/4303 (commit `c62c2ce`). This milestone added one new test module, `tests/test_canonical_live_occupancy.py` (15 tests), plus fixed two pre-existing test fixtures (`FakeOccupantManager` in `tests/test_live_perception.py`/`tests/test_live_perception_failure_modes.py`) that needed a `canonical_occupancy()` stand-in once `LiveOccupantObservationProvider` started calling it. Full suite after this milestone: see final report.

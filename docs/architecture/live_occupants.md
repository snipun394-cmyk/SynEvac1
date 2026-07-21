# Live Occupant Digital Twin

Status as of this milestone: **ONE canonical runtime object per occupant** — `live_occupants.occupant.LiveOccupant`, owned and mutated (by replacement) exclusively by `live_occupants.manager.LiveOccupantManager`. This is the first object in the entire Live pipeline that persists across cycles with a real lifecycle (`first_seen`, `last_seen`, `status`, bounded history) — everything upstream is either frozen-and-recomputed-every-cycle, or a thin registry with no behavior/position memory.

## 1. Pipeline (current)

```
CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector       (unchanged)
    -> tracking.simple_tracker.SimpleSingleCameraTracker             (unchanged)
    -> camera_calibration.projection.WorldProjector                  (unchanged)
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (unchanged)
    -> cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver  (unchanged)
    -> live_occupants.manager.LiveOccupantManager                    (NEW -- this milestone,
                                                                        a pure OBSERVER, see Sec 4)
    -> [unchanged] RawHumanDetection -> live_camera_pipeline.identity_resolver.IdentityResolver
    -> virtual_camera.detection.Detection                            (unchanged)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine              (unchanged)
    -> building_state.estimator.BuildingStateEstimator                 (unchanged)
```

## 2. Investigation findings (Phase 1)

Verified directly against the current source:

1. **Where occupant state currently lives**: scattered across five distinct types, none of which is both globally-keyed AND cross-cycle-persistent: `tracking.tracked_human.TrackedHuman` (per-camera, ephemeral, one cycle), `behavior_recognition.observation.BehaviorObservation` (per-camera, ephemeral), `cross_camera_identity.identity_registry.GlobalIdentityRecord` (**global**, but thin — only `last_camera_id`/`last_track_id`/`last_timestamp`/`created_at`, no zone/behavior/position at all), `virtual_camera.detection.Detection` (per-camera, ephemeral, recomputed every cycle), `multi_camera_fusion.track.FusedTrack` (global, but recomputed fresh every `fuse()` call — its own `TrackHistory` remembers only zone/camera *transitions*, never behavior or position/velocity history).
2. **Which classes duplicate occupant information**: `occupant_id`/`global_id`/`track_id` all name the same underlying concept in three different types (`Detection.occupant_id`, `GlobalIdentityRecord.global_id`, `FusedTrack.track_id`) — this is intentional, existing, and not something this milestone changes (they're the same string by design, per `cross_camera_identity`'s own established contract). More importantly: `WorldProjector`'s `world_position`/`projection_confidence` and `BehaviorRecognizer`'s `RecognizedBehavior`/`TemporalMetrics` are computed fresh every cycle but **never persisted** anywhere once `Detection` is built — `FusedTrack` doesn't even carry `world_position` at all. This is the actual gap: rich per-cycle signal exists, but nothing remembers it across cycles at the occupant level.
3. **Which objects are immutable**: all of them — `TrackedHuman`, `BehaviorObservation`, `GlobalIdentityRecord`, `Detection`, `FusedTrack`, `BuildingState` are frozen dataclasses, recomputed fresh every cycle (verified directly in each package).
4. **Which objects own lifetime**: `cross_camera_identity.identity_registry.IdentityRegistry` owns GLOBAL identity lifetime (create/touch/release/delete) but only for the thin `GlobalIdentityRecord` shape; `tracking.simple_tracker.SimpleSingleCameraTracker` owns LOCAL per-camera track lifetime; `multi_camera_fusion.engine.MultiCameraFusionEngine._histories` owns per-track *transition* history only. None of the three owns a rich, queryable, per-occupant runtime object.
5. **Which objects AI consumes**: `BuildingState.occupant_tracks` (`FusedTrack` objects), confirmed directly in `ai_features/feature_schema.py` (`source="building_state.occupant_tracks"`).
6. **Which objects Advisory consumes**: `FusedTrack`/`BuildingState` (`advisory_system/ai_evidence.py` references `FusedTrack.history`).
7. **Which objects Command Center consumes**: `BuildingState.occupant_tracks` for the Live path; `BuildingObservation.human_observations`/`HumanObservation` for the separate, Simulation-only `command_center/incident_data.py` path — confirmed these are two genuinely different data paths, neither of which this milestone touches.

**Conclusion**: there is no harmful "two mutable copies of the same data" to fix (Phase 8's own concern) — `LiveOccupant` introduces information (`first_seen`, lifecycle `status`, bounded behavior/position/velocity/transition history) that **does not exist anywhere else today**, rather than duplicating something `BuildingState`/`FusedTrack` already track. `BuildingState` therefore needed **zero changes** this milestone (Sec 4).

## 3. `LiveOccupant` model (Phase 3)

`live_occupants/occupant.py` — frozen, exactly like every other value object in this pipeline (`TrackedHuman`, `BehaviorObservation`, `Detection`, `FusedTrack`, `BuildingState`). Fields: `occupant_id`, `current_camera_id`, `current_track_id`, `current_zone_id`, `current_floor_id`, `world_position`, `world_velocity`, `behavior`, `confidence`, `first_seen`, `last_seen`, `status`, `history` — exactly Phase 3's suggested list. Deliberately excludes anything `Detection`/`BuildingState` already derive independently (classification, hazard severity, etc.) — see Sec 2 finding 2.

## 4. Manager design (Phase 4) and pipeline integration (Phase 7)

`live_occupants.manager.LiveOccupantManager` — O(1) `occupant_id` lookup (a plain dict), with incrementally-maintained secondary indices for zone/floor/behavior/camera queries. Is a **pure observer**: `LiveCameraPipeline._process_camera_cycle()` calls `manager.update(...)` once per currently-matched `TrackedHuman`, using whatever `occupant_id` is already in hand that cycle (the cross-camera **global** id when `cross_camera_identity_resolver` is configured, else the tracker's own per-camera-local id) — **never** altering what `RawHumanDetection`/`Detection`/`BuildingState` themselves contain. `LiveCameraPipeline.run_cycle()` calls `manager.sweep_missing(time, seen_occupant_ids)` exactly **once** per overall cycle, after every camera's own loop iteration has reported who it actually saw — this is what lets the manager detect "missing" without needing to import `tracking`'s `MISSING`/`EXPIRED` states or `cross_camera_identity`'s registry internals at all.

Both `live_occupant_manager` and `world_projector`/`behavior_recognizer`/`cross_camera_identity_resolver` before it are optional, additive constructor parameters — omitting `live_occupant_manager` reproduces every prior milestone's exact behavior (proven in `tests/test_live_camera_pipeline_occupant_integration.py::NoLiveOccupantManagerPreservesPriorBehaviorTests`).

## 5. Occupant history (Phase 5)

`live_occupants/history.py` — a deliberate, documented **addition** beyond Phase 2's suggested file list (`__init__`, `occupant`, `manager`, `state`, `lifecycle`, `events`): none of those five has an obvious home for "occupant history" (`occupant.py` is the point-in-time model; `lifecycle.py` is status-transition *policy*, not storage). `OccupantHistory` is frozen (matching `LiveOccupant`'s own immutability), with `with_*()` methods each returning a new instance, bounded by `max_length` via tuple slicing — camera transitions, zone transitions, behavior changes, position samples, velocity samples, each stored separately. "No unlimited growth" is structural (every tuple is re-sliced to `max_length` on every append), not a hoped-for convention.

## 6. Lifecycle (Phase 6)

`live_occupants/state.py::OccupantStatus`: `NEW` (first-ever sighting) → `ACTIVE` (currently observed) → `TEMPORARILY_LOST` (missing this cycle, not yet timed out) or `EXITED` (missing this cycle AND last known position was within `exit_proximity_threshold` meters of a modeled `models.exit.Exit` segment — an honest, **recoverable** geometric guess, not a certainty: a later sighting returns the occupant straight to `ACTIVE`) → `EXPIRED` (terminal; removed from the manager's active store once `now - last_seen > expire_after_seconds`).

**"Respect CrossCameraIdentity timeout policy"** (Phase 6) is satisfied as a **configuration convention**, not a live cross-package query: configure `LiveOccupantManager`'s own `expire_after_seconds` to match (or exceed) the same `cross_camera_identity.transition_model.TransitionModel.timeout_seconds` a deployment's `CrossCameraIdentityResolver` already uses. This keeps `live_occupants`' own dependency surface minimal (it never imports `cross_camera_identity` at all) while still honoring the intent — documented explicitly in `live_occupants/lifecycle.py`.

## 7. `BuildingState` relationship (Phase 8)

**No duplication was found to fix.** `BuildingState.occupant_tracks` (`FusedTrack`, produced by the unmodified `MultiCameraFusionEngine`) remains the canonical **snapshot** — untouched this milestone. `LiveOccupantManager` remains the canonical **runtime owner** of the richer, cross-cycle state (`first_seen`, lifecycle status, bounded history) that `BuildingState`/`FusedTrack` never carried in the first place. The two are parallel, sibling consumers of the same upstream signals (tracker/world-projection/behavior/cross-camera-identity output) — not one deriving from the other, avoiding "two mutable copies" of anything by construction rather than by careful synchronization.

## 8. Events (Phase 9)

**Reused the existing `live_system.event_bus.EventBus`/`EventType`** — its own docstring explicitly invites this ("extending this list is the natural extension point for a future event; nothing about EventBus itself needs to change to add one"). Added 7 new `EventType` members directly to `live_system/event_bus.py` (verified: adding enum members breaks no existing test — none enumerates the total member count): `OCCUPANT_CREATED`, `OCCUPANT_UPDATED`, `OCCUPANT_BEHAVIOR_CHANGED`, `OCCUPANT_ZONE_CHANGED`, `OCCUPANT_CAMERA_CHANGED`, `OCCUPANT_EXITED`, `OCCUPANT_EXPIRED`. Payload shapes live in `live_occupants/events.py` (the same "payload shape lives with its publisher, not inside `event_bus.py`" convention every other event already follows). `event_bus` is an **optional** `LiveOccupantManager` constructor parameter — publishing is a side effect a caller (e.g. a unit test) may not want at all.

Event ordering per `update()` call is fixed and deterministic: `OCCUPANT_UPDATED` always fires first (a general-purpose "something changed" signal), followed by `OCCUPANT_CAMERA_CHANGED`/`OCCUPANT_ZONE_CHANGED`/`OCCUPANT_BEHAVIOR_CHANGED` in that order, only for fields that actually changed — verified directly in `tests/test_live_occupants.py::EventOrderingTests`.

## 9. Detection = immutable perception result / LiveOccupant = runtime entity / BuildingState = runtime snapshot

| | `Detection` | `LiveOccupant` | `BuildingState` |
|---|---|---|---|
| Scope | One camera, one instant | One occupant, across the whole time they're known | The whole building, one instant |
| Lifetime | One cycle (recomputed fresh) | Persists across cycles (created once, updated by replacement, eventually expired) | One cycle (recomputed fresh) |
| Owner | `IdentityResolver`/pipeline glue | `LiveOccupantManager` | `BuildingStateEstimator` |
| Knows about lifecycle? | No | Yes (`NEW`/`ACTIVE`/`TEMPORARILY_LOST`/`EXITED`/`EXPIRED`) | No |
| Carries history? | No | Yes (bounded, Sec 5) | No (only `FusedTrack.history`'s narrower zone/camera transitions) |

## 10. Files created / modified

**Created:**
- `live_occupants/{__init__,occupant,manager,state,lifecycle,events,history}.py` (`history.py` is an additive file beyond Phase 2's suggested list — Sec 5)
- `tests/test_live_occupants.py` — 19 unit tests (Phase 10)
- `tests/test_live_camera_pipeline_occupant_integration.py` — 3 pipeline-integration tests (Phase 7)
- `tests/test_live_occupants_architecture_guards.py` — 3 import-guard tests (Phase 13)
- `scripts/demo_live_occupants.py` — offline demo (Phase 11)
- `scripts/benchmark_live_occupants.py` — performance benchmark (Phase 12)
- `docs/architecture/live_occupants.md` — this document

**Modified:**
- `live_system/event_bus.py` — added 7 new `EventType` members (Sec 8). No existing behavior changed.
- `live_camera_pipeline/pipeline.py` — added one optional constructor parameter (`live_occupant_manager`), `run_cycle()` now collects `seen_occupant_ids` and calls `sweep_missing()` once per cycle; `_process_camera_cycle()` now also returns the set of occupant_ids observed. Default (parameter omitted) behavior unchanged — re-verified against all 144 pre-existing pipeline-related tests.

**Unchanged (verified, not modified):** `tracking/*`, `behavior_recognition/*`, `cross_camera_identity/*`, `camera_calibration/*`, `human_detection/*`, `virtual_camera/detection.py`, `live_camera_pipeline/identity_resolver.py`, `multi_camera_fusion/*`, `building_state/*`.

## 11. Performance

`scripts/benchmark_live_occupants.py`, zero YOLO/tracker/behavior/cross-camera-identity inference:
- Creation: ~0.019 ms/occupant.
- Updates (500 occupants/cycle): ~7.2 ms/cycle (~0.014 ms/occupant).
- Queries (zone + occupant_id lookup): ~0.0024 ms/call.
- Cleanup (500 occupants/round, all expiring): ~3.05 ms/round.

## 12. What still remains

`LiveOccupantManager` is now the intended source of truth for future AI/Advisory/Command Center/RL/analytics/firefighter-dashboard/evacuation-replay consumers — none of those integrations were built this milestone (explicitly out of scope). `BuildingState` itself was not redesigned to derive from `LiveOccupantManager`; that remains a future decision once a genuine consumer need is identified, per Phase 8's own "avoid duplication" finding (Sec 7) that concluded no such derivation is needed yet.

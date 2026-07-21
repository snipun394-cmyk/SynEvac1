# Cross-Camera Identity Resolution (ReID Framework)

Status as of this milestone: the engineering framework that maintains **one persistent global occupant identity** as a person moves between multiple cameras — using only camera topology, time continuity, track age, and (when available) behavior evidence. No deep-learning ReID, no facial recognition, no appearance embeddings. `cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver` is a deterministic baseline; a future learned matcher plugs into the same `CrossCameraMatcher`/`CrossCameraIdentityResolver` seams without any downstream change.

## 1. Pipeline (current)

```
CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector           (unchanged, existing)
    -> tracking.simple_tracker.SimpleSingleCameraTracker                (unchanged, existing)
    -> tracking.tracked_human.TrackedHuman                             (unchanged, existing)
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (unchanged, existing)
    -> behavior_recognition.observation.BehaviorObservation              (unchanged, existing)
    -> cross_camera_identity.resolver.RuleBasedCrossCameraIdentityResolver  (NEW -- this milestone)
    -> cross_camera_identity.observation.ResolvedIdentity                 (NEW)
    -> [identity-stabilized] RawHumanDetection                            (same type, local_track_id = GLOBAL id)
    -> live_camera_pipeline.identity_resolver.SimulationIdentityResolver  (unchanged, existing -- reused for
                                                                            its own documented "local_track_id
                                                                            IS already the global identity" strategy)
    -> virtual_camera.detection.Detection                                (unchanged, existing)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine                 (unchanged, existing)
    -> building_state.estimator.BuildingStateEstimator                    (unchanged, existing)
```

## 2. Investigation findings (Phase 1)

Verified directly against the current source:

1. **Current identity assignment mechanism**: `live_camera_pipeline.identity_resolver.IdentityResolver.resolve(raw_detections, time) -> Tuple[Detection, ...]`. Two existing concrete strategies: `SimulationIdentityResolver` (pass-through: `occupant_id = raw.local_track_id`) and `MappingIdentityResolver` (a static `{(camera_id, local_track_id): global_id}` table, falling back to `f"{camera_id}:{local_track_id}"` for an unmapped pair).
2. **`MappingIdentityResolver`'s real limitation**: it is a purely *static*, pre-configured correspondence table — it has no dynamic mechanism to recognize "this brand-new local track on Camera B is probably the same person who just left Camera A." Its synthetic fallback actively works against cross-camera unification: `f"{camera_id}:{local_track_id}"` namespaces every id by camera, so the SAME global id from two different cameras is a contradiction in terms for it. This is exactly the gap this milestone closes.
3. **Current assumptions about `occupant_id`**: `Detection.occupant_id: str` is the ONLY identity signal `MultiCameraFusionEngine` associates on (`_group_by_occupant`, grouping purely by string equality). `FusedTrack.track_id` is always exactly `occupant_id` — fusion never invents its own identity.
4. **`MultiCameraFusion` expectations**: none about *how* `occupant_id` was produced — it has no geometric/appearance/temporal reasoning of its own and fully trusts whatever `occupant_id` arrives on each `Detection`. This is precisely what makes today's static `MappingIdentityResolver` swappable for a genuinely dynamic resolver with zero change to fusion.
5. **`Detection` identity lifecycle**: `Detection` itself is stateless/ephemeral (recomputed every cycle); the only cross-cycle identity *memory* downstream of it is `MultiCameraFusionEngine`'s own `TrackHistory`, keyed by `occupant_id`. If `occupant_id` changes for the same physical person, fusion has no way to know it's the same person — continuity depends entirely on upstream identity resolution being stable, which is exactly this milestone's job.
6. **Existing camera topology information**: no dedicated camera-to-camera adjacency model exists. `models.camera.Camera` has `zone_ids` (which zones it's assigned to, via `EngineeringAsset`); `navigation.graph.NavigationGraph` models zone-to-zone adjacency (via `Edge`, with a genuine `walking_distance` in meters) for pathfinding, derived from the real Building. Camera adjacency can therefore be *honestly derived* — two cameras are candidates for a transition if their assigned zones are the same or one real `NavigationGraph` edge apart — but nothing does this today. §4 below is the derivation this milestone adds.
7. **Existing Digital Twin camera metadata**: `Camera(EngineeringAsset)`: `id, name, floor_id, zone_ids, active, mode, position, rotation, horizontal_fov, max_range, resolution, fps`. No world-coordinate calibration linking one camera's pixel space to another's exists anywhere (confirmed via `virtual_camera.detection.Detection`'s own "no per-occupant exact position feed" disclosure) — this is why cross-camera matching in this milestone uses topology/time only, never raw geometric position.

## 3. Package (Phase 2)

New top-level package `cross_camera_identity/` — **not** nested inside `tracking/`, `behavior_recognition/`, or `multi_camera_fusion/` (verified structurally by `tests/test_cross_camera_identity_architecture_guards.py`). Files: `__init__.py`, `observation.py`, `identity_registry.py`, `topology.py`, `transition_model.py`, `matching.py`, `resolver.py`.

## 4. Identity Registry (Phase 3)

`cross_camera_identity.identity_registry.IdentityRegistry` — pure storage/bookkeeping, no matching or expiry *policy* of its own:
- `create(camera_id, local_track_id, timestamp) -> global_id` — mints sequential, deterministic ids (`"OCC-1", "OCC-2", ...`; no randomness). A deleted id is **never reused** (Phase 8's "identity reuse prevention" — the counter only increments).
- `touch(global_id, camera_id, local_track_id, timestamp)` — refreshes an identity's current binding, used both for "same track continuing" and "a cross-camera match just happened."
- `release(camera_id, local_track_id)` — the local track is gone; the `(camera_id, local_track_id) -> global_id` binding is dropped, but the `GlobalIdentityRecord` itself is **kept alive** (`last_track_id = None`) as a candidate for a future cross-camera match, until `TransitionModel` decides it has timed out.
- `unbound_records()` — the entire candidate pool for matching, sorted by `global_id` for full determinism.
- `delete(global_id)` — permanent removal, called only by expiry (§7).

## 5. Camera Topology (Phase 4)

`cross_camera_identity.topology.CameraTopology` — the lightweight, hand-buildable graph (`add_camera`, `add_transition(from, to, min_transition_time, max_transition_time)`, `possible_destinations`, `is_plausible_transition`). Never hardcodes a building layout — a caller populates it from its own deployment.

`is_plausible_transition()` distinguishes three honest cases:
1. **A registered transition exists** for this exact `(from, to)` pair — judged against its own min/max window.
2. **Neither camera is known to this topology at all** — genuinely open-world (no information to rule anything out with), so the caller's own default window is used.
3. **At least one camera IS known, but this specific pair isn't registered** — the topology explicitly has information here, and it says "no direct path" — treated as implausible (closed-world for known cameras).

`build_topology_from_navigation_graph(navigation_graph, camera_zone_ids, ...)` — the **optional, additive** Digital Twin reuse Phase 4 asks for: derives camera adjacency from a real `navigation.graph.NavigationGraph`'s zone edges (using the genuine `Edge.walking_distance`, never a fabricated distance) plus each camera's own `zone_ids`. Only the walking-speed/slack-factor *conversion* parameters are configurable assumptions — the graph shape itself always comes from the real Building. Proven directly against the real `NavigationGraph`/`Node`/`Edge` classes in `tests/test_cross_camera_identity.py::NavigationGraphTopologyDerivationTests`.

## 6. Matching Framework (Phase 5)

`cross_camera_identity.matching.CrossCameraMatcher` (abstract) → `RuleBasedCrossCameraMatcher`. Uses only:
- **Camera topology** — `CameraTopology.is_plausible_transition()`.
- **Time continuity** — elapsed time since the candidate's last sighting, scored by closeness to the transition's own `expected_transition_time`.
- **Track age/history** — `min_track_age_for_matching` guards against matching a brand-new, possibly-flickery 1-frame-old local track.
- **Behavior evidence** (optional) — `require_departure_motion` can reject a candidate whose last recognized behavior was `STATIONARY` right before disappearing (disabled by default, since behavior data may not always be configured upstream).

Deliberately **never** uses image embeddings, appearance similarity, facial recognition, or any deep-learning model — enforced by `tests/test_cross_camera_identity_architecture_guards.py`. Greedy, deterministic assignment; ties broken by `global_id` string, so results never depend on iteration order.

## 7. Transition Model (Phase 8)

`cross_camera_identity.transition_model.TransitionModel` owns exactly the policy questions `IdentityRegistry` deliberately does not: `is_expired(record, now)` (identity timeout) and `pending_transition_for(record)` (possible destinations, reusing `CameraTopology`). Supports every Phase 8 requirement: camera exit (`registry.release`), camera entry (`matcher.find_match` + `registry.touch`), temporary disappearance (an unbound-but-not-yet-expired record), multiple possible destinations (`topology.possible_destinations()`), identity timeout (`is_expired`), identity reuse prevention (`IdentityRegistry`'s ever-incrementing counter, §4).

## 8. Global Occupant ID (Phase 6)

`"OCC-N"` — structurally and visibly distinct from a `tracking.tracked_human.TrackedHuman.track_id` (`"CAM-X-Tn"`), so the two can never be confused. `CrossCameraIdentityResolver` never assigns a local track's own id as a global one; `resolver.py`'s own `_resolve_one()` always routes through `IdentityRegistry.create()`/`touch()`. Verified directly in `tests/test_live_camera_pipeline_cross_camera_integration.py` (`assertNotIn("CAM-A-T", global_id)`).

## 9. Pipeline Integration (Phase 7)

`LiveCameraPipeline` gained one further optional constructor parameter, `cross_camera_identity_resolver`, only consulted when `tracker` is also supplied. When present, `_track_and_recognize()`'s glue replaces each detection's `local_track_id` with the resolved **global** id (via `dataclasses.replace()`) before handing it to `identity_resolver.resolve()` — the same additive pattern the tracking and behavior-recognition milestones already established.

**Important configuration note**: pair a `cross_camera_identity_resolver` with `identity_resolver=SimulationIdentityResolver()` (its own existing docstring already says *"local_track_id IS already the global identity in this strategy"* — an exact, honest fit here), **not** `MappingIdentityResolver`, whose synthetic fallback (`f"{camera_id}:{local_track_id}"`) would re-namespace an already-global id per camera and defeat cross-camera unification. `IdentityResolver.resolve()` itself was **not** modified — this is purely a caller-configuration contract, documented in `live_camera_pipeline/pipeline.py`'s own docstring and proven in `tests/test_live_camera_pipeline_cross_camera_integration.py`.

## 10. Tracker ID vs. Global Occupant ID vs. Future Deep ReID vs. Future Appearance Embeddings

| | Tracker ID | Global Occupant ID (this milestone) | Future Deep ReID Model | Future Appearance Embeddings |
|---|---|---|---|---|
| Scope | One camera | All cameras | All cameras | All cameras |
| Format | `"CAM-X-Tn"` | `"OCC-N"` | (model-defined) | (feature vector) |
| Basis | IoU/centroid geometry, one camera | Topology + time + track age + behavior | Learned appearance similarity | Learned embedding distance |
| Lifetime | Until `max_missing_frames` exceeded | Until `TransitionModel` timeout | — | — |
| Status | Implemented (Single-Camera Tracking milestone) | Implemented (this milestone, rule-based) | Not implemented | Not implemented |

## 11. Files created / modified

**Created:**
- `cross_camera_identity/{__init__,observation,identity_registry,topology,transition_model,matching,resolver}.py`
- `tests/test_cross_camera_identity.py` — 24 unit tests (Phase 9)
- `tests/test_live_camera_pipeline_cross_camera_integration.py` — 2 pipeline-integration tests (Phase 7)
- `tests/test_cross_camera_identity_architecture_guards.py` — 3 import-guard tests (Phase 12)
- `scripts/demo_cross_camera_identity.py` — offline demo (Phase 10)
- `scripts/benchmark_cross_camera_identity.py` — performance benchmark (Phase 11)
- `docs/architecture/cross_camera_identity.md` — this document

**Modified:**
- `live_camera_pipeline/pipeline.py` — added one optional constructor parameter (`cross_camera_identity_resolver`), refactored `_track_and_recognize()`'s internal wiring from positional zip to a track-id-keyed lookup (needed once a THIRD optional stage joined tracker + behavior recognizer); default behavior (parameter omitted) is unchanged, re-verified against every pre-existing pipeline test.

**Unchanged (verified, not modified):** `tracking/*`, `behavior_recognition/*`, `human_detection/*`, `live_camera_pipeline/identity_resolver.py`, `live_camera_pipeline/detection_provider.py`, `virtual_camera/detection.py`, `multi_camera_fusion/*`, `building_state/*`, `navigation/*`, `models/camera.py`.

## 12. Performance

`scripts/benchmark_cross_camera_identity.py`, zero YOLO/tracker/behavior-recognizer inference:
- Registry lookup: ~0.0002 ms/call.
- Matching (200 candidates): ~0.075 ms/call.
- Transition evaluation: ~0.0015 ms/call.
- Cleanup (200 identities/round): ~0.005 ms/round.

## 13. What still remains

Deep-learning ReID, appearance embeddings, and facial recognition remain explicitly out of scope and unimplemented. `CrossCameraMatcher`/`CrossCameraIdentityResolver` (the seams) are designed so a future learned matcher can replace `RuleBasedCrossCameraMatcher` entirely — `IdentityRegistry`, `TransitionModel`, `IdentityResolver`, `Detection`, `MultiCameraFusion`, and `BuildingState` would need no changes at all.

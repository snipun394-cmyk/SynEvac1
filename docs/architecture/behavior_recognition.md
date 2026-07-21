# Human Behavior Recognition Framework

Status as of this milestone: the engineering framework for recognizing occupant behavior from **temporal tracking geometry only** — no neural network, no pose estimation, no OpenPose/MediaPipe/YOLO Pose/MoveNet. `behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer` is a deterministic velocity-threshold baseline; a future ML model plugs into the same `BehaviorRecognizer` seam without any downstream change.

## 0. Naming note (Phase 1 finding)

The milestone brief suggested a package named `behavior/`. **That name is already taken** by a completely unrelated, existing production package: `behavior/` is the Simulation's own occupant *decision-making* layer (`HumanBehaviorLayer`, pre-movement delay, route choice, intent, grouping — consumed by `simulator/`), and `behavior_library/`/`behaviour_profile_resolver/` are its companions. None of the three have anything to do with computer vision. This milestone's new package is therefore named **`behavior_recognition/`** instead — verified free of collision before creating any file.

## 1. Pipeline (current)

```
CameraFrame
    -> human_detection.yolo_human_detector.YOLOHumanDetector        (unchanged, existing)
    -> live_camera_pipeline.human_detector.RawHumanDetection        (unchanged, existing)
    -> tracking.simple_tracker.SimpleSingleCameraTracker             (unchanged, existing)
    -> tracking.tracked_human.TrackedHuman                          (unchanged, existing)
    -> behavior_recognition.rule_based_recognizer.RuleBasedBehaviorRecognizer  (NEW -- this milestone)
    -> behavior_recognition.observation.BehaviorObservation           (NEW)
    -> [stabilized + behavior-annotated] RawHumanDetection            (same type, state_evidence set)
    -> live_camera_pipeline.identity_resolver.IdentityResolver       (unchanged, existing)
    -> virtual_camera.detection.Detection                           (unchanged, existing)
    -> multi_camera_fusion.engine.MultiCameraFusionEngine             (unchanged, existing)
    -> building_state.estimator.BuildingStateEstimator                (unchanged, existing)
```

`IdentityResolver.resolve()` was **not** modified and does not receive `BehaviorObservation` objects directly — exactly the same integration pattern the Single-Camera Tracking Framework milestone already established for `TrackedHuman`/`local_track_id`. `live_camera_pipeline.pipeline.LiveCameraPipeline` converts each `BehaviorObservation` into a `RawHumanDetection.state_evidence` value (via `_map_behavior_to_human_state()`, §4) before handing the (still perfectly ordinary) `RawHumanDetection` to the unmodified `IdentityResolver`.

## 2. Investigation findings (Phase 1)

Verified directly against the current source:

1. **Existing `HumanState` enum** (`perception/models/human_observation.py`): `WALKING, RUNNING, STANDING, FALLEN, CRAWLING, WAITING, BEING_ASSISTED, HELPING_ANOTHER_OCCUPANT, NEVER_MOVING_YET`. Confident, "current observable state" claims — not hedged/probabilistic.
2. **Existing `HumanClassification` enum**: `ADULT, CHILD, ELDERLY, WHEELCHAIR_USER, FIREFIGHTER, FIRE_WARDEN, UNKNOWN` — unrelated to behavior, untouched by this milestone.
3. **Existing behavior-related models**: `BehaviorEvent` (`HELPING_ANOTHER_PERSON, DRAGGING_ANOTHER_PERSON, CARRYING_ANOTHER_PERSON, PUSHING_WHEELCHAIR, GROUPED_MOVEMENT, QUEUEING`) on `HumanObservation` — a multi-person social-interaction vocabulary, entirely out of reach for a single track's own position history (this is exactly Phase 5's "do not fabricate Helping/Following/Herding" exclusion list). Also found: the *unrelated* `behavior/` package (§0).
4. **Existing `Detection` fields**: `virtual_camera.detection.Detection` already has `human_state: Optional[HumanState]`, populated today via `IdentityResolver._to_detection(raw.state_evidence or None)`. `RawHumanDetection.state_evidence: Optional[HumanState]` already existed (added in the Real Human Detection Pipeline milestone, always set to `None` by `YOLOHumanDetector` itself — an honest "no evidence yet" placeholder this milestone is the first to actually populate).
5. **Existing `BuildingState` fields**: no behavior-specific field of any kind — `BuildingState.occupant_tracks` holds `FusedTrack` objects, and `FusedTrack.human_state` is populated by `MultiCameraFusionEngine` straight from whichever `Detection` in a fused group has the highest confidence. No change needed or made here.
6. **Existing AI assumptions**: none found referencing `HumanState`/behavior directly in `ai_decision/`, `ai_engine/`, or `ai_registry/` — the AI layer consumes `BuildingState` as a whole, not `HumanState` specifically.
7. **Existing Advisory assumptions**: `advisory_system/` was not found to reference `HumanState` directly either. The one confirmed, concrete consumer of `HumanState.FALLEN` is `command_center/building_view.py` (a `fallen_count` operator-facing tooltip) and `command_center/incident_data.py` — this is the finding that shaped the mapping design in §4.

## 3. `BehaviorObservation` and `RecognizedBehavior` (Phase 3/5)

`behavior_recognition/observation.py` defines `RecognizedBehavior`: `UNKNOWN, STATIONARY, WALKING, RUNNING, POSSIBLY_FALLEN` — deliberately a **separate** enum from `HumanState`, not a subset/alias of it (§4 explains why). `BehaviorObservation` fields: `camera_id`, `track_id`, `timestamp`, `recognized_behavior`, `confidence`, `supporting_metrics` (a `TemporalMetrics`). No `occupant_id`, no building/zone reference, no AI field — tracking-local, exactly like `TrackedHuman`.

Never fabricated, per Phase 5's explicit exclusion list: `HELPING, PANIC, LEADERSHIP, DISABILITY, INJURY, CONFUSION, FOLLOWING, HERDING`. Every one of those is either a multi-person social inference or a claim about *why* someone is moving a certain way — neither is available from one track's own position history.

## 4. Why `RecognizedBehavior` is a separate enum, and the honest mapping onto `HumanState`

`RawHumanDetection.state_evidence`/`Detection.human_state` already exist and are already displayed to a human operator as confident, current-observable facts (`HumanState.FALLEN` directly drives Command Center's `fallen_count` tooltip — verified in `command_center/building_view.py`, §2.7). Reusing `HumanState` *directly* as `RecognizedBehavior` would force one of two dishonest choices: inventing new `HumanState` members this milestone has no authority to add to a type consumed elsewhere, or overloading existing members (`STANDING`/`WAITING`/`NEVER_MOVING_YET`, none of which is an honestly equivalent claim to "velocity measured near zero" — each implies something about *why* the person stopped that pure geometry cannot assert) — and, most importantly, mapping the hedged `POSSIBLY_FALLEN` heuristic onto the unhedged `HumanState.FALLEN` would misrepresent a coarse geometric guess as the same confident signal Command Center already treats as reliable.

The resolution (`live_camera_pipeline/pipeline.py::_map_behavior_to_human_state`):

| `RecognizedBehavior` | `HumanState` mapping | Reasoning |
|---|---|---|
| `WALKING` | `HumanState.WALKING` | Honest 1:1 — both are the same plain "in motion at this pace" observation. |
| `RUNNING` | `HumanState.RUNNING` | Same reasoning. |
| `STATIONARY` | `None` | No existing `HumanState` member honestly matches "velocity ≈ 0" without implying more. |
| `UNKNOWN` | `None` | No evidence — `state_evidence`'s own existing convention. |
| `POSSIBLY_FALLEN` | `None` | Never conflated with the confident `HumanState.FALLEN` — see above. |

## 5. `BehaviorRecognizer` interface (Phase 4)

`behavior_recognition/recognizer.py`. Input: `(camera_id, timestamp, tracked_humans)` where `tracked_humans` is the **full** per-cycle output of one `SingleCameraTracker.update()` call (including `MISSING`/`EXPIRED` entries, needed for history cleanup — §7). Output: one `BehaviorObservation` per `NEW`/`TRACKED` entry, in the same relative order — `MISSING`/`EXPIRED` entries produce no observation (nothing currently observed to honestly report). Because `SingleCameraTracker.update()` itself guarantees its leading `len(detections)` entries are always exactly the `NEW`/`TRACKED` ones in original order, a caller can always `zip()` its per-cycle detections against a recognizer's output — see `live_camera_pipeline/pipeline.py::_track_and_recognize`.

## 6. `RuleBasedBehaviorRecognizer` — the deterministic baseline (Phase 5/6)

Classifies by plain velocity thresholds (`stationary_velocity_threshold` default 5.0 px/s, `running_velocity_threshold` default 80.0 px/s) computed from `behavior_recognition.metrics.compute_metrics()` over `BehaviorHistory`'s bounded per-track sample window. `UNKNOWN` whenever fewer than 2 samples exist (no honest velocity yet) — never a guessed default.

**`POSSIBLY_FALLEN` is disabled by default** (`enable_possibly_fallen_heuristic=False`). The only available geometric signal — a bounding box becoming wide/low (`height/width` below `possibly_fallen_aspect_ratio_threshold`) while stationary for `possibly_fallen_min_stationary_duration` seconds — has real, common false-positive causes this milestone cannot rule out from geometry alone: crouching, sitting, bending down, and picking something up all produce an indistinguishable box shape. Phase 5 itself calls this "optional only if genuinely reliable" — it is not, so it stays opt-in, and even when enabled its confidence is scaled down by `possibly_fallen_confidence_factor` (default 0.5) to reflect that explicitly, and it is never remapped onto `HumanState.FALLEN` (§4).

**Confidence** for `STATIONARY`/`WALKING`/`RUNNING` is `min(1.0, sample_count / confidence_saturation_samples)` — an honest "more samples seen → more confidence in the velocity measurement" engineering heuristic, explicitly *not* a claimed ML-derived probability.

## 7. Metrics (Phase 6) and history (Phase 7)

`behavior_recognition/metrics.py::TemporalMetrics` — `velocity` (px/s), `direction` (radians, most recent displacement heading), `distance_travelled` (cumulative path length over the retained window), `stationary_duration` (consecutive seconds below threshold, ending now), `track_age` (passed through from `TrackedHuman.age`), `acceleration` (optional, needs ≥3 samples). `compute_metrics()` is a pure function of a plain sample sequence — no camera/tracking/AI dependency — stored **separately** from the behavior label specifically so a future ML model can reuse these exact numbers as its own feature input, rather than recomputing them.

`behavior_recognition/behavior_history.py::BehaviorHistory` — bounded per-`(camera_id, track_id)` sample store (`collections.deque(maxlen=...)`, so no single track's memory ever grows unbounded), with `append`/`trim`/`clear`/`recent`. `clear(camera_id, track_id)` is called by `RuleBasedBehaviorRecognizer.recognize()` the moment a `TrackedHuman` reports `state=EXPIRED` — this is what prevents an unbounded dict of every track_id that ever existed, for the lifetime of the process (Phase 7's "no memory leaks" requirement, proven directly in `tests/test_behavior_recognition.py::TrackResetTests`).

## 8. Pipeline integration (Phase 8)

`LiveCameraPipeline` gained one further optional constructor parameter, `behavior_recognizer: Optional[BehaviorRecognizer] = None`, only ever consulted when `tracker` is also supplied (behavior recognition inherently needs tracking history — supplying one without the other is simply a no-op). Omitting it entirely reproduces the Single-Camera Tracking Framework milestone's exact behavior — proven by every pre-existing tracker/pipeline/RTSP/YOLO test continuing to pass unmodified, plus a dedicated backward-compatibility test in `tests/test_live_camera_pipeline_behavior_integration.py`.

## 9. Package boundaries (Phase 2/12)

New package: `behavior_recognition/` (`__init__.py`, `recognizer.py`, `observation.py`, `rule_based_recognizer.py`, `behavior_history.py`, `metrics.py`). Depends only on `tracking.tracked_human.TrackedHuman` and `tracking.track_state.TrackState` (to interpret `TrackedHuman.state` honestly), plain geometry, and time — **not** `perception.models.human_observation`/`HumanState` (that mapping lives one layer up, in `live_camera_pipeline/pipeline.py`, §4), and not AI, `BuildingState`, Command Center, Advisory, RTSP, or any YOLO backend. Enforced mechanically by `tests/test_behavior_recognition_architecture_guards.py`.

## 10. Files created / modified

**Created:**
- `behavior_recognition/{__init__,observation,recognizer,rule_based_recognizer,behavior_history,metrics}.py`
- `tests/test_behavior_recognition.py` — 21 unit tests (Phase 9)
- `tests/test_live_camera_pipeline_behavior_integration.py` — 5 pipeline-integration tests (Phase 8)
- `tests/test_behavior_recognition_architecture_guards.py` — 2 import-guard tests (Phase 12)
- `scripts/demo_behavior_recognition.py` — offline demo (Phase 10)
- `scripts/benchmark_behavior_recognition.py` — performance benchmark (Phase 11)
- `docs/architecture/behavior_recognition.md` — this document

**Modified:**
- `live_camera_pipeline/pipeline.py` — added one optional constructor parameter (`behavior_recognizer`), one small `_map_behavior_to_human_state()` helper, and folded behavior recognition into the existing `_track_and_recognize()` glue (renamed from `_stabilize()`); default behavior (`behavior_recognizer=None`) is unchanged from the tracking milestone.

**Unchanged (verified, not modified):** `tracking/*`, `human_detection/*`, `live_camera_pipeline/human_detector.py`, `live_camera_pipeline/identity_resolver.py`, `live_camera_pipeline/detection_provider.py`, `virtual_camera/detection.py`, `multi_camera_fusion/*`, `building_state/*`, `perception/models/human_observation.py`, `command_center/*`.

## 11. Performance

`scripts/benchmark_behavior_recognition.py`, zero detector/tracker involvement:
- History maintenance + metric computation (20 people/camera): ~0.56 ms/cycle.
- Full `recognize()` call including classification (20 people/camera): ~0.57 ms/cycle.

See `scripts/benchmark_yolo_human_detector.py` and `scripts/benchmark_single_camera_tracker.py` for detector-side and tracker-side overhead, measured completely separately — none of the three benchmarks combines its number with another's.

## 12. Behavior Recognition vs. Future Pose Estimation vs. Future Cross-Camera ReID

| | This milestone (Behavior Recognition) | Future Pose Estimation | Future Cross-Camera ReID |
|---|---|---|---|
| Input | `TrackedHuman` position/box history (one camera) | Individual video frames (per-frame keypoints) | Appearance features across cameras |
| Method | Deterministic velocity thresholds over geometry | A pose model (OpenPose/MediaPipe/YOLO Pose/MoveNet — **not built here**) | An appearance/re-ID model (**not built here**) |
| Claims made | STATIONARY/WALKING/RUNNING (velocity-based); POSSIBLY_FALLEN (hedged, opt-in) | Joint positions, limb angles — enables genuine fall/gesture detection | A stable global identity across cameras |
| Scope | One camera, one track | One frame, one detected person | Multiple cameras, one physical person |
| Status | Implemented (rule-based baseline) | Not implemented | Not implemented (`IdentityResolver`'s own named future seam) |

## 13. What still remains

Genuine fall detection, gesture/action recognition, and any multi-person social inference (helping, following, herding) all require information this milestone deliberately does not have access to (pose keypoints, appearance features, or multi-track social context) and remain future work. `BehaviorRecognizer` (the seam) and `TemporalMetrics` (the reusable feature set) are designed so a future ML model can replace `RuleBasedBehaviorRecognizer` entirely without any change to `tracking/`, `IdentityResolver`, `Detection`, `MultiCameraFusion`, `BuildingState`, or Command Center.

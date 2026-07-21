# Live Human State & Assistance Perception Bridge

Status as of this milestone: genuine `HumanClassification`/`HumanState` evidence, when a perception source actually supplies it, now survives Detection → Cross-Camera Identity → `LiveOccupant` → `EmergencyResponseIntelligenceEngine`, reconciled deterministically and honestly across cameras and cycles, via a new `human_evidence/` package. Today's YOLO-only pipeline still honestly reports `UNKNOWN`/no state for everyone — this milestone builds the bridge, not richer computer vision.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `perception.models.human_observation.HumanClassification`/`HumanState` already exist, and were already reused (not duplicated) by `virtual_camera.detection.Detection` and `live_camera_pipeline.human_detector.RawHumanDetection` — both already carry `classification`/`classification_evidence` and `human_state`/`state_evidence` fields, `Optional`, defaulting to `UNKNOWN`/`None` (Phase 2/4/5's own "no signal" convention).
2. `RawHumanDetection.state_evidence` **was already being populated** in the live pipeline — `live_camera_pipeline.pipeline._map_behavior_to_human_state()` maps `RecognizedBehavior.WALKING`/`RUNNING` onto `HumanState.WALKING`/`RUNNING` (nothing else). `classification_evidence` was **never** populated by anything in the live path (`human_detection.yolo_human_detector.YOLOHumanDetector` already honestly sets it to `HumanClassification.UNKNOWN` explicitly, per Phase 16's own requirement — confirmed already correct, no change needed there).
3. **A genuine, pre-existing bug found in `live_camera_pipeline/pipeline.py::LiveCameraPipeline._process_camera_cycle()`**: this method **unconditionally overwrote** `RawHumanDetection.state_evidence` with the lossy behavior-derived mapping, discarding whatever a real detector itself may have already supplied (e.g. a future fall-detection-capable `HumanDetector` genuinely reporting `FALLEN`). Fixed: the detector's own `state_evidence` is now authoritative; the behavior-derived mapping only fills in when the detector reported none. This was exactly the "where does evidence get lost before `LiveOccupant`" gap this milestone's own Phase 1 asked to find.
4. `multi_camera_fusion.engine.MultiCameraFusionEngine._fuse_group()` **already** reconciles conflicting classification/human_state across cameras — via "the single highest-individual-confidence `Detection` wins for every non-confidence field." This is a real, existing strategy, but a **stateless, per-cycle-only** one (no cross-cycle memory, no staleness handling) — exactly wrong for Emergency Response Intelligence's own need for *persistent* assistance evidence. `BuildingState.occupant_tracks` already receives this (via `FusedTrack`), unmodified by this milestone (Sec 8).
5. `cross_camera_identity.observation.CrossCameraObservation`/`ResolvedIdentity` carry **no** classification/state evidence at all — only camera topology, timing, and `RecognizedBehavior`. There is no existing cross-camera human-evidence persistence mechanism to reuse; this milestone's own reconciliation had to be built fresh (Sec 3/4).
6. `live_occupants.occupant.LiveOccupant` **deliberately excluded** classification (a documented, explicit decision from the Live Occupant Digital Twin milestone) — reversed narrowly in this milestone, with the original reasoning preserved and explained (Sec 2, `live_occupants/occupant.py`'s own updated docstring).
7. `RecognizedBehavior.POSSIBLY_FALLEN` was, and remains, structurally separate from `HumanState.FALLEN` — confirmed never mapped onto it anywhere (`_map_behavior_to_human_state()` only ever produces `WALKING`/`RUNNING`/`None`).
8. `perception.providers.human_observation_provider.GroundTruthHumanObservationProvider` (Simulation's richer ground-truth path) is structurally disjoint from `LiveOccupantManager` — it feeds `decision_policy`/`human_priority_policy` via a completely separate `person_id` identity space, never touching `LiveOccupant` anywhere. Live therefore stays honestly poorer than Simulation *by construction*, with zero risk of leakage (Sec 7, proven by test 36).

## 2. Evidence authority rule (Phase 2)

| Field | Authoritative source | Never becomes |
|---|---|---|
| `LiveOccupant.human_classification` | The perception source's own `classification_evidence` (via `human_evidence.reconciliation`) | Guessed from population statistics; `UNKNOWN` never silently becomes `ADULT` |
| `LiveOccupant.human_state` | The perception source's own `state_evidence`, reconciled | `RecognizedBehavior.POSSIBLY_FALLEN` — structurally separate, never promoted |
| `LiveOccupant.behavior` | `RecognizedBehavior`, from `behavior_recognition` (unchanged) | `HumanState` — a hedged heuristic, never conflated with confirmed evidence |
| `OccupantStatus` | `live_occupants.lifecycle` (unchanged) | `HumanState` — lifecycle (present/missing/exited) is orthogonal to physical state |

## 3. Architecture

```
HumanDetector.detect() -- classification_evidence/state_evidence,
        |                  Optional, None/UNKNOWN when the detector
        |                  genuinely has no signal (YOLO today: always)
        v
Tracker (SimpleSingleCameraTracker) -- geometry only, untouched
        v
LiveCameraPipeline._process_camera_cycle() -- Phase 1 bug FIXED here:
        |   genuine detector state_evidence is now preserved; the
        |   behavior-recognizer's own WALKING/RUNNING-or-None mapping
        |   only fills in when the detector supplied none
        v
Behavior Recognition (RecognizedBehavior, incl. POSSIBLY_FALLEN) -- unchanged
        v
Cross-Camera Identity (global occupant_id) -- unchanged, still carries
        |   no classification/state evidence of its own
        v
live_occupants.manager.LiveOccupantManager.update(
    ..., classification_evidence=, classification_confidence=,
    state_evidence=, state_confidence=,
)
        |
        v  human_evidence.reconciliation.reconcile_classification()/
        |  reconcile_state() + apply_*_staleness() (Phase 7/8)
        v
LiveOccupant.human_classification / human_state (persistent, reconciled)
        |
        v
EmergencyResponseIntelligenceEngine._assistance_signal() -- LiveOccupant.
        |   human_state is now PRIMARY (human_state_by_occupant_id
        |   override remains a fallback only, Sec 6)
        v
EmergencyResponseSnapshot.zones[...].occupant_evidence / being_assisted_count /
        |   vulnerable_person_observed (Sec 6)
        v  live_system.live_advisory_gateway.emergency_response_evidence_from_snapshot()
        v
Advisory (being_assisted_zone_ids / vulnerable_person_observed_zone_ids) -->
        Command Center (LiveEmergencyResponsePanel's own per-occupant detail)
```

`human_evidence/` never imports AI/RL/Advisory/Command Center/Voice Evacuation/Building Control/RTSP/YOLO/`decision_policy`, `live_occupants`, or `emergency_response` (mechanically enforced, `tests/test_human_evidence_architecture_guards.py`). `emergency_response/` never imports `human_evidence/` directly — it only ever reads the already-reconciled `LiveOccupant.human_classification`/`human_state` fields, exactly as it already read `LiveOccupant.behavior`.

## 4. Human Evidence model (Phase 3)

`human_evidence/models.py::HumanEvidence` — a small, plain-value, standalone shape (occupant_id, timestamp, classification + confidence + source + last_observed_at, human_state + confidence + source + last_observed_at). `classification_confidence`/`state_confidence` deliberately reuse the **same** per-detection presence-confidence value every other evidence field in this codebase already carries (`RawHumanDetection.confidence`) — investigated directly: no separate, classification-specific or state-specific confidence signal exists anywhere in this codebase today. This is documented honestly as "the detector's own presence confidence, carried through as the best available proxy," never a fabricated new number (Phase 3's own explicit "never fabricate confidence" requirement).

## 5. Multi-camera reconciliation & temporal staleness (Phase 7/8)

`human_evidence/reconciliation.py` — pure, deterministic functions, no randomness, no dependency on call order:

1. **Known beats unknown.** A new `UNKNOWN`/`None` reading never overwrites existing, known evidence.
2. **Agreement** only refreshes recency/confidence.
3. **Genuine conflict** between two different known values: compared as `(timestamp, confidence, source)` — a **strictly later timestamp always wins outright** (this is what lets `HumanState` update readily across cycles — `WALKING → RUNNING → FALLEN → BEING_ASSISTED`, Sec 6); at **equal timestamps** (a genuine same-cycle, multi-camera conflict), the **higher-confidence** reading wins, and on a further tie (or either confidence unavailable), the **lexicographically smaller camera_id** wins — a fixed, content-addressed tie-break, provably independent of which detection happened to be processed first (`tests/test_human_evidence.py::DeterministicConflictResolutionTests`).
4. **Staleness**: `HumanEvidenceConfig.classification_staleness_seconds` (default 300s) vs. `state_staleness_seconds` (default 10s) — deliberately different, because classification changes slowly in reality while state describes what a person is doing *right now*. Applied both on `update()` (staleness of the *existing* value is checked before reconciling new evidence) and every cycle inside `sweep_missing()` (so evidence for a person who simply stops being observed still honestly expires over real wall-clock time, not only on their next sighting).

## 6. POSSIBLY_FALLEN vs. FALLEN (Phase 6) and Emergency Response integration (Phase 12/13)

The hard boundary is enforced structurally, not by convention: `RecognizedBehavior` and `HumanState` are two entirely separate `LiveOccupant` fields, reconciled independently, with **no code path anywhere** that reads one to set the other (`live_occupants/manager.py`'s own module docstring states this explicitly; mechanically proven by `tests/test_live_occupants_human_evidence.py::PossiblyFallenSeparationTests`).

`EmergencyResponseIntelligenceEngine._assistance_signal()` now consults `LiveOccupant.human_state` as the **primary**, canonical live-sourced signal, falling back to the pre-existing `human_state_by_occupant_id` caller override only when the occupant carries no live-sourced state of its own. Three, now mutually-exclusive tiers:

| Tier | Source | Weight | Reason code |
|---|---|---|---|
| Possible | `RecognizedBehavior.POSSIBLY_FALLEN` | 0.20 | `POSSIBLE_ASSISTANCE_REQUIRED` |
| Confirmed (not yet assisted) | `HumanState.FALLEN`/`CRAWLING` | 0.35 | `CONFIRMED_ASSISTANCE_REQUIRED` |
| Being assisted | `HumanState.BEING_ASSISTED` | 0.25 | `ASSISTANCE_IN_PROGRESS` |

`being_assisted_weight` (0.25) is deliberately **weaker** than `confirmed_assistance_weight` (0.35, help is already underway) but clearly **stronger** than `possible_assistance_weight` (0.20, this is confirmed, not hedged, evidence) — proven distinguishable end-to-end by `tests/test_emergency_response_human_evidence.py::test_33_being_assisted_is_distinguishable_from_unassisted_fallen` and `tests/test_live_runtime_human_evidence_e2e.py`.

`HumanClassification.CHILD`/`ELDERLY`/`WHEELCHAIR_USER` contribute only a small, disclosed `vulnerable_classification_weight` (0.10 — smaller than every assistance-tier weight) with reason code `VULNERABLE_PERSON_OBSERVED`, deliberately **never** treated as an assistance/incapacity claim (Phase 12's own explicit "be conservative... assistance-awareness, not fabricated incapacity" requirement) — it never increments `possible_assistance_count`/`confirmed_assistance_count`. `FIREFIGHTER`/`FIRE_WARDEN` are deliberately excluded from this set entirely (they denote response personnel already on scene, not occupants needing rescue).

## 7. LiveOccupant / history / events (Phase 9/10/11)

`LiveOccupant` gained 8 new, all-defaulted fields (`human_classification`/`_confidence`/`_source`/`_last_observed_at`, `human_state`/`_confidence`/`_source`/`_last_observed_at`) — fully backward compatible (every existing construction site and test continues to work unchanged). `OccupantHistory` gained `classification_changes`/`state_changes` (mirroring `behavior_changes` exactly — one record per **genuine** value change only, never one per cycle). Four new `EventType` members: `OCCUPANT_CLASSIFICATION_UPDATED`, `OCCUPANT_STATE_CHANGED`, `POSSIBLE_ASSISTANCE_REQUIRED`, `CONFIRMED_ASSISTANCE_REQUIRED` — all transition-only, proven never to spam (`tests/test_live_occupants_human_evidence.py::NoSpamTests`, held-identical-for-5-cycles → exactly 1 history entry).

## 8. BuildingState relationship (Phase 14)

`BuildingState.occupant_tracks` (via `FusedTrack.classification`/`human_state`) is a **separate, unmodified** per-cycle, highest-confidence-wins snapshot — left alone, exactly as this milestone's own Phase 14 requires. `LiveOccupant` is the correct, separate owner of *persistent*, temporally-reconciled human evidence for Emergency Response Intelligence's own purposes; the two are never unified into one "truth," avoiding a redesign neither package's own callers asked for.

## 9. Simulation/Replay/Live semantics (Phase 17) and the current YOLO boundary (Phase 16)

`perception.providers.human_observation_provider.GroundTruthHumanObservationProvider`'s richer, simulation-only evidence is structurally disjoint from `LiveOccupantManager` (Sec 1 item 8) — Live stays honestly poorer than Simulation with zero code required to keep it that way. `human_detection.yolo_human_detector.YOLOHumanDetector` already, correctly, reports `classification_evidence=HumanClassification.UNKNOWN`/`state_evidence=None` for every detection (confirmed unchanged, tested by `tests/test_live_occupants_human_evidence.py::YoloUnknownBoundaryTests`).

## 10. Safety precedence & no execution authority (Phase 21/22)

Nothing in this milestone touches `decision_policy`'s own fields, `CivilianAnnouncement`/`BuildingRecommendation` safety-critical logic, or any execution path — `emergency_response`'s own, already-proven Safety Precedence discipline (candidate pools filtered by `decision_policy` status *before* any evidence is considered) is unchanged and untouched by this milestone; the new evidence simply flows into the same, already-safe scoring ladder. `human_evidence/`/`live_occupants/` contain no execution verb of any kind (mechanically verified). Proven end-to-end: `tests/test_live_runtime_human_evidence_e2e.py::test_no_automatic_execution_or_dispatch` confirms `voice_evacuation_controller`/`building_control_controller` are both `None` throughout, and the firefighter report carries no `dispatch`/`assigned_task` key.

## 11. Command Center (Phase 18/19)

`command_center.live_emergency_response_panel.LiveEmergencyResponsePanel`'s existing "Selected Zone — Reasons" detail now also renders a "Human evidence:" section, one line per occupant carrying any genuine signal (e.g. `"OCC-12 -- FALLEN"`, `"OCC-18 -- classification unknown"`) — never displaying `UNKNOWN` as though it were a real category (explicit wording only, per Phase 18's own requirement).

## 12. Offline end-to-end results (Phase 24)

`tests/test_live_runtime_human_evidence_e2e.py` drives the complete production chain with a test-only, honestly-labeled `FakeStatefulHumanDetector` (never pretending to be YOLO) across three occupants:
- **OCC-1** — `WALKING → RUNNING` (real tracking geometry) → genuine `FALLEN` evidence (the fake detector's own `state_evidence`) → `BEING_ASSISTED`. Proven: reaches `CONFIRMED_ASSISTANCE_REQUIRED` at the `FALLEN` cycle, then `ASSISTANCE_IN_PROGRESS` (distinct reason code, zero `confirmed_assistance_count`) once `BEING_ASSISTED`.
- **OCC-2** — a low, wide, stationary bounding box drives the REAL `RuleBasedBehaviorRecognizer`'s own (enabled) geometric heuristic to genuinely conclude `POSSIBLY_FALLEN`, with **zero** state evidence ever injected directly — proven to remain `possible_assistance_count == 1`, `confirmed_assistance_count == 0`, `human_state` staying `None` throughout.
- **OCC-3** — no evidence of any kind ever supplied — proven to remain `HumanClassification.UNKNOWN`/`human_state=None` throughout.

Zero network, zero physical CCTV, zero automatic voice/building-control/firefighter-dispatch execution anywhere in this file.

## 13. Performance (Phase 25)

`scripts/benchmark_human_evidence.py`, at the milestone's required scale (100 occupants, 20 cameras), zero YOLO/tracker/RTSP inference included:
- Human evidence reconciliation (100 occupants × classification+state/call): ~0.08 ms/call (mean).
- `LiveOccupantManager.update()` with evidence (100 occupants/call): ~3.46 ms/call.
- Emergency Response evidence consumption (100 occupants across 50 zones/call): ~0.51 ms/call.

All well within a 1-second live cycle budget.

## 14. Files created / modified

**Created:**
- `human_evidence/{__init__,models,reconciliation}.py`
- `tests/test_human_evidence.py` — 14 tests (reconciliation module)
- `tests/test_live_occupants_human_evidence.py` — 30 tests (Phase 23 items 1-11, 15-30)
- `tests/test_emergency_response_human_evidence.py` — 11 tests (Phase 23 items 12-14, 31-36 + conservatism)
- `tests/test_human_evidence_architecture_guards.py` — 5 tests (Phase 26)
- `tests/test_live_runtime_human_evidence_e2e.py` — 6 tests, full offline chain (Phase 24)
- `scripts/benchmark_human_evidence.py` — performance benchmark (Phase 25)
- `docs/architecture/live_human_state_assistance_bridge.md` — this document

**Modified:**
- `live_camera_pipeline/pipeline.py` — the Phase 1 bug fix (Sec 1 item 3); `pending_updates` now carries `classification_evidence`/`classification_confidence`/`state_evidence`/`state_confidence` through to `LiveOccupantManager.update()`.
- `live_occupants/occupant.py` — 8 new, defaulted fields; updated docstring explaining the deliberate, narrow reversal of the original "no classification" decision.
- `live_occupants/history.py` — `ClassificationChangeRecord`/`StateChangeRecord` + `with_classification_change()`/`with_state_change()`.
- `live_occupants/events.py` — 4 new payload dataclasses.
- `live_occupants/manager.py` — `update()` gains 4 new keyword-only parameters; reconciliation + staleness applied on every `update()`/`sweep_missing()` call; 4 new events published on genuine transitions only.
- `live_system/event_bus.py` — 4 new `EventType` members.
- `emergency_response/models.py` — `being_assisted_weight`/`vulnerable_classification_weight`; `ASSISTANCE_IN_PROGRESS`/`VULNERABLE_PERSON_OBSERVED` reason codes; `OccupantAssistanceSignal.being_assisted`; `ZoneResponsePriority.being_assisted_count`/`vulnerable_person_observed`/`occupant_evidence`; new `OccupantEvidenceSummary`.
- `emergency_response/engine.py` — `_assistance_signal()` now primarily reads `LiveOccupant.human_state`; `_score_zone()` gains the two new weighted contributions; `_compute_zone_priority()` builds `occupant_evidence`.
- `advisory_system/emergency_response_evidence.py` — `being_assisted_count`, `being_assisted_zone_ids`, `vulnerable_person_observed_zone_ids`.
- `live_system/live_advisory_gateway.py` — adapter wires the two new id sets through.
- `advisory_system/recommendation_models.py` — `IncidentCommanderDashboard.response_being_assisted_zone_ids`.
- `advisory_system/advisory_engine.py` — wires the new field into `build_commander_dashboard()`.
- `command_center/live_emergency_response_panel.py` — per-occupant "Human evidence:" detail section.

**Unchanged (verified, not modified):** `decision_policy/*`, `building_state/*` (including `FusedTrack`/`multi_camera_fusion` — Sec 8), `cross_camera_identity/*`, `behavior_recognition/*`, `human_detection/*`, `ground_truth/*`, `voice_evacuation/*`, `building_control/*`, `facp/*`.

## 15. Answers to this milestone's own closing questions

**A. Can genuine `HumanClassification` evidence now survive into `LiveOccupant`?** Yes — `tests/test_live_occupants_human_evidence.py::ClassificationSurvivalTests` (Sec 4/7).

**B. Can genuine `HumanState` evidence now survive into `LiveOccupant`?** Yes — `tests/test_live_occupants_human_evidence.py::StateSurvivalTests` (Sec 5/7).

**C. Does `UNKNOWN` remain `UNKNOWN` when no evidence exists?** Yes — proven directly (`tests/test_live_occupants_human_evidence.py::YoloUnknownBoundaryTests`) and structurally (Sec 2's own authority table).

**D. Can two cameras provide conflicting evidence without nondeterministic overwrite behavior?** Yes — `tests/test_human_evidence.py::DeterministicConflictResolutionTests` proves the same result regardless of arrival order (Sec 5).

**E. Can classification survive a cross-camera handover when appropriate?** Yes — `tests/test_live_occupants_human_evidence.py::CrossCameraHandoverTests` (Sec 5).

**F. Can human state change over time?** Yes — `WALKING → RUNNING → FALLEN → BEING_ASSISTED`, proven both in isolation (`tests/test_human_evidence.py`) and end-to-end (`tests/test_live_runtime_human_evidence_e2e.py`) (Sec 5/12).

**G. Is `POSSIBLY_FALLEN` still strictly separate from `HumanState.FALLEN`?** Yes — structurally enforced, mechanically proven (Sec 6).

**H. Does confirmed `FALLEN` produce stronger assistance evidence than `POSSIBLY_FALLEN`?** Yes — `0.35` vs. `0.20`, proven by `tests/test_emergency_response_human_evidence.py::test_32_...` (Sec 6).

**I. Can `BEING_ASSISTED` be distinguished from `FALLEN`?** Yes — two distinct counts and reason codes, proven by `test_33_being_assisted_is_distinguishable_from_unassisted_fallen` (Sec 6).

**J. Can the current YOLO-only path honestly remain classification `UNKNOWN`?** Yes — confirmed unchanged, tested directly (Sec 9).

**K. Can any of this override deterministic evacuation safety rules?** No — this milestone touches no safety-decision code path at all; the existing, already-proven Safety Precedence discipline is entirely untouched (Sec 10).

**L. Can any of this automatically dispatch firefighters, broadcast voice messages, or execute building controls?** No — mechanically verified (no execution verb in `human_evidence/`/`live_occupants/`) and proven end-to-end (Sec 10/12).

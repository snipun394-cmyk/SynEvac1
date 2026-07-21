# Live Emergency Response & Rescue Priority Intelligence

Status as of this milestone: a new deterministic runtime analytics layer, `emergency_response/`, ranks and explains zone-level emergency response/rescue priority using only genuinely live evidence. `decision_policy`, `BuildingState`, `crowd_intelligence`, and `evacuation_progress` remain unmodified as *packages* (one previously-latent bug was found and fixed *inside* `evacuation_progress/engine.py` — see Sec 1 item 4); this milestone never dispatches, broadcasts, or executes anything.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `decision_policy.rescue_policy`/`human_priority_policy` and `human_decision_engine/` are entirely simulation-derived — they read `ground_truth.human_behavior.DynamicHumanState`/`Scenario`-authored occupants, never a live perception source. No live equivalent exists there.
2. `live_occupants.occupant.LiveOccupant` carries **no** `HumanClassification`/`HumanState` field at all — only `RecognizedBehavior` (`behavior_recognition.observation`). The perception pipeline's own `perception.models.human_observation.HumanObservation`/`derive_inference_flags()` is a **separate, non-deduplicated identity space** (`person_id`, not `LiveOccupantManager`'s own `occupant_id`) — confirmed never unified anywhere in this codebase.
3. `RecognizedBehavior.POSSIBLY_FALLEN` is therefore the **one** built-in signal this milestone can honestly treat as live-reaching assistance evidence — and only as a **possible**, hedged one (the behavior-recognition package's own docstring names its false-positive causes).
4. **One real, pre-existing bug found in already-committed code, in `evacuation_progress/engine.py::_classify_zone_status()`**, discovered only because this milestone's own end-to-end test was the first to exercise the real trend→status pipeline for a genuinely stalled zone (every prior test constructed `ZoneClearance(status=STALLED)` objects directly, bypassing the classification method entirely): the method compared the already-*mapped* `trend` value against `EvacuationProgressTrend.STABLE`, but `_map_zone_trend()` maps "no change while occupants remain" to `STALLED`, not `STABLE` — so `ZoneClearanceStatus.STALLED` was **unreachable** via the real pipeline in production. Fixed: the comparison now reads `trend == EvacuationProgressTrend.STALLED`. Re-ran the full pre-existing evacuation-progress regression suite (30 tests) afterward — all still passed, confirming the fix was a pure bug fix, not a behavior change any existing test depended on.
5. Reuse opportunities: `BuildingState.facp_status`/`smoke_detector_states`/`heat_detector_states` (FACP/detector alarm evidence — read directly, never a second FACP/SensorManager query), `BuildingState.zone_severity()` (hazard evidence), `crowd_intelligence.models.IntensityLevel` (congestion evidence), `evacuation_progress.models.ZoneClearanceStatus` (stall/observability evidence) — every one of these is an already-computed sibling snapshot or canonical live state, never recomputed.

## 2. Architecture

```
LiveOccupantManager.active_occupants()  +  BuildingState (facp_status, zone_severity)
        +  CrowdIntelligenceSnapshot (already computed)  +  EvacuationProgressSnapshot (already computed)
        v
emergency_response.engine.EmergencyResponseIntelligenceEngine.compute(
    time, building_state, crowd_snapshot, evacuation_progress_snapshot, human_state_by_occupant_id=None,
)
        |
        v
emergency_response.models.EmergencyResponseSnapshot  (ZoneResponsePriority per zone, deterministic ordering)
        |
        v  (live_system.live_advisory_gateway.emergency_response_evidence_from_snapshot())
        |
advisory_system.emergency_response_evidence.EmergencyResponseEvidence  (plain values only)
        |
        v
                    Advisory (advisory_system.advisory_engine)
                   /        |         \          \
      decision_policy   AI Evidence  Crowd Ev.  Progress Ev.
        |
        v
AdvisoryReport  -->  StateManager.emergency_response / advisory_report
        |
        v
Command Center (command_center.live_emergency_response_panel.LiveEmergencyResponsePanel)
        |
        v
Human Operator / Incident Commander  (decision support only -- never automatic dispatch)
```

`emergency_response/` never imports AI, RL, Advisory, Command Center, Voice Evacuation, Building Control, RTSP, YOLO, `decision_policy`, or any simulation-only source (`ground_truth`, `human_decision_engine`, `simulator`) — mechanically enforced, see `tests/test_emergency_response_architecture_guards.py`. It is wired into `LiveOrchestrator.run_cycle()` **after** `evacuation_progress_gateway` and **before** `live_ai_gateway`/`live_advisory_gateway` — no existing stage is reordered.

## 3. Possible vs. Confirmed assistance (Phase 4/5)

Two, never-merged tiers, mirroring `evacuation_progress`'s own "never claim more certainty than the pipeline can honestly support" discipline:

| Tier | Source | Field |
|---|---|---|
| **POSSIBLE** | `LiveOccupant.behavior == RecognizedBehavior.POSSIBLY_FALLEN` — a live, geometric heuristic | `OccupantAssistanceSignal.possible` |
| **CONFIRMED** | An OPTIONAL caller-supplied `human_state_by_occupant_id: Mapping[occupant_id, HumanState]`, checked against `{FALLEN, CRAWLING, BEING_ASSISTED}` | `OccupantAssistanceSignal.confirmed` |

`human_state_by_occupant_id` is documented, in `engine.py`'s own module docstring, as the **caller's own responsibility**: it is only meaningful if the caller's perception source assigns `HumanObservation.person_id` using the *same* identity scheme as `LiveOccupant.occupant_id` — this engine never assumes that correlation exists, never reinterprets a mismatched identity space, and defaults to `{}` (every signal falls back to `possible`-only) when not supplied. `ResponseReason.CONFIRMED_ASSISTANCE_REQUIRED` outranks `POSSIBLE_ASSISTANCE_REQUIRED` in the scoring ladder (`confirmed_assistance_weight=0.35` vs. `possible_assistance_weight=0.20`) and the two are never collapsed into one ambiguous flag.

## 4. Deterministic priority scoring (Phase 7)

Every weight is a documented, configurable project assumption (`emergency_response.models.ResponseWeights`) — never an opaque constant, and `priority_score` is an explicitly **relative** ranking value, never clamped to `[0, 1]` or presented as a probability:

| Evidence | Weight | Reason code |
|---|---|---|
| Known occupants present (normalized to 3+) | 0.25 | `KNOWN_OCCUPANTS_PRESENT` |
| Confirmed assistance required | 0.35 | `CONFIRMED_ASSISTANCE_REQUIRED` |
| Possible assistance required (only if not confirmed) | 0.20 | `POSSIBLE_ASSISTANCE_REQUIRED` |
| Evacuation stalled (`ZoneClearanceStatus.STALLED`) | 0.20 | `EVACUATION_STALLED` |
| Hazard present, scaled by severity (`HazardSeverity` → 0.25/0.5/0.75/1.0) | 0.30 | `HAZARD_PRESENT` |
| High congestion actively restricting a stalled evacuation | 0.15 | `HIGH_CONGESTION_RESTRICTING_EVACUATION` |
| Uncertain occupancy (`ZoneClearanceStatus.UNKNOWN`) | 0.20 | `UNCERTAIN_OCCUPANCY` |
| FACP/detector alarm active for this zone | 0.15 | `FACP_ALARM_ACTIVE` |

`ResponsePriorityThresholds` (`critical_at=0.65`, `high_at=0.40`, `moderate_at=0.15`) classify the summed score into `LOW`/`MODERATE`/`HIGH`/`CRITICAL`. `priority_score`/`priority_level` are `None`/`UNKNOWN` **only** when a zone carries genuinely zero evidence of any kind (Phase 3's own honest floor) — poor observability (`UNCERTAIN_OCCUPANCY`) is itself a real, positive, actionable contribution, never a silent default to `LOW` (Phase 6/9's explicit requirement).

## 5. Observed-clear vs. uncertain (Phase 6/9)

`ZoneClearanceStatus.UNKNOWN` (from the already-computed `EvacuationProgressSnapshot`) contributes `UNCERTAIN_OCCUPANCY` — a genuine search/verification concern that *raises* priority. `ZoneClearanceStatus.OBSERVED_CLEAR` with zero known occupants contributes the informational `OBSERVED_CLEAR` reason code only (no score contribution) — a zone is never silently treated as safe merely because nobody is currently tracked there without confirmed camera coverage.

## 6. Double-counting protection (Phase 2/8)

Every occupant count derives exclusively from `LiveOccupantManager.active_occupants()` — never raw per-camera detections. Proven directly: `tests/test_emergency_response_double_counting.py` — 2 cameras, 3 physical occupants, 4 raw detections (one identity visible in both cameras simultaneously) → `known_occupant_count` sums to 3, never 4, across the affected zones.

## 7. Events (Phase 12/13)

`RESPONSE_PRIORITY_UPDATED` fires once per successful computation (mirroring `EVACUATION_PROGRESS_UPDATED`/`CROWD_INTELLIGENCE_UPDATED`). `ZONE_RESPONSE_ESCALATED`/`ZONE_RESPONSE_DEESCALATED`/`POSSIBLE_ASSISTANCE_DETECTED` are **transition-only**, emitted by `LiveOrchestrator._emit_emergency_response_transition_events()`, which compares each cycle's snapshot against the previous cycle's:
- Escalation/de-escalation fires only when a zone's `priority_level` **ordinal** (`LOW=0 < MODERATE=1 < HIGH=2 < CRITICAL=3`) genuinely changes between two concrete levels (never involving `UNKNOWN`, never re-firing every cycle the same level is held — proven by `tests/test_emergency_response_events.py::HeldHighNoSpamTests`, held across 5 consecutive cycles, exactly 1 event).
- `POSSIBLE_ASSISTANCE_DETECTED` fires only the cycle a zone's own assistance count transitions from zero to nonzero, never re-fired every cycle the same person stays flagged.

## 8. Reappearance correction (Phase 17)

Since every count is read fresh from `LiveOccupantManager.active_occupants()` each cycle, an occupant who goes missing and later reappears is handled correctly with zero special-case code in this package: `known_occupant_count` drops to 0 while missing and returns to its prior value on reappearance (`tests/test_emergency_response_events.py::ReappearanceTests`).

## 9. Advisory integration and Safety Precedence (Phase 16/19)

`advisory_system.emergency_response_evidence.EmergencyResponseEvidence` mirrors `CrowdDecisionEvidence`/`EvacuationProgressEvidence` exactly: plain values only, no `emergency_response` import (mechanically enforced). It is **SECONDARY, SUPPORTING evidence only**:

```
HAZARD / DETERMINISTIC SAFETY RULES  (decision_policy: CLOSE / AVOID / EVACUATE_IMMEDIATELY / SHELTER_IN_PLACE)
        >
EMERGENCY RESPONSE PRIORITY  (WAIT-zone confidence/reason only; additive-only BuildingRecommendations)
        >
AI / CROWD / PROGRESS SUPPORTING CONFIDENCE
```

Mechanically, `_response_confidence_for_wait_zone()`/`_response_wait_zone_reason_note()` only ever run when `action == WAIT` (a status `zone_policy` itself, independently, already produced) **and** only when this exact zone is one emergency response intelligence has independently flagged `CRITICAL`/`HIGH` — never creating `WAIT`, never touching `SHELTER_IN_PLACE`/`EVACUATE_IMMEDIATELY`, never a building-wide generalization. Three new, purely additive `BuildingRecommendation` categories were added — `"Priority Search: Zone {id}"` (critical/high zones only), `"Possible Assistance Required in Zone {id}"`, `"Verify Occupancy in Zone {id}"` — none of them ever reads or writes `decision_policy`'s own `CLOSE`/`AVOID`/action fields; candidate zone sets for each are drawn directly from `EmergencyResponseEvidence`'s own already-classified id sets, never re-derived from a mutable decision. Proven by the full 7-test safety matrix in `tests/test_emergency_response_advisory_safety_precedence.py` (items 25-30 of the milestone's own required matrix, plus a "no evidence preserves prior behavior" regression test).

`_RESPONSE_FINDING_CONFIDENCE = 0.80` — a documented, deterministic constant, set **above** `DETERMINISTIC_RULE_BASE_CONFIDENCE` (0.70) from the start (having learned, in the immediately preceding evacuation-progress milestone, that a constant below that floor makes corroborating evidence *decrease* blended confidence instead of increasing it) so a genuine finding always *strengthens* a recommendation's blended confidence once averaged in via `recommendation_confidence(..., response_confidence=...)`.

## 10. Firefighter/Commander intelligence (Phase 18)

`IncidentCommanderDashboard` gained three additive fields: `response_highest_priority_zone_id`, `response_critical_zone_ids`, `response_possible_assistance_zone_ids`. `FirefighterIntelligenceReport` gained `live_priority_zone_ids`/`live_possible_assistance_zone_ids` — deliberately **separate** from the pre-existing, simulation-derived `rescue_priority_areas` field (never merged, never silently replacing a simulation-derived value with a live one or vice versa).

## 11. Command Center (Phase 21)

A new `command_center.live_emergency_response_panel.LiveEmergencyResponsePanel` (mirroring `LiveEvacuationProgressPanel`/`LiveAIPanel`'s own "dumb widget, pushed updates only" convention) shows a ranked Response Priority Queue table (rank, priority, zone, floor, occupants, assistance count, clearance status, hazard severity) plus a selected-zone reason detail, wired as a new "Live Emergency Response" tab in `Dashboard`. Prominently labeled "Decision support only — this panel does not dispatch personnel, broadcast voice messages, or execute building controls." `CommandCenterSnapshot` gained `emergency_response`/`emergency_response_timestamp` fields (mirroring `evacuation_progress`, never folded into `_resolve_consistency()`).

## 12. No automatic dispatch/execution (Phase 19/25)

`emergency_response/` contains no execution verb of any kind (mechanically verified: no `.evaluate(`/`.broadcast(`/`.execute_control(`/`.dispatch(` call anywhere in the package). Proven end-to-end: `tests/test_live_runtime_emergency_response_e2e.py::test_no_automatic_execution_or_dispatch` confirms `runtime.voice_evacuation_controller`/`runtime.building_control_controller` are both `None` throughout, and `FirefighterIntelligenceReport.to_dict()` carries no `"dispatch"`/`"assigned_task"` key.

## 13. Offline end-to-end results (Phase 23)

`tests/test_live_runtime_emergency_response_e2e.py` drives the complete production chain across the full stack (Perception → BuildingState → Crowd Intelligence → Evacuation Progress → Emergency Response → Advisory → Command Center) with three zones:
- **Zone A** — already evacuated, confirmed clear throughout.
- **Zone B** — a genuine "fire + stalled evacuation" narrative: an active smoke-detector alarm plus two occupants queuing at `EXIT-1` whose clearance stalls, then both occupants actually cross and it clears. Proven: Zone B rises to the building's single highest response priority (`highest_priority_zone_id() == "zone-b"`, first in `response_priority_order`, `EVACUATION_STALLED` in its reason codes) while it is stalled, its `priority_score` measurably falls once both occupants clear, and the Advisory report carries a genuine `"response"`-sourced `BuildingRecommendation` mentioning zone-b (the "Priority Search" recommendation, reachable here because the added hazard evidence pushes the zone past the `HIGH` threshold).
- **Zone C** — no occupants ever visible; its own camera starts online (confirmed `OBSERVED_CLEAR`), then is disabled mid-run (`camera_manager.disable_camera()`) — clearance correctly reverts to uncertain (`UNCERTAIN_OCCUPANCY`), never permanently "clear." (Investigated directly: `Camera.mode`, set via `set_camera_mode()`, and `Camera.active`, set via `enable_camera()`/`disable_camera()`, are two independent fields in `camera_manager/manager.py` — only `active` governs whether a camera currently covers its zone at all.)

Zero network, zero physical CCTV, zero automatic voice/building-control/firefighter-dispatch execution anywhere in this file.

## 14. Performance (Phase 24)

`scripts/benchmark_emergency_response.py`, at the milestone's required scale (50 zones, 100 occupants), zero real YOLO/tracker/RTSP inference included:
- Complete zone-priority computation (all zones/call): ~0.37 ms/call (mean).
- Per-zone scoring ladder (50 zones/call): ~0.31 ms/call.
- Building-wide deterministic ordering (50 zones/call): ~0.01 ms/call.
- Event/change-detection (escalation/de-escalation/assistance): ~0.01 ms/call.
- Advisory response-evidence creation (adapter): ~0.02 ms/call.
- Advisory response-recommendation processing (full `AdvisoryOrchestrator.generate_report()`): ~0.06 ms/call.

All well within a 1-second live cycle budget.

## 15. Files created / modified

**Created:**
- `emergency_response/{__init__,models,engine}.py`
- `advisory_system/emergency_response_evidence.py`
- `command_center/live_emergency_response_panel.py`
- `live_system/emergency_response_gateway.py`
- `tests/test_emergency_response.py` — 20 unit tests
- `tests/test_emergency_response_events.py` — 4 tests (transition-event behavior, driven through the real `LiveOrchestrator`)
- `tests/test_emergency_response_double_counting.py` — 1 test
- `tests/test_emergency_response_advisory_safety_precedence.py` — 7 tests
- `tests/test_emergency_response_architecture_guards.py` — 7 tests
- `tests/test_live_runtime_emergency_response_e2e.py` — 5 tests, full offline chain
- `scripts/benchmark_emergency_response.py` — performance benchmark
- `docs/architecture/live_emergency_response_intelligence.md` — this document

**Modified:**
- `evacuation_progress/engine.py` — the one genuine pre-existing bug fix (Sec 1 item 4).
- `live_runtime/factory.py` — new optional `emergency_response_engine` parameter; default-constructs and wires one into `LiveOrchestrator` via `EngineEmergencyResponseGateway`.
- `live_runtime/runtime.py` — stores `emergency_response_engine` as a new attribute.
- `live_system/orchestrator.py` — new optional `emergency_response_gateway` constructor parameter; a new `run_cycle()` stage after evacuation progress; transition-based event emission; the advisory stage now also converts `snapshot.emergency_response` and passes it to `live_advisory_gateway.generate()`; a new `latest_emergency_response` forwarding property.
- `live_system/state_manager.py` — new `LiveBuildingSnapshot.emergency_response` field; `update_emergency_response()`/`latest_emergency_response()`.
- `live_system/event_bus.py` — four new `EventType` members.
- `live_system/live_advisory_gateway.py` — `emergency_response_evidence_from_snapshot()` adapter; `generate()` gains an `emergency_response_evidence` parameter (default `None`, backward compatible).
- `advisory_system/{recommendation_models,confidence_engine,advisory_engine}.py` — additive `emergency_response_evidence`/`response_confidence` fields and the WAIT-zone/BuildingRecommendation/CommanderDashboard/FirefighterReport logic described in Sec 9/10.
- `command_center/{data_source,dashboard,live_command_center_gateway}.py` — the new panel's wiring (Sec 11).
- `tests/test_ai_augmented_advisory.py` — the one existing custom `LiveAdvisoryGateway` implementation updated to accept the new parameter.

**Unchanged (verified, not modified):** `decision_policy/*`, `building_state/*`, `crowd_intelligence/*` (the engine itself), `evacuation_progress/models.py`/`ledger.py`, `ground_truth/*`, `human_decision_engine/*`, `voice_evacuation/*`, `building_control/*`, `facp/*`.

## 16. Answers to this milestone's own closing questions

**A. Can SynEvac rank zones by combined emergency response/rescue priority using only live evidence?** Yes — `EmergencyResponseSnapshot.response_priority_order`, a fully deterministic ranking (Sec 4/2).

**B. Can it distinguish a possible fallen-person signal from a confirmed one?** Yes — `OccupantAssistanceSignal.possible`/`.confirmed`, never merged (Sec 3).

**C. Can it factor in hazard severity, evacuation stall, congestion, and uncertain occupancy together, with disclosed weights?** Yes — the full scoring ladder in Sec 4, every weight documented in `ResponseWeights`.

**D. Can poor camera coverage be distinguished from confirmed safety, and does it correctly raise priority rather than default to LOW?** Yes — `UNCERTAIN_OCCUPANCY` is a genuine, positive score contribution, never a silent `LOW` default (Sec 5).

**E. Can the system avoid double-counting occupants across cameras when computing zone priority?** Yes — proven directly by `tests/test_emergency_response_double_counting.py` (Sec 6).

**F. Does priority escalate/de-escalate correctly over time without event spam?** Yes — transition-only events, proven held-high-for-5-cycles → exactly 1 event (Sec 7).

**G. Can a reappearing occupant correct a stale response-priority reading?** Yes — zero special-case code needed, since counts are recomputed fresh every cycle from `LiveOccupantManager` (Sec 8).

**H. Can emergency response priority override a deterministic safety decision?** No — structurally impossible; no code path in this milestone ever writes to `decision_policy`'s own fields (Sec 9, proven by the full safety matrix).

**I. Does this system ever automatically dispatch firefighters, broadcast voice, or execute building controls?** No — mechanically verified (no execution verb in the package) and proven end-to-end (`voice_evacuation_controller`/`building_control_controller` both `None` throughout the full E2E run) (Sec 12).

**J. Is the priority score presented honestly (a relative ranking, not a validated life-safety probability)?** Yes — `ResponseWeights`/`ResponsePriorityThresholds` are explicitly documented as disclosed, configurable project assumptions, never a validated standard, and `priority_score` is never clamped to `[0, 1]` or described as a probability anywhere in the code or this document (Sec 4).

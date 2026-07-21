# Live Evacuation Progress, Flow & Clearance Intelligence

Status as of this milestone: a new deterministic runtime analytics layer, `evacuation_progress/`, measures **observed** evacuation progress from the digital twin — never a prediction, never a simulation replay. `decision_policy`, `BuildingState`, and `crowd_intelligence`'s own engine remain unmodified; `live_occupants`/`live_runtime` each received one small, additive fix this milestone's own investigation found necessary.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `ground_truth.evacuation_metrics.compute_evacuation_metrics()` is entirely post-hoc — it reads `movement_result.occupants[...].state == OccupantState.ARRIVED` from an already-completed `MultiAgentSimulationResult`. No live equivalent exists or can exist from this function.
2. `campaign_analytics/analyzer.py` aggregates `total_evacuation_time` **statistics across many completed simulation runs** (average/min/max/percentiles) — purely post-hoc, cross-run analysis, never reusable live.
3. **No existing live cleared-occupant count, exit-crossing event, per-zone clearance, exit throughput, or observed-evacuation-duration computation existed anywhere in this codebase.**
4. **Two real, pre-existing gaps found in `live_runtime.factory.build_live_runtime()`**, both fixed as a direct prerequisite for this milestone:
   - `LiveOccupantManager()` was constructed with **no `event_bus`** — in production, it never published a single `OCCUPANT_CREATED`/`UPDATED`/`EXITED`/`EXPIRED` event. Fixed: `event_bus` is now resolved *before* `LiveOccupantManager`'s own default construction and threaded through.
   - `LiveOccupantManager()` was constructed with **no `exits=`** — `live_occupants.lifecycle.is_near_exit()` always returned `False` against an empty list, making `OccupantStatus.EXITED` **completely unreachable** in production (every missing occupant became `TEMPORARILY_LOST` → `EXPIRED` only). Fixed: the building's own `Exit` objects are now passed through.
   
   Both fixes only affect the auto-constructed default path — a caller supplying their own `LiveOccupantManager` is unaffected, exactly as before.
5. `OccupantStatus.EXITED` is already documented as "an honest, geometry-based guess … not a certainty," recoverable the moment a real sighting contradicts it — this is exactly Phase 3's own "reappearance correction" requirement, already built into `live_occupants/lifecycle.py`, not something this milestone needed to add. `OccupantStatus.EXPIRED`, however, **removes the occupant from `LiveOccupantManager`'s own store entirely** — a durable, event-driven ledger (this milestone's own `evacuation_progress.ledger.EvacuationLedger`) is required to avoid silently losing every occupant's final classification the moment they expire.
6. Reuse opportunities: `crowd_intelligence.flow.exit_sides()`/`distance_to_asset()` (exit-crossing attribution geometry), `crowd_intelligence.trends.TrendTracker` (generic, reused directly for zone/exit/overall trends — never duplicated), `crowd_intelligence`'s own per-exit queue/approach/congestion metrics (never recomputed), `BuildingState.camera_observations` (for honest zone observability — `crowd_intelligence`'s own `position_coverage_fraction` is `None` for an empty zone, which is useless for answering "is this zone even watched right now").

## 2. Architecture

```
LiveOccupantManager (event_bus=<runtime's own EventBus>, exits=<building's own Exit objects>)
        |  OCCUPANT_CREATED / OCCUPANT_UPDATED / OCCUPANT_ZONE_CHANGED / OCCUPANT_EXITED
        v
evacuation_progress.ledger.EvacuationLedger   (durable, event-driven memory --
        |                                       survives EXPIRED occupant removal)
        v
evacuation_progress.engine.EvacuationProgressEngine.compute(time, building_state, crowd_snapshot)
        |
        v
evacuation_progress.models.EvacuationProgressSnapshot
        |
        v  (live_system.live_advisory_gateway.evacuation_progress_evidence_from_snapshot())
        |
advisory_system.evacuation_progress_evidence.EvacuationProgressEvidence  (plain values only)
        |
        v
                    Advisory (advisory_system.advisory_engine)
                   /        |        \
      decision_policy   AI Evidence   Crowd Evidence
        |
        v
AdvisoryReport  -->  StateManager.evacuation_progress / advisory_report
        |
        v
Command Center (command_center.live_evacuation_progress_panel.LiveEvacuationProgressPanel)
        |
        v
Human Operator
```

`evacuation_progress/` itself never imports AI, RL, Advisory, Command Center, Voice Evacuation, Building Control, RTSP, YOLO, or `decision_policy` (Phase 22, mechanically enforced). It is wired into `LiveOrchestrator.run_cycle()` **after** `crowd_intelligence_gateway` (reads `snapshot.crowd_intelligence`, this cycle's fresh value or the previous cycle's) and **before** `live_ai_gateway`/`live_advisory_gateway` — no existing stage is reordered.

## 3. The three distinct concepts (never mixed)

| Concept | Owner | Meaning |
|---|---|---|
| **CURRENT OCCUPANCY** | `live_occupants.manager.LiveOccupantManager.active_occupants()` (unchanged, pre-existing) | Who is observed right now |
| **OBSERVED EVACUATION PROGRESS** | THIS package | What has been directly, deterministically observed **so far**, always phrased against a known-observed denominator |
| **PREDICTED EVACUATION TIME** | `live_system.live_ai_gateway`'s own, pre-existing, **EXPERIMENTAL** `evacuation_time_experimental` field | A model's forecast — this package builds no prediction of any kind and never touches that field |

`"80% of currently observed identities have exited"` (what this package can honestly say) is **never** presented as `"80% of the building population has evacuated"` (which would require knowing the true population — a number this platform cannot honestly know and never claims to).

## 4. Exit-crossing evidence semantics (Phase 3)

Only **two** evidence levels are used (`evacuation_progress.models.ExitEvidenceLevel`): `LIKELY_EXITED` and `UNKNOWN`. **No `CONFIRMED` level was added** — investigated directly: no turnstile, door sensor, or any hard-confirmation signal exists anywhere in this codebase for an exit crossing. `OccupantStatus.EXITED` is itself already an honest, recoverable geometric guess; this package never claims more certainty than that.

`evacuation_progress.ledger.EvacuationLedger` subscribes to the live occupant event stream:
- `OCCUPANT_CREATED` → records the occupant_id as "ever seen" (the honest observed-population denominator) and its initial zone.
- `OCCUPANT_ZONE_CHANGED` → grows each zone's own "ever seen here" baseline.
- `OCCUPANT_EXITED` → records a durable `ExitedRecord(occupant_id, exit_id, zone_id, timestamp)`. `exit_id` is determined by finding the **nearest** modeled `Exit` to the occupant's own last known `world_position` (reusing `crowd_intelligence.flow.exit_sides()`/`distance_to_asset()` directly — never recomputed) among exits on the occupant's own floor. `exit_id` is `None` when `world_position` is unavailable — the occupant still counts toward `known_exited_occupants` overall, but per-exit attribution is honestly unavailable (Phase 16 test 16).
- `OCCUPANT_UPDATED` with status `ACTIVE`/`NEW` → **removes** the occupant from the durable-exited ledger, if present — a real, current sighting always corrects an earlier uncertain disappearance (Phase 9, proven in `tests/test_evacuation_progress.py::ReappearanceCorrectionTests`).

## 5. Zone-clearance semantics (Phase 4/7)

`ZoneClearanceStatus`: `UNKNOWN`, `OCCUPIED`, `CLEARING`, `NEARLY_CLEAR`, `OBSERVED_CLEAR`, `STALLED`. The classification ladder in `EvacuationProgressEngine._classify_zone_status()` checks **observability first, unconditionally**: a zone with no currently-active camera coverage (from `BuildingState.camera_observations`, never from `crowd_intelligence`'s own coverage fraction) is always `UNKNOWN`, regardless of what the occupant count reads — never labeled `OBSERVED_CLEAR` merely because nobody happens to be watching. `baseline_observed_count` is the count of distinct occupant_ids **ever** tracked in that zone (an observed watermark, never a fabricated design occupancy) — `clearance_fraction = cleared_count / baseline` only when `baseline > 0`.

## 6. Exit-flow / throughput model (Phase 5/10)

`ExitFlow.queue_candidate_count`/`approaching_count`/`congestion_level` are read **directly** from `crowd_intelligence`'s own `AssetApproachMetrics` for that exit — never recomputed (Phase 10's own explicit "do not duplicate" requirement). `unique_exited_count`/`recent_flow_per_minute`/`flow_active` come from the ledger's own durable exit-attributed records within a configurable `flow_window_seconds` (default 60s). **The critical distinction**: `low_flow_exit_ids` (a "flow problem") is `queue_candidate_count > 0 and not flow_active` — a genuine bottleneck signal, deliberately **independent of trend** — proving both worked examples directly: `HIGH queue + LOW throughput` → flagged (`tests/test_evacuation_progress.py::FlowProblemTests::test_10_...`); `HIGH queue + HIGH throughput` → never incorrectly flagged (`test_11_...`), even though both cases can carry identical queue counts.

## 7. Trend model (Phase 6)

`crowd_intelligence.trends.TrendTracker` is reused directly (bounded history, configurable time window — never duplicated), but its own 4-value vocabulary (`RISING`/`STABLE`/`FALLING`/`UNKNOWN`) is deliberately **re-mapped**, not reused verbatim, into this package's own 5-value `EvacuationProgressTrend` (`IMPROVING`/`STABLE`/`SLOWING`/`STALLED`/`UNKNOWN`) via three small, context-aware functions (`_map_zone_trend`/`_map_exit_trend`/`_map_overall_trend`) — because "no change" means something **different** depending on what is measured: a `STABLE` exit flow rate is healthy (steadily processing people), but a `STABLE` zone-clearance fraction while occupants remain is a genuine problem (`STALLED`). This distinction is made in exactly one place, never duplicated or allowed to drift between the zone/exit/overall call sites. (This exact bug — comparing `crowd_intelligence`'s raw `TrendDirection` enum against this package's own string constants without ever converting between them — was caught and fixed during this milestone's own smoke-testing, before any test was written against it.)

## 8. Observability honesty (Phase 7)

`ObservabilitySummary` and every `ZoneClearance.observable`/`ExitFlow.position_available` field distinguish "no occupants observed" from "confirmed empty" using **real, independent signals** — camera active/offline state (`BuildingState.camera_observations`), not occupant count itself. Proven across 100%, offline, and no-camera-at-all coverage (`tests/test_evacuation_progress.py::ZoneClearanceObservabilityTests`).

## 9. Double-count protection (Phase 8)

Every count in this package derives exclusively from `LiveOccupantManager`'s own canonical global-occupant identity — never raw per-camera detections. Proven directly, extending the crowd-intelligence milestone's own worked example one layer further: `tests/test_evacuation_progress_double_counting.py` — 2 cameras, 3 physical occupants, 4 raw detections (one person visible in both cameras simultaneously) → `known_active_occupants == 3`; a camera handover of the shared identity never fires a fake `OCCUPANT_EXITED` transition.

## 10. Crowd Intelligence integration (Phase 10)

`EvacuationProgressEngine.compute()` accepts an already-computed `CrowdIntelligenceSnapshot` as a parameter (never constructs its own) and reads only its own per-exit `AssetApproachMetrics` — density, queue detection, approach demand, and congestion classification are never recomputed.

## 11. Runtime ownership (Phase 11) and StateManager integration (Phase 12)

Exactly **one** `EvacuationProgressEngine` per `LiveRuntime`, subscribing to the SAME `event_bus` `LiveOccupantManager` now publishes to (the Phase 1 fix). Exposed as `runtime.evacuation_progress_engine`. Result lands on `LiveBuildingSnapshot.evacuation_progress` via `StateManager.update_evacuation_progress()`/`latest_evacuation_progress()`, mirroring `crowd_intelligence`/`ai_prediction_snapshot`/`advisory_report` exactly — investigated and confirmed this is the correct placement (not a `BuildingState` field), since `BuildingState`/`BuildingStateEstimator` were not modified.

## 12. Advisory integration and Safety Precedence (Phase 13/14/15)

`advisory_system.evacuation_progress_evidence.EvacuationProgressEvidence` mirrors `CrowdDecisionEvidence`/`AIDecisionEvidence` exactly: plain values only, no `evacuation_progress` import (Phase 22, mechanically enforced). It is **SECONDARY, SUPPORTING evidence only**:

```
HAZARD / DETERMINISTIC SAFETY RULES  (decision_policy: CLOSE / AVOID / EVACUATE_IMMEDIATELY / SHELTER_IN_PLACE)
        >
EVACUATION PROGRESS / CROWD CONGESTION OPTIMIZATION   (WAIT-zone confidence/reason only; additive-only BuildingRecommendations)
        >
AI SUPPORTING CONFIDENCE
```

Mechanically, `_progress_confidence_for_wait_zone()`/`_progress_wait_zone_reason_note()` only ever run when `action == WAIT` (a status `zone_policy` itself already, independently, produced) — never creates WAIT, never touches `SHELTER_IN_PLACE`/`EVACUATE_IMMEDIATELY`. Three new, purely additive `BuildingRecommendation` categories were added (`"Review Stalled Evacuation Progress in Zone …"`, `"Review Exit … Throughput"`, `"Confirm Clearance Status for Zone …"`, plus one building-wide "slowing" monitor) — none of them ever reads or writes `decision_policy`'s own `CLOSE`/`AVOID`/action fields. Proven by the full 9-test safety matrix in `tests/test_evacuation_advisory_safety_precedence.py` (items 20-25 of the milestone's own required matrix, plus a "no evidence preserves prior behavior" regression test).

`_PROGRESS_FINDING_CONFIDENCE = 0.80` — a documented, deterministic constant, deliberately set **above** `DETERMINISTIC_RULE_BASE_CONFIDENCE` (0.70) so a genuine finding *strengthens* a recommendation's blended confidence once averaged in, never weakens it (a real bug this milestone caught and fixed during its own test-writing: an initial, lower constant was mathematically *decreasing* confidence when blended in, the opposite of the intended "corroborating evidence" effect).

## 13. Firefighter/Commander intelligence (Phase 16)

`IncidentCommanderDashboard` gained three additive fields (`evacuation_progress_fraction`, `evacuation_stalled_zone_ids`, `evacuation_clearance_unknown_zone_ids`), mirroring the `crowd_*` fields the prior milestone already added. `FirefighterIntelligenceReport` was **not** extended in this milestone — the commander-dashboard fields already satisfy Phase 16's "particularly useful for incident command" framing, and further expansion was judged out of scope given the size of this milestone; a future milestone can extend it the same way if needed.

## 14. Command Center (Phase 17)

A new, dedicated `command_center.live_evacuation_progress_panel.LiveEvacuationProgressPanel` (mirroring `LiveAIPanel`/`LiveStatusPanel`'s own "dumb widget, pushed updates only" convention exactly) was added as a new "Live Evacuation Progress" tab in `Dashboard`, wired the same way every other Live-only panel already is. `CommandCenterSnapshot` gained `evacuation_progress`/`evacuation_progress_timestamp` fields (mirroring `ai_prediction_snapshot`, never folded into `_resolve_consistency()`'s own logic, matching the precedent already set when `crowd_intelligence` was similarly added without touching that method). Every label is worded explicitly as "observed"/"tracked" (e.g. `"Observed progress: 33% of 3 tracked/observed occupant(s) cleared"`) — never `"X% of building evacuated"` (Phase 17's own explicit requirement, since total building population is never known).

## 15. Events (Phase 18)

`EVACUATION_PROGRESS_UPDATED` fires once per successful computation (mirroring `CROWD_INTELLIGENCE_UPDATED`). `ZONE_CLEARANCE_STALLED`/`ZONE_OBSERVED_CLEAR`/`EXIT_FLOW_STALLED` are **transition-only** events — `LiveOrchestrator._emit_evacuation_progress_transition_events()` compares each cycle's snapshot against the previous cycle's (captured before `update_evacuation_progress()` is called) and fires only for zones/exits that **newly** entered that status this cycle, never re-firing every cycle the status happens to still hold. Verified directly: a zone that becomes `OBSERVED_CLEAR` and stays that way across three further cycles produces exactly one `ZONE_OBSERVED_CLEAR` event, not four.

## 16. Offline end-to-end results (Phase 20)

`tests/test_live_runtime_evacuation_progress_e2e.py` drives the complete production chain across 5 cycles with two occupants and two exits (`EXIT-1`, congested; `EXIT-2`, safe and clear throughout). Proven directly: occupancy rises `1 → 1 → 2 → 2`; a queue forms at `EXIT-1` by the point both occupants are stationary near it; both then genuinely cross (`known_exited_occupants` reaches 2, `evacuation_progress_fraction` reaches 1.0); `EXIT-1`'s own throughput is measured (`flow_active=True`, `unique_exited_count=2`, a positive `recent_flow_per_minute`); the Advisory report reflects this cycle's evidence while `EXIT-2` is never touched by any progress-sourced recommendation; `runtime.voice_evacuation_controller`/`runtime.building_control_controller` are both `None` throughout. Zero network, zero physical CCTV anywhere in this file.

## 17. Performance (Phase 21)

`scripts/benchmark_evacuation_progress.py`, at the milestone's required scale (50 zones, 10 exits, 100 occupants), zero real YOLO/tracker/RTSP inference included:
- Complete progress update (all zones/exits): ~0.29 ms/call (mean).
- Zone-clearance computation (50 zones/call): ~0.17 ms/call.
- Exit-flow computation (10 exits/call): ~0.07 ms/call.
- History/trend computation (50 keys/call): ~0.06 ms/call.
- Advisory progress-evidence creation (adapter): ~0.05 ms/call.

All well within a 1-second live cycle budget. Complete `AdvisoryReport` generation cost with/without progress evidence mirrors `scripts/benchmark_crowd_advisory.py`'s own already-reported figures (both evidence sources are blended the same, cheap way).

## 18. Files created / modified

**Created:**
- `evacuation_progress/{__init__,models,ledger,engine}.py`
- `advisory_system/evacuation_progress_evidence.py`
- `command_center/live_evacuation_progress_panel.py`
- `live_system/evacuation_progress_gateway.py`
- `tests/test_evacuation_progress.py` — 17 unit tests (Phase 19 items 1, 3, 5-19)
- `tests/test_evacuation_progress_double_counting.py` — 2 tests (Phase 8/19 items 2, 4)
- `tests/test_evacuation_advisory_safety_precedence.py` — 7 tests (Phase 19 items 20-25)
- `tests/test_evacuation_progress_architecture_guards.py` — 7 tests (Phase 22)
- `tests/test_live_runtime_evacuation_progress_e2e.py` — 4 tests, full offline chain (Phase 20)
- `scripts/benchmark_evacuation_progress.py` — performance benchmark (Phase 21)
- `docs/architecture/live_evacuation_progress.md` — this document

**Modified:**
- `live_runtime/factory.py` — the two prerequisite fixes (Sec 1 item 4); new optional `evacuation_progress_engine` parameter; default-constructs and wires one into `LiveOrchestrator` via `EngineEvacuationProgressGateway`.
- `live_runtime/runtime.py` — stores `evacuation_progress_engine` as a new, deliberately untyped attribute.
- `live_system/orchestrator.py` — new optional `evacuation_progress_gateway` constructor parameter; a new, independent `run_cycle()` stage; transition-based event emission; the advisory stage now also converts `snapshot.evacuation_progress` and passes it to `live_advisory_gateway.generate()`; a new `latest_evacuation_progress` forwarding property.
- `live_system/state_manager.py` — new `LiveBuildingSnapshot.evacuation_progress` field; `update_evacuation_progress()`/`latest_evacuation_progress()`.
- `live_system/event_bus.py` — four new `EventType` members.
- `live_system/live_advisory_gateway.py` — `evacuation_progress_evidence_from_snapshot()` adapter; `LiveAdvisoryGateway.generate()`/`ReplayCompatibleAdvisoryGateway.generate()` gain an `evacuation_progress_evidence` parameter (default `None`, backward compatible).
- `advisory_system/{recommendation_models,confidence_engine,advisory_engine}.py` — additive `evacuation_progress_evidence`/`progress_confidence` fields and the WAIT-zone/BuildingRecommendation/CommanderDashboard logic described in Sec 12/13.
- `command_center/{data_source,dashboard}.py` — the new panel's wiring (Sec 14).
- `tests/test_ai_augmented_advisory.py` — the one existing custom `LiveAdvisoryGateway` implementation updated to accept the new parameter.

**Unchanged (verified, not modified):** `decision_policy/*`, `building_state/*`, `crowd_intelligence/*` (the engine itself), `ground_truth/*`, `campaign_analytics/*`, `voice_evacuation/*`, `building_control/*`, `facp/*`, `ai_registry/*`.

## 19. Answers to this milestone's own closing questions

**A. Can SynEvac now determine how many UNIQUE observed occupants remain inside?** Yes — `known_active_occupants`, from `LiveOccupantManager.active_occupants()`.

**B. Can it estimate how many observed occupants have exited without double counting?** Yes — proven directly by `tests/test_evacuation_progress_double_counting.py` (4 raw detections → 3 unique identities, never 4).

**C. Can it distinguish an empty-looking zone from a sufficiently-observed cleared zone?** Yes — `ZoneClearanceStatus.UNKNOWN` (no current camera coverage) vs. `OBSERVED_CLEAR` (confirmed by active coverage), never conflated (Sec 5/8).

**D. Can it measure exit throughput separately from exit queue size?** Yes — `ExitFlow.recent_flow_per_minute`/`unique_exited_count` (throughput, from the ledger) vs. `queue_candidate_count`/`approaching_count` (demand, reused from `crowd_intelligence`), kept as two independent fields.

**E. Can it identify a high-queue / low-throughput exit?** Yes — `low_flow_exit_ids`, proven by both worked examples (Sec 6).

**F. Can it detect stalled zone clearance?** Yes — `ZoneClearanceStatus.STALLED`, via the trend-mapping logic in Sec 7.

**G. Can a temporary camera loss falsely count someone as evacuated?** No — `TEMPORARILY_LOST` (away from any modeled exit) is never counted as exited; only geometry-confirmed `EXITED` (near a modeled Exit) is (Sec 4, proven by `tests/test_evacuation_progress.py::TemporaryOcclusionTests`/`DisappearanceAwayFromExitTests`).

**H. Can a reappearing occupant correct an earlier likely-exit classification?** Yes — proven directly by `tests/test_evacuation_progress.py::ReappearanceCorrectionTests` (Sec 4/9).

**I. Can evacuation progress override deterministic safety rules?** No — structurally impossible; no code path in this milestone ever writes to `decision_policy`'s own fields (Sec 12, proven by the full safety matrix).

**J. Can any progress recommendation automatically execute voice/building controls?** No — `evacuation_progress/` contains no execution verb of any kind (mechanically verified), and the E2E test confirms `voice_evacuation_controller`/`building_control_controller` are both `None` throughout.

# Manual Call Point → Live Emergency Response Integration

Status: a manually activated call point is now a genuine, localized live emergency signal for SynEvac's Emergency Response and Advisory layers — closing the gap `docs/architecture/designer_asset_connectivity_audit.md` found ("MCP reaches FACP but currently drops out of Emergency Response scoring"). No new Designer asset was added, FACP was not redesigned, `EmergencyResponseIntelligenceEngine`'s existing architecture was extended (not rebuilt), and MCP was never given execution authority.

## 1. The chain traced (Phase 1)

Before this milestone, `ManualCallPoint → SensorManager → EngineFACPGateway → SimulatedFACP → FACPSnapshot` already worked correctly — an activated MCP reached `BuildingState.facp_status.active_alarm_source_ids` with its identity, type, and zone all structurally intact (`facp.models.DetectorConditionReport`/`PanelEvent` already carry `asset_id`/`asset_type`/`zone_ids`). **The information was never lost in FACP itself.** It was lost one hop later: `emergency_response/engine.py::_zone_alarm_ids()` only ever iterated `building_state.smoke_detector_states`/`heat_detector_states` — there was no `manual_call_point_states` mapping on `BuildingState` at all, so an MCP-only alarm, despite being genuinely present in `active_alarm_source_ids`, was silently invisible to Emergency Response's zone-alarm computation. Emergency Response received only a building-wide boolean-shaped signal (`alarm_active` per zone, from Smoke/Heat only) — never a source-type-distinguished one, and never MCP at all.

## 2. MCP semantics (Phase 2)

An activated Manual Call Point means exactly one thing: **a human has manually reported an emergency at/associated with this call point.** It does not mean fire confirmed, smoke confirmed, an injured/trapped/fallen occupant, an impassable zone, an unsafe exit, panic, suppression activation, or an invalid evacuation route. The reason code introduced for this, `MANUAL_EMERGENCY_REPORTED`, is deliberately distinct from `FACP_ALARM_ACTIVE` (which remains scoped to automatic Smoke/Heat Detector alarms only, exactly as it already behaved before this milestone — MCP was already excluded from it).

## 3. Structured evidence, not a boolean (Phase 3)

`BuildingState` gained one additive field, `manual_call_point_states: Mapping[str, DetectorAssetState]`, mirroring `smoke_detector_states`/`heat_detector_states` exactly — same `DetectorAssetState(status, reading)` shape, populated the same way (`BuildingStateEstimator._build_detector_states()`, unchanged). The one adaptation: a Manual Call Point has no external hazard-threshold reading (its own `compute_state(time)` is already the complete, self-contained answer — a human action, not a sensor threshold), so a new, small, sibling type was added: `perception.models.manual_call_point_observation.ManualCallPointReading` (`detector_id`, `timestamp`, `alarm_active`, `confidence=None` always), derived directly from `ManualCallPoint.compute_state(time) == DetectorState.ALARM`, wired via a new `manual_call_point_status_provider`/`manual_call_point_reading_provider` pair in `live_runtime/factory.py` (built entirely from the already-in-scope `sensor_manager`, no new constructor parameter needed) and threaded through `live_system/building_state_gateway.py` the same way smoke/heat already were.

`emergency_response/models.py` gained `AlarmSourceEvidence` (`source_id`, `source_type`, `zone_ids`) — a small, structured, per-source reduction, reusing `SensorStatus.sensor_type` directly (never a string-parsed reason code) to tell Smoke/Heat/ManualCallPoint apart. `ZoneResponsePriority` gained `alarm_sources: Tuple[AlarmSourceEvidence, ...]` and `manual_emergency_reported: bool` — both purely additive.

## 4. Emergency Response integration (Phase 4)

`emergency_response/engine.py::_zone_alarm_ids()` (which only ever returned a bare set of zone ids) became `_alarm_sources_by_zone()`, returning `{zone_id: (AlarmSourceEvidence, ...)}` built from all three `BuildingState` detector-state mappings. `_compute_zone_priority()` now derives two **independent** booleans from that structured evidence — `automatic_alarm_active` (any non-MCP source) and `manual_emergency_reported` (any MCP source) — and `_score_zone()` applies each as its own additive contribution: the existing `facp_alarm_weight` (0.15, unchanged) for automatic alarms, and a new, disclosed, configurable `ResponseWeights.manual_report_weight` (0.20) for manual reports. Both combine additively with every other existing signal (occupants, assistance, hazard, congestion, stall, uncertainty, route anomaly) exactly as before — MCP is one more explainable signal, never a dominant override.

## 5. Source-type distinction preserved end to end (Phase 5)

`ZoneResponsePriority.reason_codes` now distinguishes `FACP_ALARM_ACTIVE` (automatic) from `MANUAL_EMERGENCY_REPORTED` (manual) as two separate codes, and `alarm_sources` carries the actual source id/type/zone for each. `_explain()` was extended to name the specific device: for a zone with an active MCP, the explanation text now includes *"Manual emergency report received from MCP-1 in Zone Z3."* — the milestone's own worked example, verbatim. This never claims what the person who activated the call point observed; it states only that a report was received from that device.

## 6. Multiple MCPs (Phase 6) — proven independent, latching respected

`MultipleMCPTests` (`tests/test_manual_call_point_emergency_response_integration.py`): MCP-1 in Zone A and MCP-2 in Zone B are proven to affect only their own zone, both independently when both are active, and restoring MCP-1 (without an operator FACP reset) correctly removes it from `active_alarm_source_ids` — and therefore from Zone A's evidence — while the panel itself correctly stays latched in `ALARM` until an explicit `acknowledge()`/`reset()`. This milestone never bypasses that latching; Emergency Response's own per-cycle `manual_emergency_reported` reflects *currently active* sources only, which is a distinct, honest concept from the panel's own latched display state (exactly the distinction FACP already established for Smoke/Heat, extended here to MCP without any special-casing).

## 7. Mixed alarm sources (Phase 7) — proven non-overwriting

`MixedAlarmSourcesTests`: SmokeDetector SD-1 (Zone A), HeatDetector HD-1 (Zone B), and ManualCallPoint MCP-1 (Zone C), triggered together, each remain distinguishable by `source_type` in their own zone's `alarm_sources`. A further test places both a SmokeDetector and a ManualCallPoint in the *same* zone and confirms both `FACP_ALARM_ACTIVE` and `MANUAL_EMERGENCY_REPORTED` appear together, with both scoring contributions applied — neither reason code, nor its score contribution, ever suppresses the other.

## 8. Advisory integration (Phase 8) — evidence, never an override

`advisory_system/emergency_response_evidence.py::ZoneResponseDetail` gained `manual_emergency_reported`/`manual_call_point_ids` (plain-value fields, no new package dependency — this module still imports nothing from `emergency_response/`). `EmergencyResponseEvidence` gained `manual_emergency_report_zone_ids`, mirroring the five/six existing flagged-zone-id sets exactly. `live_system/live_advisory_gateway.py::emergency_response_evidence_from_snapshot()` (the one function allowed to import both packages) now populates both from the real `ZoneResponsePriority` fields. No new Advisory engine was created; the existing `EmergencyResponseEvidence → Advisory` path was extended in place. MCP evidence can inform firefighter/commander awareness text (e.g. "Investigate manual emergency report at MCP-1 in Zone Z3") but never directly sets `EVACUATE_IMMEDIATELY`/`SHELTER_IN_PLACE`/`AVOID`/`CLOSE` or route safety — those remain governed entirely by their own existing, independent deterministic inputs, unchanged by this milestone (re-confirmed directly, §11 below).

## 9. Command Center (Phase 9) — existing panel extended, not replaced

`command_center/live_emergency_response_panel.py::LiveEmergencyResponsePanel` (display-only, no operator control, unchanged in that respect) gained one new table column ("Alarm Sources", showing e.g. `Manual: MCP-1; Auto: SD-1`) and an expanded per-zone detail view listing each alarm source with its type spelled out (`Manual Call Point MCP-1 — manual emergency report (human-reported, not a confirmed hazard)` vs. `SmokeDetector SD-1 — automatic detector alarm`). No new panel was created.

## 10. Full offline E2E (Phase 10)

`FullOfflineE2ETests::test_full_chain_mcp_then_smoke_no_automatic_execution` builds a real `LiveRuntime` (`live_runtime.factory.build_live_runtime()`) with Zone A (MCP-1) and Zone B (SD-1). Initial cycle: no alarm evidence anywhere. Activating MCP-1 and running the next cycle proves the complete real chain — `ManualCallPoint → SensorManager → EngineFACPGateway → SimulatedFACP → BuildingState → EmergencyResponseIntelligenceEngine → EmergencyResponseSnapshot → emergency_response_evidence_from_snapshot()` — reaches Zone A correctly, with MCP-1's identity and zone intact throughout. Triggering SD-1 afterward proves both source types coexist without interference. Throughout: `runtime.voice_evacuation_controller`, `runtime.building_control_controller`, and `runtime.dynamic_signage_controller` all remain `None` (never configured, never dispatched) — nothing was broadcast or executed automatically at any point.

## 11. Safety precedence (Phase 11) — MCP is evidence, never execution authority

`SafetyPrecedenceTests` proves directly: activating an MCP in a zone whose only Exit is already `is_blocked=True` never changes `Edge.traversable` for that exit; activating an MCP in a zone with an active, `"Blocked"`-traversability Obstacle covering a Door never changes that Door's traversability either (both mechanisms — Exit.is_blocked, Obstacle-aware `Edge.traversable`, see `docs/architecture/obstacle_navigation_integration.md` — have no code path through which MCP could reach them at all, confirmed by the models' own import graph, not merely by absence of a test failure). An empty `HazardSnapshot` produces an empty `hazard_summary` regardless of any MCP activity, since no code path connects `models/manual_call_point.py` to `hazard/`, `hazard_evolution/`, `fire_growth/`, or `smoke_propagation/` at all (mechanically confirmed, §14). `EVACUATE_IMMEDIATELY`/`SHELTER_IN_PLACE`/route-safety decisions remain entirely governed by their own pre-existing, independent deterministic inputs.

## 12. Failure / degradation (Phase 12)

`FailureAndDegradationTests` covers: an MCP with `zone_ids=()` (unassigned) reaches `active_alarm_source_ids` at the FACP level but is correctly **never assigned to any zone's evidence** (no fake zone is ever fabricated for it); an MCP referencing a deleted/nonexistent zone id never crashes (the zone simply isn't one of `EmergencyResponseIntelligenceEngine`'s own real building zones, so the reference is harmlessly inert); an inactive MCP (`active=False`) never produces evidence even if `activate()` was called on it (mirrors `ManualCallPoint.compute_state()`'s own pre-existing "inactive never alarms" rule); `BuildingState` with `facp_status=None` never crashes Emergency Response (`_alarm_sources_by_zone()` returns `{}`); a legacy project saved with no `"manual_call_points"` key at all loads and computes cleanly (the field already defaulted to an empty list before this milestone); duplicate MCP ids never crash `SensorManager` discovery; and an FACP `reset()` genuinely clears manual evidence going forward. No test produced a crash or a fabricated zone/location.

## 13. Events (Phase 13) — investigated, nothing added

`facp/engine.py::SimulatedFACP.evaluate()` already emits a `PanelEvent(event_type=PanelEventType.DETECTOR_ALARM, source_asset_type="ManualCallPoint", ...)` on the exact transition cycle an MCP's `DetectorConditionReport` first reaches `ALARM` (the same generic per-`DetectorConditionReport` transition logic Smoke/Heat already use — `MANUAL_ALARM` is a distinct, unrelated event type reserved for `SimulatedFACP.manual_alarm()`, a different method the MCP chain does not use). This is already transition-only (fired once per state change, never every cycle) and already carries the exact structured identity/zone metadata a new event would need to duplicate. **Conclusion: no new EventBus event was added** — `FACPSnapshot.recent_events` (already surfaced via `BuildingState.facp_status`) combined with this milestone's new per-cycle `ZoneResponsePriority.alarm_sources`/`manual_emergency_reported` fields together already provide everything a consumer needs, both historically (event log) and for the current cycle (structured evidence).

## 14. Architecture guards (Phase 14)

Mechanically proven (`ArchitectureGuardTests`): `models/manual_call_point.py` imports none of `voice_evacuation`, `building_control`, `dynamic_signage`, `decision_policy`, `hazard`, `hazard_evolution`, `fire_growth`, `smoke_propagation`, `ai_decision`, `ai_registry`, `ai_inference`, `ai_training`, or `rl_training`; `facp/engine.py`/`facp/models.py` and `emergency_response/engine.py`/`emergency_response/models.py` import none of the execution-system packages either; every `ai_decision`/`ai_registry`/`ai_inference`/`ai_training`/`rl_training`/`rl` source file was scanned and none references `ManualCallPoint` or `SimulatedFACP` by name (AI/RL cannot activate or reset either); no `socket`/`serial`/`can`/`modbus`/`bacnet` import exists anywhere in the MCP chain (no hardware protocol implementation was added).

## 15. Performance (Phase 15)

`scripts/benchmark_manual_call_point_emergency_response.py`, 50 zones / 100 mixed Smoke-Detector-and-MCP alarm sources:

| Measurement | Mean | p95 |
|---|---|---|
| `EmergencyResponseIntelligenceEngine.compute()`, 100 alarm sources | 0.637 ms | 0.784 ms |
| Baseline (no FACP status at all) | 0.457 ms | 0.505 ms |
| Incremental cost of localized FACP/MCP evidence | 0.181 ms/cycle | — |

Negligible at this milestone's own named scale. No optimization was performed.

## 16. Files created / modified

**New:** `perception/models/manual_call_point_observation.py` (`ManualCallPointReading`), `tests/test_manual_call_point_emergency_response_integration.py` (24 tests), `scripts/benchmark_manual_call_point_emergency_response.py`, this document.

**Modified (additively):** `building_state/models.py` (`manual_call_point_states`, `manual_call_point_state()`), `building_state/estimator.py` (`manual_call_point_statuses`/`readings` params, extended `_aggregate_alarm_status`/`_summarize_active_assets`), `live_system/building_state_gateway.py` (two new provider parameters), `live_runtime/factory.py` (`manual_call_point_status_provider`/`manual_call_point_reading_provider` closures), `emergency_response/models.py` (`AlarmSourceEvidence`, `ResponseReason.MANUAL_EMERGENCY_REPORTED`, `ResponseWeights.manual_report_weight`, `ZoneResponsePriority.alarm_sources`/`manual_emergency_reported`), `emergency_response/engine.py` (`_alarm_sources_by_zone()` replacing `_zone_alarm_ids()`, independent automatic/manual scoring, extended `_explain()`), `advisory_system/emergency_response_evidence.py` (`ZoneResponseDetail`/`EmergencyResponseEvidence` additive fields), `live_system/live_advisory_gateway.py` (`emergency_response_evidence_from_snapshot()` populates the new fields), `command_center/live_emergency_response_panel.py` (new "Alarm Sources" column + detail text).

**Unchanged:** `facp/engine.py`, `facp/models.py`, `EngineFACPGateway`, `PathfindingEngine`, `decision_policy/*`, `voice_evacuation/*`, `building_control/*`, `dynamic_signage/*`, every AI/RL package, `models/manual_call_point.py` itself.

## 17. MANUAL REPORT vs. AUTOMATIC DETECTOR ALARM vs. CONFIRMED HAZARD — never synonyms

| Term | What it means | What produces it | What it never means |
|---|---|---|---|
| **MANUAL REPORT** | A human manually reported an emergency at/associated with this device. | `ManualCallPoint.activate()` (an intrinsic device state, a direct human action). | Confirmed fire/smoke/injury/impassability; never inferred as any of those. |
| **AUTOMATIC DETECTOR ALARM** | A Smoke/Heat Detector's own reading crossed its alarm threshold. | `SmokeDetector`/`HeatDetector.compute_state()` against a real (or Ground-Truth) hazard reading. | A human confirmation of anything — it is a sensor's own threshold crossing, nothing more. |
| **CONFIRMED HAZARD** | An actual fire-growth/smoke-propagation physics value crossing a severity threshold (`HazardSeverity`, `hazard_summary.zone_severities`). | `hazard/`, `hazard_evolution/`, `fire_growth/`, `smoke_propagation/` — an entirely separate subsystem. | Neither a Manual Report nor an Automatic Detector Alarm changes this value in any way — no code path connects either to hazard physics. |

These three remain, and must remain, structurally distinct concepts throughout this codebase.

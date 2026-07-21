# Live Crowd Intelligence → Operational Advisory Integration

Status as of this milestone: `advisory_system` can now consume live, deterministic crowd congestion/density evidence as **secondary, supporting** operational evidence — alongside, never instead of, the existing deterministic `decision_policy`/`ground_truth` pipeline and AI evidence. `decision_policy` itself, `BuildingState`, and `crowd_intelligence`'s own engine are all **unmodified** by this milestone.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. **How `AdvisoryReport` is generated in Live mode**: `live_system.live_advisory_gateway.ReplayCompatibleAdvisoryGateway.generate()` builds an `AdvisoryInputs` (fresh `decision_policy_provider(time)` call each cycle, per that class's own docstring — it requires an already-real Building/Scenario/GroundTruth, honestly named "ReplayCompatible" not "Live") and calls `AdvisoryOrchestrator.generate_report()`. Confirmed unchanged in shape by this milestone — only two new things are threaded through it (`crowd_evidence` parameter, `crowd_decision_evidence` field).
2. **Where `AIDecisionEvidence` enters**: `AdvisoryInputs.ai_decision_evidence`, populated once per cycle by `live_system.orchestrator.LiveOrchestrator.run_cycle()` via `ai_decision_evidence_from_prediction_snapshot()`. `CrowdDecisionEvidence` now enters the exact same way, one line below it (`crowd_decision_evidence_from_snapshot()`).
3. **Recommendation fields representing zones/exits/stairs/doors/firefighter/commander**: `CivilianAnnouncement` (zone-addressed), `BuildingRecommendation` (`target_type`/`target_id`, any asset), `FirefighterIntelligenceReport` (`available_routes`/`blocked_routes`/`rescue_priority_areas`), `IncidentCommanderDashboard` (`available_exits`/`blocked_routes`/`predicted_bottlenecks`) — all confirmed by reading `advisory_system/recommendation_models.py` in full.
4. **Existing congestion vocabulary**: `decision_policy.exit_policy.HIGH_CONGESTION` and `decision_policy.stair_policy.CONGESTED` **already exist** — a critical finding. Both are computed exclusively from `GroundTruth.exits_exceeding_capacity`/`worst_exit`/`stairs_exceeding_capacity`/`stair_risk_scores` (Sec 1 items 5/6 below) — i.e. **simulation-only**, post-hoc congestion, never live-observed. Crowd Intelligence's own congestion signal is a genuinely independent, live-observed source about the same underlying concept.
5. **Decision Policy's existing congestion reasoning**: yes — `exit_policy.compute_exit_decisions()`/`stair_policy.compute_stair_decisions()` already reason about congestion, but exclusively via `GroundTruth`'s own completed-simulation fields.
6. **Which of those are simulation-only and must not be reused for Live mode**: all of them — `exits_exceeding_capacity`, `worst_exit`, `stairs_exceeding_capacity`, `stair_risk_scores` require a completed `MultiAgentSimulationResult`. This milestone never reads or writes any of these; Crowd Intelligence supplies its own, separately-provenanced live signal instead.
7. **How crowd intelligence should enter Advisory**: Option **B** — a dedicated evidence object (`CrowdDecisionEvidence`), converted from `CrowdIntelligenceSnapshot` by a dedicated adapter — chosen because it exactly mirrors the already-proven `AIDecisionEvidence` pattern (Option C's "another existing evidence pattern" IS this one, reused rather than reinvented). Passing the whole mutable `CrowdIntelligenceSnapshot`/runtime into Advisory (implicitly Option A) was rejected outright per Phase 2's own explicit instruction.

Also found: `AIDecisionEvidence`'s own module docstring explicitly explains why it carries **no** per-asset localization field — "no live-compatible signal anywhere in this codebase capable of identifying which specific stair/door/exit/zone is congested" existed at the time that milestone shipped, and names a *future* per-door aggregation layer as the eventual fix. **`crowd_intelligence/` is exactly that future layer** — so `CrowdDecisionEvidence`, unlike `AIDecisionEvidence`, is deliberately allowed to name specific exit/stair/door/zone ids.

## 2. Crowd Decision Evidence architecture

```
BuildingState
        |
        v
crowd_intelligence.engine.CrowdIntelligenceEngine.compute()
        |
        v
crowd_intelligence.models.CrowdIntelligenceSnapshot
        |
        v  (live_system.live_advisory_gateway.crowd_decision_evidence_from_snapshot())
        |
advisory_system.crowd_evidence.CrowdDecisionEvidence  (plain values only -- advisory_system
        |                                               imports NOTHING from crowd_intelligence)
        v
                    Advisory  (advisory_system.advisory_engine)
                   /        \
      decision_policy      AIDecisionEvidence
   (read, never mutated)   (ai_decision_evidence_from_prediction_snapshot())
        |
        v
AdvisoryReport (CivilianAnnouncement / FirefighterIntelligenceReport /
                BuildingRecommendation / IncidentCommanderDashboard)
        |
        v
StateManager.crowd_intelligence / StateManager.advisory_report
        |
        v
Command Center (existing panels, additive only -- Sec 14)
        |
        v
Human Operator -> optional operator-approved execution (untouched)
```

`advisory_system/crowd_evidence.py` mirrors `advisory_system/ai_evidence.py` field-for-field in spirit: a small, frozen, **plain-value-only** dataclass (`CrowdDecisionEvidence`), an `UNAVAILABLE_CROWD_DECISION_EVIDENCE` canonical instance, and two small detail records (`CrowdZoneDetail`, `CrowdAssetDetail`) populated **only** for zones/assets already flagged by `CrowdIntelligenceSnapshot.building_summary` (congested, queueing, or trending) — never one entry per zone/asset in the building (Phase 2's own "do not duplicate the entire snapshot unnecessarily"). The reduction itself (`crowd_decision_evidence_from_snapshot()`) lives in `live_system/live_advisory_gateway.py`, the one module already permitted to import both `advisory_system` and `crowd_intelligence`.

`CrowdDecisionEvidence` additionally carries `position_unavailable_asset_ids` — a field `AIDecisionEvidence` has no equivalent of, added specifically because "not congested" and "no position coverage" are two genuinely different things (Sec 10).

## 3. Files created / modified

**Created:**
- `advisory_system/crowd_evidence.py`
- `tests/test_crowd_advisory_evidence.py` — 9 tests (adapter correctness)
- `tests/test_crowd_advisory_safety_precedence.py` — the Phase 16 safety matrix, 15 tests
- `tests/test_crowd_advisory_architecture_guards.py` — 5 tests (Phase 19)
- `tests/test_live_runtime_crowd_advisory_e2e.py` — 4 tests, full offline chain (Phase 17)
- `scripts/benchmark_crowd_advisory.py` — performance benchmark (Phase 18)
- `docs/architecture/crowd_aware_advisory_integration.md` — this document

**Modified:**
- `advisory_system/recommendation_models.py` — `AdvisoryInputs.crowd_decision_evidence`; `IncidentCommanderDashboard.crowd_highest_density_zone_id`/`crowd_most_congested_asset_id`/`crowd_most_congested_level` (all additive, mirroring the existing `ai_bottleneck_*` fields).
- `advisory_system/confidence_engine.py` — `recommendation_confidence()` gains a `crowd_confidence` parameter, blended exactly like `ai_confidence`/`rl_confidence` already are.
- `advisory_system/advisory_engine.py` — the bulk of this milestone's logic (Sec 4-9 below): `_confidence_source()` recognizes `"crowd"`; `_crowd_congestion_confidence_for_wait_zone()`/`_crowd_wait_zone_reason_note()` (WAIT-zone-only, mirrors the existing AI restriction exactly); `build_building_recommendations()` gains three new, additive-only recommendation categories; `build_commander_dashboard()`/`build_firefighter_intelligence()` blend a crowd confidence signal alongside the existing AI one.
- `live_system/live_advisory_gateway.py` — `crowd_decision_evidence_from_snapshot()` (the adapter); `LiveAdvisoryGateway.generate()`/`ReplayCompatibleAdvisoryGateway.generate()` gain a `crowd_evidence` parameter (default `None`, backward compatible).
- `live_system/orchestrator.py` — `run_cycle()` now converts `snapshot.crowd_intelligence` into `CrowdDecisionEvidence` and passes it to `live_advisory_gateway.generate()`, right alongside the existing `ai_evidence` conversion.
- `command_center/recommendation_center.py` — `_confidence_label()`/`_prediction_source_line()` recognize `"crowd"`; a new `_crowd_influence_line()` mirrors `_rl_influence_line()` exactly (Sec 14).
- `tests/test_ai_augmented_advisory.py` — the one existing custom `LiveAdvisoryGateway` implementation (`_FlakyGateway`) updated to accept the new, optional `crowd_evidence` parameter.
- `tests/test_command_center.py` — two pre-existing string assertions updated for the new `"...no AI/RL/crowd signal..."` wording.

**Unchanged (verified, not modified):** `decision_policy/*` (every rule module), `building_state/*`, `crowd_intelligence/*` (the engine itself), `ground_truth/*`, `voice_evacuation/*`, `building_control/*`, `facp/*`.

## 4. Safety precedence implementation (Phase 4 — the most important requirement)

```
HAZARD / HARD SAFETY RULES        (decision_policy: CLOSE / AVOID / EVACUATE_IMMEDIATELY / SHELTER_IN_PLACE)
        >
CROWD CONGESTION OPTIMIZATION      (crowd_intelligence -> CrowdDecisionEvidence: WAIT-zone confidence/reason,
                                    "Monitor" recommendations, "Prefer" recommendations among already-safe assets)
        >
AI SUPPORTING CONFIDENCE           (ai_evidence -> AIDecisionEvidence: WAIT-zone confidence only, building-wide monitor)
```

Mechanically, every crowd-influenced code path reads `decision_policy`'s own status **first** and defers to it unconditionally:

- `_crowd_congestion_confidence_for_wait_zone()`/`_crowd_wait_zone_reason_note()` only ever run when `action == WAIT` (a status `zone_policy` itself, independently, already produced) — never creates WAIT, never touches `SHELTER_IN_PLACE`/`EVACUATE_IMMEDIATELY` at all, exactly mirroring `_ai_bottleneck_confidence_for_wait_zone`'s own restriction.
- `_crowd_prefer_alternative_recommendations()` computes `usable_ids = [id for id, status in status_by_id.items() if status != unsafe_value]` **before** congestion is even considered — an asset `decision_policy` marks `CLOSE`/`AVOID` can never appear as either the "congested" side or the "alternative" side of a preference recommendation. This is a structural guarantee (the unsafe asset is filtered out of the candidate pool entirely), not a special-cased check that could be bypassed.
- Every new crowd-sourced `BuildingRecommendation` is **appended**, never replacing or suppressing an existing recommendation (including the pre-existing, unrelated "Unlock Exit" recommendation for a `CLOSE` exit, and the AI-sourced "Monitor for Building-Wide Congestion" recommendation, which coexists independently).
- The civilian announcement's actual broadcast text (`_format_announcement()`) is **completely untouched** by crowd evidence — crowd intelligence may only enrich `CivilianAnnouncement.reason` (the audit-trail explanation), never the instruction occupants actually receive.

Proven directly by all 15 tests in `tests/test_crowd_advisory_safety_precedence.py` (Sec 6 below) and re-proven end-to-end in `tests/test_live_runtime_crowd_advisory_e2e.py`.

## 5. Exit congestion behavior (Phase 6)

Two safe (non-`CLOSE`) exits, one congested per crowd evidence, one not → a `BuildingRecommendation` (`action="Prefer Exit {clear} over Exit {congested}"`, `target_type="exit"`, `confidence_source=("crowd",)`) is added. One safe exit congested, the *only* alternative `CLOSE` → no preference is generated (there is no genuine safe alternative to prefer). An unsafe (`CLOSE`) exit is never selected as either side of this comparison, regardless of its own or an alternative's congestion (Sec 4).

## 6. Stair congestion behavior (Phase 7)

Identical mechanism, one layer over (`unsafe_value=AVOID`, `status_by_id` from `stair_decisions`). Crowd intelligence may only ever affect confidence/reason/an additive "Prefer"/"Monitor" recommendation — it never changes an `AVOID`/`USE` decision.

## 7. Zone-density behavior (Phase 8)

`build_building_recommendations()` adds one `"Monitor High Crowd Density in Zone {id}"` recommendation per zone in `CrowdDecisionEvidence.zones_above_density_threshold` — purely informational (firefighter/commander awareness), never touching `zone_policy`'s own `EVACUATE_IMMEDIATELY`/`WAIT`/`SHELTER_IN_PLACE` action (Phase 6 test in the safety matrix proves `SHELTER_IN_PLACE` is untouched even with a genuinely clear alternate route available; the symmetric "never modify emergency evacuation priority purely because a zone is crowded" is structural — no code path in this milestone ever writes to `zone_decisions`/`action` at all).

## 8. Temporal trend handling (Phase 9)

`CrowdAssetDetail.trend`/`CrowdZoneDetail.trend` carry the RISING/STABLE/FALLING/UNKNOWN classification straight through from `crowd_intelligence.models.TrendDirection` (as a plain string — `UNKNOWN` is normalized to `None`, since "no trend information" is a cleaner signal than a string literally reading `"UNKNOWN"`). `_crowd_monitor_asset_recommendation()`'s reason text includes the trend when present (`"...with a RISING trend."`), giving HIGH+RISING a visibly different, more detailed reason than HIGH+FALLING would — no numeric score is invented on top of this; the distinction is conveyed entirely through the same disclosed `_CROWD_LEVEL_CONFIDENCE` mapping (Sec 9) and the trend word itself, never an arbitrary combined "urgency score."

## 9. Confidence mapping (a documented, deterministic constant)

```python
_CROWD_LEVEL_CONFIDENCE = {"HIGH": 0.60, "VERY_HIGH": 0.75, "CRITICAL": 0.90}
```

The same "disclosed policy threshold, not a fabricated data value" convention `DETERMINISTIC_RULE_BASE_CONFIDENCE`/`risk_based_confidence` already establish. `LOW`/`MODERATE` never contribute a confidence value (`None`) — they are not operationally noteworthy enough to raise a recommendation's confidence.

## 10. Calibration/coverage handling (Phase 10)

`CrowdDecisionEvidence.position_coverage_fraction` (building-wide) and `position_unavailable_asset_ids` (per-asset) both flow into Advisory. `_crowd_coverage_caveat()` appends an explicit sentence to every crowd-sourced `BuildingRecommendation.reason` whenever coverage is below 100% — e.g. *"Position coverage: 35% -- absence of observed congestion elsewhere does not confirm those routes are clear."* Critically, `_crowd_prefer_alternative_recommendations()`'s own `clear_usable` filter excludes any asset present in `position_unavailable_asset_ids` — an asset with **zero** position coverage is never treated as a confirmed-clear alternative merely because it is absent from the congested list (this was found and fixed as a genuine correctness gap during this milestone's own testing — see `tests/test_crowd_advisory_safety_precedence.py::Test11ZeroPositionCoverageTreatedUnavailable`). Proven at 100%, partial, and 0% coverage (Tests 4, 10, 11).

## 11. AI + crowd evidence coexistence (Phase 11)

`recommendation_confidence()` blends `ai_confidence`/`rl_confidence`/`crowd_confidence` as three **independent, optional** parameters — never merged into a single opaque input before blending. Provenance is kept separate through `confidence_source: Tuple[str, ...]`, which records exactly which of `"ai"`/`"rl"`/`"crowd"` genuinely contributed to *this* recommendation's confidence — never a blanket label. `command_center/recommendation_center.py`'s `_confidence_label()`/`_prediction_source_line()`/`_crowd_influence_line()` read this tuple to produce an explanation that names each contributing source separately, and explicitly never labels crowd analytics as AI (`"live crowd intelligence (deterministic analytics, not AI)"`). Proven for AI+crowd both present, AI-only, crowd-only, and neither (Tests 9, 12–15).

## 12. Explainability (Phase 12)

Every crowd-sourced recommendation's `reason` field is built entirely from structured `CrowdDecisionEvidence` fields, never an unsupported natural-language claim: level, trend, `approaching_count`/`queue_candidate_count` (from `CrowdAssetDetail`), and the coverage caveat (Sec 10) are concatenated deterministically — e.g. *"Live crowd intelligence reports HIGH congestion at Exit EXIT-1 with a RISING trend. 2 occupant(s) approaching, 2 in an estimated queue. Position coverage: 92%."* No text is ever generated from a field that doesn't exist on the evidence object.

## 13. Live runtime integration (Phase 13)

No second Advisory engine was created. `LiveOrchestrator.run_cycle()`'s existing sequence (Perception → BuildingState → Crowd Intelligence → Live AI → **Live Advisory**) is unchanged in order; the only addition is that the Advisory stage now also converts `snapshot.crowd_intelligence` into `CrowdDecisionEvidence` (one line, mirroring the pre-existing AI evidence conversion) and passes it into the same `AdvisoryOrchestrator`/`ReplayCompatibleAdvisoryGateway` that already existed.

## 14. Command Center (Phase 14)

Investigated first: `BuildingRecommendation`/`CivilianAnnouncement` are already rendered **generically** by `command_center/recommendation_center.py` (a table driven by whatever recommendations `AdvisoryReport` happens to contain) — new crowd-sourced `BuildingRecommendation`s (the "Monitor"/"Prefer" entries) require **no new table/column** to appear; they show up automatically. The only addition made: the existing, already-generic confidence-provenance explanation functions (`_confidence_label`/`_prediction_source_line`) now recognize `"crowd"` in `confidence_source`, and a new `_crowd_influence_line()` mirrors the existing `_rl_influence_line()` exactly. The `BuildingRecommendationsPanel`'s own detail-view text block (a pre-existing, hardcoded "this category is never AI/RL-augmented" string, unrelated to this milestone) was left untouched — it predates crowd intelligence and fixing its own latent inaccuracy for AI-sourced entries is out of this milestone's scope. No automatic execution UI was added anywhere.

## 15. Automatic-execution guards (Phase 15)

Mechanically verified by `tests/test_crowd_advisory_architecture_guards.py`: every crowd-advisory file (`advisory_system/crowd_evidence.py`, `advisory_engine.py`, `recommendation_models.py`, `confidence_engine.py`) never imports `voice_evacuation`, `speaker_manager`, `building_control`, `reinforcement_learning`/`rl_training`, YOLO/RTSP/hardware modules, and never calls an execution verb (`.broadcast(`/`.execute_control(`/`.acknowledge(`/etc.). `decision_policy/*` never imports `crowd_intelligence` (the one-way dependency direction, Sec 4/16). End-to-end, `tests/test_live_runtime_crowd_advisory_e2e.py` constructs its `LiveRuntime` with `voice_output_provider`/`building_control_provider` both left `None` and asserts `runtime.voice_evacuation_controller`/`runtime.building_control_controller` are both `None` — there is nothing wired for even a hypothetical crowd-triggered action to reach.

## 16. Offline end-to-end results (Phase 17)

`tests/test_live_runtime_crowd_advisory_e2e.py` drives the complete production chain (`ReplayFrameSource` → `YOLOHumanDetector` w/ fake backend → `SimpleSingleCameraTracker` → `WorldProjector` → `RuleBasedBehaviorRecognizer` → `LiveOccupantManager` → `LivePerceptionFusionCoordinator` → `BuildingState` → `CrowdIntelligenceEngine` → (Live AI left unconfigured) → `ReplayCompatibleAdvisoryGateway`) across 5 cycles, with two safe exits (`EXIT-1`, `EXIT-2`). Proven directly:
1. Crowd Intelligence detects congestion at `EXIT-1` once two occupants converge and stop.
2. `AdvisoryReport.building_recommendations` contains crowd-sourced evidence (`confidence_source == ("crowd",)`).
3. Advisory names the specific congested exit (`"Monitor Congestion at Exit EXIT-1"`).
4. The alternate safe, uncongested `EXIT-2` receives a supporting `"Prefer Exit EXIT-2 over Exit EXIT-1"` recommendation.
5. The moment `EXIT-2` is marked `CLOSE` by `decision_policy` (mid-run), the preference **disappears immediately** in the very next cycle, regardless of `EXIT-1`'s own still-peak congestion.
6. `runtime.voice_evacuation_controller is None` — no automatic voice execution wired or possible.
7. `runtime.building_control_controller is None` — no automatic building-control execution wired or possible.
8. Once both occupants leave, the queue clears and the preference/monitor recommendations for `EXIT-1` disappear on the very next cycle.

Zero network, zero physical CCTV anywhere in this file.

## 17. Performance (Phase 18)

`scripts/benchmark_crowd_advisory.py`, at a heavily-congested worst case (50 zones, 20 doors, 10 exits, 10 stairs, 100 occupants), zero real YOLO/tracker/RTSP inference included:
- `CrowdDecisionEvidence` creation (the adapter): ~0.03 ms/call (mean).
- Complete `AdvisoryReport` generation, no AI/no crowd: ~0.43 ms/call.
- Complete `AdvisoryReport` generation, AI only: ~0.42 ms/call.
- Complete `AdvisoryReport` generation, crowd only: ~0.58 ms/call.
- Complete `AdvisoryReport` generation, AI + crowd: ~0.57 ms/call.

Crowd processing adds roughly 0.15 ms per cycle at this scale — negligible against a 1-second live cycle budget. Real per-camera/tracker inference timing is reported separately in `scripts/benchmark_yolo_human_detector.py`/`benchmark_live_perception.py`/`benchmark_crowd_intelligence.py` — never conflated with the numbers above.

## 18. Answers to this milestone's own closing questions

**A. Can live crowd congestion now influence Advisory recommendations?** Yes — via WAIT-zone confidence/reason enrichment and new, additive `BuildingRecommendation`s (Monitor/Prefer), proven end-to-end.

**B. Can crowd congestion ever make a deterministically unsafe exit/stair usable?** No — structurally impossible: `_crowd_prefer_alternative_recommendations()` filters candidates by `decision_policy` status *before* considering congestion at all; an unsafe (`CLOSE`/`AVOID`) asset can never appear on either side. Proven by Tests 1–3 of the safety matrix and the E2E test's own item 5.

**C. Can crowd intelligence change `EVACUATE_IMMEDIATELY` or `SHELTER_IN_PLACE` decisions?** No — no code path in this milestone ever writes to `zone_decisions`/`action`; crowd evidence only touches `WAIT`-zone confidence/reason (Tests 6, 7).

**D. Can AI evidence and Crowd evidence coexist without being merged into an opaque confidence score?** Yes — both are independent parameters to `recommendation_confidence()`, and `confidence_source` records each contributing source separately, never conflating "crowd" with "ai" (Test 12; Sec 11).

**E. Does the system remain honest when position/calibration coverage is incomplete?** Yes — coverage caveats are appended automatically below 100%, and an asset with zero position coverage is never treated as a confirmed-clear alternative (Tests 10, 11; Sec 10).

**F. Can any crowd-generated recommendation automatically execute voice or building controls?** No — `advisory_system` contains no execution verb of any kind (mechanically verified), and the E2E test confirms `voice_evacuation_controller`/`building_control_controller` are both `None` throughout.

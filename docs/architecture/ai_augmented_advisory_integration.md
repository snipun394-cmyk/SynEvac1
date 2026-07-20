# AI-Augmented Decision Policy & Advisory Integration

Status: **The validated Bottleneck Occurrence AI prediction now flows into the Advisory System as a supporting predictive signal. It does not flow into Decision Policy at all — that boundary is honest, investigated, and mechanically enforced, not a placeholder. AI remains advisory throughout: it cannot execute controls, broadcast messages, command firefighters, or override deterministic life-safety rules.**

## 1. Investigation: why AI does not flow into Decision Policy

`decision_policy.generate_policy()`'s six rule modules (`zone_policy`, `exit_policy`, `stair_policy`, `rescue_policy`, `announcement_policy`, `human_priority_policy`) read **exclusively** from `GroundTruth`'s own post-hoc, completed-simulation analytics — `zone_risk_scores`, `stair_risk_scores`, `zone_route_stats`, `hazard_spread_order`, `exits_exceeding_capacity`, `stairs_exceeding_capacity`, `worst_exit` — and from `Scenario`'s own designed occupant/exit-state data. Every one of these was confirmed, field by field, to have **no `building_state.models.BuildingState` equivalent**: `BuildingState` is a live, per-tick estimate of current sensor-observable conditions; `GroundTruth` is computed only after a full simulation run completes. `decision_policy` cannot run from `BuildingState` alone, and this milestone does not fabricate a `GroundTruth`-shaped stand-in to force it to.

Separately, `decision_policy.policy.DecisionPolicy` itself carries no `confidence`/`explanation` field of any kind — there is structurally nowhere on it for AI evidence to attach even if it could run live.

`advisory_system` already had a real, working, purpose-built extension point for exactly this (`AdvisoryInputs.ai_predictions`/`.ai_confidence`, `advisory_engine._ai_signal_for()`/`_confidence_source()`, `confidence_engine.recommendation_confidence()`), and the established `ai_inference.recommendation` precedent already proves the required discipline: AI only ever *annotates* an already-decided `DecisionPolicy` output, never mutates or replaces one.

**Net effect:** AI Decision Evidence flows into `advisory_system`, not `decision_policy`. `decision_policy` is not imported by any file this milestone adds, and is not modified anywhere in this milestone — the strongest possible safety guarantee available: AI cannot influence `AVOID`/`CLOSE`/`SHELTER_IN_PLACE` decisions because it never reaches the code that makes them, not merely because a rule says it must not (§5).

## 2. Bottleneck localization boundary

The production-candidate `BottleneckOccurrenceModel_LiveCompatible` predicts **occurrence only** — a single building-wide probability and boolean, never which stair/door/exit/zone, never when. Phase 2's own investigation confirmed **no live-compatible signal anywhere in this codebase** identifies which specific stair/door/exit/zone is congested:

- `ground_truth/bottleneck.py`'s per-asset findings require a completed simulation (`GROUND_TRUTH_ONLY`).
- `simulation_interactive`'s per-edge queue/occupancy state is the interactive simulator's own internal event-heap bookkeeping (`SIMULATION_ONLY`), never perception-derived.
- `building_state`/`multi_camera_fusion` have no door/edge-indexed occupancy concept at all: the Navigation Graph's `Node` type is Zone-only; Door/Exit/Stair are Edge-only, and `BuildingState.zone_occupancy` is Node-keyed — structurally incapable of carrying door/edge occupancy.

Consequently, **`AIDecisionEvidence` (§3) has no localization field** — no `suspected_bottleneck_asset_ids`, no `likely_zone_id`. This milestone never fabricates "Avoid North Stair because AI predicts a bottleneck there." A future milestone that builds a real per-door congestion aggregation layer (e.g. from `FusedTrack.history.zone_transitions`, the closest already-live-estimable building block) could add such a field, sourced from *that* subsystem — never synthesized from this building-wide occurrence model.

## 3. `AIDecisionEvidence`

`advisory_system.ai_evidence.AIDecisionEvidence` — deliberately placed in `advisory_system`, not `decision_policy`, as a direct consequence of §1/§2. Immutable, frozen dataclass:

- `available: bool` — `False` (every other field at its `None` default) is the honest "no AI evidence this cycle" value. `UNAVAILABLE_AI_DECISION_EVIDENCE` is the one canonical instance — never a fabricated all-zero/all-optimistic placeholder.
- `bottleneck_occurrence_probability`, `bottleneck_predicted`, `threshold` — the model's own raw output, unmodified.
- `model_id`, `model_version`, `model_status` (always `"PRODUCTION_CANDIDATE"` for this milestone's one authorized model — carried explicitly so a future, still-`EXPERIMENTAL` signal could never silently be treated as equally authoritative).
- `prediction_timestamp`, `building_state_timestamp`, `feature_schema_version` — provenance, for explainability (§9).

`evidence_from_bottleneck_prediction(...)` is the one constructor `live_system.live_advisory_gateway` calls; it is kept free of any import of `live_system`/`ai_registry`/`ai_inference` types (plain floats/strings/bools only), so `advisory_system` itself gains no new package dependency.

## 4. Where AI evidence attaches inside Advisory System

Four, and only four, places in `advisory_system.advisory_engine` read `AdvisoryInputs.ai_decision_evidence`:

1. **`build_civilian_announcements()`** — `_ai_bottleneck_confidence_for_wait_zone()` blends the AI probability into a zone's `recommendation_confidence()` **only when `decision_policy` itself already, independently, marked that zone `WAIT`** (i.e. its recommended exit is already in `ground_truth.exits_exceeding_capacity`, a deterministic, `GroundTruth`-derived congestion signal computed with zero knowledge of this AI model). AI never creates a `WAIT` status, never changes which exit/stair is recommended, and never touches `SHELTER_IN_PLACE`/`EVACUATE_IMMEDIATELY` confidence at all.
2. **`build_firefighter_intelligence()`** — the raw probability is blended into `confidence`, and separately exposed verbatim as `ai_bottleneck_probability`/`ai_bottleneck_model_id` (information, never a task).
3. **`build_building_recommendations()`** — may append exactly one new, additive `BuildingRecommendation`: `"Monitor for Building-Wide Congestion"` (§6).
4. **`build_commander_dashboard()`** — `ai_bottleneck_probability`/`ai_bottleneck_model_id` exposed verbatim, and folded into `recommendation_confidence` alongside every other confidence source already blended there.

No other function in `advisory_engine.py` reads `ai_decision_evidence`.

## 5. Safety precedence (mechanically guaranteed, not just tested)

Because AI attaches only at the points in §4 — all of which run *after* `decision_policy`'s zone/exit/stair decisions are already fixed, and none of which can change an `action`, a `recommended_exit`, a `recommended_stair`, an exit `status`, or a stair `status` — AI cannot override a deterministic safety decision by construction. `advisory_engine._resolve_zone_action()` (unchanged by this milestone) still degrades an `EVACUATE_IMMEDIATELY` zone whose only exit is `CLOSE` to `SHELTER_IN_PLACE`, and still drops a `recommended_stair` marked `AVOID` from the announcement, regardless of any AI probability.

`tests/test_ai_augmented_advisory.py::SafetyPrecedenceTests` proves this directly: a `CLOSE`d exit stays `SHELTER_IN_PLACE` even at AI probability 0.99; an `AVOID`ed stair is never re-announced as usable even at AI probability 0.97; firefighter `blocked_routes` still lists an `AVOID`ed stair regardless of AI.

## 6. Building recommendation: "Monitor for Building-Wide Congestion"

The **one** new recommendation AI evidence alone can add, appended only when `ai_decision_evidence.bottleneck_predicted` is `True`:

- Deliberately a **monitor** action, never a control action — no door/deluge/smoke-exhaust/stair-pressurization/voice-broadcast/exit-closure recommendation is ever generated from this signal.
- Deliberately **building-wide** (`target_type="building"`, `target_id=None`) — never a fabricated zone/stair/exit target (§2).
- Deliberately **additive** — appended only, never replacing or suppressing any deterministic recommendation (`tests/test_ai_augmented_advisory.py::BuildingRecommendationsAIIntegrationTests::test_deterministic_recommendations_unaffected_by_ai_presence` proves the deterministic recommendation list is byte-for-byte identical with and without AI evidence).
- Carries `confidence_source=("ai",)` — the one `BuildingRecommendation` in a report that is genuinely `AI_SUPPORTED` rather than `RULE_BASED` (§9).

## 7. Zone-based civilian recommendations (unchanged guarantee)

Every `CivilianAnnouncement` is addressed `"Attention occupants in <zone>..."` — no field anywhere on it references an individual `occupant_id`, with or without AI evidence present (`tests/test_ai_augmented_advisory.py::CivilianAdvisoryAIIntegrationTests::test_announcements_remain_zone_addressed_never_individual_with_ai_present`). AI evidence never adds a per-occupant instruction; it can only strengthen the confidence behind a zone-level `WAIT` message `decision_policy` already decided to issue (§4.1).

## 8. Firefighter intelligence rules (unchanged guarantee)

`FirefighterIntelligenceReport` has no `assigned_task`/`mission`/`firefighter_id` field, with or without AI evidence — `ai_bottleneck_probability`/`ai_bottleneck_model_id` are information fields sitting alongside `hazard_severity_by_zone`, `blocked_routes`, `rescue_priority_areas`, never a directive. SynEvac does not command firefighters; it never has, and this milestone adds no code path that could.

## 9. Explainability: RULE_BASED vs. AI_SUPPORTED

`confidence_source: Tuple[str, ...]` (already present on `CivilianAnnouncement`/`BuildingRecommendation`; extended this milestone onto `FirefighterIntelligenceReport`) is the one place downstream consumers (e.g. `command_center.recommendation_center`) may say "AI confidence," and only when `"ai"` actually appears in it:

- **Empty tuple** — `RULE_BASED`: confidence is entirely `DETERMINISTIC_RULE_BASE_CONFIDENCE` (+ deterministic risk-score/agreement terms, neither of which is AI).
- **`("ai",)`** (or `("ai", "rl")`) — `AI_SUPPORTED`: a genuine, non-`None` AI signal contributed to `confidence`, on top of a recommendation `decision_policy` already, independently produced. **Never `AI_GENERATED`** — no recommendation in this codebase is created by AI; AI only ever confirms or strengthens confidence in a location/action a deterministic rule already chose (`tests/test_ai_augmented_advisory.py::ExplainabilityConfidenceSourceTests`).

`AIDecisionEvidence.model_id`/`model_version`/`prediction_timestamp`/`building_state_timestamp`/`feature_schema_version` (§3) additionally let a consumer show provenance ("bottleneck-1 v1, predicted at t=10.0 from BuildingState as of t=9.5") for any `AI_SUPPORTED` recommendation.

## 10. Confidence separation

Three genuinely separate quantities, computed by three different code paths, never conflated:

| Quantity | Where computed | Meaning |
|---|---|---|
| AI model probability (`ai_decision_evidence.bottleneck_occurrence_probability`) | `ai_registry`'s trained classifier | How likely the model thinks a building-wide bottleneck is |
| Occupancy confidence (`occupancy_confidence()`) | `advisory_system.confidence_engine`, from `human_observations` | How confident Live Perception is in its own occupant detections |
| Recommendation confidence (`recommendation_confidence()`) | `advisory_system.confidence_engine`, `combine_confidence()` (unweighted mean of non-`None` sources) | How confident the Advisory System is in *this specific recommendation*, of which AI probability may be one blended input |

`IncidentCommanderDashboard` carries all three as **separate fields** (`ai_bottleneck_probability`, `occupancy_confidence`, `recommendation_confidence`) — `tests/test_ai_augmented_advisory.py::ConfidenceSeparationTests` asserts they are not silently the same float.

## 11. Live runtime graph

**Before** (after commit implementing Live AI Inference Runtime Integration):

```
BuildingState -> [live_ai_gateway] -> LiveAIPredictionSnapshot -> StateManager.ai_prediction_snapshot
                                                                          |
                                                          (nothing downstream reads it)
```

**After** (this milestone):

```
BuildingState -> [live_ai_gateway] -> LiveAIPredictionSnapshot -> StateManager.ai_prediction_snapshot
                                                                          |
                                                    ai_decision_evidence_from_prediction_snapshot()
                                                                          |
                                                                   AIDecisionEvidence
                                                                          |
                                              [live_advisory_gateway] <- ALSO requires an already-real
                                                     |                    Building/Scenario/GroundTruth,
                                                     v                    supplied by the caller (§12)
                                              AdvisoryReport -> StateManager.advisory_report
                                                     |
                                          (Phase 13: nothing downstream reads it yet)
```

`LiveOrchestrator.run_cycle()`'s stage order: read sensors → update snapshot → assemble canonical `BuildingState` → live AI inference (`BuildingState -> LiveAIPredictionSnapshot`) → **[new]** live advisory generation (`LiveAIPredictionSnapshot -> AIDecisionEvidence -> AdvisoryReport`, via `live_advisory_gateway`) → the old, still-unimplemented-in-production `ai_inference_gateway`/`decision_policy_gateway`/`recommendation_builder` stages (unchanged) → notify command center. Each stage after sensor reading only runs if its gateway is configured; every stage's absence remains a valid, working configuration, never an error.

`live_ai_gateway` is called independently of `live_advisory_gateway`'s own success — it is always invoked when configured (passing `None` if no `BuildingState` exists yet), so a deployment with live AI wired up but no `BuildingState` source yet still gets an honest `UNAVAILABLE` `LiveAIPredictionSnapshot` every cycle.

## 12. The honest limit: `live_advisory_gateway` does not run from `BuildingState` alone

Per §1, `decision_policy.generate_policy()` — and therefore `AdvisoryInputs` — requires a real `Building`/`Scenario`/`GroundTruth`, none of which has a live equivalent. `live_system.live_advisory_gateway.ReplayCompatibleAdvisoryGateway` is named honestly, not `LiveAdvisoryGateway`'s only implementation pretending otherwise: it requires a caller to supply an already-valid `Building`/`Scenario`/`GroundTruth` (exactly what a completed campaign, or a Replay session reading one, already has), and a `decision_policy_provider(time) -> Optional[DecisionPolicy]` callable invoked fresh every cycle — mirroring `command_center.incident_data.py`'s own established `_build_advisory_reports()` pattern of recomputing `zone_policy`/`exit_policy`/`stair_policy`/`announcement_policy` fresh per frame from a fixed `Building`/`Scenario`/`GroundTruth`.

What **is** genuinely live every cycle is the `AIDecisionEvidence` layered on top — built fresh from a real live/replay `BuildingState` via `live_ai_gateway`, never fabricated. This milestone does not synthesize a `GroundTruth`-shaped stand-in from `BuildingState` to force full liveness; that would fabricate exactly the analytics §1 confirmed have no live source.

## 13. Failure/degradation

`ReplayCompatibleAdvisoryGateway.generate()` wraps its entire body in `try/except Exception: return None` — Advisory failure must never crash the live cycle (mirrors `RegistryLiveAIInferenceGateway`'s own discipline). `None` is `LiveAdvisoryGateway`'s documented "no update this cycle" signal, covering both "not enough information yet" (`decision_policy_provider` itself returned `None`) and any caught internal exception uniformly — `LiveOrchestrator` never distinguishes the two; it only ever leaves the previous `AdvisoryReport` in `StateManager` untouched under its own honest, non-bumped `component_timestamps["advisory_report"]` entry, the identical staleness-detection mechanism `live_ai_gateway` already established for `ai_prediction_snapshot`.

| Scenario | `advisory_report` this cycle | Notes |
|---|---|---|
| No `live_advisory_gateway` configured | stays `None` forever | valid, working configuration |
| `decision_policy_provider(time)` returns `None` | previous report kept, timestamp not bumped | "not enough information yet" |
| `decision_policy_provider` raises | previous report kept, timestamp not bumped | caught by `generate()`'s own `try/except` |
| AI evidence unavailable (`UNAVAILABLE_AI_DECISION_EVIDENCE`) | a full report is still produced | `ai_decision_evidence=None`/unavailable degrades every AI-touched field to `None` honestly (§4); the deterministic recommendations are entirely unaffected (§6) |
| AI evidence available, `bottleneck_predicted=False` | full report, no `"Monitor..."` recommendation | §6 |

## 14. Explicit no-output-execution boundary (Phase 13)

This milestone stops at `AdvisoryReport` generation. Mechanically enforced by dependency-direction guards, not only by convention:

- `tests/test_live_system.py::LiveSystemPackageDependencyDirectionTests` (pre-existing, package-wide glob over `live_system/*.py`) already forbids `voice_evacuation`/`speaker_manager`/`building_control.controller`/`building_control.providers` imports anywhere in `live_system`, including the new `live_advisory_gateway.py` — verified passing with this milestone's own file added.
- `tests/test_ai_augmented_advisory.py::NoOutputExecutionBoundaryTests` adds an explicit, milestone-scoped guard naming `live_advisory_gateway.py` and every file in `advisory_system/` directly, plus a structural check that `AdvisoryReport`/`BuildingRecommendation`/`CivilianAnnouncement` expose no `execute`/`send`/`broadcast`/`apply`/`activate`/`trigger` method.
- `LiveOrchestrator.run_cycle()` calls neither a `VoiceEvacuationController`, a `BuildingControlController`, nor any real speaker/FACP/hardware integration — `AdvisoryReport` is stored in `StateManager` and an `ADVISORY_REPORT_UPDATED` event is emitted; nothing subscribes to that event to *act* on it in this milestone.

`AdvisoryReport -> VoiceEvacuationController`, `AdvisoryReport -> BuildingControlController`, real speakers, real FACP, hardware, and any automatic action remain **explicitly out of scope**, deferred to a later milestone.

## 15. Remaining limitations

- **No door/edge-level congestion localization exists anywhere live** (§2) — a genuine platform gap, not merely undone integration work. Any future "which stair is congested" feature requires a new live-estimable subsystem, not a reinterpretation of this model's output.
- **`live_advisory_gateway` cannot run on `BuildingState` alone** (§12) — a real deployment needs a `Building`/`Scenario`/`GroundTruth` source (a Replay session or a completed campaign) wired up alongside it; a pure live-camera-only deployment with no such source configured will never produce an `AdvisoryReport`, only an honest `None`.
- **`AdvisoryReport` is not yet visible anywhere live** — no Command Center panel, no Designer debug view, reads `StateManager.advisory_report`/`latest_advisory_report()` yet. Deferred, per this milestone's own explicit scope.
- **No automatic execution of any kind** (§14) — by design, not by omission; a later, separately-scoped milestone is required before any `AdvisoryReport` output can affect the physical or simulated building.

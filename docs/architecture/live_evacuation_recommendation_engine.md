# Live Dynamic Evacuation Recommendation Engine

Status as of this milestone: SynEvac can, for the first time, answer "if I am responsible for Zone Z3 right now, which SAFE exit should I recommend, and why?" — for every currently occupied zone, live, every cycle — via a new `evacuation_recommendation/` package that combines every completed live-intelligence subsystem (Crowd, Evacuation Progress, Trajectory, Emergency Response, AI) into one explainable, ranked recommendation. It sits AFTER deterministic safety evaluation and never replaces or modifies Decision Policy.

## 1. Investigation findings (Phase 1)

Verified directly against the current source before writing anything:

1. `decision_policy/exit_policy.py::compute_exit_decisions()` is confirmed **entirely simulation-only** — reads `scenario.exit_states`, `ground_truth.hazard_spread_order`/`maximum_hazard_zone`/`exits_exceeding_capacity`/`worst_exit`/`people_trapped`. `decision_policy/` is never imported by this milestone (mechanically enforced — see §7).
2. No new live "safe exit" concept exists beyond what the trajectory_intelligence milestone already found: zone-level hazard severity (`BuildingState.hazard_summary.zone_severities`) and structural edge traversability (`Edge.traversable`, reading `Door.active`/`locked`/`Exit.is_blocked`). `HazardSnapshot`'s own edge-level `HazardEdgeState.traversable` remains consumed only inside `BuildingStateEstimator._summarize_hazard()`, never retained on `BuildingState` itself.
3. `trajectory_intelligence.route_progress.SafeRouteCalculator` is mechanically reusable in isolation, but this codebase's own established convention — confirmed by every sibling live-intelligence package's own architecture guard (none of `crowd_intelligence`/`evacuation_progress`/`emergency_response`/`trajectory_intelligence` ever imports another's engine/calculator class, only `navigation`/`pathfinding` as the shared, decision_policy-free foundation) — is that each package owns its own copy of this kind of graph logic. `evacuation_recommendation/` follows the same discipline (§2).
4. `SafeRouteCalculator` also isn't the right shape here regardless: it finds the single nearest safe exit per zone via one Dijkstra seeded at the shared `Outside` node, which structurally cannot produce *ranked alternatives* (every exit collapses into one frontier). `PathfindingEngine.alternative_paths()` (Yen's algorithm) diversifies *routes to one goal*, not *exit choice*, so it's also not the right primitive. `evacuation_recommendation.ranking.SafeExitDistanceCalculator` instead runs one Dijkstra **per safe exit**, seeded at that exit's own zone with `Outside` itself excluded (so an in-building search can never "cheat" by walking out one exit and back in another) — see §3.
5. Nothing in `advisory_system/recommendation_models.py` previously recommended "which SPECIFIC exit a specific zone's occupants should use" as a structured field — `decision_policy.exit_policy`'s own per-zone `recommended_exit` was only ever surfaced as prose inside `CivilianAnnouncement.announcement`. This milestone is the first to expose it structurally (`advisory_system/evacuation_recommendation_evidence.py`).
6. AI evidence enters exclusively as a single, **building-wide** bottleneck probability (`live_system.live_ai_gateway.LiveAIPredictionSnapshot.bottleneck.probability`) — never per-zone/per-exit (confirmed, unchanged, by `advisory_system/ai_evidence.py`'s own documented boundary). Crowd evidence enters per-exit via `CrowdIntelligenceSnapshot.exit_metrics`/`.exit(exit_id)`. Emergency-response evidence enters per-zone via `EmergencyResponseSnapshot.zone(zone_id)`.
7. Pipeline placement: the milestone's own brief orders inputs "...Emergency Response → AI Evidence → Recommendation Engine → Advisory." Since `live_ai_gateway.predict()` runs after `emergency_response_gateway` in the existing orchestrator, the Recommendation stage is inserted immediately after the `live_ai_gateway` block and before `live_advisory_gateway` — see §4.

## 2. Architecture guard: no sibling-engine imports

`evacuation_recommendation/` imports only `navigation`, `pathfinding`, `hazard.severity`, and itself — mechanically enforced by `tests/test_evacuation_recommendation_architecture_guards.py` (mirrors `tests/test_trajectory_intelligence_architecture_guards.py` exactly, additionally forbidding imports of `crowd_intelligence.engine`/`evacuation_progress.engine`/`trajectory_intelligence.engine`/`trajectory_intelligence.route_progress`/`emergency_response.engine`). Every sibling snapshot (`crowd_snapshot`, `evacuation_progress_snapshot`, `trajectory_snapshot`, `emergency_response_snapshot`, `ai_prediction_snapshot`) arrives as a plain, duck-typed `compute()` parameter each cycle — the same pattern `emergency_response.engine.EmergencyResponseIntelligenceEngine` already established for its own sibling inputs. `decision_policy/` is never imported (Phase 2's own explicit "do NOT modify Decision Policy" requirement, satisfied by never depending on it at all).

## 3. Safe exit candidates and ranking (Phase 3/4)

`evacuation_recommendation.ranking.SafeExitDistanceCalculator`:

1. Candidate exits = every `Edge.EXIT` edge that is both structurally `traversable` AND whose own zone is not hazard-excluded (same hard node-exclusion floor as `trajectory_intelligence`, an independently-owned copy — `RecommendationConfig.hazard_unsafe_severity_floor`, default `HIGH`). An unsafe exit is **never** scored — it is not a candidate at all.
2. For each candidate exit, one `PathfindingEngine.distances_from(exit_zone, excluded_node_ids=hazardous_zones ∪ {Outside})` Dijkstra run gives every zone's in-building distance to that exit's own zone; the exit's own edge cost is added for the final outdoor step. N safe exits → N Dijkstra runs per cycle, cached by a `(excluded_zones, candidate_exit_ids, non_traversable_edges)` fingerprint — only re-run when deterministic safety state actually changes (mirrors `ai_decision.engine.AIDecisionEngine`'s and `trajectory_intelligence.route_progress.SafeRouteCalculator`'s own established per-cycle caching discipline).
3. If no candidate exits exist at all, or a specific zone cannot reach any of them, that zone's status is honestly `NO_SAFE_EXIT_AVAILABLE` — never a fabricated alternative (Phase 3).

`evacuation_recommendation.scoring.score_candidate()` combines, per candidate, per zone: route distance (normalized *relative to that zone's own other safe candidates*), crowd congestion level, exit queue size, evacuation-progress throughput (`recent_flow_per_minute`), trajectory support (majority `route_progress_status` among occupants in that zone whose own `nearest_safe_exit_id` is this exit), and an emergency-response zone-elevation penalty. Every weight (`evacuation_recommendation.models.RecommendationWeights`) is documented and configurable — no ML.

**AI evidence is deliberately applied as an *identical* contribution to every candidate for a given zone this cycle** (`ai_bottleneck_probability` is a single building-wide value): by construction, this can never change which exit ranks higher than another — it only ever shifts the absolute score/explanation, never the relative order (Phase 7/13's own "AI only supports" requirement — mechanically proven by `tests/test_evacuation_recommendation.py::RankingTests::test_10_ai_only_supports_never_changes_relative_order`).

## 4. Pipeline placement

```
BuildingState → Crowd Intelligence → Evacuation Progress → Trajectory Intelligence
→ Emergency Response → Live AI Inference → Recommendation Engine → Advisory → Command Center
```

Inserted in `live_system/orchestrator.py::LiveOrchestrator.run_cycle()` immediately after the `live_ai_gateway` block and before `live_advisory_gateway`, reading `snapshot.crowd_intelligence`/`evacuation_progress`/`trajectory_intelligence`/`emergency_response`/`ai_prediction_snapshot` — this cycle's fresh values if their own stage succeeded, or the previous cycle's otherwise, the same "read whatever is on the state manager's current snapshot" convention every prior stage already uses.

## 5. Confidence (Phase 7)

`evacuation_recommendation.scoring.compute_confidence()` reflects **evidence quality**, never AI certainty: an average of (a) the zone's own crowd-intelligence `position_coverage_fraction` (camera coverage), (b) whether a real graph route distance was derivable (vs. a cost-only fallback), (c) whether any genuine hazard evidence exists at all this cycle. Each missing component is scored neutrally (0.5), never fabricated as certain; `None` only when there is genuinely zero basis (e.g. no crowd evidence, no route, no hazard reading at all — this shouldn't normally happen for a `RECOMMENDED` zone, since a route was, by definition, found).

## 6. Dynamic updates and events (Phase 9/10)

The full `EvacuationRecommendationSnapshot` is recomputed every cycle (cheap, due to the fingerprint cache in §3) — mirroring every sibling live-intelligence package's own "full recompute, transition-gated events" convention exactly. Events fire only on genuine transitions:

- `RECOMMENDATION_CHANGED` — a zone's own top-choice `recommended_exit_id` changes while it remains `RECOMMENDED` both before and after.
- `NO_SAFE_EXIT` — a zone transitions from `RECOMMENDED` to `NO_SAFE_EXIT_AVAILABLE`.
- `RECOVERY_OF_SAFE_EXIT` — a zone transitions back from `NO_SAFE_EXIT_AVAILABLE` to `RECOMMENDED`.
- `SAFE_EXIT_CHANGED` — building-wide: the currently-safe exit *set* itself changes (an exit becomes hazard-excluded or recovers), distinct from any one zone's own recommendation.
- `EVACUATION_RECOMMENDATION_UPDATED` — the general-purpose "a fresh snapshot exists" signal, every cycle, mirroring `TRAJECTORY_INTELLIGENCE_UPDATED`/`RESPONSE_PRIORITY_UPDATED`.

## 7. Safety precedence

```
Decision Policy  >  Recommendation Ranking  >  AI support
```

Decision Policy is never imported or modified by this package (§2). Recommendation Ranking can never surface an exit that deterministic safety (hazard + structural traversability) excludes — mechanically proven end-to-end by `tests/test_live_runtime_evacuation_recommendation_e2e.py`: EXIT-1 is excluded the moment its own zone becomes hazardous and never reappears in `ranked_exit_ids` for the remainder of the scenario, even as the recommendation subsequently migrates twice more (to EXIT-3, then to EXIT-2) in response to congestion. AI evidence never drives ranking at all (§3) — it is the weakest, uniformly-applied signal, strictly subordinate to both of the above.

## 8. No execution authority

Mechanically enforced by `tests/test_evacuation_recommendation_architecture_guards.py`: no imports of `advisory_system`, `command_center`, `voice_evacuation`, `speaker_manager`, `building_control`, `decision_policy`, `ai_*`, `rl_training`, RTSP/YOLO/torch modules, or `ground_truth`/`simulator`; no execution verbs (`.broadcast(`, `.dispatch(`, `.execute_control(`, `.confirm(`, ...) anywhere in the package. `tests/test_live_runtime_evacuation_recommendation_e2e.py::test_no_automatic_execution_or_dispatch` proves `voice_evacuation_controller`/`building_control_controller`/`facp` all stay `None` through a full live cycle that produces a genuine recommendation.

## 9. What this milestone deliberately does not do

No new AI models, no AI retraining, no RL, no Decision Policy redesign, no automatic execution (voice/building-control/dispatch), no hardware integration, no physical CCTV, no pose estimation, no deep-learning route prediction. `CivilianAnnouncement`/`BuildingRecommendation`/`decision_policy.exit_policy` are all left untouched — this recommendation is additive, commander-awareness-only evidence (`IncidentCommanderDashboard.zones_with_evacuation_recommendation_ids`/`zones_without_safe_exit_ids`), never a replacement for the existing GroundTruth/DecisionPolicy-sourced civilian announcement pipeline.

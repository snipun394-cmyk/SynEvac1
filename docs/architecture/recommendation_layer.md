# The Recommendation Layer

## 1. Overview & Terminology

The Recommendation Layer (`recommendation_layer/`) is SynEvac's single, canonical public interface for evacuation recommendations. It sits downstream of every existing recommendation-shaped subsystem, reads their already-computed output, and re-expresses it as one unified vocabulary of six recommendation categories:

| Category | Meaning |
|---|---|
| `OCCUPANT_ROUTING` | Which exit a zone's occupants should use |
| `HAZARD_AVOIDANCE` | A zone or route that should be avoided due to hazard/loss of safe exit |
| `CONGESTION_MITIGATION` | A zone/exit experiencing congestion that would benefit from mitigation |
| `EXIT_UTILIZATION` | A building-wide redistribution opportunity across exits |
| `WARDEN_DISPATCH` | A zone where a human warden should be sent |
| `SYSTEM_WARNING` | A warning about the health of the recommendation pipeline itself (low confidence, AI-flagged risk, guidance inconsistency, widespread loss of safe exits) |

**"Internal providers"** refers to the existing packages this layer adapts: `evacuation_recommendation/`, `evacuation_guidance/`, `emergency_response/`, `crowd_intelligence/`, and `advisory_system/`. This layer never recomputes their logic, never modifies them, and they remain entirely unaware this layer exists.

## 2. Relationship to Existing Engines

**This is not a competing recommendation engine.** `evacuation_recommendation/` (deterministic exit-routing) and `evacuation_guidance/` (route-to-exit instructions) are the pre-existing, FROZEN, canonical live-wired recommendation core — see `docs/architecture/core_architecture_freeze_review.md`'s own Phase-12 FROZEN table, which names both explicitly. Neither package is modified, redesigned, or duplicated by this milestone.

`advisory_system/` is a separate, older, broader package producing multi-audience reports (civilian announcements, firefighter intelligence, building-systems recommendations, commander dashboard). It is absent from the Freeze Review's FROZEN table and was explicitly flagged there as unreviewed technical debt.

**A caller supplying `advisory_report=None` (or omitting it) is the honest default, not a degraded fallback.** Every category is designed to produce meaningful output using only the snapshots that are always available in a live/offline-demo session (`evacuation_recommendation`, `emergency_response`, `crowd_intelligence`). `advisory_report`/`ai_prediction_snapshot` are consumed purely as *optional enrichment* — never as the sole trigger for any recommendation.

| Category | Primary (always-available) source | Optional enrichment source(s) |
|---|---|---|
| Occupant Routing | `evacuation_recommendation` | — |
| Hazard Avoidance | `evacuation_recommendation` (no safe exit), `emergency_response` (hazard present) | `advisory_system` (commander-flagged critical zones) |
| Congestion Mitigation | `evacuation_recommendation` (congestion/queue reason codes) | `crowd_intelligence` (congested assets), `advisory_system` (congestion-flavored recommendations) |
| Exit Utilization | *self-derived* (`recommendation_layer`) — no upstream provider computes this | — |
| Warden Dispatch | `emergency_response` (response priority, assistance counts) | `advisory_system` (firefighter intelligence) |
| System Warning | `evacuation_recommendation`, `evacuation_guidance` | — |

## 8. Known Architectural Gap — `advisory_system` Live Wiring

A pre-implementation dependency analysis found: `live_advisory_gateway`/`ReplayCompatibleAdvisoryGateway` is constructed in **zero non-test call sites** anywhere in the repository. The only real, reachable `AdvisoryOrchestrator.generate_report()` call site is `command_center/incident_data.py`'s Replay/Load-Incident flow, itself fed by Campaign Studio's batch-simulation output (`ground_truth.json`) — never the always-on Live pipeline (`live_system/orchestrator.py`).

Consequently, `advisory_report` is `None` in the overwhelming majority of live/offline-demo cycles today. This milestone treats that fact as an **intentional non-goal**: the Recommendation Layer is designed so every category is still useful with `advisory_report` absent, and it does not attempt to give `advisory_system` a live-reachable path. That remains a candidate for a **future, separate** architectural milestone — out of scope here.

## 3. Architecture

```
recommendation_layer/
    models.py       -- RecommendationType, RecommendationPriority, RecommendationStatus,
                        TriggerCondition, RecommendationSource, Recommendation, RecommendationSet
    manager.py       -- RecommendationManager (dedupe / provenance merge / lifecycle / ranking)
    layer.py          -- RecommendationLayer (the public facade)
    adapters/
        occupant_routing_adapter.py
        hazard_avoidance_adapter.py
        congestion_mitigation_adapter.py
        exit_utilization_adapter.py
        warden_dispatch_adapter.py
        system_warning_adapter.py
```

Each adapter is a **thin, stateless mapping function**: `adapt(*snapshots) -> Tuple[Recommendation, ...]`. It never merges across providers, never decides which provider "wins" a same-cycle collision, and never carries a hidden scoring formula. All cross-provider merging, provenance bookkeeping, and lifecycle management happen exactly once, centrally, in `RecommendationManager`. The only two constants anywhere in this package that are not a pure passthrough of an existing provider field are named module-level constants on the two self-derived checks (`exit_utilization_adapter.EXIT_OVERUTILIZATION_RATIO`/`EXIT_OVERUTILIZATION_MIN_ZONES`, `system_warning_adapter.BUILDING_WIDESPREAD_NO_SAFE_EXIT_ZONE_COUNT`) — simple, documented thresholds, never tunable formulas.

`RecommendationSet` is a whole-collection aggregate (mirrors `advisory_system.recommendation_models.AdvisoryReport`'s own shape), not per-zone-keyed like `EvacuationRecommendationSnapshot` — one zone can carry multiple recommendation types simultaneously (e.g. `OCCUPANT_ROUTING` and `WARDEN_DISPATCH` for the same zone).

## 4. Public API

```python
from recommendation_layer.layer import RecommendationLayer

layer = RecommendationLayer(grace_period_seconds=5.0)  # defaults shown

recommendation_set = layer.compute(
    time,
    evacuation_recommendation_snapshot=...,   # from evacuation_recommendation.engine
    evacuation_guidance_snapshot=...,         # from evacuation_guidance.engine
    emergency_response_snapshot=...,          # from emergency_response.engine
    crowd_intelligence_snapshot=...,          # from crowd_intelligence.engine
    ai_prediction_snapshot=...,               # from live_ai_gateway, usually None
    advisory_report=...,                      # from live_advisory_gateway, usually None -- see §8
)

layer.latest                    # the most recent RecommendationSet, or None before the first compute()

recommendation_set.active()             # only ACTIVE recommendations
recommendation_set.by_type(type_)       # filter by RecommendationType
recommendation_set.for_zone(zone_id)    # every recommendation touching this zone
recommendation_set.to_dict()            # full serialization
```

`compute()` never raises — a bug in one adapter is individually caught and never blanks the other five categories — and always returns a `RecommendationSet` (never `None`). The `None`-means-"skip"/"failed" convention belongs one layer up, to `live_system.recommendation_layer_gateway.EngineRecommendationLayerGateway`, which has actual I/O to fail on.

**Every future consumer (a future Guidance v2, Command Center, a Dashboard) should import `RecommendationLayer`, never reach into `evacuation_recommendation`/`advisory_system` directly.**

## 5. Adapter System

See §2's table for primary/enrichment sources. Field-level detail:

- **Occupant Routing** — one `Recommendation` per zone with `status == RECOMMENDED`. `priority` is `MEDIUM` if the zone's own `reason_codes` intersect `{HIGH_CONGESTION, QUEUE_PRESENT, EMERGENCY_RESPONSE_ZONE_ELEVATED}`, else `LOW`.
- **Hazard Avoidance** — `NO_SAFE_EXIT_AVAILABLE` zones → `CRITICAL`; `emergency_response` zones with `HAZARD_PRESENT` → `HIGH`; `advisory_report.commander_dashboard.critical_zones` → `HIGH`. All three use `trigger_condition=ZONE_HAZARD_PRESENT` or `ZONE_NO_SAFE_EXIT`.
- **Congestion Mitigation** — zones/candidates with `HIGH_CONGESTION`/`QUEUE_PRESENT`/`LOW_THROUGHPUT` reason codes, or `congestion_level` at `HIGH`+; `crowd_intelligence.building_summary.congested_exits`; `advisory_report.building_recommendations` entries whose `action`/`reason` mentions congestion.
- **Exit Utilization** — counts zones routed to each exit; if one exit's load exceeds `EXIT_OVERUTILIZATION_RATIO` (2.0) times a safe alternative's load, with `EXIT_OVERUTILIZATION_MIN_ZONES` (3) as the floor, emits a paired `EXIT_OVERUTILIZED`/`EXIT_UNDERUTILIZED_ALTERNATIVE`.
- **Warden Dispatch** — `priority_level in (HIGH, CRITICAL)` → `ZONE_RESPONSE_ELEVATED`; any assistance count > 0 → separate `ZONE_ASSISTANCE_REQUIRED`; `advisory_report.firefighter_intelligence.live_priority_zone_ids`/`live_possible_assistance_zone_ids` mirror the same two trigger conditions.
- **System Warning** — low `confidence`/`coverage_fraction`/`POOR_COVERAGE` → `RECOMMENDATION_LOW_CONFIDENCE`; `AI_BOTTLENECK_RISK_ELEVATED` → `RECOMMENDATION_AI_BOTTLENECK_RISK` at `LOW` (AI is support-only, never critical alone); guidance `inconsistencies` → `GUIDANCE_INCONSISTENCY`; `BUILDING_WIDESPREAD_NO_SAFE_EXIT_ZONE_COUNT` (3) or more occupied zones with no safe exit → `BUILDING_NO_SAFE_EXIT_WIDESPREAD` at `CRITICAL`.

## 6. Recommendation Lifecycle

**Dedup key**: `type | trigger_condition | sorted(affected_zones) | sorted(affected_exits)`. Two candidates sharing this key are the same real-world claim.

**Same-cycle merge**: candidates sharing a dedup key this cycle are merged by `RecommendationManager` — a fixed provider-priority order (`evacuation_recommendation` > `emergency_response` > `evacuation_guidance` > `crowd_intelligence` > `advisory_system` > `recommendation_layer`) picks the winner; every other contributing provider is recorded in `supporting_sources`; `supporting_evidence` is merged with each key tagged in `evidence_origin` by which provider it actually came from.

**Cross-cycle identity**: a `Recommendation`'s `recommendation_id` is minted exactly once, the first cycle its dedup key is seen, and stays stable for its entire active lifetime — including through the grace-period window below. It is never re-minted while that key keeps reappearing.

**Expiration**: a key present in a previous cycle but absent this cycle: the first miss sets `expires_at = time + grace_period_seconds` (default 5.0s, ≈5 missed cycles at the live pipeline's ~1Hz cadence) while staying `ACTIVE`; reappearing before that deadline clears `expires_at` under the *same* ID; reaching the deadline flips `status` to `EXPIRED` (returned once more so a consumer sees the transition); the cycle after that, it is dropped entirely.

**Ranking**: `(-priority_ordinal, -updated_at, recommendation_id)` — highest priority first, most-recently-reaffirmed first within a tier, id as a stable final tiebreak.

## 7. Integration Points

**Live pipeline** — `live_system/recommendation_layer_gateway.py` (`RecommendationLayerGateway` Protocol + `EngineRecommendationLayerGateway` adapter) is wired into `LiveOrchestrator.run_cycle()` as the LAST optional stage, immediately after `live_advisory_gateway` and before the legacy `decision_policy_gateway`/`recommendation_builder` seams. It is "engine-shaped" (like Recommendation/Guidance) — `live_runtime/factory.py` default-constructs a `RecommendationLayer()` and unconditionally wraps it, so it runs automatically under both `build_live_runtime()` and `build_offline_demo_runtime()` with zero extra caller wiring. Stored via `state_manager.update_recommendation_set()`/`latest_recommendation_set()`, exposed as `LiveOrchestrator.latest_recommendation_set` (a forwarding property), and announced via `EventType.RECOMMENDATION_SET_UPDATED` (fired every successful cycle — no per-item transition events, since the lifecycle state machine already lives in `RecommendationManager`).

**Studio** — `designer/widgets/recommendation_panel.py::RecommendationPanel` is a new dock (tabified onto the existing Live-Runtime-family bottom-dock chain, hidden by default). It is refreshed via a new `LiveRuntimeController.on_cycle_callback` hook fired once per real `run_cycle()` tick — **not** the separate Manual Simulation Sandbox loop, which has no perception/AI pipeline behind it at all. Selecting a row calls `GraphicsScene.highlight_recommendation(zone_ids, exit_ids)` (a new method, mirroring the existing `_highlight_route`/`_clear_route_highlight` occupant-route-highlighting mechanism with its own separate tracking list, and the same disclosed single-floor-view limitation).

**Files touched as pure composition seams** (never the frozen packages themselves): `live_system/orchestrator.py`, `live_system/state_manager.py`, `live_runtime/factory.py`, `live_runtime/runtime.py`, `event_bus/bus.py`, plus the Studio-side files above and one small, self-contained visual addition to `designer/items/exit_item.py` (a `set_highlighted()` triple, mirroring `DoorItem`'s, so `affected_exits` can be highlighted). `evacuation_recommendation/`, `evacuation_guidance/`, `advisory_system/`, `emergency_response/`, `crowd_intelligence/` are never edited — mechanically enforced by `tests/test_recommendation_layer_architecture_guards.py`'s one-directional-dependency guard.

## 9. Future AI-Based Recommendation Roadmap

Today, AI enters only via `ai_prediction_snapshot.bottleneck.probability`, already folded into `evacuation_recommendation`'s own `AI_BOTTLENECK_RISK_ELEVATED` reason code before this layer ever sees it — read, never recomputed. A future, model-backed enrichment should plug into an *existing* adapter as one more optional input (exactly like `crowd_intelligence`/`advisory_report` today), never as a new seventh category that bypasses `RecommendationManager`'s dedup/lifecycle machinery, and never granted a path to `.execute()`/`.dispatch()` — this layer stays recommendation-only, permanently, mirroring every frozen provider's own explicit boundary.

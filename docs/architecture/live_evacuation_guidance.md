# Live Evacuation Guidance & Zoned Message Planning

Status as of this milestone: SynEvac can, for the first time, turn "Zone Z3 should use Exit E2" into an actual, graph-valid, ordered route ("Zone Z3 → Door D4 → Stair S2 → Exit E2"), deterministic human-readable instructions derived from that structure, and a PLANNED (never sent) voice message candidate for eventual operator approval — via a new `evacuation_guidance/` package. This is still planning only; execution remains exactly where it already lived.

## 1. Investigation findings (Phase 1)

1. `pathfinding.route.Route` already returns the full ordered path (`nodes`/`edges`, walkable in sequence) — `PathfindingEngine.distances_from()` already builds real `Route` objects internally; `evacuation_recommendation.ranking.SafeExitDistanceCalculator` confirmed to immediately collapse them to a plain float and retain nothing else — this milestone's own `route_planner.py` is the first consumer to keep the actual path.
2. `Node`/`Edge` retain full identity: `Edge.reference` is the real Door/Exit/Staircase model object with its own `.name` (present but not always descriptive in existing fixtures — often a short code like `"D1"`; `instruction_builder.asset_label()` handles both cases without duplicating a type prefix that's already implied by the name).
3. A Stair edge's two `Node`s genuinely carry different `floor_id` values (confirmed against `navigation/graph_builder.py::_add_stair_edges`) — walking `Route.nodes` and diffing `floor_id` reliably detects a floor change, and `Route.edges[i].reference` is the real `Staircase` used.
4. `PathfindingEngine._relax()` already refuses non-traversable edges unconditionally; `distances_from(start_id, excluded_node_ids=...)` is the exact pattern `trajectory_intelligence`/`evacuation_recommendation` already established for hazard exclusion — reused here as an independently-owned copy, seeded at the OCCUPANT's own zone (not the exit's, and not `Node.OUTSIDE_NODE_ID` — see §3).
5. Nothing in the codebase previously converted a `Route` into instructions or structured steps — genuinely new.
6. `voice_evacuation.models.VoiceMessageType.ROUTE_GUIDANCE` already existed, unused by any adapter until this milestone — exactly the right semantic slot (priority 50, between `EVACUATE`=90 and `ALL_CLEAR`=10).
7. `SpeakerManager.active_speakers_in_zone(zone_id)` and `Speaker.zone_ids` already existed, fully reusable — never duplicated, only received duck-typed.
8. `command_center.live_operator_action_gateway.LiveOperatorActionGateway` already established the exact "explicit operator click → adapter → `VoiceEvacuationController.broadcast()`" shape for civilian announcements — this milestone adds a parallel, identically-shaped `approve_guidance_message()`/`reject_guidance_message()` pair, reusing the SAME controller instance (never a second one).

## 2. Architecture

```
Decision Policy (unchanged, never imported here)
        |
        v
EvacuationRecommendationSnapshot  ("which exit" -- Zone Z3 -> Exit E2)
        |
        v
evacuation_guidance/                         <- THIS milestone
  route_planner.py   -- graph-valid ordered Route to the EXACT
                         recommended exit, hazard/traversability
                         excluded, never a different exit
  validation.py       -- independent re-check (Phase 10)
  instruction_builder.py -- structured NavigationSteps + deterministic
                         text ("Proceed through Door D4 toward Stair
                         S2 and continue to Exit E2.")
  message_planner.py  -- speaker coverage (delivery) + the single
                         combined voice sentence (message) -- kept
                         SEPARATE concerns
  engine.py            -- EvacuationGuidanceEngine, revision tracking
        |
        v
EvacuationGuidanceSnapshot ("how to reach it" + a PLANNED voice
                             message candidate, never sent)
        |
        +---------------------------------------+
        v                                       v
Advisory (EvacuationGuidanceEvidence,   Command Center (Live Evacuation
 commander awareness only)               Guidance panel: full structured
                                          route + Approve/Reject)
                                                        |
                                                        v
                                          Operator's own explicit click
                                                        |
                                                        v
                          voice_evacuation.adapter.guidance_plan_to_voice_message()
                                                        |
                                                        v
                          command_center.live_operator_action_gateway.
                          LiveOperatorActionGateway.approve_guidance_message()
                                                        |
                                                        v
                          voice_evacuation.controller.VoiceEvacuationController
                          (UNCHANGED -- the SAME controller civilian
                           announcements already reach)
                                                        |
                                                        v
                                            Output Provider (Simulation
                                             today; real hardware later)
```

**The critical boundary this package preserves:**

| Term | Meaning | Owner |
|---|---|---|
| Recommendation | "Use Exit E2." | `evacuation_recommendation/` (unchanged) |
| Guidance | "Zone Z3 → Door D4 → Stair S2 → Exit E2." | `evacuation_guidance/` (this milestone) |
| Message Plan | "Zone Z3: Proceed through Door D4 toward Stair S2 and continue to Exit E2." | `evacuation_guidance/message_planner.py` (this milestone) |
| Broadcast | The message actually reaching a speaker | `voice_evacuation.VoiceEvacuationController`, reached ONLY via explicit operator approval |

## 3. Route resolution (Phase 5/6)

`evacuation_guidance.route_planner.resolve_route(graph, zone_id, exit_id, building_state, config)`:

1. Finds the recommended exit's own `Edge` in the graph; if missing, `ROUTE_UNCERTAIN` (a mismatch between what the Recommendation Engine named and what this graph actually has — never fabricated).
2. If the edge is not `traversable`, or its own zone is hazard-excluded, `ROUTE_UNAVAILABLE`.
3. Runs ONE `PathfindingEngine.distances_from(zone_id, excluded_node_ids=hazardous_zones ∪ {Outside})` — seeded at the OCCUPANT's own zone, with `Outside` itself excluded so the interior search can never "cheat" by leaving through one exit and re-entering through another. This structurally guarantees the resulting route can only ever terminate via the one exit edge manually appended afterward — Guidance can never silently substitute a different exit (Phase 22).
4. If the exit's own zone isn't reachable in that search, `ROUTE_UNAVAILABLE` with `RECOMMENDED_EXIT_UNREACHABLE` — never a fabricated path.
5. Otherwise, the interior `Route` plus the manually-appended exit edge is the final, genuine, graph-valid `Route` — `ROUTE_AVAILABLE`, or `ALREADY_AT_EXIT` when the occupant's own zone already is the exit's zone.

`evacuation_guidance.validation.validate_route()` then INDEPENDENTLY re-checks the result (start zone, terminal exit, every edge traversable, every zone not hazard-excluded) before it is ever trusted — belt-and-suspenders, not merely construction-time trust (Phase 10).

## 4. Instructions and message planning (Phase 7/8/9/13/14)

`instruction_builder.build_navigation_steps()` produces the FULL structured step list (`LEAVE_ZONE`/`PASS_THROUGH_DOOR`/`ENTER_ZONE`/`USE_STAIR`/`CONTINUE_TO_EXIT`/`EXIT_BUILDING`) — always reconstructable, never lossy. `build_instructions()` derives text from that structure, one sentence per DOOR/STAIR/EXIT edge only (LEAVE_ZONE/ENTER_ZONE are never separately narrated — Phase 9's "do not over-instruct" is satisfied structurally, since every edge in this graph model already IS a meaningful decision point). Stair direction (ascend/descend) is genuinely derived from `Floor.display_order`, never invented; no left/right/landmark/distance fabrication anywhere.

`message_planner.build_voice_message_text()` walks the SAME route's edges in their true sequence order (not grouped by type — a route that descends a stair before a door says so in that order) to build one combined sentence, e.g. `"Zone Z3: Proceed through Door D4 toward Stair S2 and continue to Exit E2."` — the milestone's own worked example, produced deterministically, with no LLM anywhere.

`message_planner.speaker_coverage_for_zone()` is entirely independent — it never affects route validity (Phase 13/17): a route with `NO_SPEAKER_COVERAGE` is still `ROUTE_AVAILABLE`, just honestly flagged, and its voice plan still carries `delivery_status=NO_SPEAKER_COVERAGE` rather than being silently dropped.

## 5. Revisions (Phase 12)

Each zone's guidance carries a deterministic, monotonically-increasing `revision: int`, fingerprinted on `(recommended_exit_id, ordered_door_ids, ordered_stair_ids, route_status)`. An identical fingerprint across cycles keeps the SAME revision (never spam); any genuine change increments it by exactly 1. Never a random UUID.

## 6. No duplicate announcements (Phase 15/16) — the explicit architecture decision

**Chosen: guidance produces its own, separately-labelled candidate — never merged into `CivilianAnnouncement.announcement` text.** `voice_evacuation.adapter.guidance_plan_to_voice_message()` builds a `VoiceMessage` with `message_type=VoiceMessageType.ROUTE_GUIDANCE` (priority 50). "One final operator-visible message per zone" is enforced by `VoiceEvacuationController`'s own PRE-EXISTING per-zone priority supersession — a civilian `EVACUATE` message (priority 90) for the same zone always outranks and supersedes a `ROUTE_GUIDANCE` message, never the reverse — no new reconciliation logic was needed. `command_center.live_operator_action_gateway.LiveOperatorActionGateway` gains a parallel `approve_guidance_message()`/`reject_guidance_message()` pair, keyed on `zone_id::guidance_revision` (never message text) so a changed revision always starts fresh as `RECOMMENDED`, while an unsent old revision's own decision is simply left behind (stale, never silently promoted) and a SENT old revision remains in `VoiceEvacuationController`'s own append-only broadcast log forever (Phase 16). Proven end-to-end by `tests/test_live_runtime_evacuation_guidance_e2e.py` and `tests/test_evacuation_guidance_voice_integration.py`.

## 7. Safety precedence

```
Decision Policy / deterministic safety
        >
Evacuation Recommendation
        >
Guidance Planning
        >
Voice Planning
        >
Operator Approval
        >
Output Provider
```

Guidance can never make an unsafe asset usable (§3's hard node-exclusion + independent validation). Voice planning never reinterprets the route — `message_planner.py` only ever reads `route.edges` in order, it makes no routing decision of its own. `evacuation_guidance/` never imports `decision_policy`, `advisory_system`, `command_center`, `voice_evacuation` output providers, `building_control`, `facp`, or any sibling live-intelligence engine (mechanically enforced by `tests/test_evacuation_guidance_architecture_guards.py`).

## 8. Trajectory / Crowd relationship (Phase 21/22)

`evacuation_guidance/` never reads `trajectory_intelligence` or `crowd_intelligence` at all. A route change comes ONLY from: the recommended exit changing, or the route to that exact exit becoming unsafe/unavailable — never from "one person appears to be ignoring guidance" (no psychological inference anywhere in this package) and never from Guidance independently reranking exits (if the recommended exit becomes unreachable, Guidance reports `RECOMMENDED_EXIT_UNREACHABLE`, it never silently tries a different exit — Phase 22's own hard requirement, mechanically proven by `tests/test_evacuation_guidance.py::NoSilentSubstitutionTests`).

## 9. No execution authority

Mechanically enforced by `tests/test_evacuation_guidance_architecture_guards.py`: no imports of AI/RL training, Command Center, voice output providers, Building Control execution, FACP mutation code, or hardware protocols; no execution verbs (`.broadcast(`, `.send(`, `.dispatch(`, `.execute_control(`, `.confirm(`) anywhere in the package. `tests/test_live_runtime_evacuation_guidance_e2e.py::test_no_automatic_execution_anywhere` proves nothing is sent to the output provider merely by computing guidance.

## 10. What this milestone deliberately does not do

No automatic voice broadcast, no automatic building control, no dynamic signage hardware, no text-to-speech, no audio playback, no vendor PA protocols, no physical CCTV, no AI retraining, no RL, no Decision Policy redesign, no LLM-generated instructions, no psychological behavior inference.

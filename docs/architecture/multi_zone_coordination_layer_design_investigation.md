# Multi-Zone Coordination Layer — Narrowest Honest Design Investigation

Status: investigation/design only, zero code changes. This follows directly
from `docs/architecture/multi_zone_coordination_architecture_investigation.md`
(verdict **B**: the current RL decision path can produce multiple zone
recommendations, but never jointly coordinates them — Option B, "a separate
recommendation coordination/allocation layer," was identified there as the
smallest justified next step). This document works out *how* that layer
would actually be built, using only components verified to already exist in
this codebase, without writing any code.

---

## 0. Ground rule stated up front

**The current RL policy does not perform multi-zone optimization, and
nothing below changes that fact or claims otherwise.** The policy emits one
`Discrete` action per tick (`rl_training/action_space.py:140`), decoded to
one target broadcast identically to every zone
(`ActionMapper.decode()`, `ProductionRLEnvironment.step()`,
`LiveRLDecisionPipeline.tick()`). Its observation *does* encode every zone's
hazard/occupancy state (`rl_training/perception_observation_space.py:153-159`),
so its single choice is a judgment informed by whole-building conditions —
but it is still one scalar output, never a per-zone assignment, and its
observation contains no route topology, no capacity, and no congestion
feature of any kind (verified absent from
`PerceptionObservationSchema.feature_names`). Everything in this design
treats that single choice as exactly what it is: **one whole-building
directional preference**, not a verified, per-zone-feasible plan.

---

## 1. Definitions — coordinated allocation vs. independent per-zone recommendations

The prior investigation's Section 3 already showed the current architecture
represents neither extreme cleanly, so this design needs a precise line:

- **Independent per-zone recommendation**: a zone's target is computed with
  no reference to what any *other* zone is being told or to any shared
  resource's total load. This describes **both** of the codebase's existing
  paths today: `RLActionDecisionEngine`'s broadcast (every zone gets the
  same value, but only because it's the same *input*, not because anything
  compared zones to each other) and `AIDecisionEngine._zone_routes()`
  (`ai_decision/engine.py:222-252`, which calls `hazard_engine.nearest_exit(node.id)`
  once per zone, in isolation, and never inspects another zone's result).
- **Coordinated allocation**: a zone's *final* target is a function of (a)
  what other zones are simultaneously being told, (b) a shared resource's
  real, structural capacity, and (c) whether their routes to that resource
  genuinely overlap — such that the assignment is evaluated and adjusted as
  a *set*, not zone-by-zone in a vacuum.

The design below is coordinated in this precise sense: the step that
redirects an "overflow" zone is explicitly conditioned on what every other
zone sharing that target is also being told, not on that zone's own state
alone. It does not claim to be optimal, does not claim to be multi-agent RL,
and does not claim the RL policy contributed anything beyond the single
default target it already contributes today.

---

## 2. Insertion point

The narrowest change is a **pure interception step**, added between "the RL
policy's chosen single target is known" and "the decision engine is told
what to recommend" — exactly the seam that already exists identically in
both call sites that currently do this uniformly:

- `ProductionRLEnvironment.step()` (`rl_training/production/environment.py:265-267`):
  `recommended_edge_id, recommendation_label = self._decode_action(action)`
  → currently `self._decision_engine.set_action(recommended_edge_id, zone_ids)`.
- `LiveRLDecisionPipeline.tick()` (`live_runtime/live_rl_decision_pipeline.py:143-161`):
  `inferred = self._rl_engine.infer(...)` → currently
  `self._decision_engine.set_action(inferred.recommended_edge_id, zone_ids)`.

In both places, the single call to `set_action(recommended_edge_id,
zone_ids)` would be replaced by: build a per-zone mapping via the new
coordinator, then call the **already-existing**
`RLActionDecisionEngine.set_zone_recommendations(mapping)`
(`rl_training/production/decision_engine.py:67-70`) instead.

**This requires zero changes inside `RLActionDecisionEngine.decide()`.**
That method already branches on whether `set_zone_recommendations()` was
used (`self._explicit_zone_recommendations is not None`,
`decision_engine.py:294-297`) and, if so, builds `effective_actions` from
that per-zone mapping directly — the exact same code path every existing
`set_zone_recommendations()` test already exercises
(`tests/test_rl_action_decision_engine.py:77-99` and others). Severity,
occupancy, `is_unsafe`, and `priority_evacuation_order` all continue to be
computed exactly as today, per zone, from `observation` — the coordinator
only changes *which target string* each zone is paired with before `decide()`
runs. As a direct consequence, `RLActionDecisionEngine._zone_reason()`
(`decision_engine.py:209-261`) **already** produces the correct
`"recommended exit: <target>"` clause per zone with no changes at all,
since it already reads whatever `recommended_exit_edge_id` ended up in that
zone's `ZoneRecommendation`.

---

## 3. Components reused (nothing new invented)

| Need | Reused component | Why it's honest to reuse |
|---|---|---|
| Per-zone real route to a candidate target | `pathfinding.engine.PathfindingEngine.nearest_exit()`/`.distances_from()`/`.alternative_paths()` (`pathfinding/engine.py`) | Same, unmodified engine `AIDecisionEngine` already uses for exactly this purpose (`ai_decision/engine.py:222-252`) and the simulation's own per-occupant replanning already uses (`simulation_interactive/replanning.py:159-192`). Operates only on real `Building`/`NavigationGraph` structure. |
| Never route through a perceived hazard the coordinator has no evidence is safe | `hazard.cost_model.HazardAwareCostModel` (`hazard/cost_model.py`) wrapping `ai_decision.perception_adapter.hazard_snapshot_from_observation()` (`ai_decision/perception_adapter.py:109-128`) | Both already exist and are already perception-honest — `hazard_snapshot_from_observation()`'s own docstring: "every value placed inside the returned snapshots comes only from the BuildingObservation handed in." An edge whose perception `blocked_estimate` is `True` costs `HazardEdgeState.BLOCKED_COST` (infinite) and is never relaxed onto (`hazard/cost_model.py:38-45`) — never "assume open," matching the existing `AIDecisionEngine._hazard_aware_engine()` construction verbatim. |
| Per-zone occupant estimate | `BuildingObservation.occupancy_observation(zone_id).estimated_count` | The exact same accessor `RLActionDecisionEngine._build_zone_recommendation()` already calls (`decision_engine.py:106`) — no new perception, no duplicated derivation. |
| Real, structural per-target capacity (Door/Exit/Stair) | `crowd_intelligence.capacity.door_capacity()/exit_capacity()/stair_capacity()` (`crowd_intelligence/capacity.py`) | Reads only static `Door.width`/`Exit.width`/`Exit.capacity`/`Staircase` geometry off the `Building` model (`models/door.py:14`, `models/exit.py:14-15`) via `simulator.capacity.DefaultCapacityModel`/`StairCapacityModel` — real structural data, not simulator ground truth about current hazard/occupancy state. Already disclosed in its own docstring as "the same documented, non-validated ENGINEERING ESTIMATE... not a validated life-safety flow-rate model, just a reasonable default" — this design inherits that exact disclosure, not a stronger claim. |
| Optional secondary cross-check: is this target *already* congested right now | `crowd_intelligence.engine.CrowdIntelligenceEngine.compute()` → `AssetApproachMetrics.congestion_level`/`BuildingCrowdSummary.congested_exits`/`congested_stairs`/`congested_doors` (`crowd_intelligence/engine.py`) | Already live-wired into `build_live_runtime()` (`live_runtime/factory.py:214`), driven by real position-tracked occupants near the asset (`crowd_intelligence/queue.py:23-74`, "NEVER fabricates a queue from zone occupancy alone"). **Disclosed limitation**: this is a *separate* live composition (`LiveOrchestrator`/`build_live_runtime()`), not currently threaded into `LiveRLDecisionPipeline` at all (that pipeline's own docstring: "Standalone by design... does not import or construct LiveOrchestrator"). Using it here is a real, additional, small wiring step, not something already connected — flagged explicitly in Section 8. |
| Per-zone override channel into the decision engine | `RLActionDecisionEngine.set_zone_recommendations()` (`decision_engine.py:67-70`) | Already implemented, already tested, currently unused in production (Section 1 of the prior investigation). |
| A field to carry the real computed route, if useful for downstream consumers | `ZoneRecommendation.recommended_route: Optional[Route]` (`ai_decision/recommendation.py:29`) | Already exists on the schema, always `None` today under the RL path (never set by `RLActionDecisionEngine._build_zone_recommendation()`) — populating it is additive, not a schema change. |

No new package needs to reach into `ground_truth`, `hazard_evolution`, or any
simulator-private type. `crowd_intelligence`'s own architecture guard
(`tests/test_crowd_intelligence_architecture_guards.py:44-54`) forbids that
package from importing AI/decision/execution code — it says nothing about a
decision-layer coordinator *reading* `crowd_intelligence`'s own reporting
output, which is the intended direction here and does not violate that
guard.

---

## 4. Detecting shared-route conflicts — the narrowest honest version

**Step 1 (minimal, proposed as the actual first version): shared *final*
target edge.** `ZoneRecommendation.recommended_exit_edge_id` is already the
only per-zone target representation that exists anywhere in the traced
path — it is literally "the last edge of the route" (see
`AIDecisionEngine._zone_recommendation()`, `route.edges[-1].id`). Two zones
whose real route to the RL-chosen target both end on that same Door/Exit/
Stair edge id are, by construction, converging on the same physical
asset. This is detectable with zero new concepts: compute each zone's route
to the RL's chosen target via `PathfindingEngine.nearest_exit()`/targeted
search, and group zones by whether that route actually reaches the target
edge at all (a zone might not even be able to reach it, in which case it
was never honestly a candidate for that target in the first place — see
Section 6).

**Step 2 (explicitly out of scope for the narrowest version, disclosed, not
silently dropped): intermediate corridor-edge overlap.** Two zones could
converge on *different* final exits/stairs while still sharing an
intermediate corridor edge along the way. Detecting this is mechanically
possible (`Route.edge_ids` is a plain list; comparing two zones' edge-id
sets for intersection uses nothing not already in `PathfindingEngine`'s
output). **What is missing is not detection but honest capacity
attribution**: `crowd_intelligence.capacity` only defines a capacity
concept for Door/Exit/Stair (verified: `door_capacity()`/`exit_capacity()`/
`stair_capacity()` are the only three functions in that module); a generic
`Corridor` edge has no `width`/`capacity` field anywhere in the traced model
(`navigation/edge.py:93-107`: `width`/`capacity` are plain `getattr`
passthroughs onto `edge.reference`, which returns `None` for anything that
isn't a Door/Exit/Staircase). A corridor-overlap conflict could be
*flagged* honestly (same edges appear in two zones' routes) but not
honestly *compared against a capacity number*, since no such number exists
in this codebase today. The narrowest honest design therefore limits itself
to the Door/Exit/Stair join key from Step 1, where both "shared" and
"capacity" are real; Section 8 lists corridor-level detection as a
disclosed, deferred extension, not a silent omission.

---

## 5. Combined demand

For each candidate target `T` (starting with the RL policy's single chosen
target, then any alternate targets used in redistribution — Section 6):

```
demand(T) = sum(zone.occupancy_observation(zone_id).estimated_count
                for zone_id in zones_routed_to(T)
                if estimated_count is not None)
```

Two honesty requirements, both directly inherited from existing codebase
conventions rather than newly invented:

- **A zone with no occupancy reading this cycle must never be treated as
  contributing 0.** Exactly like `SeverityOccupancyPriorityRule`'s own
  `-(rec.occupant_count or 0.0)` sort key (`ai_decision/priority.py:79`)
  already accepts *for ranking purposes only* — but summing 0 for an
  unobserved zone in a **demand total** would produce a false-safe
  undercount, which ranking-only use does not. The coordinator's `demand(T)`
  must therefore be presented as an explicit **lower bound**, with the
  count of unobserved-but-assigned zones surfaced alongside it (e.g. "≥42
  occupants across 3 observed zones; 1 additional zone assigned this target
  has no occupancy reading this cycle"), never silently folded into one
  number that looks complete.
- **Demand is conditional on compliance, never asserted as what will
  happen.** `behaviour_profile_resolver/dynamic_replanning.py` and
  `simulation_interactive/replanning.py` both already model recommendation
  compliance as probabilistic (`derive_compliance_seed()`,
  `ComplianceDecisionStrategy` — occupants are not guaranteed to follow a
  recommendation). `demand(T)` must be phrased as "if this target's
  recommendation is followed," never as a committed headcount.

---

## 6. Resolution algorithm (design, not code)

1. Start from the RL policy's single chosen target as every zone's default
   assignment — this is the "preference, not hard constraint" decision
   (Section 7): it is *not* discarded, only *capped*.
2. For each zone assigned that target, compute its real route via the
   **hazard-aware** `PathfindingEngine` (Section 3) — i.e. built from the
   *same* `BuildingObservation` this decision cycle already has, exactly
   the way `AIDecisionEngine._hazard_aware_engine()` already builds one.
   - A zone with **no** honest route to the RL-chosen target at all (the
     hazard-aware search returns no `Route`) was never a real candidate for
     it — this is not a "conflict" this layer resolves, it is the
     pre-existing, already-disclosed `is_reachable` gap (Section 9); such a
     zone keeps whatever the base `RLActionDecisionEngine` behavior already
     is for an unreachable target (this design does not change that).
3. Compute `demand(T)` (Section 5) over the zones that *do* have a real
   route to `T`.
4. Compare `demand(T)` against `T`'s structural capacity
   (`crowd_intelligence.capacity`, Section 3) — a **disclosed, documented
   engineering-estimate threshold**, the same honesty level
   `StairCongestionPredictor.congestion_threshold` already carries, never
   presented as a validated life-safety limit.
5. If `demand(T)` is within capacity: no redistribution — every zone keeps
   the RL's chosen target, exactly as today. **This is the common case and
   must remain the common case**: this layer changes nothing about a tick
   where the RL's single choice happens to be feasible for everyone routed
   to it.
6. If `demand(T)` exceeds capacity: select overflow zones and, for each,
   search for a real, hazard-aware, reachable **alternative** Exit/Stair
   node — mechanically available today via `PathfindingEngine.distances_from(start_id)`
   (already exists, `pathfinding/engine.py:234-270`, returns a `Route` to
   *every* reachable node in one search), filtered to Exit/Stair-type goal
   nodes excluding `T`, picking the lowest-`total_cost` alternative.
   - **Overflow selection policy is a disclosed placeholder**, exactly the
     same honesty already applied to `SeverityOccupancyPriorityRule`'s "V1's
     whole ranking policy... a documented placeholder, not a validated
     life-safety prioritization model" (`ai_decision/priority.py:63-71`).
     A reasonable first rule: keep the target for the zones nearest to it
     (lowest `total_cost` to `T`), redirect the farthest-first — this
     avoids redirecting a zone that is already closest to/most committed to
     `T` while zones further away still have a meaningfully competitive
     alternative route. This is stated here as a starting policy to
     validate empirically, not as a settled answer.
   - A zone with **no** honest alternative (every other Exit/Stair is
     unreachable given current perceived hazard state) is **not** forced
     off `T` — it keeps the RL-chosen target, over-capacity or not, exactly
     mirroring `RecommendationAwareRouteChoiceStrategy`'s own existing
     "recommended exit/stair unreachable from here... fall back rather than
     fail; the occupant still gets a route" discipline
     (`simulation_interactive/replanning.py:189-192`). This layer never
     leaves a zone with *no* recommendation because it couldn't find a
     perfect answer.
7. Populate the resulting `{zone_id: target_edge_id}` mapping via
   `set_zone_recommendations()`. Optionally also populate
   `ZoneRecommendation.recommended_route` with the real `Route` computed
   (currently unused, Section 3) for any downstream consumer that wants the
   full path, not just the final edge id.
8. A small, additive, `_stair_congestion_note()`-shaped helper (same pattern
   as `decision_engine.py:152-207`, not a new mechanism) could append a
   hedged clause to a redirected zone's `PriorityEvacuationEntry.reason` —
   e.g. "redirected from Stair-2 to Exit-1: projected combined demand
   (≥58) exceeds Stair-2's structural capacity estimate (50)" — using only
   the same "Monitor... is projected to..." hedging discipline already
   established, never a certainty claim.

---

## 7. Should the RL recommendation be a preference or a hard constraint?

**Preference — capped by real capacity, not overridden by default, and
never silently discarded.** The evidence for this, all drawn from what is
actually verified about the policy (not assumed):

- The policy's observation genuinely includes every zone's hazard/occupancy
  state (Section 0), so its single choice is a real, evidence-informed
  judgment about which target is generally best *given hazard conditions* —
  this is worth preserving as the default, not thrown away.
- The policy's observation contains **zero** route/topology/capacity/
  congestion feature (verified absent from
  `PerceptionObservationSchema.feature_names` — the only exit/stair
  features are `is_active_recommendation_target`, a self-referential
  bookkeeping bit, never occupancy or capacity). It therefore has no basis
  on which to have reasoned about whether its single choice is
  *simultaneously feasible for every zone* — treating its output as a hard,
  unoverridable per-zone constraint would attribute joint-feasibility
  reasoning to a component that structurally cannot have performed it.
- Overriding it *unconditionally* (e.g., always redistributing zones across
  several targets regardless of demand) would equally be dishonest in the
  other direction — it would discard the one piece of real, trained,
  hazard-informed judgment the system has, replacing it with an arbitrary
  rule that has no comparable evidence behind it.
- The design in Section 6 therefore treats the RL choice as the default for
  every zone, changes nothing when it is honestly feasible (the common
  case), and reroutes only the *provable* overflow — using the *same*
  hazard-aware routing evidence, so a zone is never redirected into
  something the perceived hazard state itself would flag as unsafe.

---

## 8. Explicitly out of scope for this narrowest design (disclosed, not silently dropped)

- **Intermediate corridor-edge overlap** (Section 4, Step 2) — detectable,
  not honestly capacity-comparable with what exists today.
- **`CrowdIntelligenceEngine`'s live congestion signal as a primary input**
  — real and already live-wired, but on a separate composition path from
  `LiveRLDecisionPipeline` today (Section 3); wiring it in as an *additional*
  cross-check ("this target is also already showing HIGH congestion right
  now, independent of this recommendation") is a natural, small next
  extension once the base coordinator exists, not a prerequisite for it.
- **Sequential/staggered delivery across ticks** (Option C from the prior
  investigation) — a refinement on top of this design (delay when a
  redirected zone's assignment takes effect), not required for the base
  version.
- **Changing the RL action space or retraining** (Option A from the prior
  investigation) — this design deliberately requires neither; see the
  summary table below.

---

## 9. Safety and honesty constraints — checked against this specific design

- **No route safety is invented.** Reachability for both the default target
  and any alternative is decided exclusively by the existing, unmodified
  `PathfindingEngine` search returning `None` or a `Route` — never by this
  layer asserting a route is open/closed on its own authority.
- **`is_reachable` is not repurposed as route verification.** This design
  does not change `RLActionDecisionEngine.is_reachable`'s existing,
  disclosed always-`True` placeholder status (`decision_engine.py:137-148`);
  a coordinator built with a real `PathfindingEngine` reference *could*
  compute a genuine `is_reachable` (the way `AIDecisionEngine` already
  does), but that is a separate, larger change this document does not fold
  in silently — flagged here explicitly rather than assumed.
- **Capacity is a disclosed engineering estimate, not a validated
  life-safety number**, inherited verbatim from `crowd_intelligence.capacity`'s
  own existing disclosure — this design does not strengthen that claim.
- **Demand is always presented as a hedged, compliance-conditional,
  possibly-undercounted lower bound** (Section 5), never a committed
  headcount or a certainty.
- **Confirmed hazard remains distinguishable from conservative
  uncertainty.** This design does not touch `ConservativeSafetyState`
  (`live_runtime/operator_advisory_adapter.py`) or `is_unsafe`'s existing
  derivation in `RLActionDecisionEngine._build_zone_recommendation()` — a
  redirected zone's `is_unsafe`/`severity` are computed exactly as before,
  from the same perception, independent of whether this layer redirected
  its target.
- **The RL policy keeps recommending doors/exits, stairs, or NOOP exactly
  as today** — this design only decides, after the fact, which real,
  reachable target each zone is paired with when the policy's single choice
  would honestly overload it; it introduces no new action vocabulary and no
  new policy behavior.

---

## 10. Summary

| Question | Answer for this design |
|---|---|
| Influences the RL policy itself? | No — same single categorical choice, unchanged. |
| Requires retraining? | No. |
| New information required? | None not already genuinely available: real `Building`/`NavigationGraph` topology, perception-honest hazard state (already used by `AIDecisionEngine`), perception-honest per-zone occupancy (already used by `RLActionDecisionEngine`), structural Door/Exit/Stair capacity (`crowd_intelligence.capacity`, real model fields). |
| Genuinely available in the live perception stack today? | Yes, for every input listed above. `CrowdIntelligenceEngine`'s live congestion (optional secondary signal) is real but not yet wired to this pipeline (Section 8). |
| Introduces route certainty or privileged information? | No — same reachability/hazard honesty level `AIDecisionEngine` already relies on; capacity/demand are explicitly hedged, not asserted. |
| RL recommendation treated as | A capped **preference/default**, not a hard constraint (Section 7) — preserved for every zone where it is honestly feasible, overridden only for provable, capacity-based overflow, and never silently discarded. |
| Is this "coordinated allocation"? | Yes, in the precise sense defined in Section 1 — the redirect decision for any one zone explicitly depends on what other zones sharing its target are also being told and on that target's real capacity, not on that zone's state alone. |
| Does the existing RL policy already do this? | No — restated explicitly: it emits one scalar choice per tick with no per-zone output and no visibility into topology, capacity, or congestion. This design adds coordination *around* it; it does not claim the policy already provides it. |

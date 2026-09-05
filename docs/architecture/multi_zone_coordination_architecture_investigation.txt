# Multi-Zone Coordination Architecture Investigation

Status: investigation only, zero code changes. Read-only trace of the actual
decision path, verified from source (not from naming or memory), for a
question the OperatorAdvisoryReport capability-contract work surfaced but did
not itself answer: **when SynEvac recommends actions for more than one zone,
does anything in the codebase reason about those recommendations jointly, or
about the physical infrastructure (stair/exit/corridor) they share?**

Scope: the live/production RL decision path —

```
BuildingObservation
  -> PerceptionObservationEncoder.encode()          (rl_training/perception_observation_space.py)
  -> SB3 policy .predict()                          (rl_training/production/live_inference.py)
  -> ActionMapper / action-index decode              (rl_training/action_space.py,
                                                        rl_training/production/environment.py,
                                                        rl_training/production/live_inference.py)
  -> RLActionDecisionEngine.set_action()/decide()    (rl_training/production/decision_engine.py)
  -> ZoneRecommendation (per zone)                    (ai_decision/recommendation.py)
  -> priority_evacuation_order                        (rl_training/production/decision_engine.py)
  -> OperatorAdvisoryReport                           (live_runtime/operator_advisory_adapter.py)
```

`ai_decision.engine.AIDecisionEngine` (the ground-truth-shaped, non-RL sibling
implementing the same `DecisionEngine` contract) is examined only as a point
of comparison, since it is not the engine `live_rl_decision_pipeline.py` /
`ProductionRLEnvironment` actually run.

---

## 1. Current decision granularity

**One RL inference produces exactly one building-wide discrete decision, not
per-zone decisions.**

- `ActionMapper.space` (`rl_training/action_space.py:140`) is a single
  `gymnasium.spaces.Discrete(len(entries))` — one categorical choice per
  policy call, full stop. There is no per-zone action dimension anywhere in
  the action space.
- The entry table (`ActionMapper.__init__`, lines 81-128) has exactly one
  `NOOP`, one entry per `Exit` (`RECOMMEND_EXIT:<id>`), one entry per `Stair`
  (`RECOMMEND_STAIR:<id>`), and three whole-building broadcasts
  (`BROADCAST_EVACUATE`, `BROADCAST_SHELTER_IN_PLACE`, `DEPLOY_STAFF`). None
  of these entries are zone-scoped.
- `ActionMapper.decode()` (lines 152-177) turns a chosen `RECOMMEND_EXIT`/
  `RECOMMEND_STAIR` entry into **one `simulation_interactive.Action` per
  zone, all carrying the identical `target_id`** (`for zone_id in
  self._zone_ids`). This is explicitly disclosed in the class's own comment
  (lines 71-79): *"A 'recommend exit/stair' action is applied to every zone
  in the building... since this environment's action space is
  building-scoped."*
- The production path (`ProductionRLEnvironment.step()`,
  `rl_training/production/environment.py:263-308`, and the live counterpart
  `LiveRLDecisionPipeline.tick()`, `live_runtime/live_rl_decision_pipeline.py:112-184`)
  both decode the single action index to one `(recommended_edge_id,
  recommendation_label)` pair, then call
  `RLActionDecisionEngine.set_action(recommended_edge_id,
  <every zone id known to the encoder>)` — the same edge id, broadcast
  identically to the whole zone list.
- `RLActionDecisionEngine.decide()` (`rl_training/production/decision_engine.py:290-323`)
  then builds `effective_actions = {zone_id: self._recommended_edge_id for
  zone_id in self._zone_ids}` — again, one value replicated across every key.

**How many recommendations can one inference produce?** Structurally, one
per zone in the building (a `ZoneRecommendation` object is built for every
zone via `_build_zone_recommendation()`), but their
`recommended_exit_edge_id` values are **not independent outputs** — they are
N copies of the single value the policy chose this tick (or `None` for
NOOP/broadcast entries, which have no target at all). `severity`,
`occupant_count`, and `is_unsafe` genuinely differ per zone (they come from
that zone's own perception), but the *actioned target* does not.

**Produced by the RL policy, derived afterward, or supplied independently?**
Derived afterward, mechanically, by broadcast — not chosen independently per
zone, and not a joint output of the policy itself (the policy never emits
more than one categorical value).

There *is* a real per-zone differentiation path —
`RLActionDecisionEngine.set_zone_recommendations(Mapping[str,
Optional[str]])` (`decision_engine.py:67-70`) — which is mutually exclusive
with `set_action()` and does allow each zone to carry a genuinely distinct
target. It is fully implemented, unit-tested
(`tests/test_rl_action_decision_engine.py:77-99` and others), and used
directly by `tests/test_live_decision_coordinator.py`,
`tests/test_operator_advisory_adapter.py`, and
`tests/test_stair_congestion_decision_integration.py`. **But no production
caller — `ProductionRLEnvironment`, `LiveRLDecisionPipeline`, or anything
else — ever calls it.** It is a real, working escape hatch with zero live
wiring today (confirmed by grepping every non-test call site of
`set_zone_recommendations(` — the only production-shaped candidate is
`LiveDecisionCoordinator`, and its own docstring says explicitly: *"it is
the caller's own separate responsibility to have already called that
engine's `set_action()`/`set_zone_recommendations()`"* — `LiveDecisionCoordinator`
itself never chooses between them).

---

## 2. Shared-target coordination

**Can the system know multiple zones are recommended toward the same
stair/exit?** Only in the degenerate sense that, in the actual production
path, *every* zone is always recommended toward the same target (or none) —
because it is the same broadcast value. There is no code anywhere that
detects, counts, or even represents "N zones share target T" as a fact,
because in the only path that runs today N is always "all of them" and this
is true by construction, not by any comparison. Nothing computes `{target:
[zone_ids]}` groupings — verified by search: no `groupby`/`defaultdict`/
`by_target`/`by_exit`/`by_stair` grouping logic exists anywhere in
`ai_decision/`, `rl_training/`, or `live_runtime/`.

**Can it know how many occupants from different zones would use the same
target?** No aggregation of `occupant_count` across zones exists anywhere.
Each `ZoneRecommendation.occupant_count` is that zone's own perceived
estimate (`ai_decision/recommendation.py:23-29`); nothing sums it, groups it
by `recommended_exit_edge_id`, or compares it to any capacity/threshold.

**Can one zone's recommendation influence another zone's?** In the current
production path this question is close to malformed: there is only one
recommendation per tick, mechanically copied to every zone — zone A's
"recommendation" and zone B's are the same value by construction, not by one
influencing the other. Nothing in `RLActionDecisionEngine`,
`ActionMapper`, or `ProductionRLEnvironment` reads one zone's state to change
what another zone is told. The observation the *policy* consumes does
include every zone's hazard/occupancy features
(`rl_training/perception_observation_space.py:153-159`, one hazard/occupancy
block per zone_id), so the policy's *single* choice is informed by the whole
building's state jointly — but its output is still one value, not a
per-zone assignment, so there is no mechanism by which zone A's evidence
could push zone B toward a *different* target than zone A gets.

**Is there any building-wide assignment/allocation state?** No. There is no
object anywhere (checked `ai_decision/`, `rl_training/`, `live_runtime/`,
`advisory_system/`, `live_system/`) that holds a mapping like "target T's
total assigned demand so far this tick" or "targets already saturated this
cycle." `RLActionDecisionEngine` holds exactly one scalar
(`self._recommended_edge_id`) or one already-fully-formed mapping
(`self._explicit_zone_recommendations`, populated only via the unused
`set_zone_recommendations()` escape hatch) — never a running allocation it
builds up across zones.

---

## 3. Contradiction and overload analysis

Walking each scenario against the traced code:

- **Two or more zones sent to the same stair.** This is not a *possible*
  failure mode in the current path — it is the *default outcome* of every
  non-NOOP RL decision, by construction (Section 1). Whatever stair the
  policy picks is recommended to literally every zone in the building
  simultaneously. Nothing downstream distinguishes "all zones happen to
  share a stair because the policy chose one target" from "each zone
  independently, coincidentally converged on the same stair."
- **Multiple zones sent to the same exit.** Same mechanism, same conclusion.
- **Recommendations competing for the same downstream path/corridor.**
  Cannot be represented at all: `RLActionDecisionEngine` has no
  `Building`/`NavigationGraph`/`PathfindingEngine`/`Route` reference of any
  kind — confirmed by its own import list (`decision_engine.py:1-16`) and
  restated explicitly in its comments (lines 52-59, 118-122: *"this engine
  has no Building/NavigationGraph/Route reference of any kind"*). It knows
  only bare edge-*id strings* (`recommended_exit_edge_id`), never which
  edges are structurally the same corridor, which edges are downstream of
  which, or which routes overlap. A corridor-overlap contradiction is
  invisible to this engine by construction, not merely undetected.
- **One zone's recommendation making another zone's worse.** Cannot arise
  causally in the current path (all zones get the same value, so there is no
  "one zone's recommendation" distinct from another's to interact) — but the
  *emergent effect* of the single broadcast (all zones funneled to one
  target at once) is exactly the overload scenario the question is
  ultimately about, and nothing detects or prevents it.

**Can the architecture detect, prevent, or represent these conflicts?**
No, on all three. It cannot *represent* them (no shared-edge/shared-route
data structure exists at the decision-engine level), so it cannot detect
them, so it cannot prevent them. The one place a real per-zone route *is*
computed against the real graph — `AIDecisionEngine._zone_routes()`
(`ai_decision/engine.py:222-252`), which calls `hazard_engine.nearest_exit(node.id)`
independently per zone — still computes each zone's route **in isolation**;
nothing compares the resulting `Route` objects across zones for shared
edges, and this engine is not even the one the RL pipeline runs.

---

## 4. Current RL action semantics

Verified from `ActionMapper.decode()`, `ProductionRLEnvironment._decode_action()`/
`step()`, and `LiveRLDecisionPipeline.tick()` (not inferred from naming):

The action space represents **a single building-wide target choice, applied
identically to every zone** — not a per-zone recommendation, not a
simultaneous *differentiated* assignment across zones, and not an
independent per-zone route choice. It is closest to "one whole-building
policy decision per tick" (comparable to a single incident commander calling
one instruction over the PA system), where the *individual occupant's*
route to that shared target is then computed separately and independently
per occupant by the simulation's `RecommendationAwareRouteChoiceStrategy`
(`simulation_interactive/replanning.py:159-192`) — which does real,
per-occupant `PathfindingEngine.alternative_paths()` search from that
occupant's own current node toward the recommended edge, falling back to the
occupant's base route choice if the recommended edge is unreachable from
there. That per-occupant fallback is the *only* mechanism in the entire
traced path that keeps a shared broadcast target from being nonsensical for
zones/occupants who cannot actually reach it — and it operates at
simulation-occupant granularity, entirely downstream of and invisible to the
decision engine and the RL policy itself.

---

## 5. Route knowledge

| What | Genuinely known? | Where | By whom |
|---|---|---|---|
| Which exit/stair id was chosen this tick | Yes | `ActionSpaceEntry.target_id` | RL policy (indirectly, via its own encoded choice), `RLActionDecisionEngine` |
| Which zone connects to which exit/stair via a real edge | Yes, real structural fact | `Building`/`NavigationGraph`, read via `PathfindingEngine` | `AIDecisionEngine._zone_routes()` (ground-truth engine) and `RecommendationAwareRouteChoiceStrategy` (per-occupant simulation routing) — **never** `RLActionDecisionEngine` or the RL policy |
| Corridor/edge structure between zones and exits | Yes, real | Same `NavigationGraph` | Same two consumers above |
| Whether two recommendations share physical infrastructure | **No** | — | Nothing computes this; would require comparing two `Route.edges` lists, which nothing does |
| Downstream route overlap | **No** | — | Same |
| Current stair/exit occupancy or congestion | Yes, real, but **not visible to the RL policy** | `StairCongestionPredictor` (`stair_congestion/predictor.py`), fed by real perception (`compute_asset_occupancy_snapshot`) | Only reaches `RLActionDecisionEngine._stair_congestion_note()`, purely as **after-the-fact free text** appended to `PriorityEvacuationEntry.reason`; verified absent from `PerceptionObservationSchema.feature_names` (`rl_training/perception_observation_space.py:94-167` — stair/exit features are only `is_active_recommendation_target`, a self-referential bookkeeping bit, never occupancy/congestion) |

Clear distinction: **route topology (zone↔exit/stair connectivity, corridor
structure) is real, non-fabricated structural data** — it lives in
`Building`/`NavigationGraph` and is genuinely queryable via
`PathfindingEngine`, exactly the way `AIDecisionEngine` and the simulation's
occupant-level replanning already use it. It is not simulator-only or
privileged information; it is the same building model the Designer/Builder
edits. **What is missing is not the data — it is any consumer at the
decision-engine or RL-policy level that reads it.** `RLActionDecisionEngine`
deliberately has zero access to it (a documented, tested property — see its
own comments and `GroundTruthIndependenceTests`-style guards), and the RL
observation encoder never encodes it either.

---

## 6. Occupancy and demand

Per-zone occupant estimates are real and already present:
`ZoneRecommendation.occupant_count` comes from
`observation.occupancy_observation(zone_id).estimated_count`
(`decision_engine.py:106`), a genuine perception-derived estimate, not
ground truth or a fabricated number.

**Can the system estimate demand toward a shared stair/exit, aggregated
across zones?** No such aggregation exists anywhere in the traced path.
It would require, at minimum: (a) grouping zones by their (currently
identical) `recommended_exit_edge_id`, and (b) summing `occupant_count`
within each group. Neither step is implemented. This is a genuinely small
gap to close — the raw per-zone numbers already exist honestly; only the
grouping/summation is missing — but as of today it is missing, and nothing
downstream (priority ordering, the stair congestion note, the operator
advisory adapter) performs it either.

This finding does not require introducing individual occupant tracking,
route blockage detection, or privileged simulation occupancy to close —
consistent with the constraint in Section 9 — because the inputs
(`occupant_count` per zone, `recommended_exit_edge_id` per zone) are already
perception-honest values sitting on the existing `ZoneRecommendation`
objects.

---

## 7. Relationship to staircase congestion prediction

**`StairCongestionPredictor` only ever predicts from a single stair's own
past observed occupancy trend — it has no channel for demand caused by
recommendations.**

Verified from `stair_congestion/predictor.py`:

- `StairCongestionPredictor.observe(stair_id, timestamp, occupancy, status)`
  (lines 102-206) takes a bare `(stair_id, timestamp, occupancy)` triple. Its
  own module docstring (lines 10-29) explicitly scopes it as "a standalone,
  input-agnostic perception/estimation layer" that "has no opinion about,
  and no access to, WHERE that occupancy number came from."
- The prediction itself (`_predict()`, lines 210-228) is deterministic
  linear extrapolation of the *existing observed trend*
  (`predicted_value = latest_value + rate * horizon_seconds`), where `rate`
  is derived purely from a rolling window of past observed occupancy
  samples (lines 138-185). There is no term for "N zones with total
  occupant_count M were just recommended toward this stair" — the predictor
  has no parameter through which that fact could even be passed.
- Its sole consumer, `RLActionDecisionEngine._stair_congestion_note()`
  (`decision_engine.py:152-207`), only reads an already-computed
  `StairCongestionState` per `recommended_edge_id` and turns it into hedged
  free text (`"Monitor Stair ... congestion is rising..."`). It never feeds
  anything back into the predictor, and the predictor is never told the
  policy's action at all — confirmed by `LiveRLDecisionPipeline`'s own
  documented ordering contract (`live_rl_decision_pipeline.py:50-62`): the
  RL action is selected and recorded via `set_action()` **before** stair
  congestion state is computed or published, specifically so congestion
  "has no mechanism to reach back and change an action already recorded."
  That same one-way ordering also means the reverse never happens either:
  the action taken has no path back into what the congestion predictor
  extrapolates from.

**This is the critical distinction the investigation asked to pin down:**
the existing predictor is a pure *retrospective/extrapolative* monitor of
observed stair occupancy. It cannot represent, and today has no interface
to represent, *anticipated additional demand* that a joint building-wide
recommendation might be about to create. A stair could look STABLE or even
FALLING right up until the moment a broadcast recommendation sends every
zone toward it, because the predictor's linear extrapolation has no notion
of "recommended" demand at all — only occupancy already realized in past
ticks.

---

## 8. Architectural options (smallest honest next steps)

All four options below are evaluated against the same six questions the
investigation asked for. None require retraining, multi-agent RL, or a
redesign of the action space — the existing architecture does not force
that; see the verdict in Section 10 for why.

### Option A — Joint/global action representation (e.g. `MultiDiscrete` per zone)

- Influences the RL policy itself? **Yes** — this is the one option that
  does. The action space would need to become e.g. one categorical choice
  per zone (`MultiDiscrete([n_targets] * n_zones)`), which changes
  `ActionMapper`, the environment's `action_space`, and every existing
  trained artifact's action schema.
- Requires retraining? **Yes**, unavoidably — a saved policy's output
  distribution is shaped by `action_space`; changing it invalidates every
  existing checkpoint (`training_runs/production_rl_positive_guidance_v1`
  etc.).
- Information required: same per-zone perception already encoded, no new
  data.
- Does that information genuinely exist in the live perception stack?
  Yes for the inputs; the *output* shape is new.
- Introduces route certainty or privileged information? No, but it is by
  far the largest, most invasive option of the four — explicitly the kind
  of "major redesign/retraining" the investigation asked to avoid unless
  the smaller options are genuinely impossible. They are not (see B).

### Option B — Separate recommendation coordination/allocation layer (recommended smallest step)

Insert a stateless coordination step **between** `LiveRLInferenceEngine.infer()`
(or `RLActionDecisionEngine.decide()`) and the consumer of its output,
using components that already exist:

1. Read the RL policy's single chosen target exactly as today (no policy
   change).
2. For each zone, use `PathfindingEngine.nearest_exit()` /
   `alternative_paths()` — the same, already-existing, already-used-for-this-
   exact-purpose machinery `AIDecisionEngine._zone_routes()` and
   `RecommendationAwareRouteChoiceStrategy` already call — to find whether
   that zone's real route to the RL-chosen target shares an edge with other
   zones' routes to it, and to find each zone's next-nearest alternative
   target if it does.
3. Sum each zone's already-perception-honest `occupant_count` (Section 6)
   by shared target/edge.
4. Compare that aggregate against a caller-supplied capacity/occupancy
   threshold — the same convention `StairCongestionPredictor`'s own
   `congestion_threshold` already establishes (a documented, disclosed
   placeholder assumption, not a validated life-safety standard).
5. When the aggregate exceeds the threshold, redistribute the *overflow*
   zones to their nearest reachable alternative target, and hand the
   **result** to `RLActionDecisionEngine` via `set_zone_recommendations()` —
   the exact per-zone escape hatch that already exists, is already fully
   tested, and is currently wired to nothing in production (Section 1).

Evaluated:
- Influences the RL policy itself? **No** — the policy still makes exactly
  the same single-target choice it does today; this layer only decides how
  to *distribute* that choice (or override individual zones' copy of it)
  before it reaches occupants.
- Requires retraining? **No.**
- Information required: per-zone route-to-target (real, from
  `PathfindingEngine`/`NavigationGraph`), per-zone `occupant_count` (already
  on `ZoneRecommendation`), a capacity/occupancy threshold (a disclosed
  assumption, same honesty level already accepted for
  `StairCongestionPredictor`).
- Does that information genuinely exist in the live perception stack
  today? The route/topology piece: yes, real Building/NavigationGraph data,
  already used elsewhere for exactly this kind of query. The occupancy
  piece: yes, already perception-derived and already present on
  `ZoneRecommendation`. Nothing new needs to be sensed.
- Introduces route certainty or privileged information? No more than
  `AIDecisionEngine` already relies on — `PathfindingEngine` results are
  real structural facts about the building graph, not simulator ground
  truth about hazard state or occupant behavior, and this option does not
  touch `is_reachable`'s existing disclosed-placeholder status on
  `RLActionDecisionEngine`'s own output.

### Option C — Sequential decision-making (stagger recommendations across zones/ticks)

- Influences the RL policy itself? No, if implemented as an external
  scheduler that simply delays *when* a given zone's copy of the broadcast
  is applied (e.g. via `set_zone_recommendations()` again, phased across a
  few ticks). Yes, if instead implemented by feeding the policy multiple
  sequential observations per tick to force a sequence of single-zone
  choices (that would be a real per-step contract change).
- Requires retraining? Only in the "yes" sub-case above; not in the
  external-scheduler framing.
- Information required: same as Option B, plus a notion of tick/time
  budget to stagger over.
- Genuinely available? Same as Option B.
- Introduces privileged information? No.
- Weaker than Option B on its own: staggering *when* zones receive a
  recommendation reduces simultaneity but still does not know *whether* two
  zones share a target/route unless it also does Option B's route/edge
  comparison — so in practice this is a refinement layered on top of B, not
  an independent alternative to it.

### Option D — Evaluate candidate recommendation sets against shared capacity/congestion

- A generalization of Option B: instead of only reacting to the RL policy's
  one chosen target, generate a small number of candidate *zone→target*
  assignments (e.g. "broadcast target as today" vs. "distribute zones
  across their own nearest reachable target") and score each by the same
  aggregate-demand-vs-capacity check as Option B, picking the lowest-risk
  candidate.
- Influences the RL policy itself? No — still a downstream selection among
  candidates, not a change to how the policy is trained or what it outputs.
- Requires retraining? No.
- Information required / genuinely available: identical to Option B.
- Introduces privileged information? No.
- This is strictly more machinery than Option B for the same underlying
  data; Option B (react to the RL choice, redistribute only the overflow)
  is the smaller, more directly justified first step. Option D becomes
  worth the extra complexity only if evidence later shows single-target
  broadcast plus overflow-redistribution is insufficient — e.g. because the
  RL policy's single choice is itself already a poor fit for the
  building's topology on a given tick, which Option B alone cannot fix
  (Option B can only reroute the zones that don't fit; it cannot ask the
  policy to have picked a different primary target in the first place).

---

## 9. Safety and honesty constraints — preserved by this investigation and by every option above

- No ground-truth or simulator-internal information was read to answer any
  question above; every claim is sourced to real production code
  (`rl_training/`, `ai_decision/`, `live_runtime/`, `stair_congestion/`,
  `simulation_interactive/`) or to the `Building`/`NavigationGraph`
  structural model, never to `hazard_evolution`, `ground_truth`, or a
  `HazardSnapshot`/`OccupancySnapshot` read independently of perception.
- No claim is made that SynEvac can detect a blocked route unless a real
  sensor/perception signal exists. `RLActionDecisionEngine.is_reachable`
  remains, and is explicitly documented in this report, as the same
  disclosed always-`True` placeholder it already is
  (`decision_engine.py:137-148`) — never treated here as route
  verification.
- `is_reachable` is not used anywhere in this report as evidence of route
  verification, matching its own existing disclosure.
- Stair congestion, as traced in Section 7, remains based purely on
  observable/live signals (`StairCongestionPredictor`'s own observed-
  occupancy history) — this investigation does not propose changing that;
  Option B's proposed *demand* aggregate is a distinct, separately-labeled
  quantity (perception-derived `occupant_count` toward a target), not
  folded into or presented as the same thing as observed stair occupancy.
- Any future congestion/demand comparison (Option B step 4) must remain
  hedged, exactly as `StairCongestionPredictor`'s own predictions already
  are (`exceeds_threshold`, never a certainty) — this report does not
  propose otherwise.
- Confirmed hazard vs. conservative uncertainty remains distinguishable
  exactly as `live_runtime/operator_advisory_adapter.py`'s own
  `ConservativeSafetyState` (`CONFIRMED_HAZARD` vs.
  `UNCERTAIN_CONSERVATIVELY_UNSAFE`) already keeps it — none of the options
  above touch that logic.
- All four options preserve the RL policy's existing ability to recommend
  doors/exits, stairs, or NOOP — none of them (Option A included) remove or
  narrow the target vocabulary; Option A would only change how many such
  choices are made per tick (one per zone instead of one for the building).

---

## 10. Verdict

**B — The current architecture can produce multiple zone recommendations,
but they are not jointly coordinated.** A coordination layer is needed.

The single strongest piece of evidence is structural, not incidental: the
production action space is one `Discrete` choice per tick, mechanically
broadcast to every zone (`ActionMapper.decode()`,
`ProductionRLEnvironment.step()`, `LiveRLDecisionPipeline.tick()` all
confirm this identically). This is not "independent per-zone recommendations
that happen not to be coordinated" (which might have argued for verdict C,
since fixing independent-but-uncoordinated outputs after the fact is harder)
— it is "one whole-building recommendation, applied uniformly," which is
actually an easier starting point to add coordination to than genuinely
independent per-zone RL outputs would be, precisely because a coordination
layer does not need to reconcile N independently-reasoned decisions — it
only needs to *redistribute* the one decision already made, using route and
occupancy data that already exists in the codebase (Section 5/6) and a
per-zone override mechanism (`set_zone_recommendations()`) that is already
built, tested, and simply unused in production.

This also rules out C: the architecture is not *fundamentally incapable* of
coordinated multi-zone planning without redesign/retraining. Every piece
Option B needs — per-zone route computation via `PathfindingEngine`
(proven, already used for this exact kind of query elsewhere), per-zone
occupant estimates (already perception-honest and already on
`ZoneRecommendation`), and a per-zone override channel into the decision
engine (`set_zone_recommendations()`, already implemented and tested) —
exists today. None of it requires touching the RL policy or retraining a
model.

**Recommended smallest justified next milestone (not implemented here):**
Option B — a stateless "Zone Recommendation Coordination" layer sitting
between the RL policy's single chosen target and
`RLActionDecisionEngine.set_action()`/`set_zone_recommendations()`, which
(1) computes each zone's real route to the policy's chosen target via the
already-existing `PathfindingEngine`, (2) detects when multiple zones' routes
share the target edge (or a downstream edge), (3) sums their already-honest
`occupant_count` estimates, (4) compares that aggregate to a disclosed,
caller-supplied capacity/occupancy threshold in the same spirit as
`StairCongestionPredictor.congestion_threshold`, and (5) redistributes only
the overflow zones to their own nearest reachable alternative target via the
already-built, already-tested `set_zone_recommendations()` call. This closes
the gap identified in Sections 2, 3, and 6 without retraining, without a new
action space, and without introducing any information the codebase does not
already genuinely have.

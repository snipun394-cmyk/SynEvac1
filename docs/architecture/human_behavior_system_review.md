# Human Behavior System Review

Investigation-only technical audit of SynEvac's occupant-behavior subsystem. Zero code was
changed to produce this document. Every claim below is traced to a specific file; where the
repository does not document a source or justification for a modeling choice, this report says
so explicitly rather than inferring one.

Packages read in full for this review: `behavior/`, `behavior_library/`,
`behaviour_profile_resolver/`, `human_decision_engine/`, `behavior_recognition/`, plus the
consuming/producing edges in `simulator/`, `simulation_interactive/`, `simulation_runtime/`,
`ground_truth/human_behavior.py`, `scenario/occupant.py`, `hazard/severity.py`, and
`docs/architecture/reproducibility_review.md`.

---

## PHASE 1 — Architecture

### Package inventory

**`behavior/`** — the Human Behavior Layer (orchestrator + interfaces)
- **Purpose**: Defines the three-stage Decision → Navigation(route+delay) pipeline interface and
  runs it for one occupant at a time. Owns no concrete behavior logic itself.
- **Owner concept**: "Human Behavior Layer" (per its own docstrings, part of the chain
  `Scenario Runner → SimulationContext → Behaviour Profile Resolver → Human Behavior Layer →
  Simulation`, restated in `behaviour_profile_resolver/registrar.py` and
  `docs/architecture/reproducibility_review.md`).
- **Public API** (`behavior/__init__.py`): `HumanBehaviorLayer`, `DecisionContext`,
  `BehaviorProfile`, `Role`, `BehaviorGroup`, `DecisionStrategy`/`AlwaysEvacuateDecisionStrategy`,
  `PreMovementDelayStrategy`/`NoPreMovementDelay`, `RouteChoiceStrategy`/
  `ShortestRouteChoiceStrategy`, `ActionIntent`.
- **Main classes**: `HumanBehaviorLayer` (`orchestrator.py`), `BehaviorProfile`/`Role`
  (`profile.py`), `DecisionContext` (`context.py`), `ActionIntent`/`DecisionStrategy`
  (`intent.py`), `PreMovementDelayStrategy` (`pre_movement.py`), `RouteChoiceStrategy`/
  `RouteChoice` (`route_choice.py`), `BehaviorGroup` (`group.py`).
- **Inputs**: a `NavigationGraph`, a `PathfindingEngine`, a `BehaviorProfile`, a `start_id`, and
  optionally a `HazardSnapshot`, a prior `MultiAgentSimulationResult`, and a `random.Random`
  (`context.py:17-63`).
- **Outputs**: one immutable `BehaviorDecision` per occupant, submitted to
  `simulation.submit_decision()` (`orchestrator.py:107-132`).
- **Consumed by**: `simulator.coordinator.MultiAgentSimulation` (via `submit_decision`).
- **Its own inputs are produced by**: `behaviour_profile_resolver` (which builds `BehaviorProfile`
  instances and picks concrete strategies) and `hazard_evolution`/`hazard` (which produces
  `HazardSnapshot`).

**`behavior_library/`** — the concrete strategy implementations
- **Purpose**: Every actual `DecisionStrategy`/`RouteChoiceStrategy`/`PreMovementDelayStrategy`
  implementation lives here — compliance, helping, herding, familiarity, hesitation,
  leader-following, assistance, firefighting, social-group awareness, attribute sensitivity.
- **Public API**: no `__init__.py` re-exports (file is empty); consumers import directly from each
  module (`behavior_library.decision_strategies`, `.pre_movement_strategies`,
  `.route_choice_strategies`, `.assistance_strategies`, `.firefighter_strategies`,
  `.dynamic_human_strategies`, `.dynamic_firefighter_strategies`, `.attribute_aware_strategies`).
- **Main classes**: see the full behavior inventory in Phase 2.
- **Inputs**: a `DecisionContext` per call to `decide()`/`choose()`/`delay()`.
- **Outputs**: `ActionIntent`, `RouteChoice`, or a `float` delay.
- **Consumed by**: `behaviour_profile_resolver` (wires strategies into `BehaviorProfileTemplate`
  and per-occupant wrapper chains) and `simulation_interactive` (replanning).
- **Its inputs are produced by**: `behavior/context.py`'s `DecisionContext`, populated by whichever
  caller is orchestrating (`HumanBehaviorLayer.register()`,
  `behaviour_profile_resolver.dynamic_registrar._register_with_hazard()`, or
  `simulation_interactive.replanning.replan_occupant()`).

**`behaviour_profile_resolver/`** — the registration/wiring layer
- **Purpose**: The only place a `behaviour_profile_id` string (from `ScenarioOccupant`/
  `ScenarioFirefighter`) is turned into a concrete `BehaviorProfile` + strategy set, and the only
  place per-occupant traits (attributes, groups, assistance pairings) are derived and merged in.
- **Public API** (`__init__.py`): `register_occupants`, `register_deferred_occupants`,
  `resolve_profile`/`UnknownBehaviourProfileError`, `BehaviorProfileTemplate`,
  `DEFAULT_PROFILE_REGISTRY`, `OccupantCategory`/`occupant_category`/`register_category`,
  `register_firefighters`, `DEFAULT_FIREFIGHTER_PROFILE_REGISTRY`, `register_population`.
- **Main classes/functions**: `BehaviorProfileTemplate` (`template.py`), `DEFAULT_PROFILE_REGISTRY`
  (`registry.py`), `resolve_profile` (`resolver.py`), `OccupantCategory` (`category.py`),
  `register_occupants`/`_register_one` (`registrar.py`), `OccupantAttributes`/
  `derive_occupant_attributes` (`occupant_attributes.py`), `assign_occupant_groups`
  (`occupant_grouping.py`), `DEFAULT_FIREFIGHTER_PROFILE_REGISTRY`/`register_firefighters`
  (`firefighter_registry.py`/`firefighter_registrar.py`), `register_population`
  (`combined_registrar.py`), `register_population_dynamic` (`dynamic_registrar.py`).
- **Inputs**: a `SimulationContext` (from `scenario_runner`), holding `ScenarioOccupant`/
  `ScenarioFirefighter` records and `metadata.seed`.
- **Outputs**: a populated `HumanBehaviorLayer`/`MultiAgentSimulation` (every occupant registered
  and resolved to a `BehaviorDecision`).
- **Consumed by**: `dataset_builder`, `research_framework`, `validation_framework`,
  `simulation_interactive.route_manager`, `designer/campaign`, `ground_truth`.
- **Its inputs are produced by**: `scenario_runner` (`SimulationContext` construction) and, one
  layer further up, `scenario_generator`/the Designer's authored `Scenario`.

**`human_decision_engine/`** — dynamic (in-simulation) decision-making, opt-in
- **Purpose**: Replaces *scenario-authored* assistance/rescue assignments
  (`assisting_occupant_id`/`rescue_target_occupant_id`, fixed at authoring time) with *dynamic,
  rule-based* decisions computed from structural proximity + hazard state at registration time.
  Per its own module docstring, it "contains no DecisionStrategy/RouteChoiceStrategy of its own" —
  it decides *what trait values* get written, and delegates all actual movement mechanics to the
  existing, unmodified `behavior_library` strategies.
- **Public API**: no `__init__.py` re-exports (empty file); consumers import directly.
- **Main classes**: `HumanDecisionEngine`/`CivilianDecision` (`engine.py`),
  `FirefighterDecisionEngine`/`FirefighterTaskDecision` (`firefighter_engine.py`),
  `compute_rescue_priority`/`RescuePriorityFactors` (`priority.py`),
  `compute_dynamic_pairings`/`AssistancePairing` (`pairing.py`), `GroupRegistry`/`Group`
  (`groups.py`), `DecisionEventLog`/`DecisionEvent` (`events.py`), `build_human_decisions_view`
  (`view.py`).
- **Inputs**: the civilian/firefighter roster (occupant_id, zone_id, category), an optional
  `HazardSnapshot` via `DecisionContext`, and (for the firefighter engine) externally-supplied
  `known_fallen_ids`/`known_possible_injury_ids`.
- **Outputs**: `CivilianDecision`/`FirefighterTaskDecision` objects, plus a shared
  `DecisionEventLog` and `GroupRegistry` for Command Center/dataset display.
- **Consumed by**: `behavior_library.dynamic_human_strategies`/`dynamic_firefighter_strategies`
  (which write its output onto `BehaviorProfile.traits`), and
  `behaviour_profile_resolver.dynamic_registrar` (the orchestration entry point).
- **Its inputs are produced by**: `behaviour_profile_resolver.category.occupant_category()` (roster
  classification) and the hazard layer (`HazardSnapshot`).

**`behavior_recognition/`** — a distinct, non-simulation package (perception-side, not decision-side)
- Included here per the task's "any other packages that participate in ... behavioral simulation"
  instruction, but it is important to disclose precisely what it is: **it does not participate in
  occupant decision-making at all.** It infers a *label* (`STATIONARY`/`WALKING`/`RUNNING`/
  `POSSIBLY_FALLEN`/`UNKNOWN`) for a *camera-tracked* human from bounding-box motion history alone
  — used on the live/perception side (`live_camera_pipeline`), never by the simulated occupant
  decision pipeline described above. Its own docstring (`recognizer.py:9-21`) states it is
  "Completely independent of AI, BuildingState, Command Center, Advisory, RTSP, and any YOLO
  backend" and is mechanically enforced by `tests/test_behavior_recognition_architecture_guards.py`.
  It explicitly excludes HELPING/PANIC/LEADERSHIP/DISABILITY/INJURY/CONFUSION/FOLLOWING/HERDING as
  "either a multi-person social inference or an interpretation of WHY a person is moving a certain
  way — neither is available from a single track's own position history"
  (`behavior_recognition/observation.py:43-49`). Mentioned again in Phase 6 for clarity.

**Packages investigated and found to be *out of scope*** (they sit adjacent to "decision" in name
but do not model occupant behavior):
- **`ai_decision/`** / **`decision_policy/`** — post-hoc *operator-recommendation* engines
  (`AIDecisionEngine`, `generate_policy`) that consume a completed simulation's hazard/occupancy
  snapshots to recommend zone/exit/rescue guidance to Command Center operators. They do not
  influence, and are not influenced by, any simulated occupant's decision-making.
- **`human_evidence/`** — a live-perception classification/state reconciliation bridge
  (`HumanEvidence`, `reconcile_classification`/`reconcile_state`) between camera observations and
  `LiveOccupant` state. It never touches `behavior/`, `behavior_library/`, or
  `behaviour_profile_resolver/`.

### The complete behavior pipeline

```
ScenarioOccupant / ScenarioFirefighter (authored: behaviour_profile_id, zone_id,
    assisting_occupant_id, assistance_type, rescue_target_occupant_id, ...)
            |
            v
scenario_runner.SimulationContext  (bundles occupants + firefighters + metadata.seed
                                     + an empty MultiAgentSimulation + PathfindingEngine)
            |
            v
behaviour_profile_resolver
   - resolver.resolve_profile(behaviour_profile_id, registry) -> BehaviorProfileTemplate
   - occupant_attributes.derive_occupant_attributes(seed, occupant_id, profile_id)
        -> OccupantAttributes (deterministic per-occupant physical/behavioral spread)
   - occupant_grouping.assign_occupant_groups(scenario) -> social Family/Friends/... groups
   - registrar._assistance_traits_by_occupant_id() / _helping_likelihood_traits_by_id()
        -> HELPER/ASSISTED trait pairs
   - registrar._register_one() builds a BehaviorProfile, wraps the template's
     decision/route/pre-movement strategies in attribute-aware / social-group-aware /
     assistance-aware / crowd-following-aware wrapper strategies
            |
            v
behavior.orchestrator.HumanBehaviorLayer.register(start_id, profile, decision_strategy,
                                                    route_choice_strategy, pre_movement_strategy)
   Stage 1 - Decision:   decision_strategy.decide(context)      -> ActionIntent
   Stage 2 - Navigation: route_choice_strategy.choose(context)  -> RouteChoice   (if movement required)
             Pre-move:   pre_movement_strategy.delay(context)   -> float seconds
            |
            v
simulator.decision.BehaviorDecision (immutable) -> simulation.submit_decision()
            |
            v
simulator.coordinator.MultiAgentSimulation
   - add_occupant() plans/accepts a fixed Route (no further rerouting inside this class)
   - discrete-event queueing over shared edge capacity (DefaultCapacityModel) and
     congestion-based speed slowdown (DefaultCongestionModel)
   - produces MultiAgentSimulationResult: per-occupant OccupantTimeline, arrival times,
     peak edge/node occupancy, queue events
            |
            v
   (optional, opt-in) simulation_interactive.route_manager.RouteManager.maybe_replan()
   re-runs Decision -> Navigation for one occupant, at AT_NODE/STATIONARY points only, when
   an engineering change, an operator recommendation, a broadcast, or a hazard-signature
   change is observed — via simulation_interactive.replanning.replan_occupant()
            |
            v
ground_truth.human_behavior.compute_occupant_behavior() (post-hoc, from the completed
   MultiAgentSimulationResult + tick-by-tick hazard snapshots) derives DynamicHumanState
   (WALKING/RUNNING/WAITING/FALLEN/POSSIBLE_INJURY/HELPING/BEING_ASSISTED) per occupant
            |
            v
dataset_builder / Command Center / research_framework (consumers of the finished record)
```

A parallel, **opt-in** path exists (`behaviour_profile_resolver.dynamic_registrar` +
`human_decision_engine`) that replaces the scenario-authored assistance/rescue *assignment* step
with a dynamically-computed one (structural pairing + hazard-aware decision), while reusing every
downstream stage (`behavior_library` strategies, `MultiAgentSimulation`) unmodified.

---

## PHASE 2 — Complete behavior inventory

For each behavior: location, mechanism, math, randomness, parameters, limitations/assumptions,
determinism.

### 1. Compliance (with an evacuation instruction)
- **Where**: `behavior_library/decision_strategies.py::ComplianceDecisionStrategy`;
  gated a second time by `behavior_library/attribute_aware_strategies.py::
  AttributeAwareComplianceDecisionStrategy`.
- **How**: draws `roll = rng.random()`; if `roll <= profile.compliance_level`, delegates to a
  compliant sub-strategy (default `AlwaysEvacuateDecisionStrategy`), else to a noncompliant one
  (default `AlwaysWaitDecisionStrategy`).
- **Math**: single Bernoulli trial, `p = compliance_level` (a static per-profile scalar, e.g. 0.9
  for `Adult_Default`, 0.5 for `Child_Default` — `registry.py:40-107`), or `p = compliance` (a
  per-occupant, category-ranged uniform draw from `occupant_attributes.py`, used only for profiles
  that have no existing `ComplianceDecisionStrategy`, e.g. `Wheelchair_Default` —
  `attribute_aware_strategies.py:190-241`).
- **Randomness**: stochastic, one Bernoulli draw per occupant per registration/replan.
- **Configurable parameters**: `compliance_level` (per `BehaviorProfileTemplate`),
  `compliant_strategy`/`noncompliant_strategy` (composable).
- **Limitations/assumptions**: a static, author-chosen probability, not derived from any
  compliance-behavior study; two independent compliance gates exist (template-level and
  attribute-level) but are explicitly prevented from compounding via an `already_gated` flag
  (`attribute_aware_strategies.py:202-216`) — disclosed as a deliberate anti-double-penalty design,
  not an oversight.
- **Deterministic or stochastic**: stochastic, but reproducible given a fixed per-occupant seed
  (see Phase 2's cross-cutting reproducibility note below).

### 2. Pre-movement delay (probabilistic / lognormal)
- **Where**: `behavior_library/pre_movement_strategies.py::ProbabilisticPreMovementDelay`.
- **How**: `rng.lognormvariate(mu, spread)` with `mu = log(median_delay)`.
- **Math**: lognormal distribution, parameterized by a **median** delay (seconds) and a **spread**
  (the underlying normal's sigma). Its own comment: "lognormal is the shape commonly used in
  evacuation literature for human response/pre-movement time, but this is a simple, two-parameter
  draw, not a validated RSET pre-movement-time model."
- **Randomness**: stochastic.
- **Configurable parameters**: `median_delay`, `spread`; used per-profile, e.g. 30s for Adult, 45s
  for Child, 50s/spread 0.6 for Elderly, 60s/spread 0.7 for Wheelchair, 40s for Visitor
  (`registry.py`).
- **Limitations/assumptions**: the median/spread values are documented as "illustrative,
  documented starting points," not measured data.
- **Deterministic/stochastic**: stochastic, reproducible.

### 3. Hesitation (route-ambiguity-driven pre-movement delay)
- **Where**: `behavior_library/pre_movement_strategies.py::HesitationPreMovementDelayStrategy`.
- **How**: queries `PathfindingEngine.alternative_paths()` for up to `max_alternatives` routes;
  counts how many are "tied" with the cheapest (`total_cost` within `tie_margin` fraction of the
  minimum); adds `hesitation_per_tied_option` seconds per tied option beyond the first, capped at
  `max_hesitation`.
- **Math**: `hesitation = min((tied_count - 1) * hesitation_per_tied_option, max_hesitation)`.
- **Randomness**: **none** — entirely deterministic given the current graph/cost-model state; its
  own comment states this explicitly ("Entirely deterministic — no `random` anywhere").
- **Configurable parameters**: `max_alternatives` (5), `tie_margin` (0.15),
  `hesitation_per_tied_option` (3.0s), `max_hesitation` (20.0s).
- **Limitations/assumptions**: models hesitation as *genuine route ambiguity*, explicitly not "at
  a blocked exit" — the module's own comment explains traversability is baked into Dijkstra
  relaxation itself, so there is "no honest way ... to ask 'would my preferred exit have been
  reachable if nothing were blocked.'" An honestly-scoped simplification, not a full
  decision-conflict model.
- **Deterministic/stochastic**: deterministic.

### 4. Follow-leader pre-movement delay
- **Where**: `behavior_library/pre_movement_strategies.py::FollowLeaderPreMovementDelayStrategy`.
- **How**: reads `leader_occupant_id` trait; looks up the leader's already-resolved
  `BehaviorDecision.depart_time` in `context.decisions_so_far`; returns
  `max(0, leader_depart_time) + follow_gap`.
- **Math**: linear offset from the leader's own departure time.
- **Randomness**: none.
- **Configurable parameters**: `follow_gap` (default 1.0s).
- **Limitations/assumptions**: requires the leader to already be registered/resolved (an ordering
  dependency enforced upstream by `_ordered_for_assistance()`); falls back gracefully if not.
- **Deterministic/stochastic**: deterministic.

### 5. Attribute-aware pre-movement delay (reaction speed / panic susceptibility scaling)
- **Where**: `behavior_library/attribute_aware_strategies.py::
  AttributeAwarePreMovementDelayStrategy`.
- **How**: multiplies a base delay (from whatever fallback strategy runs first, typically
  `ProbabilisticPreMovementDelay`) by `pre_movement_delay_multiplier`, itself computed in
  `occupant_attributes.py::attribute_traits()`.
- **Math**: `multiplier = (1.4 - 0.8*reaction_speed) * (0.7 + 0.6*panic_susceptibility)`, floored
  at 0.2.
- **Randomness**: none at this layer (the randomness is upstream, in the attribute draw itself,
  Behavior 12 below); this layer is a deterministic function of already-derived attributes.
- **Limitations/assumptions**: hand-authored functional form combining two independently-drawn
  [0,1] attributes into one multiplier; not validated against human-factors data.
- **Deterministic/stochastic**: deterministic given the attributes.

### 6. Attribute-sensitivity-aware pre-movement delay (route familiarity / smoke & visibility tolerance / risk aversion / panic scaling of hesitation)
- **Where**: `behavior_library/attribute_aware_strategies.py::
  AttributeSensitivityAwarePreMovementDelayStrategy` + `_sensitivity_multiplier()`.
- **How**: computes a `[0.5, 2.0]` multiplier by averaging deviation-from-neutral across five
  `[0,1]` attributes (`route_familiarity`, `smoke_tolerance`, `visibility_tolerance`,
  `risk_aversion`, `panic_susceptibility`, each defaulting to neutral 0.5), then scales *only* the
  hesitation component from a nested `HesitationPreMovementDelayStrategy`.
- **Math**: `multiplier = 0.5 + 1.5 * average(familiarity_component, tolerance_component,
  risk_component, panic_component)` (see `_sensitivity_multiplier`, lines 247-268).
- **Randomness**: none at this layer.
- **Limitations/assumptions**: only has an effect when `HesitationPreMovementDelayStrategy` itself
  returns a nonzero value (i.e., only when genuine route ambiguity exists) — a documented
  narrow-scope decision, not a general panic/fear delay model.
- **Deterministic/stochastic**: deterministic given the attributes.

### 7. Social-group-aware pre-movement delay ("wait for your group leader")
- **Where**: `behavior_library/attribute_aware_strategies.py::
  SocialGroupAwarePreMovementDelayStrategy`, composed with `FollowLeaderPreMovementDelayStrategy`.
- **How**: reads `social_leader_occupant_id`; if present, waits for that leader like Behavior 4
  above.
- **Deterministic/stochastic**: deterministic (the delay math), but *whether* a group forms at all
  is itself stochastic (see Behavior 15, Social Groups).

### 8. Route choice: shortest path (default / baseline)
- **Where**: `behavior/route_choice.py::ShortestRouteChoiceStrategy`.
- **How**: `context.engine.nearest_exit(context.start_id)`.
- **Randomness**: none. Deterministic.

### 9. Route choice: familiarity-based
- **Where**: `behavior_library/route_choice_strategies.py::FamiliarityBasedRouteChoiceStrategy`.
- **How**: among up to `max_alternatives` distinct routes to Outside, scores each by the mean
  `BehaviorProfile.familiarity` value across its nodes, and picks the highest-scoring one.
- **Math**: `score(route) = mean(familiarity.get(node_id, 0.0) for node_id in route.node_ids)`.
- **Randomness**: none.
- **Limitations**: bounded to `max_alternatives` (default 3) candidates, "not exhaustive"
  (documented); degrades to shortest-route when no familiarity data exists.
- **Assumption**: familiarity itself has no in-repo generator today outside test fixtures/registry
  defaults (e.g. Staff having above-average familiarity) — `BehaviorProfileTemplate.familiarity`
  is documented as the only source, since `ScenarioOccupant` carries no familiarity data
  (`template.py:36-41`).
- **Deterministic/stochastic**: deterministic.

### 10. Route choice: static herding
- **Where**: `behavior_library/route_choice_strategies.py::StaticHerdingRouteChoiceStrategy`.
- **How**: tallies which Exit *edge* the majority of already-resolved occupants
  (`context.decisions_so_far`, this registration session only) actually used; with probability
  `follow_probability`, steers this occupant toward that majority exit (if reachable within
  `max_alternatives` candidates), else falls back to shortest route.
- **Math**: `Counter` over `decision.route.edges[-1].id`; `most_common(1)`; a single Bernoulli gate
  (`rng.random() > follow_probability` bypasses following).
- **Randomness**: stochastic (the follow/no-follow gate).
- **Configurable parameters**: `follow_probability` (default 1.0), `max_alternatives` (5).
- **Limitations/assumptions**: explicitly "static" — reads one fixed snapshot of this session's
  own decisions only, never a multi-pass/adaptive read of `context.prior_result` (documented as a
  future "Dynamic Herding" extension point, not implemented). Depends on registration order.
- **Deterministic/stochastic**: stochastic, reproducible.

### 11. Route choice: crowd-following (per-occupant tendency)
- **Where**: `behavior_library/attribute_aware_strategies.py::
  CrowdFollowingAwareRouteChoiceStrategy`.
- **How**: delegates to a freshly-built `StaticHerdingRouteChoiceStrategy`, using the occupant's
  own `crowd_following_tendency` attribute as `follow_probability` instead of one shared constant.
- **Randomness**: stochastic (inherits Behavior 10's Bernoulli gate, parameterized per-occupant).
- **Deterministic/stochastic**: stochastic, reproducible.

### 12. Load balancing (anti-herding)
- **Where**: `behavior_library/route_choice_strategies.py::LoadBalancingRouteChoiceStrategy`.
- **How**: among up to `max_alternatives` candidates, scores each by
  `(assigned_count + 1) / effective_capacity` (using `Edge.capacity` when modeled, else an
  undivided count of 1), and picks the least-loaded, cost-tiebroken candidate.
- **Randomness**: none.
- **Limitations**: an explicit *anti*-herding counterpart to Behavior 10, not a validated
  wayfinding-under-crowding model; ties resolve toward the cheaper route.
- **Deterministic/stochastic**: deterministic.

### 13. Hazard-aware route choice
- **Where**: `behavior_library/route_choice_strategies.py::HazardAwareRouteChoiceStrategy`.
- **How**: among up to `max_alternatives` candidates, scores each by
  `route.total_cost + hazard_score_weight * sum(hazard_score at every node on the route)`, picks
  the minimum.
- **Math**: linear cost/hazard tradeoff, `hazard_score_weight` default 50.0.
- **Randomness**: none.
- **Limitations**: only active once a `HazardSnapshot` is supplied to `DecisionContext` (never true
  in the base `HumanBehaviorLayer.register()` path — see Phase 6); falls back to shortest-route
  when absent.
- **Deterministic/stochastic**: deterministic.

### 14. Leader/follower (route + departure timing)
- **Where**: `behavior_library/route_choice_strategies.py::FollowLeaderRouteChoiceStrategy` +
  Behavior 4 above.
- **How**: reads `leader_occupant_id`; copies the leader's already-resolved `goal_id`/`route`
  wholesale.
- **Randomness**: none.
- **Limitations**: requires the leader already registered/resolved (ordering dependency);
  falls back to shortest route if not.
- **Deterministic/stochastic**: deterministic.

### 15. Social groups (families/friends/coworkers/school groups/etc.)
- **Where**: `behaviour_profile_resolver/occupant_grouping.py::assign_occupant_groups`.
- **How**: per zone, shuffles occupants with a seeded RNG; a fraction `_GROUPED_FRACTION = 0.65`
  are marked "sociable"; sociable occupants are chunked into groups of size 2–4
  (`_MIN_GROUP_SIZE`/`_MAX_GROUP_SIZE`); a leader is chosen as the member with highest `leadership`
  attribute (tie-broken by occupant_id); group "type" (Family/School Group/Hospital Staff/Visitors
  Together/Friends) is inferred from the mix of `OccupantCategory` present.
- **Math/randomness**: `random.Random(seed)`, `rng.shuffle`, per-member Bernoulli inclusion at
  `p=0.65`, `rng.randint(2, min(4, remaining))` for group size.
- **Configurable parameters**: `_MIN_GROUP_SIZE`, `_MAX_GROUP_SIZE`, `_GROUPED_FRACTION`.
- **Limitations/assumptions**: purely zone-and-seed driven, not authored by a scenario designer;
  documented as "illustrative... not measured data."
- **Deterministic/stochastic**: stochastic but fully reproducible from `(scenario_seed)` alone (one
  shared RNG stream for the whole grouping pass, distinct from the per-occupant attribute stream).

### 16. Assistance behaviors (HELP / ESCORT / PUSH_WHEELCHAIR / DRAG / CARRY)
- **Where**: `behavior_library/assistance_strategies.py` (`AssistanceAwareDecisionStrategy`,
  `AssistanceAwareRouteChoiceStrategy`).
- **How**: a helper's own `walking_speed` is multiplied by a fixed, per-assistance-type factor
  (`_SPEED_MULTIPLIER_BY_ASSISTANCE_TYPE`: HELP 0.85, ESCORT 0.80, PUSH_WHEELCHAIR 0.75, DRAG 0.50,
  CARRY 0.55) applied to a once-captured `base_walking_speed` (idempotent across repeated calls);
  the assisted occupant's route/speed is then set to exactly the helper's resolved route/speed
  (recomputed as the shortest sub-path from the assisted occupant's own start to the helper's
  goal — proven to be the exact tail of the helper's route, not an approximation).
- **Randomness**: none — the multipliers are fixed constants.
- **Assignment**: either scenario-authored (`ScenarioOccupant.assisting_occupant_id`/
  `assistance_type`) or dynamically decided (`human_decision_engine`, Behavior 21 below), or
  attribute-driven (a group leader with `helping_likelihood >= 0.6` opportunistically helps their
  own group's least-mobile member — `registrar.py::_helping_likelihood_traits_by_id`, threshold
  `_HELPING_LIKELIHOOD_THRESHOLD = 0.6`).
- **Limitations/assumptions**: explicitly documented as "pure cost to the helper's own pace," "not
  a validated mobility-impact model."
- **Deterministic/stochastic**: deterministic given who is paired with whom (the pairing decision
  itself may be stochastic — see Behavior 21).

### 17. Basic helping (opt-in, non-proximity-based)
- **Where**: `behavior_library/decision_strategies.py::BasicHelpingDecisionStrategy`.
- **How**: reads `helping_occupant_id` trait; if set, returns `ActionType.HELP` targeting that id;
  otherwise falls back.
- **Randomness**: none.
- **Limitations**: explicitly scoped as "basic" — no proximity, capability, capacity-limit, or
  mutual-consent reasoning; its own comment states a "more sophisticated helping model ... would be
  a *different* DecisionStrategy."
- **Deterministic/stochastic**: deterministic.

### 18. Firefighter search / rescue / guide / report-hazard
- **Where**: `behavior_library/firefighter_strategies.py::FirefighterDecisionStrategy`/
  `FirefighterRouteChoiceStrategy`; task variants added by
  `behavior_library/dynamic_firefighter_strategies.py`.
- **How**: a firefighter with an assigned `rescue_target_zone_id` gets a combined route
  (entry point → target zone → nearest exit, stitched by `_combined_rescue_route()`) and a fixed
  0.70x speed multiplier for the whole trip; one without an assignment "searches" by walking toward
  the nearest exit from their entry point (documented as "the only 'which area to cover' signal
  available without a real search algorithm").
- **Randomness**: none in the static path. In the dynamic path
  (`FirefighterDecisionEngine._best_candidate`), task assignment is a deterministic
  highest-priority-score selection (ties broken by occupant_id), not randomized.
- **Limitations/assumptions**: "searching rooms" is explicitly disclosed as a documented
  simplification, not a real search algorithm; walking speed/no-pre-movement-delay values are
  disclosed as "not a validated fire-service performance figure."
- **Deterministic/stochastic**: deterministic.

### 19. Rescue priority scoring
- **Where**: `human_decision_engine/priority.py::compute_rescue_priority`.
- **How**: a fixed-weight linear score: category base weight (Child 0.25, Elderly 0.20,
  Wheelchair 0.20, Visitor 0.05, others 0.0) plus wheelchair (+0.15), possible injury (+0.30),
  fallen (+0.35), smoke exposure (×0.25), distance-from-hazard inverse (×0.15), inaccessibility
  (×0.10), isolation (×0.15).
- **Math**: see `priority.py:64-82`; every weight is a fixed, documented, never-learned constant.
  Its own comment: "the same 'illustrative, not validated' honesty this codebase applies to every
  other scoring threshold."
- **Randomness**: none — proven deterministic by construction (two calls with identical inputs
  always return the identical score).
- **Deterministic/stochastic**: deterministic.

### 20. Own-safety hesitation before helping (hazard-gated)
- **Where**: `human_decision_engine/engine.py::HumanDecisionEngine._evaluate_helper`.
- **How**: reads the would-be helper's own zone hazard severity; at `HazardSeverity.HIGH` or above,
  returns `DELAY` ("hazard_present_hesitates"); at `CRITICAL`, returns `IGNORE`
  ("own_safety_hazard_critical"); otherwise `HELP`.
- **Randomness**: none — a fixed severity-ordinal threshold gate.
- **Limitations/assumptions**: distance is captured only structurally (same-zone-only pairing,
  Behavior 21); "relationship" (a named future extension in the source phase) has no signal
  available and is explicitly not modeled ("never faked").
- **Deterministic/stochastic**: deterministic.

### 21. Dynamic assistance pairing ("who could help whom")
- **Where**: `human_decision_engine/pairing.py::compute_dynamic_pairings`.
- **How**: purely structural — pairs occupants who start in the *same zone*: mobile helper
  categories (Adult/Staff/Fire Warden) to wheelchair-user targets (`PAIRING_ASSIST`) and to
  child/elderly targets (`PAIRING_GROUP_FOLLOW`), one helper claimed by at most one target,
  deterministic by construction (sorted by zone_id/occupant_id, never by dict/set iteration order).
- **Randomness**: none.
- **Limitations/assumptions**: "nearby" is modeled as "starts in the same zone" — explicitly "a
  documented simplification, not a validated proxemics model."
- **Deterministic/stochastic**: deterministic.

### 22. Broadcast override (shelter-in-place / evacuate-immediately)
- **Where**: `simulation_interactive/replanning.py::BroadcastAwareDecisionStrategy`.
- **How**: reads `broadcast_override` trait; `"SHELTER_IN_PLACE"` forces `ActionType.WAIT`,
  `"EVACUATE_IMMEDIATELY"` forces `ActionType.EVACUATE`, otherwise delegates to the base strategy.
- **Randomness**: none.
- **Deterministic/stochastic**: deterministic.

### 23. Recommendation-aware rerouting (operator/signage-driven)
- **Where**: `simulation_interactive/replanning.py::RecommendationAwareRouteChoiceStrategy`.
- **How**: reads `recommended_exit_edge_id`/`recommended_stair_edge_id` traits (set externally,
  e.g. by dynamic signage / Command Center); if the recommended edge appears among the top-5
  alternative routes, routes through it; else falls back.
- **Randomness**: none.
- **Deterministic/stochastic**: deterministic.

### 24. Ignore alarm / non-compliance outcome
- **Where**: `behavior_library/decision_strategies.py::AlwaysWaitDecisionStrategy`/
  `AlwaysIgnoreDecisionStrategy` — the "otherwise" branch of Behavior 1 (Compliance).
- **How**: fixed `ActionType.WAIT`/`ActionType.IGNORE`, `requires_movement=False`.
- **Note**: `ActionType`'s own docstring states "Simulation never branches on this value" — WAIT
  and IGNORE are behaviorally identical to the coordinator (both are simply non-movement), the
  distinction is descriptive metadata only.
- **Deterministic/stochastic**: deterministic (the outcome, given a compliance roll already made).

### 25. Occupant physical/behavioral attribute generation (the substrate under most of the above)
- **Where**: `behaviour_profile_resolver/occupant_attributes.py::derive_occupant_attributes`.
- **How**: `random.Random(sha256(f"{seed}|behaviour_profile_resolver.occupant_attributes|
  {occupant_id}"))`; draws 14 independent `rng.uniform(low, high)` values (walking_speed_multiplier,
  reaction_speed, stamina, smoke_tolerance, visibility_tolerance, fatigue_resistance,
  mobility_factor, leadership, risk_aversion, route_familiarity, compliance, helping_likelihood,
  panic_susceptibility, crowd_following_tendency) from a category-conditioned range table
  (`_RANGES_BY_CATEGORY`, one row per `OccupantCategory`).
- **Randomness**: stochastic, fully reproducible per `(seed, occupant_id)` — independent of
  registration order or which other occupants exist.
- **Limitations/assumptions**: every range is disclosed as "illustrative, documented starting
  points ... not measured real-world data." Wheelchair users are given a near-zero
  `mobility_factor` range (0.0–0.15) as "the disclosed signal for 'cannot use stairs'" — but the
  module's own comment states plainly: "This module only ever generates and exposes the value; it
  does not itself restrict routing" — i.e. **a wheelchair user's inability to use stairs is a
  disclosed data value, not an enforced pathfinding constraint** anywhere in this pipeline.
- **Deterministic/stochastic**: stochastic, reproducible.

### Cross-cutting reproducibility note (applies to every stochastic behavior above)
`docs/architecture/reproducibility_review.md` documents that this system previously had a real bug:
strategy instances in `DEFAULT_PROFILE_REGISTRY` were constructed once at module-import time with
no `rng=` argument, so every occupant sharing a profile shared one unseeded, mutable
`random.Random()` — non-reproducible and order-dependent. The fix (already applied, not part of
this review's own findings) added an optional `DecisionContext.rng` field, threaded a
per-occupant, seed-derived `random.Random` through `HumanBehaviorLayer.register()`, and updated
`ComplianceDecisionStrategy`/`ProbabilisticPreMovementDelay`/`StaticHerdingRouteChoiceStrategy` to
prefer `context.rng` over `self.rng`. Every stochastic behavior in this inventory now draws from a
seed deterministically derived as `sha256(f"{scenario_seed}|<namespace>|{occupant_id}")`,
independently re-implemented in at least four places (`registrar.py`, `occupant_attributes.py`,
`occupant_grouping.py`, `ground_truth/human_behavior.py`, `simulation_interactive/replanning.py`)
rather than shared from one helper — a repeated, deliberate pattern, not duplication by accident.

---

## PHASE 3 — Decision process trace (one occupant, alarm to evacuation)

```
Alarm / scenario start
   |
   v
[REGISTRATION] behaviour_profile_resolver.registrar._register_one()
   - resolve_profile(occupant.behaviour_profile_id, registry) -> BehaviorProfileTemplate
   - derive_occupant_attributes(seed, occupant_id, profile_id) -> OccupantAttributes
   - assign_occupant_groups(context) -> GroupAssignment (may already be resolved earlier
     in the batch, since grouping is computed once for the whole scenario)
   - assistance/helping-likelihood traits merged into BehaviorProfile.traits
   - BehaviorProfile constructed (walking_speed, familiarity, compliance_level, role, traits)
   |
   v
[PRE-MOVEMENT] wrapped pre_movement_strategy chain (registrar.py:335-362):
   FollowLeaderPreMovementDelayStrategy (if ASSISTED)   -- OR --
   SocialGroupAwarePreMovementDelayStrategy
     -> AttributeAwarePreMovementDelayStrategy
       -> AttributeSensitivityAwarePreMovementDelayStrategy
         -> template.pre_movement_strategy (e.g. ProbabilisticPreMovementDelay)
   [Called at Stage 2 below, but the chain is fixed here at registration]
   |
   v
[DECISION] HumanBehaviorLayer.register() Stage 1 (orchestrator.py:53)
   decision_strategy.decide(context) -- wrapped chain (registrar.py:378-384):
   AttributeAwareComplianceDecisionStrategy
     -> AssistanceAwareDecisionStrategy
       -> template.decision_strategy (e.g. ComplianceDecisionStrategy)
         -> AlwaysEvacuateDecisionStrategy / AlwaysWaitDecisionStrategy
   Returns ActionIntent(action_type, requires_movement, metadata)
   |
   +-- requires_movement == False --> BehaviorDecision(no goal/route) --> STATIONARY
   |
   v (requires_movement == True)
[ROUTE SELECTION] HumanBehaviorLayer.register() Stage 2 (orchestrator.py:56-58)
   route_choice_strategy.choose(context) -- wrapped chain (registrar.py:385-389):
   AssistanceAwareRouteChoiceStrategy
     -> SocialGroupAwareRouteChoiceStrategy
       -> CrowdFollowingAwareRouteChoiceStrategy
         -> template.route_choice_strategy (e.g. ShortestRouteChoiceStrategy)
   Returns RouteChoice(goal_id, route) via PathfindingEngine.nearest_exit()/alternative_paths()
   |
   v
[PRE-MOVEMENT DELAY EVALUATED] pre_movement_strategy.delay(context) -> float seconds
   effective_speed = walking_speed * effective_walking_speed_multiplier
                     (skipped for ASSISTED occupants -- see orchestrator.py:100-105)
   |
   v
BehaviorDecision(occupant_id, action_type, start_id, goal_id, route, walking_speed,
                  depart_time = base_depart_time + delay, route_unavailable) -- immutable
   |
   v
simulation.submit_decision(decision) -> MultiAgentSimulation._register()/.add_occupant()
   |
   v
[MOVEMENT] simulator.coordinator.MultiAgentSimulation (discrete-event)
   - TRY_ENTER_EDGE: admitted if edge occupancy < capacity, else QUEUED
   - on admission: speed_factor = congestion_model.speed_factor(edge, other_occupants,
     capacity, opposing_occupants) -- linear degradation, floored at 0.3x
   - effective_speed = walking_speed * speed_factor; duration = distance / effective_speed
   - ARRIVE_AT_NODE: advances current_edge_index, dequeues next occupant if any
   |
   v (only if RouteManager is active -- interactive/live path, not the batch pipeline)
[REPLANNING] simulation_interactive.route_manager.RouteManager.maybe_replan()
   Triggered ONLY at AT_NODE/STATIONARY decision points (never mid-edge), when:
     - an edge on the occupant's remaining route changed this tick, OR
     - a new recommendation was posted for the occupant's current zone, OR
     - a new broadcast was posted (zone-scoped or global), OR
     - the occupant's own hazard signature changed AND their route strategy is
       HazardAwareRouteChoiceStrategy
   -> simulation_interactive.replanning.replan_occupant() re-runs Decision+Navigation+
      Pre-movement against a fresh DecisionContext (with hazard_snapshot, prior_result),
      submits a new BehaviorDecision that supersedes the old one (generation-counter based)
   |
   v
ARRIVED (reached final node) / UNREACHABLE (no route ever existed) / STATIONARY (never moved)
   |
   v
[POST-HOC] ground_truth.human_behavior.compute_occupant_behavior()
   - per-step: WAITING (if queued), FALLEN (seeded Bernoulli, gated on queue_wait_time >= 20s
     AND hazard_score >= 0.5), POSSIBLE_INJURY (seeded Bernoulli given FALLEN),
     RUNNING (hazard_score >= 0.35) vs WALKING otherwise
   - ever_helping / ever_assisted (from scenario assistance fields)
   - summarized into one DynamicHumanState by fixed priority order
   |
   v
Evacuation record (dataset_builder / Command Center / research_framework)
```

**Every function/class involved**, in call order: `resolve_profile` → `derive_occupant_attributes`
→ `attribute_traits` → `assign_occupant_groups` → `_assistance_traits_by_occupant_id` /
`_helping_likelihood_traits_by_id` → `BehaviorProfile.__init__` →
`AttributeAwareComplianceDecisionStrategy.decide` → `AssistanceAwareDecisionStrategy.decide` →
(profile's own `decision_strategy.decide`, e.g. `ComplianceDecisionStrategy.decide`) →
`AssistanceAwareRouteChoiceStrategy.choose` → `SocialGroupAwareRouteChoiceStrategy.choose` →
`CrowdFollowingAwareRouteChoiceStrategy.choose` → (profile's own `route_choice_strategy.choose`) →
`PathfindingEngine.nearest_exit`/`alternative_paths` → (pre-movement chain, as above) →
`HumanBehaviorLayer.register` → `BehaviorDecision.__init__` →
`MultiAgentSimulation.submit_decision` → `add_occupant`/`_register` →
`OccupantSimulator.simulate_to_goal`/`evacuate` (only if no route was already attached) →
`_handle_try_enter_edge` → `_admit_onto_edge` → `DefaultCapacityModel.capacity` →
`DefaultCongestionModel.speed_factor` → `_handle_arrive_at_node` → (optionally)
`RouteManager.maybe_replan` → `replan_occupant` → `compute_occupant_behavior`.

---

## PHASE 4 — Behavior profiles

### Civilian profiles (`behaviour_profile_resolver/registry.py::DEFAULT_PROFILE_REGISTRY`)
| Profile id | Walking speed (m/s) | Compliance | Role | Decision strategy | Pre-movement |
|---|---|---|---|---|---|
| `Adult_Default` | 1.2 | 0.9 | INDEPENDENT | ComplianceDecisionStrategy | lognormal, median 30s |
| `Child_Default` | 0.9 | 0.5 | FOLLOWER | ComplianceDecisionStrategy (noncompliant→WAIT) | lognormal, median 45s |
| `Elderly_Default` | 0.75 | 0.85 | INDEPENDENT | ComplianceDecisionStrategy | lognormal, median 50s, spread 0.6 |
| `Wheelchair_Default` | 0.7 | 0.9 | INDEPENDENT | AlwaysEvacuateDecisionStrategy | lognormal, median 60s, spread 0.7 |
| `Staff_Default` | 1.3 | 1.0 | LEADER | AlwaysEvacuateDecisionStrategy | none |
| `FireWarden_Default` | 1.3 | 1.0 | LEADER | BasicHelpingDecisionStrategy | none |
| `Visitor_Default` | 1.1 | 0.6 | INDEPENDENT | ComplianceDecisionStrategy | lognormal, median 40s |

### Firefighter profile (`behaviour_profile_resolver/firefighter_registry.py::
DEFAULT_FIREFIGHTER_PROFILE_REGISTRY`, separate registry, separate resolution path)
| Profile id | Walking speed | Compliance | Role | Decision | Route | Pre-movement |
|---|---|---|---|---|---|---|
| `Firefighter_Default` | 1.4 | 1.0 | LEADER | FirefighterDecisionStrategy | FirefighterRouteChoiceStrategy | none |

### How profiles are stored
As `BehaviorProfileTemplate` dataclass instances (`template.py`) in a plain
`MappingProxyType[str, BehaviorProfileTemplate]` — one shared, frozen registry object per registry
type. A **single template instance, including its strategy objects, is shared across every
occupant** resolved to the same `behaviour_profile_id` (explicitly documented, and the reason the
reproducibility fix above was needed).

### How profiles are assigned
By an opaque `behaviour_profile_id` string authored on `ScenarioOccupant`/`ScenarioFirefighter` at
scenario-generation/authoring time, looked up via `resolve_profile()`. An unrecognized id is a
**hard error** (`UnknownBehaviourProfileError`) — "no repair, no invented default," per its own
docstring. `register_category()` lets a caller register a custom id's `OccupantCategory`
classification, and `register_occupants(context, registry=...)` accepts any custom
`Mapping[str, BehaviorProfileTemplate]` in place of the default registry.

### Are profiles randomly assigned?
No — the *profile id* assignment is authored (deterministic, from the Scenario). What *is*
randomly (but reproducibly) derived per-occupant on top of the shared profile is: the 14
`OccupantAttributes` values (Phase 2 §25), social group membership (Phase 2 §15), and every
stochastic behavior's own dice roll (compliance, herding, etc.) — all seeded from
`(scenario.metadata.seed, occupant_id)`.

### Can multiple behaviors coexist on one occupant?
Yes, extensively, via composition (decorator-pattern wrapper strategies, not a single monolithic
class). A typical civilian occupant's actual `decision_strategy` at registration time is a
5-layer-deep composition: `AttributeAwareComplianceDecisionStrategy` →
`AssistanceAwareDecisionStrategy` → `ComplianceDecisionStrategy` → (`AlwaysEvacuateDecisionStrategy`
or `AlwaysWaitDecisionStrategy`), and `route_choice_strategy` is similarly:
`AssistanceAwareRouteChoiceStrategy` → `SocialGroupAwareRouteChoiceStrategy` →
`CrowdFollowingAwareRouteChoiceStrategy` → `ShortestRouteChoiceStrategy`. Each layer is a
documented no-op when its own trait is absent, so an ordinary occupant with no assistance/group/
crowd-following trait behaves identically to the innermost strategy alone.

### Are profiles reusable across scenarios?
Yes — `DEFAULT_PROFILE_REGISTRY`/`DEFAULT_FIREFIGHTER_PROFILE_REGISTRY` are process-wide constants,
reused unmodified across every Scenario that uses the default profile ids. Per-occupant realism
(attributes, groups) is *derived on demand* from `(seed, occupant_id, profile_id)` rather than
stored anywhere, so it is reproducible across runs of the same scenario/seed but is not "the same
value" across different scenarios/seeds using the same profile id.

---

## PHASE 5 — Human realism classification

Per behavior (from Phase 2's inventory), classified against what the repository itself documents:

| Behavior | Evidence-based? | Heuristic? | Probabilistic? | Rule-based? | From published literature? | Source disclosed in code |
|---|---|---|---|---|---|---|
| Compliance (Bernoulli gate) | No | Yes | Yes | Yes | No | "illustrative... not a validated life-safety model" |
| Pre-movement delay (lognormal) | Partial (distribution *shape*) | Yes | Yes | Yes | Partial — lognormal shape is cited as "commonly used in evacuation literature," but the parameters are not | Explicitly: "not a validated RSET pre-movement-time model" |
| Hesitation (route ambiguity) | No | Yes | No (deterministic) | Yes | No | "an honestly-observable signal instead" (own framing) |
| Herding (static) | No | Yes | Yes | Yes | No | Undocumented assumption (no literature citation) |
| Load balancing (anti-herding) | No | Yes | No | Yes | No | Undocumented assumption |
| Familiarity-based routing | No | Yes | No | Yes | No | Undocumented assumption; no data source for familiarity exists |
| Hazard-aware routing | No | Yes | No | Yes | No | Undocumented assumption (linear cost/hazard tradeoff weight is arbitrary) |
| Assistance speed multipliers | No | Yes | No | Yes | No | Explicitly: "not a validated mobility-impact model" |
| Firefighter rescue/search | No | Yes | No | Yes | No | Explicitly: "not a validated fire-service performance figure" |
| Rescue priority scoring | No | Yes | No | Yes | No | Explicitly: "illustrative, not validated" |
| Occupant attribute ranges (speed, stamina, panic, etc.) | No | Yes | Yes | Yes (category-conditioned) | No | Explicitly: "not measured real-world data" |
| Social group formation | No | Yes | Yes | Yes | No | Explicitly: "illustrative... not measured data" |
| Own-safety hazard hesitation | No | Yes | No | Yes (threshold) | No | Undocumented assumption (thresholds are placeholders, same convention as `HazardSeverity`) |
| Fall/injury derivation (ground_truth) | No | Yes | Yes | Yes | No | Explicitly: "Neither threshold nor probability below is a validated life-safety figure" |
| Wheelchair `mobility_factor` (stairs) | No | — | — | — | No | Disclosed as data-only, not enforced |

**Summary judgment**: with the single partial exception of pre-movement delay's lognormal *shape*
(cited as literature-consistent), **every quantitative parameter in this subsystem — every speed,
probability, threshold, weight, and range — is an author-disclosed illustrative placeholder**, not
a value drawn from or validated against published evacuation research, and the repository's own
comments say so at nearly every site. This is not a hidden gap; it is unusually consistently
self-disclosed throughout the codebase (the recurring phrase is some variant of "illustrative, not
a validated life-safety model"). Nothing in this review found a case where the repository claims
literature backing that the code does not have.

---

## PHASE 6 — Simulation capabilities

| Capability | Verdict | Justification |
|---|---|---|
| Panic (as an emotional/behavioral state) | **PARTIAL** | `panic_susceptibility` is a per-occupant [0,1] attribute that scales pre-movement delay (Behavior 5/6). There is no panic *state machine*, no panic-driven route irrationality, no panic spread/contagion between occupants. |
| Family groups | **YES** | `occupant_grouping.assign_occupant_groups` explicitly forms "Family" groups (adult+child mix) with a designated leader and following members (Behaviors 7, 15). |
| Disabled occupants (mobility-impaired) | **PARTIAL** | `Wheelchair_Default` profile + `WHEELCHAIR_USER` category exist, with reduced walking speed and a near-zero `mobility_factor` attribute. But `mobility_factor` is data-only and does **not** restrict routing — a wheelchair user can be routed onto a Stair edge exactly like anyone else (explicitly disclosed, `occupant_attributes.py:160-165`). |
| Fire wardens / crowd leaders | **YES** | `FireWarden_Default`/`Staff_Default` are `Role.LEADER` profiles with `BasicHelpingDecisionStrategy`/`AlwaysEvacuateDecisionStrategy`; leader/follower mechanics (Behavior 14) and social-group leadership (Behavior 15, via the `leadership` attribute) both exist. |
| Social influence (herding) | **YES** | Static herding and crowd-following-tendency strategies (Behaviors 10–11) route occupants toward the majority-chosen exit. |
| Smoke hesitation | **PARTIAL** | `smoke_tolerance` attribute scales hesitation magnitude (Behavior 6) and gates fall/running derivation (`ground_truth/human_behavior.py`), but there is no direct "occupant refuses to enter a smoke-filled node" behavior. |
| Exit switching / dynamic rerouting | **YES** | `simulation_interactive.route_manager.RouteManager` triggers a full replan (Decision+Navigation) on hazard-signature change, operator recommendation, or broadcast — but **only** in the interactive/live pipeline, and **only** at AT_NODE/STATIONARY points (never mid-edge). The batch `SimulationRuntime` pipeline used for dataset generation/research does **not** replan at all — routes are fixed once at registration (its own docstring: "Occupant movement is NOT ticked... every occupant's route is already fixed by the time a SimulationContext exists"). |
| Emotional state (beyond panic_susceptibility) | **NO** | No emotion model, no fear/anxiety/relief state machine anywhere in this subsystem. |
| Risk perception | **PARTIAL** | `risk_aversion` attribute exists and feeds the sensitivity multiplier (Behavior 6), but there is no risk-based route re-evaluation independent of hazard data already present in `HazardAwareRouteChoiceStrategy`. |
| Occupant fatigue | **PARTIAL** | `stamina`/`fatigue_resistance` attributes exist and feed the one-time `effective_walking_speed_multiplier` computed at registration (`registrar.py::_effective_walking_speed_multiplier`) — this is a **static** per-occupant speed derating, not a dynamic fatigue model that degrades over the course of a long evacuation. |
| Running | **YES** (as a post-hoc label, not a distinct movement model) | `ground_truth.human_behavior` labels a step RUNNING when zone hazard_score ≥ 0.35, but this does not change the occupant's actual simulated `walking_speed` — it is a descriptive label applied after the fact, not a distinct kinematic behavior. |
| Crawling | **NO** | Not modeled anywhere in this subsystem. |
| Wheelchairs | **PARTIAL** | See "Disabled occupants" above — represented as a profile + attribute set, not a distinct mobility/routing model. |
| Firefighter interaction (rescue, guidance, hazard reporting) | **YES** | Full `FirefighterDecisionStrategy`/`FirefighterDecisionEngine` task set: search, rescue, assist-wheelchair-user, carry-fallen, guide civilians, report hazard, continue-search, return-outside (Phase 2 §18–19, §21). |
| Crowd/pedestrian dynamics realism (density-speed relationship) | **PARTIAL** | `DefaultCongestionModel` implements a linear speed-degradation curve based on edge occupancy vs. capacity, explicitly disclosed as "not a validated pedestrian-dynamics/fundamental-diagram model." No social-force, cellular-automaton, or validated flow-density curve is used. |

Note on the perception-side `behavior_recognition/` package (Phase 1): it can recognize
STATIONARY/WALKING/RUNNING/POSSIBLY_FALLEN from live camera tracks, but this is a completely
separate capability from occupant *simulation* — it observes real/simulated video, it does not
drive any simulated occupant's decisions, and it explicitly cannot and does not attempt HELPING,
PANIC, LEADERSHIP, DISABILITY, INJURY, CONFUSION, FOLLOWING, or HERDING recognition (its own
documented exclusion list).

---

## PHASE 7 — Research value: what the current system can genuinely answer

Based only on what Phases 1–6 verified is actually implemented:

- **Effect of compliance rate on evacuation time/bottlenecks** — `compliance_level`/`compliance`
  are real, independently variable inputs feeding a real Bernoulli gate whose outcome (WAIT vs.
  EVACUATE) genuinely changes simulated movement and timing.
- **Effect of pre-movement delay distribution on total evacuation time** — `median_delay`/`spread`
  per profile are real levers over a real lognormal draw that shifts real departure times.
- **Effect of herding vs. anti-herding (load-balancing) wayfinding policy on exit utilization and
  congestion** — both strategies exist, are swappable per-scenario, and genuinely change which
  exits get used and how queue events distribute (`peak_edge_occupancy`, `total_queue_events` are
  real, measured simulation outputs).
- **Exit/queue congestion formation under a fixed population and building topology** — the
  discrete-event coordinator's capacity/congestion model produces genuine, measurable queueing
  (`OccupantState.QUEUED`, `queue_wait_time`, `peak_edge_occupancy`).
- **Impact of assistance pairing (helper/assisted) on overall evacuation time** — the speed
  coupling between helper and assisted occupant is a real, measurable mechanic, not just a label.
- **Impact of social-group formation and leader-following on departure synchronization** — group
  members' departure times are genuinely coupled to their leader's resolved decision.
- **Comparing scenario-authored vs. dynamically-computed assistance/rescue assignment strategies**
  — both `register_population` (static) and `register_population_dynamic` (dynamic,
  `human_decision_engine`-driven) produce comparable, fully-logged outcomes (`DecisionEventLog`).
- **Rescue-priority-driven triage under a fixed firefighter count** — `FirefighterDecisionEngine`'s
  priority-score-based claiming is a real, deterministic mechanism whose sensitivity to the
  fixed weights (Phase 2 §19) is directly testable.
- **Effect of route-choice policy on exit-utilization balance** — comparing
  `ShortestRouteChoiceStrategy` vs. `FamiliarityBasedRouteChoiceStrategy` vs.
  `LoadBalancingRouteChoiceStrategy` vs. `HazardAwareRouteChoiceStrategy` across identical building
  topologies and populations is a directly supported experiment (this is, in fact, exactly what
  the project's own `predictive_dataset`/`localized_predictive_model` research lineage already
  does downstream of this subsystem, per memory of related milestones).

**What this system cannot support as a research question today** (see Phase 8 for the
full list): claims about panic *contagion*, claims about real-world compliance/pre-movement-delay
*magnitudes* (only their *relative* effect, since the input distributions are undisclosed
placeholders, not measured data), or claims that require dynamic mid-evacuation behavior change in
the batch/dataset-generation pipeline (which never replans).

---

## PHASE 8 — Limitations

**Already solved**
- Non-reproducible/shared-RNG behavior across occupants sharing one profile template — fixed by
  threading a per-occupant seeded `random.Random` through `DecisionContext`
  (`docs/architecture/reproducibility_review.md`).
- Ambiguity between "no route exists" and "chose not to move" — resolved by the explicit
  `BehaviorDecision.route_unavailable` flag (`simulator/decision.py:88-96`).

**Minor limitations**
- `BasicHelpingDecisionStrategy` has no proximity, capacity, or mutual-consent reasoning (disclosed
  in its own docstring as a deliberate scope boundary).
- `HesitationPreMovementDelayStrategy` cannot honestly detect "was my preferred exit blocked" (only
  genuine route-cost ambiguity), because traversability is baked into Dijkstra relaxation itself,
  not exposed to strategies as a separate signal.
- `LoadBalancingRouteChoiceStrategy`/`StaticHerdingRouteChoiceStrategy` are both bounded to
  `max_alternatives` candidate routes (default 3–5), not an exhaustive search.

**Major research limitations**
- **No dynamic mid-run behavior change in the batch/dataset pipeline.** `SimulationRuntime`'s own
  docstring states occupant movement is "NOT ticked" — every route is fixed before the runtime
  even exists. Dynamic rerouting (`RouteManager.maybe_replan`) only exists in the separate
  `simulation_interactive` package, used for the live/interactive Designer path, not for the
  research/dataset-generation pipeline that produces most of this project's published research
  artifacts (per the milestone history recorded in project memory).
- **No panic contagion / no crowd emotional-state propagation.** `panic_susceptibility` is a
  static, individually-drawn attribute; it never spreads, updates, or reacts to nearby occupants'
  states.
- **No validated pedestrian dynamics.** `DefaultCongestionModel`'s linear degradation is explicitly
  not a fundamental-diagram or social-force model; there is no lateral avoidance, no jamming
  transition, no realistic bottleneck-oscillation behavior.
- **No dynamic fatigue.** Stamina/fatigue_resistance apply once, at registration, as a static speed
  derating — not a function of elapsed evacuation time or distance already travelled.

**Architectural limitations**
- **Two entirely separate registration/orchestration paths** (`behaviour_profile_resolver.registrar`
  / `combined_registrar` for static assignment vs. `dynamic_registrar` for
  `human_decision_engine`-driven dynamic assignment) that share downstream `behavior_library`
  strategies but duplicate significant orchestration logic (both independently re-derive seeds,
  attributes, and traits).
- **Mobility constraints (e.g. wheelchair-cannot-use-stairs) are disclosed data, not enforced
  routing constraints** — `occupant_attributes.py` computes `mobility_factor` but nothing in
  `pathfinding`/`navigation` reads it to exclude Stair edges from a wheelchair user's route.
  Explicitly documented as an intentional boundary of that milestone's own scope, not a bug — but
  it means "can SynEvac simulate a wheelchair user being physically unable to use stairs" is
  currently **NO** at the routing level, despite the attribute existing.
- **`ActionType` is documented as purely descriptive** — "Simulation never branches on this value."
  This means the rich vocabulary of intents (HELP, ESCORT, FIREFIGHTER_GUIDE, etc.) has no causal
  effect on the coordinator beyond whatever `goal_id`/`route`/`walking_speed` a strategy already
  computed; two behaviorally-identical decisions with different `action_type` labels produce
  identical simulated movement.

**Evidence limitations**
- With the single partial exception of the lognormal pre-movement-delay *shape*, this review found
  **no citation to any published evacuation-behavior study** anywhere in `behavior/`,
  `behavior_library/`, `behaviour_profile_resolver/`, or `human_decision_engine/`. Every numeric
  parameter (speeds, probabilities, thresholds, weights, attribute ranges) is repository-authored
  and self-disclosed as illustrative. Per this task's own instruction, this is reported as
  **"Undocumented assumption"** for every such parameter rather than inferring a justification the
  repository does not state.

---

## PHASE 9 — Future research opportunities (identified, not designed)

Based only on where this investigation found the greatest gap between existing structure and
existing capability:

- The **dynamic-registration path** (`human_decision_engine` + `dynamic_registrar`) already
  computes structural pairings and hazard-aware decisions at registration time; the largest
  unrealized potential is that this same machinery is not connected to the **replanning path**
  (`simulation_interactive.route_manager`) that already exists for a different purpose
  (recommendation/broadcast/hazard-triggered rerouting) — the scientific potential of *combining*
  dynamic decision-making with in-run replanning (rather than only at initial registration) is
  structurally close but not wired together anywhere in the current codebase.
- The `OccupantAttributes` substrate (14 independently-drawn, category-conditioned values per
  occupant) is richer than what most of the current strategies actually consume — only a subset
  (`reaction_speed`, `panic_susceptibility`, `route_familiarity`, `smoke_tolerance`,
  `visibility_tolerance`, `risk_aversion`, `crowd_following_tendency`, `compliance`,
  `helping_likelihood`, `walking_speed_multiplier`, `stamina`, `fatigue_resistance`,
  `mobility_factor`) are read anywhere in `behavior_library`; `leadership` is consumed only by
  group-leader selection. This is a candidate area for sensitivity-analysis-style research using
  data the system already generates but does not yet fully exploit behaviorally.
- The batch/dataset-generation pipeline's fixed-route assumption (`SimulationRuntime`) versus the
  interactive pipeline's replanning capability (`RouteManager`) is a clean, already-existing
  architectural seam that would let a comparative study ("static vs. dynamically-replanned
  evacuation outcomes under identical scenarios") be built without inventing new mechanisms —
  both pipelines already consume the same `behaviour_profile_resolver`/`behavior_library` layer.

---

## FINAL SUMMARY

1. **What human behaviors can SynEvac currently model?** Compliance/non-compliance with
   evacuation orders, lognormal pre-movement delay, route-ambiguity hesitation, herding and
   anti-herding (load-balancing) route choice, familiarity-based route choice, hazard-aware route
   choice, leader/follower movement, social group formation (family/friends/coworkers/school/
   hospital-staff groups) with synchronized departure, civilian-to-civilian assistance (help,
   escort, push-wheelchair, drag, carry) with coupled speed and route, firefighter search/rescue/
   guide/report-hazard tasking with priority-based triage, dynamic (structural + hazard-gated)
   assistance and rescue decision-making as an alternative to scenario-authored assignment, and
   (in the interactive/live pipeline only) mid-evacuation rerouting driven by hazard change,
   operator recommendations, or broadcasts.

2. **How realistic is the behavior model?** Structurally plausible and internally consistent
   (composable, deterministic-where-claimed, reproducible), but **not empirically validated**. With
   one partial exception (the lognormal shape of pre-movement delay), every numeric parameter is
   an author-disclosed illustrative placeholder, not a value derived from or checked against
   published evacuation-behavior data.

3. **What assumptions are hard-coded?** All speed multipliers (assistance types, firefighter
   rescue), all compliance/pre-movement-delay parameters per profile, all attribute value ranges
   per occupant category, all rescue-priority weights, all congestion/capacity constants
   (`PEOPLE_PER_METER_OF_WIDTH`, `MINIMUM_SPEED_FACTOR`), all hazard-severity score cutoffs, the
   fall/injury probability thresholds, and the social-grouping fraction/size bounds. All are
   explicitly labeled in-code as illustrative/undisclosed-source rather than validated.

4. **Which behaviors are strongest** (best-specified, most internally consistent, most reusable)?
   The compositional strategy architecture itself (Decision/Route/Pre-movement interfaces with
   graceful no-op fallback chaining) and the assistance/leader-follower coupling mechanics (speed
   and route genuinely, provably synchronized between paired occupants) are the most
   rigorously-reasoned parts of the subsystem, per the depth of their own in-code justification and
   the exactness of the route-tail-recovery proof in `AssistanceAwareRouteChoiceStrategy`.

5. **Which behaviors are weakest?** Wheelchair mobility restriction (attribute exists, not
   enforced in routing), the batch pipeline's total absence of dynamic rerouting, the lack of any
   panic-state or fatigue-over-time model (both are static, one-time attribute effects rather than
   dynamic processes), and the disclosed non-exhaustive/bounded-candidate-set nature of every
   route-choice strategy that reasons over "alternatives."

6. **Is the behavior engine suitable for publishable research today?** For **comparative,
   architecture-level studies** (how does policy X vs. policy Y affect a measured simulation
   outcome, holding the building/population fixed) — yes, the mechanisms are real, measurable, and
   reproducible. For **studies claiming real-world behavioral validity or calibrated magnitudes**
   (e.g. "occupants take on average N seconds to respond") — no, because the underlying parameter
   values are self-disclosed as illustrative, not measured or literature-derived.

7. **What kinds of evacuation papers could be written TODAY using the existing behavior system?**
   Comparative wayfinding-policy studies (herding vs. load-balancing vs. hazard-aware vs.
   familiarity-based route choice) on congestion/evacuation-time outcomes; the effect of
   compliance-rate and pre-movement-delay-distribution parameters on aggregate evacuation time;
   assistance-pairing impact on evacuation time and exit congestion; static-authored vs.
   dynamically-computed assistance/rescue assignment comparison; firefighter rescue-priority
   policy sensitivity analysis. All of these are *relative*/*sensitivity* studies, not
   *calibration*-against-reality studies.

8. **What is the single biggest weakness of the current behavior engine?** The near-total absence
   of empirical grounding: virtually every quantitative parameter governing occupant behavior in
   this subsystem is an internally-consistent but self-acknowledged placeholder, not a value drawn
   from or validated against real evacuation data or published human-behavior-in-fire research.

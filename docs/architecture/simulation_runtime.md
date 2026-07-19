# Simulation Runtime — Architecture Proposal

Status: **proposal, open for review**. No code changes accompany this document — architecture only,
per this pass's explicit instruction. This document closes the gap `docs/architecture/scenario_runner.md`
§10 first named ("Scheduling Scenario Events... a future Simulation Loop") and
`docs/architecture/scenario_event_execution.md` §10/§18 re-flagged, narrower each time, but never
designed: **nothing in this codebase, today, drives simulation time forward across occupant movement,
hazard evolution, and event execution together.** This is that design.

## 1. Purpose

`SimulationRuntime` is the single orchestrator that advances a complete SynEvac simulation from an
already-built `SimulationContext` (`scenario_runner/context.py`, frozen) to a stop condition, calling
into every already-approved subsystem at the correct time, in the correct order, without modifying any
of them. It coordinates:

- `MultiAgentSimulation` (`simulator/`, frozen) — occupant movement.
- `HazardEvolutionEngine` / `EvolutionBackedHazardProvider` (`hazard_evolution/`, frozen) — hazard state.
- `ScenarioEventExecutor` (`scenario_event_executor/`, frozen) — engineering-state mutation.
- The Perception Layer (`perception/`, frozen primitives, no concrete `PerceptionProvider` yet — §13).
- `AIDecisionEngine` (`ai_decision/`, frozen) — the Rule-Based Decision Engine.
- A future Dataset Logger (no precedent exists anywhere in this codebase — confirmed by repository-wide
  search; §18).

It owns exactly one new thing none of the above owns today: **the tick loop itself** — a `target_time`
value that advances by a fixed `dt`, and the fixed sequence of subsystem calls made at each tick. It
introduces no new simulation physics, no new routing, no new hazard model, no new decision logic — it
only decides *when* to call the seams those packages already expose.

## 2. The most consequential finding: two of the three "clocks" are not steppable, and don't need to be

**This must be understood before anything else in this document makes sense**, the same way
`scenario_event_execution.md` §2 had to be understood before its own design made sense.

Repository inspection (§3 of that document, reconfirmed here) already established that occupant routes
are fixed once at registration and never revisited, and that `behaviour_profile_resolver.
register_occupants()` registers **every** occupant — via `HumanBehaviorLayer.register()` →
`MultiAgentSimulation.submit_decision()` (`behavior/orchestrator.py:85`) — in one synchronous pass,
before any simulation time has progressed. **Consequence, confirmed this pass**:
`simulator/coordinator.py::MultiAgentSimulation.run()` is atomic (`while self._event_heap:
self._process_next_event()`, `simulator/coordinator.py:254`) and has no `step(dt)`/`advance_to(time)` —
by design, not omission (§3 of `scenario_event_execution.md` already established this) — but because
**nothing perturbs an occupant's route after registration**, there is nothing a per-tick `step()` could
usefully do differently from running the whole heap to completion once. The entire occupant-movement
timeline is fully determined the instant registration finishes, before this Runtime's tick loop even
begins.

**This means occupant movement is not one of the things this Runtime ticks.** It is resolved exactly
**once**, during Initialization (§7), by calling `context.simulation.run()` — a call that, per
`scenario_runner/runner.py` and `behaviour_profile_resolver/registrar.py`, **nobody in the codebase
currently makes**: the Runner explicitly leaves `simulation` empty (`scenario_runner/context.py`'s own
docstring: "NOT run-ready"), and the Resolver only calls `submit_decision()`, never `run()`. **This
Runtime is the first component whose job is to actually run the simulation it was handed.** The
resulting `MultiAgentSimulationResult` (`simulator/multi_agent_result.py`) — specifically each
`OccupantTimeline.steps: List[OccupantTimelineStep]`, each carrying `start_time`/`end_time`/`from_node`/
`to_node` — is a complete, queryable record of where every occupant is at any time `t`, computed once,
read many times during the tick loop (§11).

The other two clocks are genuinely pull-based and *are* ticked, because both already expose exactly that
shape: `EvolutionBackedHazardProvider.snapshot_at(time)` (§12) and `ScenarioEventExecutor.advance_to
(target_time)` (§13), which was deliberately modeled on the former's calling convention. **Three
different advancement shapes for three different subsystems is not an inconsistency this Runtime needs
to paper over — it is a fact about each subsystem's own nature** (occupant movement has no mid-course
inputs to react to; hazard evolution and event execution do have a meaningful notion of "state as of
time t" that must be queried incrementally). The Runtime's job is to drive the two that need driving,
and to correctly *read* (not re-run) the one that doesn't.

## 3. Grounding in existing code

Confirmed this pass, directly from source, not inherited from a prior document's summary:

- **`AIDecisionEngine.decide(hazard_snapshot, occupancy_snapshot, time)`** (`ai_decision/engine.py:97`)
  is the actual, current signature — **unchanged from Ground Truth**, despite
  `docs/architecture/perception_layer_review_2.md` §7.1 explicitly listing "`AIDecisionEngine,
  unchanged`" in its own dependency diagram and §8.2 only *recommending* ("once `AIDecisionEngine.
  decide()` takes a `BuildingObservation`...") a future migration that has **not happened**. This
  Runtime must call `decide()` exactly as it exists today — with `HazardSnapshot`/`OccupancySnapshot`
  Ground Truth, not `BuildingObservation`. §14 details the consequence.
- **`PerceptionProvider.observation_at(time)`** (`perception/providers/provider.py:18`) is `raise
  NotImplementedError` — an interface only, exactly as `perception_layer.md` scoped it. The individual
  pieces that would compose a working implementation all exist (`GroundTruthCameraProvider`,
  `GroundTruthSmokeDetectorProvider`, `GroundTruthHeatDetectorProvider`, `OccupancyEstimator`,
  `SensorFusion.fuse()`), but **no class in this codebase wires them together behind
  `PerceptionProvider`'s own seam.** §13 addresses this directly.
- **No `OccupancyProvider` implementation in this codebase is time-varying.** `occupancy/provider.py`
  defines only `ManualOccupancyProvider` — one fixed `OccupancySnapshot` returned regardless of `time`.
  There is no `simulator`-backed equivalent of `EvolutionBackedHazardProvider` for occupancy. §11
  addresses this directly — it is the same *shape* of gap `scenario_event_execution.md` §4 resolved for
  events, and resolved the same way.
- **`context.engine` (`SimulationContext.engine: PathfindingEngine`) is already the correct,
  hazard-unaware `base_engine`** `AIDecisionEngine.__init__` expects (`scenario_runner/
  navigation_initializer.py:35`: `PathfindingEngine(graph)`, no cost model override) — no adapter
  needed here.
- **`fire_initializer.py`'s own docstring is explicit that `initial_hazard_snapshot` has never had
  `evolve()` called on it: "the first evolve() call is the caller's job, never this package's."** This
  Runtime is that caller, for the same reason it is the first caller of `simulation.run()` (§2) — both
  are "the last-mile activation step no previous phase was scoped to perform."
- **`ScenarioEventExecutor` mutates `context.building`'s `Camera.active`/`Detector.active` fields
  directly** (`scenario_event_executor/handlers.py:88-105`, reusing `scenario_runner.apply_camera_state`/
  `apply_detector_state`), and **`GroundTruthCameraProvider`/`GroundTruthSmokeDetectorProvider`/
  `GroundTruthHeatDetectorProvider` read those same fields live** (`if not detector.active: continue`,
  `perception/providers/ground_truth_smoke_detector_provider.py:80`; `if not camera.active: ...`,
  `ground_truth_camera_provider.py:123`). **This is a genuine cross-subsystem ordering dependency**,
  the first one this Runtime must get right — §8/§13.

## 4. Runtime ownership

`SimulationRuntime` owns:

- **The tick clock**: `current_time`, `dt`, `end_time` (§6/§9).
- **Construction-time references** to every subsystem seam it calls, injected or built from
  `SimulationContext` at Initialization (§7) — it does not reach into any subsystem's private state, it
  calls only already-public methods (`snapshot_at`, `advance_to`, `decide`, `observation_at`, `run`).
- **The per-tick call sequence and its ordering** (§8) — the one piece of orchestration logic that
  exists nowhere else in this codebase.
- **A per-tick result/hook shape** for the future Dataset Logger (§18) — the Runtime defines *what is
  available* each tick; it implements no persistence itself.

`SimulationRuntime` does **not** own, and must never reach into:

- Route planning, capacity/congestion modeling, or occupant state transitions (`simulator/` — read-only,
  via the precomputed `MultiAgentSimulationResult`, §11).
- Hazard physics or merge policy (`hazard_evolution/`, `fire_growth/` — called only through
  `snapshot_at(time)`).
- Engineering-state mutation semantics (`scenario_event_executor/` — called only through
  `advance_to(target_time)`).
- Sensor fusion, occupancy estimation, or observation classification (`perception/` — called only
  through `observation_at(time)`, once §13's composition gap is closed).
- Zone-recommendation, priority-ranking, or announcement logic (`ai_decision/` — called only through
  `decide(...)`).

This mirrors the exact ownership boundary `scenario_runner/runner.py` and `scenario_event_executor/
executor.py` already established for their own scopes: a thin coordinating layer over already-complete,
unmodified subsystems, never a reimplementation of any of them.

## 5. Simulation clock

**One clock, one unit: seconds elapsed since scenario `t=0`** — the same unit `ScenarioEvent.time`,
`HazardSnapshot.timestamp`, and `OccupantTimelineStep.start_time`/`end_time` already share. This Runtime
does not introduce a tick-index or step-count as a competing time representation anywhere in its public
API — `scenario_event_execution.md` §3's own finding ("two independent clocks already coexist,
unintegrated") is resolved here **not** by forcing hazard evolution and event execution to share one
clock *object* (neither has one to share — each is stateless with respect to time except for its own
internal cache), but by this Runtime being the one place that holds the single authoritative
`current_time` value and passes it, identically, to both `snapshot_at(time)` and `advance_to
(target_time)` every tick. The clock is real now, in the sense the prior documents meant when they
observed one didn't yet exist anywhere — it lives here, in this Runtime, and nowhere else.

## 6. Fixed timestep policy

**A single, constructor-supplied `dt: float`, applied uniformly.** This directly reuses
`EvolutionBackedHazardProvider`'s own `dt`-capped advancement (`hazard_evolution/provider.py:36-37`:
`step = min(self.dt, time - current_time)`) rather than inventing a second stepping convention — the
Runtime's `dt` **is** the same `dt` value passed to `EvolutionBackedHazardProvider.__init__`, not a
distinct Runtime-level setting that happens to have the same name. `ScenarioEventExecutor.advance_to()`
needs no `dt` of its own (§13's "no sorting, no dt-capping — it fires everything due by `target_time`");
the Runtime simply calls it with the same `target_time` it computed for the hazard provider this tick.

No adaptive/variable timestep is designed here — every existing time-based component in this codebase
(`EvolutionBackedHazardProvider`, the Scenario Engine's own reproducibility requirements) already
assumes a fixed, deterministic step; introducing variability would be a new capability, not a
coordination decision, and is out of scope.

## 7. Initialization

In order, each step calling only already-approved, unmodified entry points:

1. `context = scenario_runner.run(scenario, building)` — produces the immutable `SimulationContext`
   (building copy, graph, hazard-unaware `engine`, empty `simulation`, `hazard_engine` +
   `initial_hazard_snapshot` at `t=0`, `occupants`, `scheduled_events`, `metadata`). **Not this
   Runtime's own step** — performed by the caller before constructing a `SimulationRuntime`, exactly as
   every prior phase already assumes (`SimulationRuntime.__init__` takes a `SimulationContext`, it does
   not take a raw `Scenario`/`Building`).
2. `behavior_layer = behaviour_profile_resolver.register_occupants(context)` — resolves every
   `behaviour_profile_id`, constructs `BehaviorProfile`s, and synchronously populates `context.
   simulation`'s event heap via `submit_decision()` (§2/§3). Also owned by the caller, for the identical
   reason as step 1 — the Runtime consumes an already-registered `SimulationContext`, it does not import
   `behaviour_profile_resolver` itself (§19).
3. **`movement_result = context.simulation.run()`** — the first genuinely new activation step (§2/§3):
   drains the fully-populated event heap exactly once, producing the complete
   `MultiAgentSimulationResult` this Runtime will query, never re-run, for the rest of its life. This
   *is* owned by `SimulationRuntime` — it is the one call in this entire pipeline nothing upstream was
   ever scoped to make.
4. `event_executor = ScenarioEventExecutor(context)` — cursor at 0, per its own frozen design.
5. `hazard_provider = EvolutionBackedHazardProvider(context.hazard_engine, context.
   initial_hazard_snapshot, dt)` — the first `evolve()` call is deferred to the first tick, per
   `fire_initializer.py`'s own documented contract (§3).
6. `occupancy_provider = <a movement_result-backed OccupancyProvider>` — §11's flagged, required new
   component; constructed from `movement_result` here.
7. `decision_engine` — an already-constructed `AIDecisionEngine(base_engine=context.engine, ...)`,
   supplied by the caller (mirrors `EvolutionBackedHazardProvider`/`ScenarioEventExecutor` both being
   constructed from `context`-derived pieces rather than the Runtime reaching past its own inputs to
   build a decision engine's tuning parameters itself, which are deployment/scenario-difficulty
   concerns, not orchestration ones).
8. `perception_provider` (optional, §13) and `dataset_logger` (optional, §18) — both injected, both
   `None`-able; a `SimulationRuntime` with neither still ticks correctly (occupant movement, hazard,
   events, and decisions have no dependency on either).
9. `end_time` resolved (§9) — either caller-supplied, or defaulted from `movement_result.
   total_evacuation_time` and `context.scheduled_events[-1].time` (§9's exact formula).
10. `current_time = 0.0`. Runtime is now ready for its first `tick()`.

Nothing in this sequence mutates `SimulationContext` itself (still frozen); every artifact constructed
here (`movement_result`, the three providers, `end_time`) is new Runtime-owned state, exactly the
"immutable container, mutable runtime companion" split `SimulationContext.simulation` and
`ScenarioEventExecutor`'s own cursor already established, extended one level up.

## 8. Per-tick update sequence and subsystem execution order

One `tick()` call advances `current_time` by `dt` (capped at `end_time`) and performs, **in this exact
order**:

1. **`fired_events = event_executor.advance_to(next_time)`** — engineering-state mutation happens
   *first*, deliberately, because of §3's confirmed ordering dependency: a `camera`/`detector` event
   firing at exactly this tick must be visible to Ground Truth Perception providers reading `Camera.
   active`/`Detector.active` later in the same tick (step 4). Running this step after hazard/perception
   would mean a same-tick device-availability change is invisible until the *next* tick — an observable,
   avoidable one-tick lag this ordering eliminates for free.
2. **`hazard_snapshot = hazard_provider.snapshot_at(next_time)`** — advances
   `HazardEvolutionEngine.evolve()` internally by up to `dt`. Order relative to step 1 has no
   *functional* effect on the hazard result itself (`scenario_event_execution.md` §11: no event category
   produces a `HazardContribution` or touches `hazard_evolution`), but is sequenced after event execution
   for one consistent convention rather than two arbitrary ones.
3. **`occupancy_snapshot = occupancy_provider.snapshot_at(next_time)`** — a pure read against the
   already-complete `movement_result` (§2/§11); this step never advances anything, it only resolves
   "where is everyone at `next_time`."
4. **`decision = decision_engine.decide(hazard_snapshot, occupancy_snapshot, next_time)`** — the current,
   real `AIDecisionEngine` signature (§3/§14), fed this tick's just-computed Ground Truth snapshots
   directly.
5. **`observation = perception_provider.observation_at(next_time)` if configured** — reads the *same*
   `hazard_snapshot`/`occupancy_snapshot` this tick (via whatever Ground Truth providers the composition
   in §13 wraps) and the now-current `context.building` Camera/Detector `active` flags already updated
   by step 1. A sibling of step 4, not a dependency of it (§14) — both branch off the same tick's Ground
   Truth, neither waits on the other's result.
6. **`if dataset_logger: dataset_logger.on_tick(next_time, fired_events, hazard_snapshot,
   occupancy_snapshot, decision, observation)`** — a pure observer call (§18); never influences steps
   1-5, never mutates Runtime or subsystem state.
7. `current_time = next_time`; stop-condition check (§9).

Occupant movement is not a step in this sequence (§2) — it was fully resolved once, in Initialization
(§7 step 3), and this loop only *reads* its result (step 3). Restating this because it is the single
most important departure from a naive "coordinate five subsystems every tick" reading of this document's
brief: **the Runtime coordinates four subsystems per tick, plus one that was already finished before
ticking began.**

## 9. Stop conditions

No existing subsystem defines a natural "simulation is over" signal the Runtime can simply forward —
`HazardEvolutionEngine.evolve()` will run forever if asked to (fire growth curves have no terminal
state built in), and `ScenarioEventExecutor.is_complete` only means "no more *events* to fire," not "no
more simulation worth advancing" (hazard keeps evolving after the last scheduled event). The Runtime
therefore owns an explicit `end_time`, resolved at Initialization (§7 step 9):

- **Caller-supplied `end_time`** takes precedence if given — the deployment/training context (a fixed
  RL episode length, a fixed dataset-generation window) is not something this Runtime should guess at.
- **Default, if omitted**: `max(movement_result.total_evacuation_time or 0.0, context.
  scheduled_events[-1].time if context.scheduled_events else 0.0)` — the later of "every occupant who
  will ever arrive has arrived" and "every scheduled event has fired." This is a documented, honest
  default (deliberately not "run until hazard stabilizes," which has no defined meaning for an
  open-ended growth curve), not a claim that it is correct for every use case.

**Primary condition**: `current_time >= end_time`. **Error stop**: any subsystem exception propagates
immediately and halts the loop (§15) — there is no partial-tick recovery. A caller-supplied early-stop
predicate (e.g., "stop once every zone is unsafe" or "stop once an RL episode's own termination
criterion fires") is a plausible future extension but is **not designed here** — flagged into §21,
consistent with how `scenario_event_execution.md` §17 flagged "AI action" events as structurally
accommodated but undesigned.

## 10. Interaction with MultiAgentSimulation

**Read-only, and only through the precomputed `MultiAgentSimulationResult`** (§2/§7/§8). The Runtime
never calls `add_occupant()`, `submit_decision()`, or any other mutating method on `context.simulation`
after Initialization step 3 — doing so would re-trigger the heap/generation machinery
`scenario_event_execution.md` §3 already established is the wrong shape for anything this Runtime needs.
`MultiAgentSimulation` itself is not modified, extended, or subclassed anywhere in this design.

## 11. A required new component: movement-timeline-backed OccupancyProvider

**Flagged explicitly, the same way `scenario_event_execution.md` §4 flagged and then resolved the
"second scheduler" question — this is not a redesign of `occupancy/` or `simulator/`, it is a new
implementation of an already-frozen interface, the exact extension mechanism `EvolutionBackedHazardProvider`
already demonstrates for `HazardProvider`.**

No existing type answers "how many occupants are at zone Z at time T" from a `MultiAgentSimulationResult`.
This Runtime needs that answer twice per tick (§8 step 3: once to feed `decide()`, again indirectly
through Perception's `GroundTruthCameraProvider`, §13). The shape required:

- **Input**: `MultiAgentSimulationResult.occupants: Dict[str, OccupantTimeline]`, each `OccupantTimeline.
  steps: List[OccupantTimelineStep]` (`from_node`/`to_node`/`start_time`/`end_time`), computed once,
  never mutated.
- **`snapshot_at(time) -> OccupancySnapshot`**: for each occupant, scan their `steps` to find where they
  are at `time` — `AT_NODE`-equivalent (between two steps, or before their first/after their last) counts
  toward that node's `occupant_count`; mid-traversal (`time` strictly between one step's
  `start_time`/`end_time`) is a modeling decision this document flags but does not resolve (candidates:
  count toward `from_node`, toward `to_node`, split fractionally, or omit — each has a defensible
  reading; **left to the implementation phase**, consistent with this document's "architecture, not
  implementation" scope).
- **Determinism**: a pure function of `(movement_result, time)`, no caching required (unlike
  `EvolutionBackedHazardProvider`, there is no expensive `evolve()` step to amortize — every query is a
  fresh scan of already-fully-known data), though an implementation may cache for performance without
  changing this contract.
- **Where it lives**: this document takes no position on package placement (`simulation_runtime/` itself,
  or a small addition to `occupancy/`) — a decision for the implementation phase, flagged into §21.

This is the occupancy-side counterpart to §3's hazard-side finding ("the first `evolve()` call is the
caller's job") — occupancy, too, has had no caller until now.

## 12. Interaction with HazardEvolutionEngine

Exactly `EvolutionBackedHazardProvider.snapshot_at(next_time)`, once per tick (§8 step 2), with the
Runtime's own `dt` (§6). No new interaction beyond what that class already provides — the Runtime is
simply the first regular caller of a seam that has existed, unused end-to-end, since `hazard_evolution/`
was built. `HazardEvolutionEngine`/`fire_growth/` are not modified.

## 13. Interaction with the Perception Layer — a genuine, flagged composition gap

**Every individual Perception primitive exists and is frozen; no class assembles them behind
`PerceptionProvider.observation_at(time)`.** This is structurally the same situation
`scenario_event_execution.md` opened with for event execution — a fully-designed set of parts with no
assembly point — except here the *design* for the assembly point already exists
(`perception_layer_review_2.md` §3's `PerceptionFusionEngine`, holding injected provider lists,
mirroring `HazardEvolutionEngine.sources`), it has simply never been built.

**This Runtime does not build it either** — constructing a `PerceptionFusionEngine`/concrete
`PerceptionProvider` is Perception's own implementation phase's job, not an orchestration concern, and
building it here would mean this "architecture-only" document silently deciding Perception implementation
details (provider wiring, zone-assignment resolution) that belong to that package, not to this one.
**What this Runtime does design**: the seam it calls *if* a `PerceptionProvider` is supplied —
`observation_at(next_time)`, called once per tick (§8 step 5), fed nothing but a `time` value, exactly
per `perception/providers/provider.py`'s own already-frozen contract. A concrete `PerceptionFusionEngine`
implementation, once built, needs no Runtime-side change to plug in — this is the same "the interface
was the contract all along" property `docs/architecture/perception_layer_review_2.md` §2's "hybrid"
recommendation was designed to guarantee.

`perception_provider` is `Optional` at Runtime construction (§7 step 8) specifically because of this gap
— a `SimulationRuntime` must be usable today, correctly, for everything except Perception-dependent
consumers (Dataset Logger fields that want a `BuildingObservation`, a future Firefighter Dashboard),
without waiting on Perception's own composition work to land.

## 14. Interaction with the Rule-Based Decision Engine

**Ground Truth in, `DecisionRecommendation` out — exactly as it exists today, not as
`perception_layer_review_2.md` §7.1 recommends it eventually become.** `AIDecisionEngine.decide(
hazard_snapshot, occupancy_snapshot, time)` (§3) is called once per tick (§8 step 4) with this tick's
`hazard_snapshot`/`occupancy_snapshot` — the same two objects §8 step 5 independently feeds into
Perception. **This is a deliberate, load-bearing reading of the current codebase, not an oversight**:
had this document instead routed Perception's `BuildingObservation` into `decide()`, it would be
redesigning `AIDecisionEngine`'s own signature — exactly the "without redesigning any approved subsystem"
constraint this pass's brief rules out. The Decision Engine and the Perception Layer are therefore
**parallel siblings in this Runtime's tick sequence, not a pipeline**, until a future, separate
`ai_decision/` migration (already recommended, not yet approved-and-built) changes `decide()`'s own
signature — at which point only step 4's call site changes, nothing else in this document.

## 15. Error handling

**Fail-fast, no silent recovery anywhere in the tick loop** — the same discipline every subsystem this
Runtime calls already enforces on itself (`ScenarioEventExecutor.advance_to()` raises `ValueError` on
backward time rather than no-op; `execute_event()` raises `UnsupportedEventTargetTypeError` rather than
skipping an unrecognized `target_type`; `resolve_profile()` raises `UnknownBehaviourProfileError` rather
than falling back to a default profile). This Runtime adds no new swallowing behavior on top of any of
them: an exception raised by any subsystem call in §8's sequence propagates immediately out of `tick()`,
halting the loop mid-tick. `current_time` is **not** advanced past a tick that raised — a caller
inspecting the Runtime after a failure sees `current_time` still at the last successfully completed
tick, consistent with the "no partial-tick recovery" principle in §9.

No new exception types are introduced by this document. Subsystem-specific exceptions
(`UnsupportedEventTargetTypeError`, `ValueError` from either provider's backward-time guard,
`UnknownBehaviourProfileError` — raised only during Initialization, before `tick()` exists) are allowed
to surface to the Runtime's own caller unwrapped, exactly as `scenario_event_executor`'s own design
already established for its callers.

## 16. Replay compatibility

Follows directly from every already-established determinism guarantee this Runtime composes, not from
any new mechanism of its own: `Scenario` generation determinism (`scenario_generator/`'s seed hierarchy),
`SimulationContext` construction determinism (`scenario_runner.md` §9), `MultiAgentSimulation.run()`'s
own determinism given fixed inputs (no randomness in `simulator/coordinator.py`), and
`ScenarioEventExecutor`'s determinism (`scenario_event_execution.md` §8/§13). **Replaying a run** means:
reconstruct `SimulationContext` from the same `Scenario`+`Building`, re-run `register_occupants()`,
construct a fresh `SimulationRuntime` (which re-runs `simulation.run()` once, §7 step 3 — itself
deterministic), and re-call `tick()` the same number of times. Every subsystem call in §8 is a pure
function of `(subsystem state, next_time)`, and `next_time` itself is a pure function of `(current_time,
dt)` — there is no wall-clock time, no unseeded randomness, and no external I/O anywhere in the tick
loop. `AIDecisionEngine.decide()` is likewise deterministic (no randomness in `ai_decision/engine.py`,
confirmed this pass) — a replayed run produces byte-for-byte identical `DecisionRecommendation`s at
every tick, not merely an equivalent occupant/hazard trajectory.

The one caveat: an eventual concrete `PerceptionProvider` implementation (§13) must itself preserve this
property (no randomness in a future vision-model noise simulation, or a seeded one) for replay to extend
through Perception's own output — flagged as a requirement on that future implementation, not resolved
by this document.

## 17. Future RL compatibility

This Runtime's `tick()` (§8) is already shaped as the seam a future RL training loop needs — one call
advances exactly `dt` and the result is available immediately after (via the Dataset Logger hook, §18,
or by a training loop reading `hazard_snapshot`/`occupancy_snapshot`/`observation`/`decision` directly
from whatever the Runtime exposes post-tick). **This document does not add an action-injection point** —
per `scenario_event_execution.md` §17, no `target_type`/schema exists yet for an "AI action" or
"firefighter action" event, and inventing one here would mean extending frozen `scenario`/
`scenario_definition` schema, explicitly out of bounds for an architecture pass scoped to *coordination*.
A future RL loop that wants to inject actions between ticks needs that schema question resolved first,
upstream of this Runtime, not a Runtime-level change.

Consistent with `perception_layer_review_2.md` §7.4/§8.2's own dependency-direction rule: **this Runtime
must never import Gymnasium, PyTorch, or any RL/ML framework.** A future `ObservationEncoder` (§7.2 of
that document) consumes whatever `BuildingObservation` this Runtime's Perception call (§13) produces,
entirely downstream of and outside this package — this Runtime's own responsibility ends at producing
correct, correctly-timed domain objects (`HazardSnapshot`, `OccupancySnapshot`, `BuildingObservation`,
`DecisionRecommendation`), never at shaping them for any specific model.

## 18. Future Dataset Logger integration

**No Dataset Logger precedent exists anywhere in this codebase** — confirmed by repository-wide search
this pass (no `DatasetLogger`, `dataset_logger`, or similar symbol found). This document therefore
defines only an **extension point**, not an implementation: an optional, constructor-injected object
(§7 step 8) whose `on_tick(...)` method (§8 step 6) the Runtime calls once per tick, after every other
subsystem call has completed, with exactly the artifacts that tick produced: `next_time`, `fired_events`
(`Tuple[ScenarioEvent, ...]`), `hazard_snapshot`, `occupancy_snapshot`, `decision`
(`DecisionRecommendation`), and `observation` (`BuildingObservation`, possibly `None` if no
`PerceptionProvider` was configured, §13). This mirrors the same "constructed once, injected, called
through one narrow method" pattern `HazardEvolutionEngine.sources`/`PerceptionFusionEngine`'s provider
lists already establish — the Runtime defines the *shape* of what a logger receives, never how (or
whether) it persists it. A concrete Dataset Logger — file-based, database-backed, in-memory for a
training loop — is future work, entirely outside this document's scope, and can be built without any
change to this Runtime's own design once the extension point above exists.

## 19. Dependency direction

- `simulation_runtime/` **may** import: `scenario_runner` (`SimulationContext`, a one-way downstream
  dependency — `scenario_runner` never imports this package), `scenario_event_executor`
  (`ScenarioEventExecutor`), `hazard_evolution` (`HazardEvolutionEngine`, `EvolutionBackedHazardProvider`),
  `hazard` (`HazardSnapshot`, `HazardProvider` — transitively required by the above), `simulator`
  (`MultiAgentSimulation`, `MultiAgentSimulationResult` — read-only, §10), `occupancy`
  (`OccupancyProvider`, `OccupancySnapshot` — for §11's new component), `ai_decision`
  (`DecisionEngine`/`AIDecisionEngine`, `DecisionRecommendation`), `perception` (`PerceptionProvider`,
  `BuildingObservation` — optional dependency, §13), `scenario` (`ScenarioEvent`, for typing `fired_
  events`), `models`/`navigation`/`pathfinding` (transitively, via the above).
- `simulation_runtime/` **must not** import `scenario_generator`, `scenario_validator`,
  `scenario_pipeline`, or `scenario_storage` — it never generates, validates, retries, or persists a
  `Scenario`; it consumes an already-built `SimulationContext`, exactly the same boundary
  `scenario_event_executor/` already established for itself.
- `simulation_runtime/` **must not** import `behaviour_profile_resolver` or `behavior`/`behavior_library`
  directly — occupant registration is complete before a `SimulationRuntime` is ever constructed (§7 steps
  1-2, explicitly the caller's responsibility, not this package's).
- `simulation_runtime/` **must not** import `sandbox` or `designer`.
- `simulation_runtime/` **must not** import Gymnasium, PyTorch, NumPy-as-a-training-dependency, or any
  other RL/ML framework (§17) — the same rule `perception_layer_review_2.md` §8.2 already places on
  `ai_decision/`, `perception/`, and any future RL package, extended here because this Runtime sits even
  closer to a future training loop than either.
- `simulation_runtime/` **must not** import `random` — every subsystem it calls is already deterministic
  given its own inputs (§16); this package introduces no sampling of its own.

## 20. Suggested package structure

Not a commitment to implement (this milestone is architecture-only):

```
simulation_runtime/
    __init__.py
    clock.py            # tick/dt/end_time bookkeeping (§5/§6/§9) -- no subsystem calls
    occupancy_bridge.py  # the movement_result-backed OccupancyProvider (§11) -- new, additive
    runtime.py           # SimulationRuntime: owns Initialization (§7) and tick() (§8)
```

`occupancy_bridge.py` could equally live under `occupancy/` as a new provider implementation, mirroring
where `ManualOccupancyProvider` already lives — left to the implementation phase (§21), the same way
`scenario_event_execution.md` §16 left `building_initializer.py`'s public-API question open.

## 21. Open questions for a future review

- **Mid-traversal occupancy resolution** (§11) — whether an occupant strictly between two
  `OccupantTimelineStep`s at query time counts toward `from_node`, `to_node`, both, or neither. A real
  behavioral decision with more than one defensible answer; not resolved here.
- **Package placement of the movement-timeline-backed OccupancyProvider** (§11/§20) — `occupancy/` vs.
  `simulation_runtime/` itself.
- **`PerceptionProvider` composition** (§13) — this document defines the seam this Runtime calls, not
  the `PerceptionFusionEngine`-shaped implementation Perception's own architecture already anticipates
  but has not built. A dedicated Perception implementation phase, not this Runtime, should close this.
- **`AIDecisionEngine`'s eventual migration to `BuildingObservation`** (§14) — recommended by
  `perception_layer_review_2.md` §8.2, not yet approved or built. When it happens, only §8 step 4's call
  site in this Runtime changes.
- **Early-stop predicates beyond `end_time`** (§9) — e.g., an RL episode-termination condition, or a
  "stop once every zone is safe" criterion. Structurally could be a caller-supplied predicate checked
  alongside §9's primary condition; not designed here.
- **Action-injection for a future RL/firefighter-action loop** (§17) — blocked on a `scenario`/
  `scenario_definition` schema question already flagged, unresolved, in `scenario_event_execution.md`
  §17, not something this document can resolve on its own.
- **Dataset Logger's actual persistence mechanism** (§18) — this document defines only the call shape;
  no storage format, batching policy, or file layout is designed here.

## 22. Status

Proposal only. Nothing in this document has been implemented; `scenario_runner`, `scenario_event_
executor`, `behaviour_profile_resolver`, `simulator`, `hazard_evolution`, `perception`, and `ai_decision`
are all unmodified by it. The design resolves the "how do three different advancement shapes become one
coordinated tick" question (§2/§8) by recognizing occupant movement does not need ticking at all, and
flags — rather than silently resolves — two genuine, pre-existing composition gaps this pass's repository
inspection surfaced: no time-varying `OccupancyProvider` exists (§11), and no concrete `PerceptionProvider`
implementation exists (§13). Both are new, additive components consistent with this codebase's own
established extension pattern (a new implementation of an already-frozen interface — exactly what
`EvolutionBackedHazardProvider` already demonstrates), not redesigns of any approved subsystem — but
building either is implementation work, explicitly out of scope for this document, and is not done here.

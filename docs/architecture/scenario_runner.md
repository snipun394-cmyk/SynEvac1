# Scenario Runner — Architecture Proposal

Status: **proposal, open for review**. No code changes accompany this document — this is an
architecture review only, per this milestone's own instruction ("Do NOT implement anything. Stop
after the architecture review.").

**This pass's refinement**: two changes from the prior revision, both narrowing the Runner's own
scope further. (1) The Runner no longer owns, consumes, or even imports anything Behaviour-Profile-
shaped — §7 is rewritten to move that responsibility to a separate, future **Behaviour Profile
Resolver**, downstream of the Runner, not inside it. (2) The Runner's output is no longer something
already wired up to `HumanBehaviorLayer` — it is a new, explicitly intermediate, immutable
**`SimulationContext`** (§4), and completing occupant registration is now a distinct phase the Runner
does not perform. Both changes are recorded inline, section by section, rather than only summarized
here — search this document for "this pass" to find every touched section.

## 1. Purpose

The Scenario Runner is the bridge between the Scenario Engine (`scenario/`, `scenario_definition/`,
`scenario_generator/`, `scenario_validator/`, `scenario_pipeline/`, `scenario_storage/` — all frozen,
approved, and untouched by this document) and the Simulation Engine (`simulator/`, `behavior/`,
`behavior_library/`, `hazard_evolution/`, `fire_growth/`, `navigation/`, `pathfinding/` — all
pre-existing, likewise untouched).

**One governing principle, mirrored from every prior Scenario Engine freeze pass**: the Runner is a
*translation and wiring* layer — it takes an already-accepted `Scenario` and turns its stored,
resolved state into the concrete objects the existing Simulation stack already knows how to consume.
It contains no decision of its own about what a Scenario *should* look like (that is Generation/
Validation, both frozen, both untouched) and no decision about what happens *during* a run (that is
Simulation execution, Behaviour decision-making, Fire physics — none of which this document designs
or touches). Four things the Runner is explicitly not, mirroring §4.1/§5.1's own structure from the
Scenario Engine documents:

- **Not a generator.** It never samples a value, never invents an occupant, a fire origin, or a door
  state — every value it applies is already sitting on the `Scenario` it was handed.
- **Not a validator.** It never checks whether the `Scenario` it receives is acceptable — that
  already happened, upstream, before the `Scenario` was ever persisted (`scenario_validator`,
  `scenario_pipeline`). The Runner trusts its input completely; if that trust is ever misplaced, that
  is a defect in whatever produced the `Scenario`, not something this layer re-checks.
- **Not a simulator.** It never steps time forward, never resolves a movement decision, never
  computes a hazard contribution. It assembles the *structural* pieces a simulation needs (graph,
  engine, hazard source, occupant data) and stops there — short even of full run-readiness, since
  behaviour registration remains outside its scope (§4/§7) — the same boundary §4.1 already drew for
  the Scenario Generator, one layer further downstream.
- **Not a decision-maker, and — this pass — not an interpreter either.** The prior revision had the
  Runner perform a *lookup* to turn a `behaviour_profile_id` into a concrete behavioural configuration.
  This pass corrects that: the Runner performs **no lookup at all**. `behaviour_profile_id` passes
  through it exactly as opaque as it arrives — the Runner does not know, and must never come to know,
  that the string means anything. Interpretation belongs entirely to a separate, future **Behaviour
  Profile Resolver** (§7), downstream of the Runner, which in turn hands concrete strategies to
  `HumanBehaviorLayer`/`DecisionStrategy` exactly as today. The full chain, this pass:

```
Scenario
   │
   ▼
Scenario Runner            (this document -- produces a SimulationContext, §4;
                             behaviour_profile_id carried through, never read)
   │
   ▼
SimulationContext            (immutable hand-off artifact, §4)
   │
   ▼
Behaviour Profile Resolver   (a future, separate subsystem -- §7; the ONLY place
                             a behaviour_profile_id is ever interpreted)
   │
   ▼
Human Behavior Layer         (unchanged, existing -- behavior/orchestrator.py)
   │
   ▼
Simulation                   (unchanged, existing -- simulator/coordinator.py;
                             now fully registered and ready to run())
```

## 2. Grounding in existing code

This section records what already exists and what it does and does not already support — the
starting point every subsequent section reasons from.

**`scenario_storage/`** (frozen, §5 of the Scenario Engine documents' package list) already provides
exactly "Loading a Scenario" (this milestone's first responsibility): `load_scenario_by_id`/
`load_scenario_by_filename` return a fully deserialized `Scenario`. The Runner does not re-implement
loading; it consumes this package's existing public API directly.

**`navigation/graph_builder.py::NavigationGraphGenerator.build(building) -> NavigationGraph`** derives
a graph from a `Building` purely from Door/Exit/Staircase connectivity references
(`zone_a_id`/`zone_b_id`, `zone_id`, `from_zone_id`/`to_zone_id`) — **never** from Door.normally_open/
locked, Exit.is_blocked, or any other state field. Rebuilding from the same `Building` always
produces the same graph (its own documented guarantee).

**`navigation/edge.py::Edge.traversable`** is a property that reads state **live off `self.reference`**
— literally `reference.locked`/`reference.active` for a Door, `reference.is_blocked` for an Exit,
always `True` for a Stair (Staircase has no blocking/availability field at all in
`models/staircase.py` — confirmed again during this review, matching what
`scenario_validator/navigation_validation.py` already found and worked around the same way §4
below does). This is the mechanism `pathfinding/engine.py::PathfindingEngine._relax()` uses
(`if not edge.traversable: continue`) to decide what is walkable — and it reads the **Building's own
live state**, not anything Scenario-shaped. There is no pluggable seam here (no `CostModel`-style
override for traversability, only for `cost()`) — `scenario_validator` already discovered this exact
limitation and built its own parallel BFS rather than mutate the Building it was checking (§5.1: "the
Validator never modifies anything"). The Runner faces the identical limitation but, unlike the
Validator, mutating engineering state *is* one of its stated responsibilities — §4 resolves this
directly, using the one mechanism the Validator was correctly forbidden from using.

**`simulator/coordinator.py::MultiAgentSimulation(engine, ...)`** — `engine` is a `PathfindingEngine`.
Registration is either `add_occupant(start_id, goal_id=None, walking_speed=None, occupant_id=None,
depart_time=0.0, route=None)` (direct, bypassing Behaviour entirely) or `submit_decision(decision:
BehaviorDecision)` (the Human Behavior Layer's sole entry point). **Critically: `simulator.occupant.
Occupant` carries no `(x, y)` position field at all** — registration only ever names a `start_id`
(a Navigation Graph **node** id, i.e. a `Zone.id`). Position is a concept that exists nowhere in
`simulator/`.

**`sandbox/occupant.py::SandboxOccupant.origin_position`** is the one existing precedent for *why*
`(x, y)` position matters even though the simulation core never touches it — its own comment is
explicit: "the Zone a route starts in has a single Navigation Graph node with one center point, but
many occupants can be placed at many distinct points within that same Zone. Interpolating the first
edge from the Zone's shared center rather than from here would collapse every occupant in the same
Zone onto one identical starting position." `SandboxOccupant` itself is explicitly documented as
belonging only to the Manual Simulation Sandbox and must never be reused for this purpose — but the
*reason* it carries a position is exactly the Runner's own reason to carry one too (§6).

**`behavior/orchestrator.py::HumanBehaviorLayer.register(start_id, profile, decision_strategy,
route_choice_strategy=None, pre_movement_strategy=None, base_depart_time=0.0)`** requires the *caller*
to already have a concrete `BehaviorProfile` (`behavior/profile.py`) plus concrete strategy instances
— it has no notion of a named preset. This is the exact gap the Scenario Engine architecture already
flagged and explicitly deferred: `docs/architecture/scenario_engine.md` §13, "Behaviour Profile
registry design ... where the id-to-config resolution lives ... flagged, not designed." **This pass's
correction**: the prior revision of this document resolved that gap by having the *Runner* consume the
registry. That placement is wrong — it would require this package to import `behavior`/
`behavior_library` and know what a `BehaviorProfile`/`DecisionStrategy` even is, directly
contradicting "the Runner must remain completely agnostic to the meaning of behaviour profiles." §7
now places this resolution in a separate Behaviour Profile Resolver, downstream of the Runner, that
this package neither imports nor depends on in any way.

**`behavior/context.py::DecisionContext`** already has an optional `hazard_snapshot:
Optional[HazardSnapshot]` field — but `HumanBehaviorLayer.register()`, as it exists today, never
populates it; every call site constructs a `DecisionContext` with `hazard_snapshot` left at its
default `None`. Wiring a Fire-derived `HazardSnapshot` through `register()` would require a change to
`behavior/orchestrator.py` itself — outside every `scenario_*` package and outside this document's
scope to design (see §12).

**`fire_growth/model.py::FireGrowthModel(ignition_node_id, ignition_time, growth_curve=None)`** plus
`fire_growth/growth_curve.py::TSquaredFireGrowthCurve(growth_time)` together are exactly the object
`docs/architecture/scenario_engine.md` §4.2 predicted the Simulator would build from a `ScenarioFire`'s
plain data ("it is the Simulator's job, at simulation start, to turn that data into a
`FireGrowthModel`"). This document is where that prediction resolves concretely (§8):
`ignition_node_id` is `ScenarioFire.ignition_zone_id` (already-established equivalence, §2 of the
Scenario Engine documents: "`FireGrowthModel.ignition_node_id` already equals `Zone.id` for zone
nodes"), and `growth_time` is `ScenarioFire.growth_parameters["growth_time"]` (the one key
`scenario_generator` populates, per its own documented convention).

**`hazard_evolution/engine.py::HazardEvolutionEngine(sources=[...])`** is a plain, stateless-between-
constructions orchestrator over whatever `HazardSource` instances it is given (a `FireGrowthModel`
satisfies this interface, §2 of `fire_growth/model.py`'s own docstring: "implements `HazardSource`").
It has no built-in notion of a *schedule* — `evolve(snapshot, time, dt)` is a pure, single-step
function; nothing about it or `MultiAgentSimulation`'s own event heap (confirmed, again, to be
occupant-movement-only — `TRY_ENTER_EDGE`/`ARRIVE_AT_NODE`, hardcoded) can execute an arbitrary
Scenario-level scheduled event ("door closes at t=90s"). **No mechanism for executing a
`ScenarioEvent` exists anywhere in the codebase today.** §10 treats this as a genuine, currently-
unclosable gap, not something this document can silently paper over.

**`models/building.py`** has `to_dict()`/`from_dict()` (used elsewhere, e.g. `Floor`'s own deep-copy
via `Floor.from_dict(floor.to_dict())`, per `models/building.py`'s own comment) — a ready-made,
already-precedented deep-copy mechanism (§5).

## 3. Input contract

The Runner receives exactly two things:

1. **A `Scenario`** (`scenario/`, frozen) — either handed directly by a caller that already has one
   in memory (e.g. straight from `scenario_pipeline.run_pipeline()`), or loaded via
   `scenario_storage.load_scenario_by_id()`/`load_scenario_by_filename()` — "Loading a Scenario" is
   satisfied by reusing that existing API, not by this package re-implementing it.
2. **A `Building`** (`models/`) — the same building the `ScenarioDefinition` that produced this
   `Scenario` was defined against. The Runner does not resolve *which* Building that is (no
   `building_id` exists on `Scenario`/`ScenarioMetadata` any more than it existed on
   `ScenarioDefinition` — the same caller-supplies-it resolution already used throughout the Scenario
   Engine, §4.2/§5.2 of the Scenario Engine documents, applies identically here).

Nothing else. No seed, no `ScenarioDefinition`, no `accepted_hashes` — those belong entirely to
Generation/Validation and have no role once a `Scenario` is already accepted and being handed to the
Simulation Engine.

## 4. Output contract — `SimulationContext`

**Reviewed, this pass: should the Runner produce something already wired up to
`HumanBehaviorLayer`/`MultiAgentSimulation` registration, or a dedicated, immutable initialization
object that stops short of that?** The second — for a reason stronger than tidiness: **the prior
revision's output could not actually be built without the Runner first resolving
`behaviour_profile_id`**, since `HumanBehaviorLayer.register()`/`MultiAgentSimulation.submit_decision()`
both require an already-concrete behavioural configuration to register an occupant at all. Once §7
moves that resolution out of the Runner, the Runner **structurally cannot** produce a fully-registered,
run-ready simulation any more — it can only get as far as a simulation whose graph, engine, hazard
source, and Building state are correctly assembled, with occupants *described* but not yet
*registered*. This is not a stylistic choice, it is the direct, mechanical consequence of §7's own
refinement: **initialization and execution-readiness are now two separate phases because they have to
be** — the Runner alone no longer has enough information to produce the second.

The Runner therefore produces one **`SimulationContext`** — an immutable (frozen) bundle, matching
every other Scenario Engine model's own convention (`scenario/`, `scenario_definition/`), rather than
a live, already-populated simulation:

| Field | Type | Source |
|---|---|---|
| `graph` | `NavigationGraph` | Built from the Scenario-applied Building copy (§5) |
| `engine` | `PathfindingEngine` | Wraps `graph` |
| `simulation` | `MultiAgentSimulation` | Constructed and wrapping `engine` — **empty**: no `add_occupant()`/`submit_decision()` call has been made by the Runner (§6) |
| `occupants` | `Tuple[ScenarioOccupant, ...]` | `scenario.occupants`, verbatim — `zone_id`/`position`/`behaviour_profile_id` all carried through unread (§6) |
| `hazard_engine` | `HazardEvolutionEngine` | Sources include the Scenario's `FireGrowthModel` (§8) |
| `initial_hazard_snapshot` | `HazardSnapshot` | `timestamp=0.0`, empty — the Fire source has not yet been evolved even one step |
| `scheduled_events` | `Tuple[ScenarioEvent, ...]` | Time-ordered, prepared but not executed (§10) |
| `building` | `Building` | The **copy** the graph was built from, never the caller's original (§5) |

Note what is **absent**, deliberately, compared to the prior revision: no `behavior_layer` field (the
Runner never constructs a `HumanBehaviorLayer` — §7), no separate `occupant_positions` mapping (each
`ScenarioOccupant` already carries its own `position`, so a parallel structure would only duplicate
it, §6).

`simulation` being a **mutable** `MultiAgentSimulation` instance living inside an **immutable**
`SimulationContext` is the same pattern already established elsewhere in this codebase (e.g.
`scenario_pipeline.PipelineResult`, frozen, holding a mutable `ScenarioValidationReport`) —
`SimulationContext`'s own fields are never reassigned after construction; what a *later* phase does to
the objects those fields point to (registering occupants onto `simulation`) is expected, ordinary
runtime mutation of a component designed to accumulate state, not a violation of the container's own
immutability.

`SimulationContext` is not run-ready. Turning it into a fully registered, executable simulation
requires the Behaviour Profile Resolver (§7) to run first — assembling `SimulationContext` performs no
simulation step and no behavioural registration; both remain entirely outside this document's scope.

## 5. Building-state mutation and the deep-copy rule

**Compliance finding, this review.** "Applying engineering object states" (this milestone's second
responsibility) can only be realized through `Edge.traversable` reading `reference.locked`/`.active`/
`.is_blocked` *live off the Building's own engineering objects* (§2) — there is no other seam. This
means the Runner **must** write the Scenario's resolved door/exit/obstacle/camera/detector state onto
`Door.normally_open`/`.locked`, `Exit.is_blocked`, `Obstacle.active`, `Camera.active`, `Detector.active`
before `NavigationGraphGenerator.build()` ever runs.

**This is a mutation of engineering-model state — the exact operation §5.1 of the Scenario Validator
document forbade the Validator from performing, for exactly the reason that still applies here: doing
it to the caller's live `Building` would silently corrupt whatever else holds a reference to it (a
Designer window, a Project the user has open, a `Building` some other in-flight Runner call is also
reading).** The frozen resolution: **the Runner always operates on a deep copy**, never the original.

```
building_copy = Building.from_dict(building.to_dict())
```

(matching the exact pattern `models/floor.py` already uses internally for its own deep-copy need,
§2) — an implementation detail; the *rule* (never mutate the caller's `Building`) is what this
document fixes, not the specific copy mechanism.

**The six categories, concretely:**

- **Door** — `door_states` entries set `Door.locked = (state == LOCKED)`,
  `Door.normally_open = (state == OPEN)` on the matching `door_id` in `building_copy`.
- **Exit** — `exit_states` entries set `Exit.is_blocked = not is_open`.
- **Obstacle** — `obstacle_states` entries set `Obstacle.active = (presence == ACTIVE)`. No
  `NavigationGraph` edge currently reads this field at all (Obstacles are not connectivity elements,
  §5.3 of the Scenario Validator document already established this) — the Runner still applies it
  uniformly (a future fine-grained pathing/perception layer may consume it; the Runner's job is to
  make the Building copy *faithfully reflect* the Scenario, not to know which downstream reader cares
  about which field).
- **Camera / Detector** — `camera_states`/`detector_states` entries set `Camera.active`/
  `Detector.active`. Not consumed by Navigation at all — consumed (if anywhere yet) by `perception/`,
  which this document does not touch or design against ("must never perform perception").
- **Stair — the one exception.** `models/staircase.py` has no availability/blocked field of any kind
  (confirmed again this review) — there is *nothing on the Building copy to set*. A closed stair
  cannot be represented as engineering-object state the way the other five can. **Resolution**: the
  Runner instead removes/excludes the corresponding `Edge` (`edge_type == Edge.STAIR`, matching
  `stair_id`) from the **already-built** `NavigationGraph`, after `NavigationGraphGenerator.build()`
  runs, for every `stair_id` whose resolved `StairAvailability` is `CLOSED`. This is a graph-level
  post-processing step, not a Building mutation — the one place in this document's design where
  "applying engineering state" happens after graph construction rather than before it, and the
  document records this as a deliberate, narrow exception rather than folding it silently into the
  uniform Building-copy mechanism above.

## 6. Occupant spawning and positioning

**Compliance finding, this review — "positioning uniformly" does not mean re-sampling.** A
`ScenarioOccupant.position` is **already** a uniformly-sampled `(x, y)` point inside its zone's
bounding box — sampled once, by `scenario_generator` (`docs/architecture/scenario_engine.md` §4.4's
"Generate Occupants" stage), and frozen onto the `Scenario` from that point on. Read literally
against this milestone's own Non-Responsibilities ("must never generate scenarios"), the Runner
**must not** draw a new random position — doing so would be exactly the kind of Scenario-content
sampling that belongs exclusively to `scenario_generator`, one layer upstream, already done. Both
"Positioning occupants uniformly" and "Applying Behaviour Profile identifiers" resolve, in this
document, to the same thing: **verbatim carry-through of already-resolved Scenario data**, never new
computation.

**This pass narrows spawning further still.** The prior revision had this section resolve
`behaviour_profile_id` and call `HumanBehaviorLayer.register()` directly. Per §7's refinement, the
Runner now performs **no registration at all** — "spawning" reduces to exactly one mechanical step:

- `SimulationContext.occupants` is set to `scenario.occupants`, **unchanged, field for field**. No new
  type is introduced for this — `ScenarioOccupant` (`occupant_id`, `zone_id`, `floor_id`, `position`,
  `behaviour_profile_id`) already has exactly the shape a later phase needs; inventing a parallel
  Runner-owned occupant type here would only duplicate it for no reason.

That is the entirety of this section's responsibility now. Two things worth stating explicitly since
they are easy to lose after this narrowing:

- `occupant.zone_id` is already a Navigation Graph node id (`Node.id` reuses `Zone.id` throughout, §2
  of the Scenario Engine documents) — whatever later constructs a `start_id` for registration (the
  Behaviour Profile Resolver, or whatever drives it) needs no translation, only `occupant.zone_id` as
  already stored.
- `occupant.position` needs no separate tracking structure any more (§4) — it travels with the rest of
  `ScenarioOccupant` inside `context.occupants`. It remains the Runner's/this pipeline's equivalent of
  `SandboxOccupant.origin_position` (§2) for whichever future replay/visualization layer needs it, to
  avoid every occupant in a Zone visually collapsing onto that Zone's single graph-node center — the
  *reason* it must be carried through is unchanged from the prior revision; only *where* it lives
  (inline on each occupant record, not a parallel mapping) has simplified.

## 7. Behaviour Profile interpretation belongs to a separate Behaviour Profile Resolver

**Revised, this pass.** The prior revision of this document placed the id-to-config resolution
(`docs/architecture/scenario_engine.md` §13's own previously-flagged, previously-undesigned gap)
inside the Scenario Runner, reasoning that the Runner was "the first and only point in the whole
system where a `behaviour_profile_id` string is ever interpreted." That placement is withdrawn. On
reflection it conflated two genuinely different jobs that happen to run one after another:
**initializing the mechanical shape of a simulation** (this document's actual scope) and
**interpreting what a behaviour profile means** (a Behaviour-Layer-adjacent concern, sharing nothing
with graph-building, Building-state mutation, or Fire initialization). Housing both in one package
would have made `scenario_runner/` depend on `behavior`/`behavior_library` and know what a
`BehaviorProfile`/`DecisionStrategy` is — directly at odds with keeping the Runner "completely
agnostic to the meaning of behaviour profiles."

**The corrected chain** (already shown in §1): `Scenario → Scenario Runner → SimulationContext →
Behaviour Profile Resolver → Human Behavior Layer → Simulation`. The Scenario Runner's own
responsibility for behaviour profiles ends exactly where this milestone's brief says it should:
`SimulationContext.occupants` carries each `ScenarioOccupant.behaviour_profile_id` through, string for
string, string being the entire extent of what the Runner ever knows about it (§6).

**The Behaviour Profile Resolver — a future, separate subsystem, not designed here.** Conceptually, it
is whatever consumes a `SimulationContext` and, for each `occupant` in `context.occupants`:

1. Resolves `occupant.behaviour_profile_id` into a concrete `(BehaviorProfile, DecisionStrategy,
   RouteChoiceStrategy, PreMovementDelayStrategy)` bundle (`docs/architecture/scenario_engine.md`
   §13's still-open "where do the registry's entries come from" question — unresolved by this
   document, and now not this document's question to resolve at all).
2. Constructs a `HumanBehaviorLayer(context.simulation, context.engine)` (trivial wiring — no
   behavioural interpretation happens at that constructor call) and calls `.register(start_id=
   occupant.zone_id, profile=..., decision_strategy=..., route_choice_strategy=...,
   pre_movement_strategy=...)` for each occupant, which in turn drives
   `context.simulation.submit_decision(...)` — completing registration onto the same `simulation`
   instance the Runner already constructed and handed over inside `SimulationContext`.
3. An unrecognized `behaviour_profile_id` is a **hard error** at this stage — no repair, no invented
   default, mirroring the "no repair, anywhere" principle already frozen throughout the Scenario
   Engine. This is now the Resolver's failure mode to define, not the Runner's.

**Why this is a strict improvement, not just a relocation.** `scenario_runner/` no longer needs to
import `behavior` or `behavior_library` at all (§12) — its dependency surface shrinks, and its output
(`SimulationContext`) becomes usable by *any* future behaviour-resolution strategy without the Runner
itself ever having been written with one particular strategy in mind. A `Scenario` generated today
can be re-run under a completely different Behaviour Profile Resolver implementation later without
touching `scenario_runner/` at all — the same "same Scenario, different Behaviour Model" independence
`docs/architecture/scenario_engine.md` §8 already established between Generation and Behaviour, now
extended one layer further downstream to Running and Behaviour as well.

**Still genuinely out of scope, unchanged from the prior revision**: where the registry's own entries
come from, how `"Adult_Default"` specifically maps to a walking-speed distribution or a
`ShortestRouteChoiceStrategy` instance, and how a `BehaviorGroup` (`behavior/group.py`) — which
`ScenarioOccupant` has no equivalent field for at all — would ever get populated from Scenario data.
All three are now explicitly the Behaviour Profile Resolver's own future architecture to design, not
a Scenario Runner concern even indirectly.

## 8. Fire initialization

Concretely resolves the deferral `docs/architecture/scenario_engine.md` §4.2 left open:

```
growth_curve = TSquaredFireGrowthCurve(scenario.fire.growth_parameters["growth_time"])
fire_source  = FireGrowthModel(
    ignition_node_id = scenario.fire.ignition_zone_id,
    ignition_time    = 0.0,   # scenario-relative start; the Runner never
                               # invents a non-zero ignition delay
    growth_curve     = growth_curve,
)
hazard_engine = HazardEvolutionEngine(sources=[fire_source])
```

`fire_source` is registered as one `HazardSource` among possibly others (`hazard_evolution/engine.py`
already supports multiple sources; the Runner does not preclude a future Smoke Propagation source
being added alongside it, it simply does not construct one itself — smoke is not part of a
`Scenario`'s stored state). `initial_hazard_snapshot = HazardSnapshot(timestamp=0.0)` — empty, since
the fire has not yet been evolved even one step at initialization time; the *first* `evolve()` call
(the caller's job, not the Runner's) is what actually produces a non-trivial snapshot.

**`scenario.fire.fire_profile` remains genuinely unconsumed.** `docs/architecture/scenario_engine.md`
§9 already documented this as deferred ("still unconsumed by anything beyond echoing the sampled
profile onto the Scenario; its effect on `FireGrowthCurve` parameterization remains deferred") — this
review confirms nothing in `fire_growth/` has grown a profile-aware curve selection mechanism since,
so the finding stands unchanged. The Runner carries `fire_profile` through onto whatever provenance/
metadata a caller might want (it is not silently dropped), but has no mechanism to let it influence
*which* `FireGrowthCurve` gets constructed or how. Flagged again in §13, not resolved here.

## 9. Resetting Simulation state before initialization

**Reviewed: what does "reset" concretely mean, given the current stack?** Every object in §4's
`SimulationContext` is constructed **fresh, from scratch, on every Runner call** — `NavigationGraph`
(`NavigationGraphGenerator.build()` already guarantees "rebuilding the graph from the same Building
always produces the same graph," §2), `MultiAgentSimulation.__init__()` (already initializes every
internal collection to empty, §2 of `simulator/coordinator.py` — and, this pass, the Runner never
calls `add_occupant()`/`submit_decision()` on it at all, §4/§6, so it stays empty until the Behaviour
Profile Resolver populates it), `HazardEvolutionEngine`, `HazardSnapshot` (immutable). **None of these
carry hidden global or module-level mutable state that would need an explicit `reset()` call** —
unlike `sandbox/manager.py::SandboxManager`, which is a genuinely long-lived, stateful object with its
own documented reset concept for exactly that reason. The Runner is not that: it is a **stateless
function of `(Scenario, Building)`** — call it twice with the same inputs and it must produce two
independently-constructed, behaviourally-identical bundles, neither aware the other exists. (This
pass's narrowing makes this an even smaller claim than before: the Runner no longer constructs a
`HumanBehaviorLayer` at all, §7, so there is one fewer object whose freshness this section needs to
account for.)

**The one concrete rule "reset" resolves to, this review**: the Runner must never accept, or be
tempted to reuse, a caller-supplied `NavigationGraph`/`PathfindingEngine`/`MultiAgentSimulation` from
a prior run — it always rebuilds every layer of §4's bundle from the `(Scenario, Building)` pair it
was given, even if a superficially "compatible" graph already exists somewhere in the caller's
process. This, combined with §5's deep-copy rule, is what "reset" means for a component that has no
mutable state of its own to clear: starting completely fresh *is* the reset.

## 10. Scheduling Scenario Events — a genuine, unresolved gap

**This is the review's most significant finding.** "Scheduling Scenario Events" is listed as a
Runner responsibility, and "executing" one is explicitly excluded ("must never execute simulation
logic") — a clean split *in principle*. In practice, §2 already established that **no mechanism to
execute an arbitrary `ScenarioEvent` exists anywhere in the current codebase**:
`MultiAgentSimulation`'s event heap is hardcoded to two occupant-movement event kinds only;
`HazardEvolutionEngine.evolve()` has no concept of a schedule, only a single `(snapshot, time, dt)`
step. There is no "Scenario Event Scheduler" component for the Runner to hand a prepared schedule to.

**What the Runner can do today, and what this document commits to**: `scenario.events` (already
resolved, already time-ordered data — `scenario_validator`'s own Event Validation already checks this
ordering, §5.3 of the Scenario Validator document) is carried into `SimulationContext`'s
`scheduled_events` field verbatim. This satisfies "scheduling" in the narrowest honest sense —
*preparing* the data in the shape a future executor would need — without inventing an execution path
that does not exist. **What this document explicitly does not do**: design that future executor.
Doing so would require either extending `MultiAgentSimulation`'s event heap to a general-purpose
scheduled-callback mechanism, or building a new, separate component that steps `scenario.events`
forward in lockstep with whatever drives `hazard_engine.evolve()`/`simulation.run()` — both are
Simulation-execution-layer design questions, squarely outside a Scenario Runner's own scope, and are
flagged into §13 rather than decided here.

## 11. Suggested package structure

Not a commitment to implement (this milestone is architecture-only) — recorded for continuity with
how every other Scenario Engine package's suggested structure has been proposed and then followed:

```
scenario_runner/
    __init__.py
    runner.py              # run(scenario, building) -> SimulationContext; the one entry point
    simulation_context.py   # SimulationContext (§4)
    building_state.py       # deep-copy + apply door/exit/obstacle/camera/detector state (§5)
    occupant_spawner.py     # carries scenario.occupants through verbatim (§6) -- no behaviour
                             # resolution, no registration; see §7 for why neither lives here
    fire_initializer.py     # FireGrowthModel/HazardEvolutionEngine construction (§8)
    event_scheduler.py      # prepares scheduled_events; contains no execution (§10)
```

**This pass removes `behaviour_registry.py`** from the prior revision's suggested structure entirely
— there is no registry-consuming code left in this package to house (§7).

## 12. Dependency direction

- `scenario_runner/` **may** import: `scenario` (the `Scenario` shape it consumes), `scenario_storage`
  (for the "loading" convenience path only), `models` (`Building` and its engineering objects),
  `navigation`, `pathfinding`, `simulator`, `hazard_evolution`, `fire_growth` — every one of these is a
  *consumption* of an already-existing construction API (`NavigationGraphGenerator.build()`,
  `PathfindingEngine(...)`, `MultiAgentSimulation(...)`, `HazardEvolutionEngine(...)`,
  `FireGrowthModel(...)`), never a reimplementation of what any of them do.
- **`scenario_runner/` must not import `behavior` or `behavior_library` — corrected, this pass.** The
  prior revision listed both as allowed, needed for the Runner's now-withdrawn registry-consumption
  role (§7). With that role removed, there is nothing left in this package that has any reason to
  import either: it never constructs a `BehaviorProfile`, a `DecisionStrategy`, or a
  `HumanBehaviorLayer`. This is the direct, mechanical dependency-graph consequence of §7's
  refinement, not a separate decision — the "completely agnostic to the meaning of behaviour
  profiles" requirement is enforced exactly here, by there being no import path to violate it through.
- `scenario_runner/` **must not** import `scenario_generator`, `scenario_validator`, or
  `scenario_pipeline` — the Runner never generates, never validates, and never orchestrates a
  retry loop; a `Scenario` reaching it is already a finished, accepted artifact. (`scenario_storage`
  is the one exception, imported only for its read-side loading convenience, not for anything
  persistence-adjacent to writing.)
- `scenario_runner/` **must not** import `ai_decision`, `perception`, `sensors`, `occupancy`, or `rl`
  — none of those are part of *initializing* a simulation; every one of them is either a runtime
  consumer of the Building/Simulation state the Runner sets up, or entirely unrelated to this
  milestone's scope.
- `scenario_runner/` **must not** import `sandbox` or `designer` — `SandboxOccupant` is explicitly
  documented as belonging only to the Manual Simulation Sandbox (§2); `SimulationContext.occupants`
  reuses `scenario.ScenarioOccupant` directly (§4/§6), never that type.
- `scenario_runner/` **must not** import `random` — every value it applies is already resolved on
  the `Scenario` it was handed; it has no sampling of its own to perform (§6's compliance finding).

## 13. Open questions for a future review

- **Scenario Event execution** (§10) — the largest gap this review found. Needs a dedicated design
  pass, most likely as a Simulation-layer (not Scenario-layer) concern: either `MultiAgentSimulation`
  grows a general scheduled-callback mechanism, or a new component steps events forward alongside
  `hazard_engine.evolve()`. Not designed here.
- **The Behaviour Profile Resolver itself** *(new, this pass, §7)* — this document now firmly places
  id-to-config resolution *outside* the Scenario Runner, but the Resolver's own architecture (its
  package location, its registry's contents and authoring mechanism, its own input/output contract
  against `SimulationContext`) is not designed here at all — a genuinely separate, future milestone,
  not merely an unresolved detail within this one.
- **`BehaviorGroup` from Scenario data** (§7) — `ScenarioOccupant` has no group-membership field;
  group-aware behaviour (leader/follower, families) cannot currently be expressed from a generated
  Scenario at all. Flagged, not designed — would need either a convention layered on top of
  `behaviour_profile_id` or a (frozen, so not proposed lightly) `scenario/` schema addition. Now
  squarely the Behaviour Profile Resolver's question, not the Runner's.
- **`hazard_snapshot` threading through `HumanBehaviorLayer.register()`** (§2) — `DecisionContext`
  already supports it; `register()` does not populate it yet. Whether/how `SimulationContext.
  initial_hazard_snapshot` (§8) should reach occupant registration is a `behavior/orchestrator.py`
  change, outside every `scenario_*` package and, now, outside the Runner even conceptually — it
  would be the Behaviour Profile Resolver's own concern to pass `context.initial_hazard_snapshot`
  through when it calls `register()`, not something the Runner threads through itself.
- **`fire_profile` → `FireGrowthCurve` selection** (§8) — still deferred, unchanged from
  `docs/architecture/scenario_engine.md` §9.
- **Stair-edge exclusion mechanics** (§5) — that closed stairs must be excluded post-graph-build is
  fixed; whether that is a `NavigationGraph.remove_edge()` method added to `navigation/`, or a
  Runner-local filter over `graph.edges`, is an implementation detail not decided here.
- **Multi-source Hazard registration** (§8) — the Runner registers only the Scenario's Fire source
  today; whether/how a future Smoke Propagation source gets added alongside it (and whether that
  becomes the Runner's job or a caller's) is not decided here.

## 14. Status

Proposal only. Nothing in this document has been implemented; no existing package (Scenario Engine or
Simulation Engine) is modified by it.

**This pass's two refinements, both adopted**: (1) Behaviour Profile interpretation is removed from
the Scenario Runner entirely and reassigned to a separate, future Behaviour Profile Resolver (§7) —
the Runner now carries `behaviour_profile_id` through as opaquely as every upstream Scenario Engine
package already does, and no longer imports `behavior`/`behavior_library` at all (§12). (2) The
Runner's output is a dedicated, immutable `SimulationContext` (§4) rather than an already-registered,
run-ready simulation — a structural consequence of (1), not an independent stylistic choice: once the
Runner can no longer resolve behaviour profiles itself, it can no longer produce a fully-registered
`MultiAgentSimulation` either, so initialization (this document's scope) and execution-readiness (the
Resolver's, then the caller's) are now necessarily two separate phases.

The single most consequential finding overall, unchanged by this pass, is §10 (no execution mechanism
exists for Scenario Events) — any future implementation pass on `scenario_runner/` should either
accept "scheduling only, no execution" as a real, honest V1 boundary, or treat closing that gap as a
prerequisite Simulation-layer milestone in its own right. The Behaviour Profile Resolver (§7) is now
an equally significant prerequisite: without it, a `SimulationContext` this package produces has no
path to becoming a runnable simulation at all.

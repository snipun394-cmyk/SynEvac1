# Scenario Event Execution — Architecture Proposal

Status: **proposal, open for review**. No code changes accompany this document — this is an
architecture review only, requested specifically because implementation was attempted and correctly
halted: no execution architecture for `ScenarioEvent` exists anywhere in the codebase
(`docs/architecture/scenario_runner.md` §10 already flagged this; this document is the "dedicated
design pass" that finding called for).

## 1. Purpose

This document designs the runtime component that executes already-resolved `ScenarioEvent`s
(`scenario/event.py`, frozen) during a running simulation — the piece
`docs/architecture/scenario_runner.md` §10 explicitly declined to design ("Scheduling Scenario
Events... What this document explicitly does not do: design that future executor").

**Scope, precisely**: this document designs *execution* — turning a due `ScenarioEvent` into a state
mutation at the correct simulation time. It does **not** design, and explicitly defers, the
still-missing "Simulation Loop" that would actually drive simulation time forward across occupant
movement, hazard evolution, and event execution together (§10). That gap is real, was already known,
and is not closed here — see §10 below for exactly what remains open and why closing it is out of
scope for a Scenario Event Executor specifically.

## 2. The most consequential finding: events currently cannot affect occupant movement at all

**This must be understood before anything else in this document makes sense.**
`simulator/occupant.py::Occupant`'s own docstring is explicit: *"the route is fixed for the lifetime
of this Occupant instance... there is still no dynamic rerouting of an Occupant already in flight."*
Every occupant's route is computed **once**, at `HumanBehaviorLayer.register()`/
`MultiAgentSimulation.add_occupant()` time, and never revisited. `behaviour_profile_resolver.
register_occupants()` (approved, prior phase) already registers **every** occupant in `context.
occupants` in one pass, before any simulation time has progressed at all.

**Consequence**: a Door/Exit/Stair/Obstacle Scenario Event executed at, say, t=90s has **zero
observable effect on any already-registered occupant's movement**, no matter how correctly this
document's design executes it — every occupant's route was already fixed at t=0, before the event
ever fires. This is not a limitation this document introduces or is responsible for closing; it is a
pre-existing property of the already-frozen `simulator/` package, which this document does not modify
(§9 below explains why extending it is architecturally wrong, independent of this finding).

**What executing an event still legitimately accomplishes, honestly stated**: it keeps the Building's
*observable* engineering state correct and current for anything that reads it *after* the event fires
— a future PathfindingEngine query (an AI/firefighter agent deciding *now*, not an already-departed
occupant), a future Perception system reading `Camera.active`/`Detector.active`, and the persisted
record of what actually happened during the incident. This document is scoped to making that state
mutation correct, ordered, and deterministic — not to retrofitting dynamic rerouting into
`MultiAgentSimulation`, which stays completely out of scope and unmodified.

## 3. Grounding in existing code

**`simulator/coordinator.py::MultiAgentSimulation`'s event heap** (`_event_heap`, `_schedule()`,
`_process_next_event()`) is not a general-purpose scheduler:

- Every scheduled item is `(time, counter, kind, occupant_id, generation)` — **intrinsically keyed to
  an `occupant_id`**, with a "generation" staleness check tied to that occupant's own registration
  count (`_begin_registration()`). A Scenario Event ("door-1 closes at t=90") has no natural
  `occupant_id` — it is a Building-level fact, not an occupant-level one. Forcing it through this
  mechanism would mean inventing a fake occupant identity for every event, a semantic mismatch, not a
  reuse.
- `_process_next_event()`'s dispatch is a hardcoded binary `if kind == self.TRY_ENTER_EDGE: ... else:
  ...` — not a pluggable dispatch table. Adding a third kind means editing this method's own logic.
- `run()` is **atomic**: `while self._event_heap: self._process_next_event()` drains the entire heap
  in one call. There is no `step(dt)` or `advance_to(time)` method — `MultiAgentSimulation` cannot be
  driven incrementally from outside at all, today.

**`hazard_evolution/provider.py::EvolutionBackedHazardProvider`** is the one place in the entire
codebase that already drives time-based state progression *without* using `MultiAgentSimulation`'s
heap — and it is the **only existing precedent** for "how does something advance forward in
simulation time, independent of occupant movement, in this codebase today." Its shape: `snapshot_at
(time)` advances `HazardEvolutionEngine.evolve()` forward in increments of at most a fixed `dt`,
caching every timestamp landed on (`self._timeline`), forward-only, raising on an attempt to go
backwards to a time never exactly visited. This is a **pull-based, lazily-advancing, timeline-caching**
pattern — not a heap, not a push-based scheduler.

**Two independent clocks already coexist, unintegrated, today.** `MultiAgentSimulation`'s own
event-heap time (internal to one atomic `run()` call) and `EvolutionBackedHazardProvider`'s
lazily-advancing hazard time are **already two separate, non-unified time domains** in the current
codebase — there is no single existing "simulation clock" object spanning both. "The existing
simulation clock should remain the single source of truth" (this pass's brief) is satisfied here not
by there being one shared clock *object* (there isn't one to share), but by every component — occupant
events, hazard evolution, and now Scenario Event execution — using the **same units** (seconds elapsed
since scenario t=0) and never introducing a competing time representation (a tick counter, a step
index, anything not directly comparable to `ScenarioEvent.time`).

**`scenario_runner/building_initializer.py`'s and `navigation_initializer.py`'s existing mechanism is
directly reusable, not just analogous.** `docs/architecture/scenario_runner.md` §5 already established
that `navigation/edge.py::Edge.traversable` reads `reference.locked`/`.active`/`.is_blocked` **live**
off the Building's own engineering objects, on every call, never cached — which is exactly why the
Scenario Runner mutates a Building *copy* once, at t=0, to seed initial state. **The same live-read
property is what makes mid-simulation event execution possible without touching `navigation/` or
`pathfinding/` at all**: mutating `door.locked = True` on the same Building object the graph already
references is immediately visible to any *future* `PathfindingEngine` query, with zero additional
wiring — the mechanism this document needs already exists and is already exercised by the Scenario
Runner, just invoked again, later, one object at a time, instead of once, for every object, at t=0.

The one exception, carried forward unchanged: **Staircase has no availability field**
(`models/staircase.py`, reconfirmed again this pass) — `scenario_runner/navigation_initializer.py`
already handles a *closed-at-t=0* stair by removing its `Edge` from the built `NavigationGraph`
directly. A stair-availability-change *event* needs the identical treatment, applied later: this is
the one event category that mutates `context.graph.edges` rather than a Building field, and the one
place this Executor needs `navigation.Edge`, not just `models`.

## 4. The "second scheduler" question — resolved by repository inspection

**Reviewed, per this pass's explicit instruction not to create a second scheduler unless proven
superior.** The repository inspection above proves it, on two independent grounds:

1. **`MultiAgentSimulation`'s heap is structurally the wrong shape for this problem.** Its
   `occupant_id`-keyed generation/staleness mechanism exists to let a *later* `BehaviorDecision` for
   the same occupant supersede an *earlier* one — a concern that only exists because occupant events
   are scheduled *dynamically*, mid-execution, as a side effect of processing earlier events
   (`_admit_onto_edge()` schedules `ARRIVE_AT_NODE` as a result of admitting an occupant onto an
   edge). **Scenario Events have no equivalent dynamism** — every `ScenarioEvent` in `context.
   scheduled_events` is fully known upfront, already time-ordered (`scenario_validator`'s own Event
   Validation already guarantees this, §5.3 of the Scenario Validator document). A **priority queue
   is solving a problem Scenario Events don't have**: nothing is ever inserted after construction,
   nothing is ever superseded, nothing needs re-ordering. A simple, already-sorted-cursor walk is not
   merely simpler than a heap here, it is the *architecturally correct* structure for a static,
   pre-sorted sequence — reaching for a heap would be over-engineering for this specific problem
   shape, not an equivalent alternative.
2. **Extending `MultiAgentSimulation` itself would require modifying an approved, frozen package**
   for a fit that is semantically wrong anyway (finding 1) — precisely the "genuine flaw" bar this
   pass's own instruction sets for touching existing packages, which this is not (there is no flaw in
   `MultiAgentSimulation`; it is simply not the right owner for a concern it was never designed to
   hold).

**Conclusion**: this is not "creating a second scheduler" in the sense of duplicating
`MultiAgentSimulation`'s own mechanism — it is applying the *simpler, already-precedented* pattern
(`EvolutionBackedHazardProvider`'s pull-based, dt-capped advancement) to a problem that specifically
does not need a priority queue at all. The new component is a peer to `EvolutionBackedHazardProvider`,
not a competitor to `MultiAgentSimulation`'s heap.

## 5. Input contract

The Scenario Event Executor consumes exactly one thing at construction: a **`SimulationContext`**
(`scenario_runner/context.py`, frozen, approved). From it, it reads (never writes, never replaces):

- `context.scheduled_events` — the already time-ordered `Tuple[ScenarioEvent, ...]`, trusted as-is
  (this component "never validates," mirroring every prior Scenario Engine layer's "the Runner trusts
  its input completely" principle — it does not re-check ordering or re-detect conflicts;
  `scenario_validator` already did, upstream).
- `context.building` — mutated in place for five of the six frozen engineering-state categories
  (§6).
- `context.graph` — mutated (edges added/removed) for the one stair-availability exception (§6).

It does not read `context.simulation`, `context.hazard_engine`, or `context.occupants` — per §2/§11,
there is currently no data dependency from event execution into any of them.

## 6. Event application — reusing, not duplicating, existing state-mutation logic

Per this pass's explicit instruction ("reuse existing engineering object models wherever possible...
do not invent new event semantics"), event application is **the same field-level mapping
`scenario_runner/building_initializer.py`/`navigation_initializer.py` already established for initial
state**, invoked again per-event instead of once per-Scenario:

| `target_type` | Mechanism | Precedent |
|---|---|---|
| `door` | `Door.locked`/`Door.normally_open` set from the event's resolved `DoorState` | `building_initializer.py::_apply_door_states` |
| `exit` | `Exit.is_blocked` set from the resolved open/closed value | `building_initializer.py::_apply_exit_states` |
| `obstacle` | `Obstacle.active` set from the resolved `PresenceState` | `building_initializer.py::_apply_obstacle_states` |
| `camera` | `Camera.active` set from the resolved `DeviceAvailability` | `building_initializer.py::_apply_camera_states` |
| `detector` | `Detector.active` set from the resolved `DeviceAvailability` | `building_initializer.py::_apply_detector_states` |
| `stair` | The matching `Edge` (`edge_type == Edge.STAIR`) added to/removed from `context.graph.edges` | `navigation_initializer.py::_exclude_closed_stair_edges` |

**A `ScenarioEvent`'s `parameters: Mapping[str, Any]` (§`scenario/event.py`) is what carries the
resolved target value** (e.g. `{"state": "LOCKED"}`) — `event_type` (a free-form string, e.g.
`"close"`) remains descriptive/provenance only, exactly as `scenario_validator`'s own Event Validation
already treats it (its keyword-matching heuristic reads `event_type` for *contradiction detection*
only, never to decide *what value* to apply). This document does not change that convention.

**Compliance finding, this pass**: `building_initializer.py`'s per-category apply functions are
currently module-private (leading underscore) — not meant for external reuse as written.
**Recommendation for the implementation phase** (not designed further here, since this document makes
no code changes): expose them as a small, additive public API on `scenario_runner` (a rename, not a
behavior change) so this Executor calls the *exact same* mapping code rather than re-stating the same
five trivial field assignments a second time, which would silently drift out of sync if the mapping
convention ever changes (e.g. a new `DoorState` member). This is an *addition* to an approved package,
not a redesign — flagged into §16 as a decision the implementation phase should make, not decided
unilaterally here.

## 7. Event scheduling and queue ownership

**No heap, no priority queue** (§4). The Executor owns a single piece of runtime state: a cursor
(conceptually, "how many of `context.scheduled_events`, taken in order, have already fired"). This
state:

- Is **not** stored on `SimulationContext` — `SimulationContext` is immutable and represents
  initialization, not runtime progress (`docs/architecture/scenario_runner.md` §4's own frozen
  distinction). The Executor is itself the new, separate, explicitly mutable runtime object this
  cursor belongs to — the same "immutable container, mutable runtime companion" split
  `SimulationContext.simulation` (a mutable `MultiAgentSimulation` inside a frozen bundle) already
  established.
- Is **not** shared with or visible to `MultiAgentSimulation` or `HazardEvolutionEngine` — neither
  needs to know it exists (§11/§12).
- Is trivially re-derivable, never a source of truth in its own right (§14).

**The advancement operation**: given a `target_time`, the Executor applies every not-yet-fired event
whose `time <= target_time`, **in the order they already appear in `context.scheduled_events`**
(already sorted, trusted, §5), advancing the cursor past each one as it fires.

## 8. Event ordering and deterministic execution

Ordering is **entirely inherited**, not invented: `scheduled_events` arrives already sorted by `time`
ascending (`scenario_validator`'s `EVENTS_NOT_ORDERED` check, §5.3 of the Scenario Validator document,
already guarantees this before a `Scenario` is ever accepted). The Executor performs no sorting and no
re-validation of order — it walks the tuple exactly as given.

**Determinism** follows directly: given the same `scheduled_events` and the same sequence of
`target_time` values passed to the advancement operation, execution is byte-for-byte identical every
time — there is no randomness anywhere in this component (no `random` import, matching every prior
Scenario Engine/Runner layer's own dependency rule) and no dependency on anything outside its two
inputs (`context`, `target_time`).

## 9. Simultaneous event resolution

Two events sharing the exact same `time` are already constrained by an upstream, frozen guarantee:
`scenario_validator`'s `CONFLICTING_EVENTS` check (§5.3 of the Scenario Validator document) already
**rejects** any two events targeting the same `(target_type, target_id)` at the same instant before a
`Scenario` is ever accepted. **Consequence**: any two events this Executor ever sees sharing a
timestamp necessarily target *different* objects — there is no meaningful ordering dependency between
them (setting `door-1.locked` and `camera-1.active` in either order produces the identical end state).
The Executor applies same-timestamp events in the order they appear in `scheduled_events` (stable,
deterministic, arbitrary only in the sense that it doesn't matter which of two independent mutations
happens a nanosecond "first") — no additional tie-breaking mechanism is needed or designed.

## 10. Interaction with simulation stepping

**Honestly, there is no existing "simulation stepping" mechanism spanning occupant movement, hazard
evolution, and now event execution to interact with** — this is the same gap
`docs/architecture/scenario_runner.md` §10 already identified, not a new one. What exists:
`MultiAgentSimulation.run()` (atomic, occupant-only), `EvolutionBackedHazardProvider.snapshot_at(time)`
(pull-based, hazard-only). Neither drives the other today.

**This document's Executor is designed to be a third peer of that same shape** — a
`target_time`-driven advancement operation (§7), matching `EvolutionBackedHazardProvider`'s own
calling convention exactly, so that whatever **future** orchestration layer ends up unifying all three
(occupant movement, hazard evolution, event execution) can drive all three the same way, at the same
`target_time`, without this component needing to change. **Building that future orchestration layer is
explicitly out of scope for this document**, exactly as it was out of scope for
`docs/architecture/scenario_runner.md` — this document closes the "can an event be executed correctly
at all" question; it does not close "what calls this, and when, during a full simulation run." That
remains flagged, not designed, into §16.

## 11. Interaction with HazardEvolutionEngine

**None, beyond a shared time axis.** Repository inspection (§3) confirms none of the six frozen event
categories (door/exit/obstacle/camera/detector/stair) produce a `HazardContribution` or touch
`hazard/`/`hazard_evolution/` in any way — `scenario_validator`'s own Navigation Validation already
established that Obstacles have no `NavigationGraph` effect at all, and Camera/Detector failures are a
Perception-layer concern, not a Hazard one. **This Executor does not import `hazard_evolution` or
`fire_growth`.** The only relationship is that both this Executor's `target_time` and
`HazardEvolutionEngine.evolve()`'s `time` parameter are the same unit, so a future orchestration layer
can call both at the same `target_time` and have "what's true about the Building" and "what's true
about the hazard" stay mutually consistent at every point — without either component needing to know
the other exists.

## 12. Interaction with HumanBehaviorLayer

**None, and this is a direct consequence of §2's finding, not an independent design choice.** Every
occupant is registered — meaning every route is already fixed — before any simulation time
progresses, under the currently-approved `behaviour_profile_resolver.register_occupants()` design.
There is no "later" registration point in the current architecture for a Scenario Event to influence.
This Executor does not import `behavior` or `behavior_library` for the same reason
`scenario_runner`/§12 of its own document does not: there is nothing behavioural for it to touch, and
touching it would mean inventing a dynamic-rerouting or late-registration mechanism that does not
exist and is explicitly out of scope to design here (it would be a `simulator`/`behavior` redesign,
forbidden by this pass's own instruction).

## 13. Replay compatibility

Follows directly from §8's determinism: replaying a simulation (re-driving the same `SimulationContext`
— itself already deterministic, per `docs/architecture/scenario_runner.md` §9 — with the same sequence
of `target_time` calls) re-executes the identical events at the identical times, producing an
identical sequence of Building-state mutations. Nothing in this design introduces wall-clock time,
randomness, or external state — the same properties that already make `Scenario` generation and
Scenario Runner initialization replay-safe extend unchanged to event execution.

## 14. Save/load compatibility

**The Executor's own cursor state needs no bespoke persistence, because it is not truly independent
state — it is a pure function of `(scheduled_events, current_time)`.** "How many events have fired by
time T" is always re-derivable by counting how many entries of the already-sorted, already-persistable
`scheduled_events` tuple have `time <= T` — nothing about *which* events have fired needs to be saved
separately from *what time it is*. A future save/resume capability therefore does not need to
serialize this Executor at all: it needs to persist `current_time` (a single float) and reconstruct a
fresh Executor against the same `SimulationContext`, which will correctly re-derive exactly the same
fired/not-fired state on its first `advance_to()` call. This is the same "no new persistence format
needed" property `scenario_storage`'s reuse of `Scenario.to_dict()` already established one layer up
— consistent, not coincidental.

## 15. Dependency direction

- `scenario_event_executor/` **may** import: `scenario` (the `ScenarioEvent` shape it reads),
  `scenario_runner` (the `SimulationContext` shape it consumes — a one-way, downstream dependency;
  `scenario_runner` does not import this package, mirroring the exact precedent
  `behaviour_profile_resolver`'s own dependency-direction rule already established), `models`
  (`Door`/`Exit`/`Obstacle`/`Camera`/`Detector` fields it mutates), `navigation` (`Edge`, for the one
  stair exception, §6).
- `scenario_event_executor/` **must not** import `scenario_generator`, `scenario_validator`, or
  `scenario_pipeline` — it never generates, never validates, never orchestrates a retry loop; the
  events it executes already passed through all of that, upstream.
- `scenario_event_executor/` **must not** import `behavior` or `behavior_library` (§12) or
  `hazard_evolution`/`fire_growth` (§11) — no data dependency exists in either direction.
- `scenario_event_executor/` **must not** import `ai_decision`, `perception`, `sensors`, `occupancy`,
  or `rl` — none of those are part of executing an engineering-state change; every one of them is
  either a future consumer of the Building state this Executor updates, or unrelated to this
  milestone's scope.
- `scenario_event_executor/` **must not** import `sandbox` or `designer`.
- `scenario_event_executor/` **must not** import `random` — execution is a pure function of its
  inputs (§8), nothing here samples anything.

## 16. Suggested package structure

Not a commitment to implement (this milestone is architecture-only):

```
scenario_event_executor/
    __init__.py
    executor.py       # ScenarioEventExecutor: owns the cursor, advance_to(target_time)
    handlers.py       # per-target_type application functions (§6) -- a pluggable
                       # {target_type: handler} dispatch table, extensible for
                       # future event types (§17) without changing executor.py
```

## 17. Future event types — Door/Exit/Stair/Obstacle/Camera/Detector, and beyond

The six frozen categories (§6) are each one entry in a `{target_type: handler}` dispatch table, not
six special cases hardcoded into the Executor's own advancement loop (§7) — adding a new category is
adding a new table entry, never touching the cursor/ordering logic itself.

**"Future AI actions" and "Future firefighter actions" are explicitly named in this pass's brief but
cannot be designed here**, for a reason worth stating plainly: every existing `target_type`
(door/exit/stair/obstacle/camera/detector) names an **engineering object with a corresponding resolved
state category** already frozen in `scenario/`/`scenario_definition/` (§3.2 of the Scenario Engine
architecture document). An "AI action" or "firefighter action" is not an object-state change at all —
it is an *agent decision*, with no corresponding `target_type`/state-category convention anywhere in
the frozen schema. Representing one would require either a `scenario_definition`/`scenario` schema
extension (touching frozen packages — explicitly out of bounds for this document) or a fundamentally
different kind of `target_type` this document is not positioned to invent without redesigning upstream,
frozen contracts. **This document's dispatch-table design accommodates them structurally whenever
that schema question is resolved** (a new `target_type` key, a new handler, no change to §7's core
loop) — but designing what an "AI action" or "firefighter action" event actually *contains* is
explicitly flagged, not decided, here.

## 18. Open questions for a future review

- **The Simulation Loop that actually drives `advance_to()`** (§10) — still the single largest gap
  carried over from `docs/architecture/scenario_runner.md` §10, now slightly narrower (event
  execution itself is fully designed; only *what calls it, and when, alongside occupant movement and
  hazard evolution* remains open).
- **Exposing `building_initializer.py`'s apply functions publicly** (§6) — a small, additive change to
  the approved `scenario_runner` package, recommended but not designed in code here.
- **Dynamic rerouting / late occupant registration** (§2/§12) — the actual, load-bearing reason events
  cannot yet affect occupant behavior. Closing this is a `simulator`/`behavior` redesign question, not
  a Scenario Event Executor one — flagged, explicitly not proposed here.
- **AI action / firefighter action event schema** (§17) — structurally accommodated, not designed;
  would need a dedicated review of its own, likely touching frozen `scenario_definition`/`scenario`
  schema.
- **Whether stair-edge re-addition (an event turning a previously-CLOSED stair back to AVAILABLE) needs
  the original `Edge` object cached somewhere, or can be cheaply reconstructed** — `navigation_
  initializer.py`'s existing mechanism only ever *removes* edges (§5's original scope was t=0
  initialization only, where "closed" only ever subtracts from a freshly-built graph); an event
  re-opening a stair mid-simulation needs to *re-add* an edge, which the existing precedent alone
  doesn't cover. Flagged as an implementation-level detail, not resolved here.

## 19. Status

Proposal only. Nothing in this document has been implemented; no existing package (Scenario Engine,
Scenario Runner, or Simulation Engine) is modified by it. The design resolves the "second scheduler"
question definitively (§4: not a second scheduler in any meaningful sense — a simpler, already-
precedented pattern applied to a problem that never needed a priority queue) and reuses the Scenario
Runner's own established Building-mutation mechanism without alteration (§6). It does **not** resolve,
and does not claim to resolve, the absence of a unified Simulation Loop (§10) or the fact that
occupant movement cannot yet react to a mid-simulation event at all (§2) — both remain genuine,
explicitly flagged gaps for future architectural work, not silently assumed away.

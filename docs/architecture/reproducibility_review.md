# Reproducibility Pipeline — Focused Architecture Review

Status: **review only, no redesign**. No code changes accompany this document. Scope is deliberately
narrow, per this pass's explicit instruction: only **Behaviour Profile Resolver**
(`behaviour_profile_resolver/`), **Human Behavior Layer** (`behavior/`), **`behavior_library/`**, and
**Simulation Runtime** (`simulation_runtime/`) are audited. `scenario_definition`, `scenario_generator`,
`scenario_validator`, `scenario_pipeline`, `scenario_storage`, and `scenario_runner` are treated as
already-confirmed-clean upstream inputs (per `integration_validation.md` §9 point 1) and are not
re-audited here.

## 1. Purpose

`docs/architecture/integration_validation.md` §6.1/§9 identified, and this document confirms in full
detail, that the pipeline

```
Scenario Definition → Scenario Generator → Scenario Runner → Behaviour Profile Resolver
    → Human Behavior Layer → Simulation Runtime
```

does **not** currently guarantee identical results across repeated executions of the same `Scenario`,
despite every stage up to and including `scenario_runner.run()` being fully deterministic. This document
traces every source of randomness in the four named subsystems, classifies each one, and proposes the
smallest architectural change that closes the gap without altering `BehaviorProfileTemplate`'s
one-instance-per-profile sharing model, without changing any strategy's public constructor, and without
removing any existing behavioural capability.

## 2. Method

Every file in the four target packages was read in full this pass: `behaviour_profile_resolver/
{template,resolver,registry,registrar}.py`; `behavior/{orchestrator,profile,context,intent,pre_movement,
route_choice}.py`; `behavior_library/{decision_strategies,pre_movement_strategies,
route_choice_strategies}.py`; `simulation_runtime/{clock,occupancy_bridge,result,runtime}.py`. Every
`import random` and every call to a `random.Random` method was located and traced to its caller.

## 3. Complete randomness inventory

| # | Location | Uses randomness? | Reached by `DEFAULT_PROFILE_REGISTRY`? | Classification |
|---|---|---|---|---|
| 1 | `behaviour_profile_resolver/*.py` (all four files) | No | — | Clean |
| 2 | `behavior/orchestrator.py::HumanBehaviorLayer` | No | — | Clean |
| 3 | `behavior/profile.py`, `behavior/context.py` | No (plain data) | — | Clean |
| 4 | `behavior/intent.py::AlwaysEvacuateDecisionStrategy` | No | Yes (Staff, FireWarden fallback path) | Clean |
| 5 | `behavior/pre_movement.py::NoPreMovementDelay` | No | Yes (Staff, FireWarden) | Clean |
| 6 | `behavior/route_choice.py::ShortestRouteChoiceStrategy` | No (delegates to `PathfindingEngine`, itself deterministic) | Yes (**all six** default profiles) | Clean |
| 7 | `behavior_library/decision_strategies.py::AlwaysWaitDecisionStrategy`, `AlwaysIgnoreDecisionStrategy`, `BasicHelpingDecisionStrategy` | No | `BasicHelpingDecisionStrategy` yes (FireWarden) | Clean |
| 8 | **`behavior_library/decision_strategies.py::ComplianceDecisionStrategy`** | **Yes** — `self.rng.random()` in `.decide()` | **Yes** — Adult, Child, Visitor | **Bug** (§4.1) |
| 9 | **`behavior_library/pre_movement_strategies.py::ProbabilisticPreMovementDelay`** | **Yes** — `self.rng.lognormvariate()` in `.delay()` | **Yes** — Adult, Child, Wheelchair, Visitor | **Bug** (§4.1) |
| 10 | `behavior_library/route_choice_strategies.py::FamiliarityBasedRouteChoiceStrategy`, `FollowLeaderRouteChoiceStrategy`, `HelpTargetRouteChoiceStrategy` | No | No | Clean |
| 11 | **`behavior_library/route_choice_strategies.py::StaticHerdingRouteChoiceStrategy`** | **Yes** — `self.rng.random()` in `.choose()` | No (not used by any default profile) | **Latent bug** — same root cause as #8/#9, currently unreached only because no default profile happens to select it (§4.2) |
| 12 | `simulation_runtime/{clock,occupancy_bridge,result,runtime}.py` | No | — | Clean (§6) |
| 13 | `behaviour_profile_resolver/registry.py::DEFAULT_PROFILE_REGISTRY` construction | Constructs #8/#9/(#11-capable) strategy objects with no `rng=` argument, exactly once, at module-import time | — | **The architectural gap** (§5) — the actual root cause; #8/#9/#11 are its symptoms, not independent bugs |

**Three call sites draw from an unseeded `random.Random()`; one root cause explains all three.**
`Simulation Runtime` itself, and every line of `behaviour_profile_resolver`/`behavior/orchestrator.py`,
is confirmed clean — the entire gap lives inside `behavior_library`'s three affected strategy classes
and, more precisely, inside how `behaviour_profile_resolver/registry.py` constructs them.

## 4. Classification detail

### 4.1 `ComplianceDecisionStrategy` / `ProbabilisticPreMovementDelay` — bug, not intentional

Both classes' own source comments state the *intended* design precisely: "`rng` is constructor-injected
(defaulting to an unseeded `random.Random()`) so tests can supply a seeded instance for deterministic,
reproducible draws." This is a **correct, well-designed seam** — the bug is not that these classes are
capable of unseeded behavior (that is the documented, intentional fallback for standalone/test use), it
is that **the one registry every production campaign actually uses never supplies the seed the seam was
built to accept**. `behaviour_profile_resolver/registry.py::DEFAULT_PROFILE_REGISTRY` constructs
`ComplianceDecisionStrategy()` and `ProbabilisticPreMovementDelay(median_delay=...)` with no `rng=`
argument, for `Adult_Default`, `Child_Default`, `Wheelchair_Default`, and `Visitor_Default` — four of the
six profiles a real campaign is most likely to actually use (`Staff_Default`/`FireWarden_Default` are
the exception profiles, not the typical case for ordinary building occupants).

### 4.2 `StaticHerdingRouteChoiceStrategy` — an architectural gap, not (yet) a bug

Not reached by any of the six default profiles today, so no current campaign is silently affected by it
— but it shares the identical `rng or random.Random()` pattern and the identical vulnerability the
moment any custom or future registry (e.g. a `Rescue Training`/`Firefighter Training` intent profile
wanting herding behavior, `dataset_intent.md` §7) selects it. Classified separately from #8/#9 because
it is not *currently* producing wrong results for anyone — it is a gap in the sense that nothing
prevents the same bug from recurring the next time this pattern is reused, unless the fix in §7 covers
the pattern generally rather than patching the two currently-symptomatic classes specifically.

### 4.3 What is correctly intentional, for contrast

`scenario_generator/seed_manager.py`'s `category_rng()` (out of this review's scope, but the correct
reference point) is the existing, working example of the *same* underlying need — "give this piece of
sampling logic its own reproducible random stream" — solved correctly: every `random.Random` instance
`scenario_generator` ever constructs is seeded from the Master→Scenario→Attempt→Category hierarchy,
never left to its constructor default. `behavior_library`'s `rng or random.Random()` pattern is not a
worse design than that — it is the *identical* injectable-seam design, just never actually wired to a
seed source at the one call site (`registry.py`) that matters for a real campaign.

## 5. Root cause: a module-level singleton registry, constructed before any Scenario exists

**This is the one finding this whole review turns on.** `DEFAULT_PROFILE_REGISTRY` is a `MappingProxyType`
constant, defined at module scope in `behaviour_profile_resolver/registry.py`. Python imports a module's
top-level code **exactly once** per process, the first time it is imported, and caches the result for
every subsequent `import` anywhere else in that process. Consequence, precisely:

- Every strategy object inside `DEFAULT_PROFILE_REGISTRY` — including each affected `ComplianceDecisionStrategy`/
  `ProbabilisticPreMovementDelay` instance — is constructed **before any `Scenario` has been generated,
  before any seed is known, and before any occupant exists to resolve a profile for.** There is no seed
  to inject at this point even if `registry.py` wanted to — the registry is built at process-startup
  time (or first-import time), structurally prior to any per-Scenario data.
- Because it is a singleton, **every occupant across every `Scenario`, across every `SimulationRuntime`
  run, for the entire lifetime of the process**, resolving to (say) `Adult_Default` shares the *same*
  `ComplianceDecisionStrategy` instance and its one, slowly-advancing, never-reset `self.rng` state. This
  is a stronger claim than "unseeded" alone: it means running the *same* `Scenario` twice in the *same*
  process produces two *different* results even if a future fix seeded the process's global `random`
  module at startup, because the second run's draws continue from wherever the first run's left the
  shared RNG, not from a fresh, scenario-specific starting point.
- **`BehaviorProfileTemplate`'s own docstring confirms this sharing is deliberate and load-bearing for a
  different reason**: "a single template instance — including its strategy objects — is shared across
  every occupant resolved to the same `behaviour_profile_id`," matching `behavior_library`'s own
  documented "one instance is reusable across every follower in every group" convention. **The sharing
  itself is correct and must not change** (§7 preserves it exactly) — the randomness source reached
  through that shared instance is what must move.

## 6. Simulation Runtime — confirmed clean, and why its own cleanliness cannot fix this

Re-confirmed this pass: no `random` import anywhere in `simulation_runtime/`, and this package's own
dependency-direction test (`tests/test_simulation_runtime.py::
SimulationRuntimePackageDependencyDirectionTests`) already enforces this structurally, not just by
inspection. `SimulationRuntime` never calls `register_occupants()` itself — that call is the
caller's responsibility, completed *before* a `SimulationRuntime` is constructed
(`docs/architecture/simulation_runtime.md` §7, unchanged). `SimulationRuntime.__init__` calls
`context.simulation.run()` exactly once, draining whatever event heap `register_occupants()` already
populated. **`SimulationRuntime`'s own determinism claim (`simulation_runtime.md` §16) is real and
correctly scoped — "no wall-clock time, no unseeded randomness... in the tick loop" — but it is
conditional on its input, not a guarantee about the whole pipeline.** A `SimulationRuntime` fed a
non-deterministically-populated `context.simulation` faithfully and deterministically simulates
*whatever it was handed* — the non-determinism entered one layer upstream, at occupant registration, and
`SimulationRuntime` has no visibility into it and no opportunity to correct it. This is precisely why
the fix (§7) cannot live in `simulation_runtime/` at all.

## 7. Proposed minimal architectural change

**Three additive changes, no removals, no signature-breaking edits, no change to any package's
dependency-direction rules.**

### 7.1 `DecisionContext` gains one new optional field: `rng`

`behavior/context.py`'s own docstring already states its design intent precisely: "Deliberately
extensible: this is the one place designed to grow additively (new optional fields) as future Dynamic
Hazard/AI systems are integrated, without breaking existing strategy implementations that don't look at
those new fields." Adding `rng: Optional[random.Random] = None` is exactly the kind of change this field
was built to accommodate — not a new mechanism, an instance of an already-designed-for one. Every
existing `DecisionStrategy`/`RouteChoiceStrategy`/`PreMovementDelayStrategy` implementation that does not
read `context.rng` (all of them today) continues working unmodified.

### 7.2 The three affected `behavior_library` strategies prefer `context.rng` when present

`ComplianceDecisionStrategy.decide()`, `ProbabilisticPreMovementDelay.delay()`, and
`StaticHerdingRouteChoiceStrategy.choose()` each change their one line of randomness use from
`self.rng.random()`/`self.rng.lognormvariate(...)` to consulting `context.rng` first, falling back to
`self.rng` only when `context.rng is None`:

```
rng = context.rng if context.rng is not None else self.rng
```

**`self.rng` is not removed.** A test or standalone caller that constructs `ComplianceDecisionStrategy
(rng=seeded_rng)` and calls `.decide(context)` directly, with a `context.rng` left `None`, gets exactly
today's behavior — the documented "tests can supply a seeded instance" use case is unaffected. Only the
*production* path (through `HumanBehaviorLayer.register()`, §7.3) changes what it actually receives.

### 7.3 `HumanBehaviorLayer.register()` gains one new optional parameter: `rng`

```
def register(
    self, start_id, profile, decision_strategy,
    route_choice_strategy=None, pre_movement_strategy=None, base_depart_time=0.0,
    rng=None,                                              # NEW, optional, default None
):
    context = DecisionContext(
        graph=self.graph, engine=self.engine, profile=profile, start_id=start_id,
        decisions_so_far=dict(self._decisions_so_far),
        rng=rng,                                             # NEW — threaded straight through
    )
    ...
```

Backward compatible by construction: every existing caller of `.register()` that does not pass `rng`
gets `rng=None`, `context.rng` stays `None`, and §7.2's fallback means every strategy behaves exactly as
it does today. `test_behavior_layer.py` and any other direct test of `HumanBehaviorLayer` needs no
change.

### 7.4 `behaviour_profile_resolver.register_occupants()` derives and supplies a per-occupant seed

The one genuinely new piece of logic, and it is small: for each occupant, derive a seed from data already
present on `SimulationContext` — `context.metadata.seed` (`ScenarioMetadata.seed`, the resolved, already-
reproducible Scenario Seed — already flowing all the way down to this function's own parameter today,
simply unused for this purpose) and `occupant.occupant_id` (already a deterministic, content-derived
string, `scenario_generator.generator`'s `f"occ-{zone_id}-{index}"`) — and construct a fresh,
per-occupant `random.Random(derived_seed)` to pass as `rng=` into `behavior_layer.register(...)`:

```
def _derive_occupant_seed(scenario_seed: int, occupant_id: str) -> int:
    key = f"{scenario_seed}|behaviour_profile_resolver|{occupant_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
```

**Restated locally, not imported from `scenario_generator.seed_manager`** — the same "same algorithm,
independently implemented per package" precedent `integration_validation.md` §7.1 already documented for
content-hashing, applied here for the identical reason: `behaviour_profile_resolver` gains no new
dependency, and its one-way relationship with `scenario_runner` (the only package it currently imports
downstream of, per `docs/architecture/scenario_runner.md` §12) is unchanged. **`integration_validation.md`
§7.1's own recommendation — extracting the shared hashing/derivation primitive into `serialization/` —
would let this restatement become a genuine, safe import instead of a copy, and is worth doing in the
same pass as this fix if that recommendation is taken up; not required for this fix to be correct on its
own.**

### 7.5 What does *not* change

- `BehaviorProfileTemplate` and `DEFAULT_PROFILE_REGISTRY`: **zero changes.** Strategy objects remain
  singletons, shared across every occupant of a profile, exactly as designed (§5). Their `self.rng`
  fields remain exactly as constructed today — now simply unused in the production path, not removed.
- `resolve_profile()`, `UnknownBehaviourProfileError`: unchanged.
- Every strategy's public constructor signature: unchanged. No new required parameter anywhere.
- `simulation_runtime/`: **zero changes** — confirmed clean in §6, correctly has nothing to fix.
- Group/leader/follower/herding *ordering* logic (`context.decisions_so_far`): unaffected — that
  mechanism depends on registration order, which `register_occupants()`'s existing deterministic
  iteration over `context.occupants` already governs, and this fix does not touch.

## 8. Does this reduce realism or flexibility? No — and it is a strict realism improvement

The user's constraint is answered directly, not just satisfied incidentally:

- **Flexibility is unchanged.** Every existing `DecisionStrategy`/`RouteChoiceStrategy`/
  `PreMovementDelayStrategy` — including every one in `behavior_library` not discussed here
  (`FamiliarityBasedRouteChoiceStrategy`, `FollowLeaderRouteChoiceStrategy`, `HelpTargetRouteChoiceStrategy`,
  `AlwaysWaitDecisionStrategy`, `BasicHelpingDecisionStrategy`, ...) — is composable exactly as before; a
  custom registry (`register_occupants(context, registry=my_registry)`) still works exactly as today,
  and any future strategy needing its own randomness gets the identical, already-designed-for `context.
  rng` seam for free, without repeating this investigation.
- **Realism is, if anything, improved.** Today, because one shared, never-reset `random.Random()`
  advances across every occupant of a profile *and* across every scenario run in the process, occupant
  compliance rolls and pre-movement delays are not actually independent draws — they are one long,
  order-dependent, cross-scenario-contaminated sequence. §7's per-occupant, per-scenario-seeded stream
  makes each occupant's draw genuinely independent of every other occupant's and of any prior scenario
  run — a strictly more correct statistical model of "many independent people each deciding
  independently," not merely a determinism patch layered on top of the current behavior.

## 9. Status

Review only. No file in `behaviour_profile_resolver/`, `behavior/`, `behavior_library/`, or
`simulation_runtime/` has been modified. Three sites use unseeded randomness (`ComplianceDecisionStrategy`,
`ProbabilisticPreMovementDelay` — confirmed live bugs against five of six default profiles;
`StaticHerdingRouteChoiceStrategy` — confirmed latent, same root cause, not yet triggered by any default
profile) with one shared root cause: `DEFAULT_PROFILE_REGISTRY`'s module-level, pre-Scenario construction
never exercises the `rng=` seam each strategy already correctly exposes. `Simulation Runtime` is
confirmed clean and requires no change — its determinism is real but conditional on an input this review
shows is not currently deterministic. The proposed fix (§7) is three additive changes — one new optional
`DecisionContext` field, one new optional `HumanBehaviorLayer.register()` parameter, and one new small,
locally-implemented seed-derivation function inside `behaviour_profile_resolver` — none of which alters
any existing public contract, removes any capability, or touches `simulation_runtime/` at all. Nothing in
this document has been implemented.

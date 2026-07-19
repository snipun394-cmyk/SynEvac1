# Integration & Validation — End-to-End Pipeline Architecture Review

Status: **review only**. No code changes accompany this document. This pass does not design a new
package — it audits the eleven-stage pipeline every prior document in this lineage built one piece of,
checking whether the pieces actually fit together at scale (**tens of thousands of complete simulation
runs, unattended**), and states plainly where they do not yet.

## 0. Scope and current implementation status

```
Dataset Intent → Scenario Definition → Scenario Generator → Scenario Validator → Scenario Storage
    → Scenario Runner → Behaviour Profile Resolver → Simulation Runtime → Perception
    → Scenario Outcome Labels → Dataset Builder
```

| Stage | Package | Status |
|---|---|---|
| Dataset Intent | `dataset_intent/` | **Architecture only** (`dataset_intent.md`) — no code |
| Scenario Definition | `scenario_definition/` | Implemented, frozen |
| Scenario Generator | `scenario_generator/` | Implemented, frozen |
| Scenario Validator | `scenario_validator/` | Implemented, frozen |
| Scenario Storage | `scenario_storage/` | Implemented, frozen |
| Scenario Runner | `scenario_runner/` | Implemented, frozen |
| Behaviour Profile Resolver | `behaviour_profile_resolver/` | Implemented, frozen |
| Simulation Runtime | `simulation_runtime/` | Implemented, frozen |
| Perception | `perception/` | **Partially implemented** — models, providers, fusion primitives exist; no concrete `PerceptionProvider` composition exists (`simulation_runtime.md` §13, reconfirmed §5 below) |
| Scenario Outcome Labels | `scenario_outcome/` | **Architecture only** (`dataset_intent.md` §10-§14) — no code |
| Dataset Builder | `dataset_builder/` | **Architecture only** (`dataset_builder.md`) — no code |

Confirmed this pass by repository search: no `scenario_outcome/`, `scenario_outcome_storage/`,
`dataset_intent/`, or `dataset_builder/` directory exists yet; no campaign/orchestration entry point
(`main.py` is the Designer GUI's entry point, unrelated) exists anywhere in this repository. **This
review is therefore auditing a pipeline that is real end-to-end only in its first eight stages** — the
last three exist as frozen designs this review treats as fixed contracts, the same way it treats already-
implemented code.

## 1. Every integration point

| # | Boundary | Producer call | Consumer call | Shared type |
|---|---|---|---|---|
| 1 | Intent → Definition | `resolve_intent(intent, building, parameters)` | `GenerationRequest(definition=..., definition_id=...)` | `ScenarioDefinition` |
| 2 | Definition → Generator | `GenerationRequest`/`BatchGenerationRequest` | `generate_scenario(request)` | `ScenarioDefinition`, `Building`, `seed` |
| 3 | Generator → Validator | `generate_scenario()` returns `Scenario` | `validate(candidate, definition, building, accepted_hashes)` | `Scenario` |
| 4 | Validator → Pipeline | `ScenarioValidationReport.accepted` | `run_pipeline()`'s accept/reject branch | `ScenarioValidationReport` |
| 5 | Pipeline → Storage | `PipelineResult.scenario` | `scenario_storage.save_scenario(scenario, storage_root)` | `Scenario` |
| 6 | Storage → Runner | `load_scenario_by_id(scenario_id, storage_root)` | `scenario_runner.run(scenario, building)` | `Scenario`, `Building` |
| 7 | Runner → Resolver | `scenario_runner.run()` returns `SimulationContext` | `register_occupants(context)` | `SimulationContext` |
| 8 | Resolver → Runtime | `register_occupants()` mutates `context.simulation` in place | `SimulationRuntime(context, decision_engine, dt, ...)` | **the same `SimulationContext` instance** (§2.3) |
| 9 | Runtime → Perception | `SimulationRuntime.tick()` calls `perception_provider.observation_at(next_time)` if configured | — | `BuildingObservation` (§2.1 — the one genuinely missing adapter) |
| 10 | Runtime → Outcome Labels | `runtime.context`, `runtime.movement_result`, `runtime.run()` | `record_outcome(context, movement_result, tick_results, ...)` | `SimulationContext`, `MultiAgentSimulationResult`, `Tuple[TickResult, ...]` |
| 11 | Outcome Labels → Dataset Builder | `ScenarioOutcomeLabels` | Ingestion Adapter → `SimulationRunRecord` | **no shared type by design** (§2.2 — the second genuinely missing adapter) |

**Nine of eleven boundaries need no adapter at all** — each pair of neighboring packages was, across
this whole design lineage, deliberately built to produce exactly what its neighbor consumes (boundary 6
in particular: `Scenario.from_dict()`'s output is directly `scenario_runner.run()`'s input, with zero
translation). This is not an accident this review is merely confirming — it is the direct, compounding
payoff of every prior document's explicit "produces exactly what X already expects" compliance
findings. §2 covers the two boundaries where that discipline runs out.

### 1.1 Boundary 8 deserves its own callout

`register_occupants(context)` and `SimulationRuntime(context, ...)` **must receive the identical
`SimulationContext` object**, not two independently-constructed ones from the same `Scenario`+`Building`
— `register_occupants()`'s entire effect is populating `context.simulation`'s event heap in place
(`behaviour_profile_resolver/registrar.py`), and `SimulationRuntime.__init__` immediately calls
`context.simulation.run()` on whatever heap state it finds. Calling `scenario_runner.run()` twice (even
with identical inputs) produces two `SimulationContext`s with two distinct, both-empty
`MultiAgentSimulation` instances — passing the *second* one to `SimulationRuntime` after only the
*first* was registered would silently simulate zero occupants. Nothing in either package's type
signature prevents this caller error; it is a **documentation-only invariant** today, worth surfacing
explicitly for whatever orchestration code eventually calls this sequence (§4).

## 2. Required adapters

Exactly two — both already flagged in their respective source documents, restated here because an
end-to-end integration review is precisely where their absence becomes *blocking* rather than merely
noted.

### 2.1 Perception composition (blocks boundary 9)

`simulation_runtime.md` §13 flagged this as an optional dependency `SimulationRuntime` tolerates being
absent (`perception_provider: Optional[PerceptionProvider] = None`) — correct for that document's own
scope. **For this review's stated goal** (tens of thousands of *complete* runs), "Perception" is a named
pipeline stage the workflow diagram places between Simulation Runtime and Scenario Outcome Labels — a
campaign that never configures a `PerceptionProvider` produces `TickResult.observation is None` for
every tick, which is a legitimate, documented configuration (§13 again: `dataset_builder.md`'s IR
already treats this as sparse-by-design, §5 of that document) but is **not** "Perception" running as a
pipeline stage, it is Perception being skipped. Closing this gap requires exactly the composition
`perception_layer_review_2.md` §3/§8.1 already designed and never built: a `PerceptionFusionEngine`-
shaped adapter wiring `GroundTruthCameraProvider`/`GroundTruthSmokeDetectorProvider`/
`GroundTruthHeatDetectorProvider` (all three already implemented) + `OccupancyEstimator` +
`SensorFusion.fuse()` (all already implemented) behind `PerceptionProvider.observation_at(time)`, fed
from the same `context.building` + the Runtime's own `hazard_provider`/`occupancy_provider`. **This is
now the single largest concrete gap this review identifies** — every other stage either has working code
or a fully-specified, uncontested design; Perception has both halves built and no adapter joining them.

### 2.2 Dataset Builder Ingestion Adapter (blocks boundary 11)

Already fully specified, field-by-field, in `dataset_builder.md` §6 — deliberately placed outside
`dataset_builder/` itself (§3 of that document: the independence requirement is structural, not
aspirational). This review adds one operational note §2.1 doesn't need: this adapter is the **one place**
in the whole pipeline that must import from every one of Scenario Generation, Simulation, and Perception
simultaneously (`scenario`, `simulation_runtime`, `perception`, `scenario_outcome`) — every other
package in this architecture imports at most two or three upstream neighbors. It is not accidental that
this is also the one place `dataset_intent.md` §17 already named as belonging to "the future Dataset
Generation Pipeline" rather than to any single-purpose package — §4 makes this explicit.

### 2.3 Everywhere else: no adapter, by design

Confirmed for the record, boundary by boundary (§1's table) — `GenerationRequest` accepts a
`ScenarioDefinition` with zero transformation; `generate_scenario()` produces a `Scenario` that
`validate()` consumes unmodified; `scenario_storage` persists and reloads a `Scenario` byte-identically
(`Scenario.to_dict()`/`from_dict()`, no lossy step); `scenario_runner.run()` produces exactly the
`SimulationContext` shape `register_occupants()`/`SimulationRuntime` both already expect (by explicit,
documented design in `scenario_runner.md`'s own refinement history); `record_outcome()`'s three
parameters are read directly off a completed `SimulationRuntime` with no intermediate step. Nine
adapter-free boundaries out of eleven is the headline positive finding of this whole review.

## 3. Required data transformations

Beyond the two adapters above, every *within-stage* transformation already exists and needs no new
work — listed here for completeness, since "required data transformations" is a distinct review
question from "required adapters" even where the answer overlaps:

| Transformation | Where | Status |
|---|---|---|
| Intent + Building → per-id `Distribution` maps | `resolve_intent()` (§4 of `dataset_intent.md`) | Designed, not built |
| `Distribution` → sampled value | `scenario_generator.sampling.sample()` | Implemented |
| Sampled fields → resolved `Scenario` | `generate_scenario()` | Implemented |
| `Scenario` engineering states → live `Building` copy mutation | `scenario_runner.building_initializer` | Implemented |
| `behaviour_profile_id` → `BehaviorProfile` + strategies | `behaviour_profile_resolver.registrar` | Implemented |
| `ScenarioEvent` → `Building`/`NavigationGraph` mutation | `scenario_event_executor.handlers` | Implemented |
| `HazardSnapshot`(t) → `HazardSnapshot`(t+dt) | `HazardEvolutionEngine.evolve()` | Implemented |
| `MultiAgentSimulationResult` → per-time occupancy | `MovementTimelineOccupancyProvider` | Implemented |
| Ground Truth → `BuildingObservation` | `SensorFusion.fuse()` + `OccupancyEstimator` | Implemented, **not composed** (§2.1) |
| `TickResult` stream + `MultiAgentSimulationResult` → `ScenarioOutcomeLabels` | `record_outcome()` | Designed, not built |
| Real domain objects → `SimulationRunRecord` IR | Ingestion Adapter | Designed, not built (§2.2) |
| `SimulationRunRecord`s → CSV/Parquet/RL Episodes/... | `dataset_builder` shape+format builders | Designed, not built |

No transformation on this list requires touching an already-frozen package's own internals — every one
is either already-implemented, or already-specified as new, additive code in a package that does not
yet exist. This is the same finding §2 already established, restated at the data level rather than the
package-boundary level.

## 4. Required orchestration

**No orchestration layer exists anywhere in this codebase today** — every package reviewed in §0-§3 is a
library, correctly scoped to one stage; nothing currently calls the full sequence end to end, and
nothing currently decides *how many* scenarios to generate, *which* of them to simulate, or *when* a
campaign is resumable versus must restart. This section designs that layer's shape (still
architecture-only — no code).

**Three phases, matching a natural cost/resumability boundary already visible in the pipeline itself:**

```
PHASE A -- Generation Campaign          (cheap per-scenario, already resumable)
    resolve_intent() → run_batch_pipeline() → scenario_storage.save_scenario() [xN]

PHASE B -- Simulation Campaign           (expensive per-scenario, NOT currently resumable -- §10)
    for each stored Scenario not yet in the Outcome catalog:
        scenario_runner.run() → register_occupants() → SimulationRuntime(...).run()
            → record_outcome() → scenario_outcome_storage.save_outcome()

PHASE C -- Dataset Export                (cheap relative to A/B, idempotent, re-runnable)
    Ingestion Adapter over the full Outcome catalog → dataset_builder.export_dataset() [xM formats]
```

A single top-level **Dataset Generation Pipeline** entry point (named, not designed in code, by
`dataset_intent.md` §17) owns invoking Phase A, then B, then C, against one campaign configuration —
`(intent, building, count, dt, decision_engine config, casualty_policy, perception_provider config,
export shapes/formats)`. This is the one place in the entire architecture where every package from
`dataset_intent` through `dataset_builder` is legitimately allowed to be imported together, since
orchestrating all of them *is* its job — no other package should ever import this many upstream
neighbors (§2.2's own observation about the Ingestion Adapter applies doubly to whatever module actually
calls it).

**Phase A already has everything it needs** — `scenario_pipeline.run_batch_pipeline()`'s index-keyed
seeding and `scenario_storage.load_accepted_hashes()` together already make Phase A resumable (§10).
**Phase B is the one phase with no existing resumability mechanism at all** — designed in §10, since
"required orchestration" and "failure recovery" are the same design question for this phase
specifically.

## 5. Missing components

Consolidated across every document in this lineage plus this review's own findings — nothing here is
newly invented, only newly *counted together*:

| Component | First flagged | Blocking? |
|---|---|---|
| Concrete `PerceptionProvider` composition | `simulation_runtime.md` §13 | **Yes** — §2.1, blocks Perception as a real pipeline stage |
| Dataset Builder Ingestion Adapter | `dataset_builder.md` §6 | **Yes** — §2.2, blocks the final boundary |
| `scenario_outcome/` package itself | `dataset_intent.md` | Yes — no code exists |
| `scenario_outcome_storage/` package itself | `dataset_intent.md` §16 | Yes — no code exists |
| `dataset_intent/` package itself | `dataset_intent.md` | Yes — no code exists |
| `dataset_builder/` package itself | `dataset_builder.md` | Yes — no code exists |
| Definition Storage/Catalog | `dataset_intent.md` §2 | No — worked around (§6 of that doc), but see §6.4 below for the consequence of leaving it unworked-around |
| `CasualtyPolicy` implementation | `dataset_intent.md` §13 | No — deliberately deferred, `fatalities=None` is a valid, honest campaign output |
| **Phase B resume mechanism** (new, this review) | this document, §10 | **Yes**, for the stated tens-of-thousands-of-runs goal specifically |
| **Dataset Generation Pipeline entry point** (new, this review) | this document, §4 | Yes — nothing currently calls any of this end to end |
| **A shared content-hash utility** (new, this review) | this document, §7 | No — current duplication is a maintainability risk, not a blocker |
| **A default-preserving round-trip integration test** (new, this review) | this document, §6.3 | No — a test gap, not a blocking gap |

## 6. Hidden architectural mismatches

Four findings, ordered by severity. The first is verified empirically (this session's own
`simulation_runtime/` test suite hit it directly), not merely inferred from reading code.

### 6.1 Unseeded randomness inside `behavior_library` — verified, severe (see §9 for the full reproducibility analysis)

`behavior_library/decision_strategies.py`, `pre_movement_strategies.py`, and `route_choice_strategies.py`
each default `self.rng = rng or random.Random()` — an injectable-but-unseeded RNG. `DEFAULT_PROFILE_
REGISTRY` (`behaviour_profile_resolver/registry.py`) constructs every strategy **once, at module-import
time**, never passing `rng=`. Five of its six profiles (`Adult_Default`, `Child_Default`,
`Wheelchair_Default`, `FireWarden_Default`, `Visitor_Default`) use at least one such strategy; only
`Staff_Default` (`AlwaysEvacuateDecisionStrategy` + `NoPreMovementDelay`, both argument-free and
random-free) is deterministic. Because the registry is a module-level singleton, the *same* unseeded
`random.Random()` instance is shared across **every occupant, every scenario, and every run for the
lifetime of the process** — not merely "unseeded across process restarts," but stateful and advancing
unpredictably *within* a single campaign run too. This is not a hypothetical: this session's own
`tests/test_simulation_runtime.py` had to switch its fixture from `Adult_Default` to `Staff_Default`
specifically because the determinism/replay tests failed non-deterministically against the default
profile.

### 6.2 `ScenarioDefinition.seed` is not read by the Generator

`scenario_definition/definition.py`'s own docstring: "Stored so a later Generator run is reproducible —
not used by anything inside this package." Confirmed this pass: `generate_scenario()`
(`scenario_generator/generator.py`) derives its attempt seed exclusively from `GenerationRequest.seed`
— `request.definition.seed` is never read anywhere in the sampling path. A caller who sets
`ScenarioDefinition.seed` expecting it to control generation, without *separately* also passing the
matching value as `GenerationRequest.seed`/`run_pipeline(..., seed=...)`, silently gets a different,
unrelated seed's worth of sampling. Low severity (nothing currently misuses this, and every existing
test/fixture passes `seed` to the request explicitly) but worth documenting plainly so a future Dataset
Generation Pipeline implementation does not assume the field is load-bearing.

### 6.3 Two independently-authored, uncoupled inverse mappings for engineering default state

`scenario_generator/generator.py::_generate_door_states`'s `default_for(door)` reads `door.locked`/
`door.normally_open` to decide the *sampling default* for an unconstrained Definition; `scenario_runner/
building_initializer.py::apply_door_state` writes `door.locked`/`door.normally_open` from a resolved
`DoorState`. These are meant to be exact inverses of each other (generate-default should read precisely
what apply-state writes) but are hand-coded independently in two packages that do not import each other
and share no common mapping table. The same shape recurs for Exit/Stair/Obstacle/Camera/Detector.
**No integration test in the current suite exercises the round-trip** ("a Definition with no stated
constraint for door X, run through the full Generator→Runner pipeline, must leave door X in exactly its
original Building-authored state") — a future change to either mapping (e.g. a new `DoorState` member)
could silently desynchronize the two without any existing test failing. §7.2 proposes the same fix as
the content-hash duplication below: extract the mapping, don't just test around it.

### 6.4 `definition_id` staleness has no detection mechanism

A consequence of the still-missing Definition Catalog (§5): `dataset_intent.md` §6 mints `definition_id`
from `(intent, building_id, parameters)`, **not** from the resolved `ScenarioDefinition`'s own content.
If an `IntentProfile`'s tuning changes between two campaign runs (a plausible, ordinary maintenance
change — adjusting a `WeightedOptions` weight), the *same* `definition_id` now denotes a *different*
actual `ScenarioDefinition`. This does **not** break `Scenario`-level reproducibility — `ScenarioMetadata.
definition_content_hash` is computed from the actual resolved content and remains correct regardless —
but it does mean `definition_id` alone is not a safe cross-campaign identity claim, and nothing in the
current or proposed architecture would ever notice or flag the drift, because nothing catalogs which
`definition_id` has ever meant which content. Low severity given the hash-based safety net; worth
documenting so `definition_id` is never mistaken for a stronger guarantee than it is.

## 7. Duplicated responsibilities

### 7.1 Content-hash computation — three independent implementations of one recipe

`scenario_generator/metadata_builder.py::compute_definition_content_hash()`, `scenario_validator/
dataset_validation.py::compute_candidate_content_hash()`, and (proposed) `dataset_intent`'s own
`definition_id`-minting helper (`dataset_intent.md` §6) all implement the identical recipe —
`json.dumps(..., sort_keys=True, separators=(",", ":"))` → `hashlib.sha256`. This was, each time,
a *deliberate* choice (§2 of `dataset_intent.md`: "the established precedent... same algorithm,
independently implemented per-package, not cross-imported," preserving each package's one-way
dependency direction). **This review's recommendation, not a reversal of that reasoning**: extract the
shared recipe (not the domain-specific *what to hash*, only the generic `canonical_json_sha256(data:
dict) -> str` primitive) into `serialization/` — already established in this architecture as the one
dependency-free, domain-agnostic utility location every layer may safely import (`dataset_builder.md`
§9.3 reuses it for exactly this reason). This removes the duplication without reintroducing any
cross-package dependency, since `serialization/` sits *below* every domain package, not beside one of
them.

### 7.2 Building-object-id enumeration — two independent traversals of the same shape

`scenario_generator/generator.py`'s private `_zone_lookup`/`_doors_by_id`/`_exits_by_id`/`_stairs_by_id`/
`_obstacles_by_id`/`_cameras_by_id`/`_detectors_by_id` each walk `building.floors` to build an
`id -> object` map. `dataset_intent`'s `IntentProfile.resolve()` (§4 of `dataset_intent.md`, "enumerate
a Building's own zones/doors/exits/...") needs the **identical** traversal to build its per-category
distribution maps, and — per that document's own dependency-direction rule — is forbidden from
importing `scenario_generator` to reuse its private helpers. Recommendation: promote this traversal to
a small set of `models`-level convenience functions or methods (e.g. `Building.zones_by_id()`) — `models`
is the one package both `scenario_generator`'s construction module and `dataset_intent` are already
permitted to import, making this the natural, dependency-direction-safe home, not a new cross-cutting
utility package.

### 7.3 Checked and cleared — not duplication

Three superficially-similar-looking cases were checked this pass and found to be genuinely different
problems, not duplicated logic: (a) `GroundTruthCameraProvider`'s point-in-polygon coverage test vs.
`GroundTruthSmokeDetectorProvider`'s bounding-box `zone.contains()` test — different geometric
questions (camera field-of-view vs. point containment), not the same logic twice; (b) occupant-position-
to-zone resolution appears at generation time (`scenario_generator`, geometry-based, zone known first),
runtime (`MovementTimelineOccupancyProvider`, timeline-based, no geometry at all), and export time
(`dataset_builder`, already zone-keyed, no resolution needed) — three different sub-problems at three
different pipeline stages, not one problem solved three times; (c) seed derivation (`scenario_generator.
seed_manager`, for sampling randomness) vs. content hashing (§7.1, for identity) both use `sha256` but
for unrelated purposes — sharing a primitive is not the same as duplicating a responsibility.

## 8. Performance considerations

Ordered by expected impact at "tens of thousands of runs" scale:

### 8.1 `AIDecisionEngine.decide()` — O(zones) pathfinding searches, every tick, every run

`ai_decision/engine.py::_zone_recommendation()` calls `hazard_engine.nearest_exit(node.id)` once per
zone node, inside `decide()`, called once per `SimulationRuntime` tick (§8 of `simulation_runtime.md`).
Total cost for one campaign: **O(zones × ticks_per_run × runs)** independent pathfinding searches. This
is the single largest, most direct multiplier in the entire pipeline — a building with 50 zones, a
`dt` producing 100 ticks per run, and a 20,000-run campaign is 100 million pathfinding searches, before
counting anything else. **Recommendation, not a redesign**: campaign configuration should treat `dt` as
a cost-control knob, not merely a fidelity one — coarser `dt` for large campaigns is the only lever
available today without touching the already-frozen `AIDecisionEngine`/`SimulationRuntime`.

### 8.2 `scenario_runner.run()` — full Building deep-copy and `NavigationGraph` rebuild, every run

`building_initializer`'s deep-copy-before-mutate pattern and `navigation_initializer`'s
`NavigationGraphGenerator().build()` call are both, correctly, per-run — the *state* differs every run,
but the underlying Building *topology* (which zones connect to which via which doors/exits/stairs) is
typically identical across an entire campaign against one Building. `scenario_runner` has no mechanism
to distinguish "topology changed, must rebuild" from "only states changed, topology is identical" and
always does the full rebuild — architecturally correct (it cannot safely assume topology is unchanged
without checking, and checking would itself add complexity to an already-frozen, approved package) but
worth flagging as a real, repeated cost at scale, and a plausible target for a **future**, separate
performance-focused architecture pass — explicitly not proposed as a change to `scenario_runner` here.

### 8.3 `scenario_storage.load_accepted_hashes()` — O(n) full-catalog rehash at every campaign start

Already self-flagged in `scenario_storage/storage.py`'s own docstring ("a future phase could add a
content_hash column"). At tens of thousands of stored scenarios, every *new* campaign run against the
same `storage_root` re-reads and re-hashes every previously-accepted `Scenario`'s full JSON file before
generating a single new one. `dataset_intent.md` §19 already flagged the identical concern for the
proposed `scenario_outcome_storage` catalog — this is one shared recommendation covering both: add a
`content_hash` column to both catalogs, so accepted-hash/dedup lookups become an O(n) *CSV row scan*
(cheap) instead of an O(n) *JSON file load + rehash* (expensive), without changing either catalog's
public read/write contract.

### 8.4 Dataset Builder export — in-memory materialization at extreme scale

`dataset_builder.md` §16 already flagged this as an open question ("streaming/incremental export"). This
review reconfirms it as the right question to prioritize once campaign sizes reach "tens of thousands":
the baseline design (§11 of that document) takes `Sequence[SimulationRunRecord]` — a whole-campaign-in-
memory contract. For CSV/JSON export this is a real but bounded cost; for the tabular id-flattening step
(§9.1 of that document) it also requires a full pre-scan of every run to compute the shared column
ordering before any row can be written, meaning even a streaming *writer* would still need two passes
over the data unless the id universe is known in advance (e.g. from the Building alone, independent of
which runs happened to touch which ids) — flagged as a refinement worth resolving in tandem with any
future streaming-export design, not resolved here.

## 9. Reproducibility considerations

**A tiered reproducibility claim, stated precisely because the current architecture does not uniformly
support one:**

1. **Scenario generation is fully reproducible.** `(definition_content_hash, seed, generation_version)`
   jointly determine a `Scenario` byte-for-byte (`scenario_engine.md` §4.3, unchanged, reconfirmed).
   `scenario_generator`, `scenario_validator`, `scenario_pipeline` all correctly avoid `random` outside
   the one sanctioned, seeded path.
2. **Simulation of a given `Scenario` is reproducible only if every registered occupant's
   `behaviour_profile_id` resolves to a strategy combination with zero unseeded randomness.** Today,
   that is true only for `Staff_Default` (and any future profile built exclusively from
   `AlwaysEvacuateDecisionStrategy`/`NoPreMovementDelay`/`ShortestRouteChoiceStrategy`-style
   argument-free, deterministic strategies) — **false** for five of the six default profiles (§6.1). A
   campaign using the default registry as-is cannot honestly claim its simulation phase is reproducible,
   even though every layer *below* behaviour resolution (`HazardEvolutionEngine`, `ScenarioEventExecutor`,
   `MultiAgentSimulation`'s own coordination logic, `SimulationRuntime`'s own clock) is itself fully
   deterministic (`simulation_runtime.md` §16, unaffected by this finding — the non-determinism enters
   at occupant registration, upstream of everything `SimulationRuntime` itself does).
3. **Outcome Labels inherit whatever reproducibility level the run beneath them had** — `record_outcome()`
   itself introduces no new randomness (`dataset_intent.md` §14), so a non-reproducible simulation
   produces non-reproducible labels, and a (hypothetically fixed, §9 point 2) reproducible simulation
   produces reproducible labels.
4. **A full Run's configuration is not itself a reproducibility-capturable value** — already flagged
   (`dataset_intent.md` §10.1/§19): `decision_engine`, `perception_provider`, `casualty_policy` are live
   Python objects, not serializable `Distribution`-shaped data, so even a fully-deterministic simulation
   phase has no equivalent of `definition_content_hash` capturing "which exact run configuration produced
   this outcome." Flagged, not resolved, consistent with that document's own treatment.

**This review's one concrete recommendation**: either (a) construct `behaviour_profile_resolver`'s
registry with explicitly seeded `rng` instances derived from the same seed hierarchy
`scenario_generator.seed_manager` already establishes (e.g. a new, dedicated `"behaviour"` category
child stream, extending — not modifying — the existing `CATEGORY_KEYS` mechanism, since
`RESERVED_CATEGORY_KEYS` already demonstrates this extension point exists), or (b) explicitly document,
for any campaign wanting reproducible simulation, that it must restrict itself to deterministic-only
behaviour profiles until (a) is designed. Neither option is designed further here — flagged into open
questions (§11) as the single highest-value follow-up this whole review identifies.

## 10. Failure recovery and resume behavior

### 10.1 Phase A (Generation) — already resumable, no new work needed

`run_batch_pipeline()`'s index-keyed (not stream-position-keyed) seed derivation and `scenario_storage.
load_accepted_hashes()` together already provide exactly what unattended, resumable generation needs:
re-running Phase A against the same `storage_root` after a crash, having first called
`load_accepted_hashes()`, reproduces the identical remaining scenarios without re-generating or
re-storing anything already accepted. **One operational hazard worth stating explicitly**: an
orchestrator that forgets to call `load_accepted_hashes()` before resuming risks `scenario_storage.
save_scenario()`'s `FileExistsError` guard firing mid-campaign on a scenario whose `scenario_id` was
already stored — not a design flaw (the guard is correct and intentional), but a real operational
requirement any Phase A orchestration code must honor.

### 10.2 Phase B (Simulation) — no resume mechanism exists; this review designs one

Nothing today records "which stored Scenarios have already been simulated." A crash at scenario #7,342
of 10,000 has no way to resume at #7,343 without re-simulating from #1 — unattended, safely-resumable
operation at "tens of thousands of runs" scale genuinely requires this. **Design, mirroring Phase A's
own pattern exactly**: before starting/resuming Phase B, scan `scenario_outcome_storage`'s catalog
(§16 of `dataset_intent.md`) for the set of `scenario_id`s already present, and skip them — the same
`accepted_hashes`-style pre-scan `run_batch_pipeline()` already performs, applied to the Outcome catalog
instead of the Scenario catalog. This requires no change to `scenario_outcome_storage`'s own design
(§16 of `dataset_intent.md` already specifies `scenario_id` as an ordinary catalog column, sufficient
for exactly this scan) — only a new orchestration-level function (`load_simulated_scenario_ids(storage_
root) -> FrozenSet[str]`, structurally identical to `load_accepted_hashes()`) belonging to whatever
package eventually owns Phase B orchestration (§4), not to `scenario_outcome_storage` itself.

**Granularity limitation, stated rather than silently accepted**: resume operates at whole-run
granularity, not whole-tick granularity — a crash mid-run (after tick 40 of a 100-tick run) loses that
run's in-progress `TickResult`s entirely (`SimulationRuntime` holds them in memory only, §8 of
`simulation_runtime.md`; nothing persists intermediate ticks), and the *entire* run is redone from
scratch on resume. Acceptable given §8.1's own finding that a single run's cost, while real, is
bounded and small relative to a 10,000-run campaign — flagged as a deliberate trade-off, not proposed
for finer-grained checkpointing here.

### 10.3 Phase C (Export) — cheap to redo, no resume mechanism required

`dataset_builder`'s exports are pure functions of already-persisted data (§13 of `dataset_builder.md`) —
a crash mid-export requires only re-running the export, not re-simulating anything. At extreme scale
this could still be wasteful (§8.4); a streaming/chunked export (already an open question in that
document) would make even this cheaper, but is not required for correctness the way Phase B's resume
mechanism is.

## 11. Open questions carried forward from this review

- **Seeded `rng` for `behaviour_profile_resolver`'s default registry** (§9) — the single highest-value
  follow-up; two options sketched, neither designed in code.
- **Perception composition** (§2.1) — now the largest concrete implementation gap; already fully
  specified by `perception_layer_review_2.md`, simply never built.
- **Dataset Builder Ingestion Adapter and its package placement** (§2.2) — field mapping already
  specified (`dataset_builder.md` §6); placement (dedicated package vs. folded into the Dataset
  Generation Pipeline) still open.
- **The Dataset Generation Pipeline entry point itself** (§4) — named and shaped here and in
  `dataset_intent.md` §17; not yet a package, not yet a module, not yet code.
- **`content_hash` columns for both the Scenario and Outcome catalogs** (§8.3) — one shared
  recommendation, affecting two documents' worth of previously-independent "future phase" notes.
- **Extracting the shared content-hashing primitive into `serialization/`** (§7.1) and **Building-id
  enumeration into `models/`** (§7.2) — both low-risk, low-urgency refactors of already-frozen packages;
  neither blocks anything, both reduce future drift risk.
- **A default-preserving round-trip integration test spanning `scenario_generator` ↔ `scenario_runner`**
  (§6.3) — the cheapest possible mitigation for that finding, addable without touching either package's
  production code.

## 12. Status

Review only. No package audited in this document has been modified. The pipeline's first eight stages
integrate cleanly, with the two gaps in §2 the only load-bearing missing adapters; the last three stages
are sound, unimplemented designs this review found no reason to reopen. The single most consequential
finding is §6.1/§9's verified, non-hypothetical reproducibility gap in the default behaviour profile
registry — everything else in this document is either a real but bounded performance concern (§8), a
genuinely new but small piece of missing orchestration (§4/§10.2), or a low-severity documentation/
maintainability recommendation (§6.2-§6.4, §7). Nothing found here contradicts any prior frozen
architecture; every recommendation is additive, and every one is explicitly flagged as not designed
further in this pass.

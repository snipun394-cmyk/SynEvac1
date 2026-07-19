# Dataset Intent & Scenario Outcome Labels — Architecture Proposal

Status: **proposal, open for review**. No code changes accompany this document — architecture only,
per this pass's explicit instruction. Two related, but structurally distinct, systems are designed
together because the user brief asked for both in one review and because they sit at opposite ends of
the same pipeline (Intent shapes what gets *generated*; Outcome Labels record what *happened* when it
was *simulated*) with the entire, unmodified Scenario Engine + Simulation Runtime between them.

## 1. Purpose

**Dataset Intent** is a high-level policy layer sitting above `scenario_definition/` (frozen,
unmodified). A caller states *why* they want a dataset — `Standard Evacuation`, `Mixed Realistic
Incidents`, `Rescue Training`, `Firefighter Training`, `RL Training`, or `Custom` — instead of
hand-authoring every `Distribution`-valued field on a `ScenarioDefinition` themselves. Each named
intent deterministically *translates into* a `ScenarioDefinition` (§4-§8); it never bypasses, wraps, or
duplicates anything `scenario_generator`/`scenario_validator`/`scenario_pipeline` already do with that
`ScenarioDefinition` once produced.

**Scenario Outcome Labels** are a new, small, structured record — `reachable/evacuated/trapped
occupants`, `fatalities`, `rescue required`, `building cleared`, `maximum congestion`, `worst hazard
zone`, `simulation end reason` — computed *after* a `SimulationRuntime` run completes (§10-§14), from
artifacts that already exist (`MultiAgentSimulationResult`, the `TickResult` stream, `SimulationContext`).
This is new data with no existing precedent anywhere in this codebase; §3 explains why it cannot be
computed *during* generation or validation, and why one label in the requested list (`Fatalities`)
cannot be honestly computed at all without a new, explicitly-flagged policy decision this document does
not make.

## 2. Grounding in existing code

Confirmed this pass, directly from source:

- **`ScenarioDefinition`** (`scenario_definition/definition.py`) is `fire: FireDefinition`,
  `engineering: EngineeringConstraints`, `occupant: OccupantDefinition`,
  `event_templates: Tuple[EventTemplate, ...]`, `seed: Optional[int]` — every constrained field is a
  `Distribution` (`FixedValue`/`UniformRange`/`WeightedOptions`, `scenario_definition/distributions.py`),
  every per-object field is `Mapping[id, Distribution]` keyed by the *Building's own* object ids
  (`occupancy_distribution`, `door_state_distribution`, `stair_state_distribution`, ...). Nothing in
  this package samples anything, imports `random`, or imports `navigation` (§12 of
  `docs/architecture/scenario_engine.md`, reconfirmed unchanged this pass).
- **`scenario_generator.generate_scenario(request: GenerationRequest)`** (`scenario_generator/
  generator.py`) is the *only* place a `ScenarioDefinition`'s distributions are ever sampled — an id
  absent from a Definition's distribution map falls back to the Building's own currently-authored state
  (`_generate_door_states`'s `default_for`, etc.), never to a fabricated default. `GenerationRequest`
  (`scenario_generator/request.py`) requires a caller-supplied `definition_id: str` — confirmed:
  **`ScenarioDefinition` itself carries no id of its own**, "reproducibility depends on
  `definition_content_hash` *and* a stable `definition_id`... which only a Definition-storage/catalog
  layer above this package can supply." No such layer exists anywhere in this codebase today (confirmed
  by repository-wide search) — every current caller of `generate_scenario()` (test fixtures, this
  session's own work) invents a `definition_id` string by hand. §6 makes Intent Resolution that missing
  layer's first legitimate occupant, for Definitions it produces.
- **`scenario_generator.metadata_builder.compute_definition_content_hash()`** and
  **`scenario_validator.dataset_validation.compute_candidate_content_hash()`** are two *independently
  implemented* functions using the identical recipe (`json.dumps(..., sort_keys=True)` →
  `hashlib.sha256`) — confirmed this is the established precedent for "the same hashing algorithm,
  restated locally per package that needs it," not a shared import. §6 follows this precedent rather
  than importing `scenario_generator` from the new Intent package.
- **`docs/architecture/scenario_engine.md` §1's pipeline diagram** (`Definition → Generator → Validator
  → Accepted (stored) / Rejected → Generate Again`) is unchanged by this document. §10 there: "Difficulty
  ... not a `ScenarioDefinition` field, computed only after the Validator accepts a candidate" — the one
  existing precedent for "a value computed about a `Scenario` after the fact," and explicitly *not* a
  simulated-outcome value (difficulty needs no simulation to compute). Outcome Labels are a different
  kind of after-the-fact value — computed after *simulating*, not after *validating* — §3 draws this
  distinction precisely because it determines where each concept is allowed to live.
- **`HazardSeverity.from_score()`** (`hazard/severity.py`) is explicitly documented: "a placeholder
  classification, not a validated life-safety threshold model." **No fatality, casualty, tenability, or
  life-safety model exists anywhere in this codebase.** `AIDecisionEngine.DEFAULT_UNSAFE_SEVERITY_
  THRESHOLD = HazardSeverity.HIGH` (`ai_decision/engine.py`) is the closest existing concept — a zone is
  `is_unsafe` if severity ≥ HIGH or unreachable — and even that is a routing/priority signal, never
  represented as "occupants here have died." §13 treats this honestly.
- **`MultiAgentSimulationResult`** (`simulator/multi_agent_result.py`, read this pass) already carries
  `unreachable_occupant_ids`, `peak_edge_occupancy: Dict[str, int]`, `peak_node_occupancy: Dict[str,
  int]`, and `total_evacuation_time` — **"Maximum congestion" is already computed by the already-frozen
  `MultiAgentSimulation` and needs no new derivation** (§11). Per-occupant `OccupantTimeline.arrival_
  time`/`.state` (`OccupantState.ARRIVED`/`.UNREACHABLE`/...) is what "Reachable"/"Evacuated"/"Trapped"
  are built from.
- **`SimulationRuntime`** (`simulation_runtime/runtime.py`, approved, this session's prior phase) runs
  `context.simulation.run()` exactly **once**, atomically, at construction — before any tick — and
  `.run()` returns `Tuple[TickResult, ...]`, each carrying `hazard_snapshot`/`occupancy_snapshot`/
  `decision`/`fired_events`/`observation`. There is exactly one stop condition today
  (`current_time >= end_time`, `docs/architecture/simulation_runtime.md` §9) — no early-stop predicate,
  no labeled abort path. §12 works within this, unmodified.

## 3. The most consequential finding: Outcome Labels split into two families with different lifetimes

**This must be understood before §10-§14 make sense.** Because `SimulationRuntime` resolves the entire
occupant-movement timeline **once, atomically, before a single tick runs** (`docs/architecture/
simulation_runtime.md` §2 — routes are fixed at registration, unaffected by ticking), some requested
labels are **fully determined the instant a `SimulationRuntime` is constructed**, independent of `dt`,
independent of how many ticks actually ran, independent of `end_time`:

- **Reachable / Unreachable occupants** — a pure fact about `MultiAgentSimulationResult.
  unreachable_occupant_ids`, known before tick 1.
- **Evacuated / Trapped occupants**, **Building cleared** — depend on comparing each occupant's
  `arrival_time` (if any) against an **evaluation time**, not against "how far the Runtime happened to
  tick." §10 makes this evaluation time an explicit, documented parameter (defaulting to `runtime.
  end_time`) precisely so a label never silently leaks information the Runtime's own configured window
  did not actually cover — an occupant whose precomputed `arrival_time` is 500s does not count as
  "evacuated" for a run configured with `end_time=120`, even though `movement_result` already "knows"
  they eventually would have escaped. Reporting otherwise would make Outcome Labels lie about what a
  short, cheap `end_time` actually observed.

The other family genuinely requires the tick stream to have run at all:

- **Maximum congestion** is a `movement_result` fact too (§2 above), available immediately — but is
  listed here because, unlike Evacuated/Trapped, it needs no evaluation-time comparison; it is already
  a whole-run peak, matching `end_time`'s own role as "the window this run considered."
- **Worst hazard zone** needs the actual sequence of `hazard_snapshot`s produced across ticks — a
  `SimulationRuntime` constructed but never ticked (`runtime.run()` never called, or `end_time == 0.0`)
  has no hazard history beyond `context.initial_hazard_snapshot`, and Outcome Label computation must
  say so (`None`), never fabricate a "worst zone" from a single, always-`t=0` snapshot.
- **Simulation end reason** is a classification of *how the run's own window concluded* — §12.
- **Fatalities / Rescue required** need a policy over the hazard timeline that does not exist (§13).

Design consequence: `record_outcome()` (§10) takes the **full `Tuple[TickResult, ...]`** a caller
already has from `SimulationRuntime.run()` (or has accumulated tick-by-tick), not just the final one —
"worst hazard zone" and "simulation end reason" are computed by scanning that whole sequence, reusing
data the Runtime already produced rather than re-deriving it from scratch or requiring `SimulationRuntime`
itself to change shape.

## 4. Dataset Intent — design principles

1. **Intent translates; it does not sample.** An `IntentProfile`'s output is a `ScenarioDefinition` —
   still pure declarative data, still zero randomness, still "declares what may be sampled, samples
   nothing" (§3's own frozen ethos, `scenario_engine.md` §1). Variety within an intent (e.g. "Mixed
   Realistic Incidents" spanning many different actual fires) comes entirely from the *existing*
   Generator sampling a *wide* `Distribution` this layer hands it — never from Intent itself drawing
   values.
2. **Building-generic, not per-building-authored.** The whole point of an intent ("specify the purpose...
   rather than configuring every constraint manually," this pass's own brief) is that a caller picks
   `Rescue Training` for *any* Building and gets a sensible `ScenarioDefinition` back — an `IntentProfile`
   enumerates a Building's own zones/doors/exits/stairs/cameras/detectors (the same `_zone_lookup`-style
   traversal `scenario_generator.generator` already performs) and applies one **uniform, intent-specific
   policy function per category**, never a hand-authored per-id table. A new Building needs no new
   Intent-side authoring at all.
3. **Intent cannot reason about reachability — a real, load-bearing constraint, not an oversight.**
   `scenario_definition/` is forbidden from importing `navigation` (§12 of `scenario_engine.md`,
   reconfirmed §2 above) precisely because reachability is exclusively a Validator/simulation concern.
   Intent Resolution produces a `ScenarioDefinition` and therefore **inherits this same restriction** —
   it structurally cannot ask "which zone is farthest from an exit" or "which door, if locked, strands
   the most zones" (that is `BuildingAnalysisEngine.critical_connectors()`, reachable only from
   `pathfinding`/`navigation`, both off-limits here). Where a "harder" intent (`Rescue Training`,
   `Firefighter Training`) wants scenarios more likely to trap occupants, it achieves this **honestly**
   — by widening the *probability* of engineering-object failure states (more doors/stairs plausibly
   `CLOSED`/`LOCKED`, lower `min_open_exits`) and letting rejection sampling and the eventual simulation
   surface the emergent difficulty — never by computing "difficulty" itself at Definition-construction
   time. This is the same "no repair, no feasibility-awareness in the Generator" philosophy already
   frozen in `scenario_engine.md` §1, applied one layer earlier.
4. **An intent's own parameters are an open, per-profile-interpreted bag** — `Mapping[str, Any]`,
   mirroring `ScenarioEvent.parameters`/`EventTemplate.parameters`'s own established "opaque, consumer-
   interpreted" convention. Each `IntentProfile` documents which keys it understands (e.g.
   `occupancy_scale: float`, `target_floor_ids: FrozenSet[str]`); an unrecognized key is ignored, not an
   error — the same tolerant convention `ScenarioMetadata.extra` already commits to for a materially
   different reason (forward-compatible schema growth) applied here for a similar one (a profile need not
   enumerate every key every caller might send).

## 5. `IntentProfile` / `resolve_intent()` — reusing the Behaviour Profile Resolver's own shape

**Deliberately the same structure `behaviour_profile_resolver/` already established for an unrelated,
but structurally identical, problem**: an opaque, named policy id (`behaviour_profile_id` there,
`DatasetIntent` here) resolved through a registry into something that knows how to produce a concrete
object, with a dedicated `LookupError` subclass for an unknown id, and an explicit escape hatch for a
caller who wants to bypass the registry entirely.

```
class DatasetIntent(Enum):
    STANDARD_EVACUATION = auto()
    MIXED_REALISTIC_INCIDENTS = auto()
    RESCUE_TRAINING = auto()
    FIREFIGHTER_TRAINING = auto()
    RL_TRAINING = auto()
    CUSTOM = auto()

class IntentProfile:                                    # interface, mirrors DecisionStrategy/
    def resolve(self, building, parameters) -> ScenarioDefinition:   # HazardSource-style markers
        raise NotImplementedError

DEFAULT_INTENT_REGISTRY: Mapping[DatasetIntent, IntentProfile] = {...}   # 5 entries -- CUSTOM excluded,
                                                                          # exactly as CUSTOM is not a
                                                                          # profile to look up (§8)

class UnknownDatasetIntentError(LookupError): ...        # mirrors UnknownBehaviourProfileError exactly

def resolve_intent(
    intent: DatasetIntent,
    building,
    parameters: Mapping = None,
    base_definition: Optional[ScenarioDefinition] = None,
    registry: Mapping[DatasetIntent, IntentProfile] = None,
) -> ScenarioDefinition:
    ...
```

`resolve_intent()` is the one entry point — a caller never talks to an `IntentProfile` directly, the same
way nothing outside `behaviour_profile_resolver` talks to a `BehaviorProfileTemplate` directly. §8 covers
`CUSTOM`'s distinct handling inside this same function.

## 6. Minting `definition_id` for an Intent-produced `ScenarioDefinition`

Since `resolve_intent()` is, for the five named intents, the **first** thing to construct a given
`ScenarioDefinition`, it is the natural (not merely convenient) place to also mint the `definition_id`
`GenerationRequest` requires and that no package in this codebase currently supplies (§2). Reusing the
established, independently-implemented-per-package precedent (§2) rather than importing
`scenario_generator`:

```
def _derive_definition_id(intent: DatasetIntent, building_id: str, parameters: Mapping) -> str:
    canonical = json.dumps(
        {"intent": intent.name, "building_id": building_id, "parameters": parameters},
        sort_keys=True, separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"def-{digest[:16]}"
```

Deterministic and reproducible — the same `(intent, building, parameters)` always mints the same
`definition_id`, matching every other identity-derivation rule in this codebase (never `uuid4()` for
anything meant to be reproducible). `CUSTOM` never mints one — a caller supplying their own
`ScenarioDefinition` already owns whatever `definition_id` they choose (§8).

## 7. Per-intent translation semantics

Each row is a *policy*, not a formula — the exact `Distribution` parameters (e.g. which `WeightedOptions`
weights) are an implementation-phase tuning decision, explicitly not fixed by this document, the same
"placement decided, formula not" caveat `scenario_engine.md` §10 already applies to Difficulty scoring.

| Intent | Occupancy | Fire | Engineering | Events |
|---|---|---|---|---|
| **Standard Evacuation** | Moderate, realistic per-zone `UniformRange` | Unconstrained ignition zone, one fire, common profile(s) | Left **unstated** — every id defaults to the Building's own authored state via the Generator's existing fallback (§2); no failures | None, or a small `FixedValue(True)` fire-alarm-style set |
| **Mixed Realistic Incidents** | Wide `UniformRange` per zone (empty-to-full) | Unconstrained ignition, wide `UniformRange` growth-rate, all `allowed_fire_profiles` | `WeightedOptions` across every category — some doors/stairs/devices plausibly degraded, most not | A representative mix of `EventTemplate`s with moderate `occurs` probabilities |
| **Rescue Training** | Skewed toward higher occupancy in zones with longer/likelier-degraded egress (by `zone_type`, §4 point 3 — never by computed distance) | Unconstrained or `zone_type`-weighted ignition | `WeightedOptions` skewed toward failure (lower `min_open_exits`, higher LOCKED/CLOSED weight) — **does not guarantee** a rescue-required outcome (§9) | Engineering-failure events more heavily weighted toward `occurs=True` |
| **Firefighter Training** | Realistic, varied, enough occupants to exercise `priority_evacuation_order` | Unconstrained ignition, varied growth rates for multi-zone escalation | Camera/detector `DeviceAvailability` distributions include real `FAILED` probability (exercises `BuildingObservation` `UNOBSERVED`, §13 of `perception_layer_review_2.md`) | Mid-simulation engineering-failure events timed via `UniformRange` on `EventTemplate.time`, to exercise `ScenarioEventExecutor` |
| **RL Training** | Maximally wide `UniformRange`/`WeightedOptions` on every field — coverage/diversity is the goal, not realism | Unconstrained, wide | Unconstrained, wide | Wide `occurs`/`time` distributions across the full space of `EventTemplate`s |
| **Custom** | Caller-supplied `ScenarioDefinition`, unchanged (§8) | — | — | — |

**RL Training's real distinguishing need is downstream, not in the Definition** (§4 point 1's own logic
applied to this specific intent): maximizing sampling *coverage* is fully expressible as "wide
distributions," which is all `ScenarioDefinition` can express. What RL training additionally wants —
every accepted `Scenario` run with a `PerceptionProvider` configured and a Dataset Logger attached
(`docs/architecture/simulation_runtime.md` §13/§18) — is a **Run Configuration** concern, entirely
outside what a `ScenarioDefinition`/`GenerationRequest` can express (`SimulationRuntime`'s constructor
arguments are live Python objects — a `DecisionEngine`, an optional `PerceptionProvider` — never
serializable `Distribution`-shaped data). §16 places this correctly: it belongs to the future Dataset
Generation Pipeline's per-run configuration step, not to Intent.

## 8. `CUSTOM` and the override mechanism

`CUSTOM` is not "an empty profile" — it is the explicit statement that **Intent is optional sugar, never
a mandatory gate**. `resolve_intent(DatasetIntent.CUSTOM, building, base_definition=my_definition)`
returns `my_definition` unchanged (identity), and `CUSTOM` is deliberately absent from
`DEFAULT_INTENT_REGISTRY` — looking it up as if it were a named profile is a caller error caught before
ever consulting the registry, not something `UnknownDatasetIntentError` needs to handle.

**"Start from a named intent, then override specific fields" needs no new mechanism.** `ScenarioDefinition`
is already a frozen dataclass; a caller (or a thin, optional convenience wrapper this document does not
mandate) already has everything needed via the standard library:

```
resolved = resolve_intent(DatasetIntent.MIXED_REALISTIC_INCIDENTS, building)
pinned = dataclasses.replace(resolved, fire=my_custom_fire_definition)
```

This is not designed further here because there is nothing to design — `dataclasses.replace()` already
does exactly this, for any field, and inventing a bespoke patch/override DSL on top would duplicate a
mechanism the language already provides for free.

## 9. Where Intent sits in the existing pipeline — no modification, confirmed

```
DatasetIntent + Building + parameters
        │
        ▼
  resolve_intent()                    <-- NEW (dataset_intent/, this document)
(produces ScenarioDefinition + definition_id)
        │
        ▼
  ScenarioDefinition ──────────────────────────────────────────────────┐
        │                                                              │
        ▼                                                              │
  GenerationRequest / BatchGenerationRequest          <-- UNCHANGED    │  (scenario_generator/,
        │                                                              │   frozen, §2)
        ▼                                                              │
  scenario_pipeline.run_pipeline() / run_batch_pipeline()  <-- UNCHANGED (scenario_pipeline/, frozen)
        │
        ├── Accepted ──► scenario_storage.save_scenario()   <-- UNCHANGED (scenario_storage/, frozen)
        │
        └── Rejected ──► Generate Again                      <-- UNCHANGED
```

Every box below `ScenarioDefinition` is exactly as `scenario_engine.md` §1 already froze it — Intent
Resolution's entire footprint is producing the one value (`ScenarioDefinition`) and one string
(`definition_id`) that already had to come from *somewhere* before a `GenerationRequest` could ever be
built. **A batch generated with an Intent and a batch generated by hand-authoring the identical
`ScenarioDefinition` are indistinguishable to every package below this line** — this is the compliance
finding this section exists to state explicitly: `scenario_generator`/`scenario_validator`/
`scenario_pipeline`/`scenario_storage` gain **zero** new code paths, zero new fields, zero new
dependencies as a result of this design.

**Outcome-based filtering (§16) never becomes a new Validator rejection category.** `scenario_validator`
checks structural validity *before* any simulation exists to have an outcome (`scenario_engine.md` §1:
"checks the candidate against the Definition and Building/Nav Graph integrity — sole gate") — teaching
it to reject based on a simulated outcome would require running a full `SimulationRuntime` *during*
validation, an enormous, explicitly out-of-scope redesign of the already-frozen rejection-sampling
pipeline. Outcome-based curation happens **after** storage, over already-accepted, already-stored
Scenarios (§16) — a `Scenario` that turns out "boring" once simulated is still a legitimately generated,
legitimately stored `Scenario`; a Dataset Generation Pipeline choosing not to keep it in a specific
*dataset* is a downstream curation decision, not a retroactive validation failure.

## 10. Scenario Outcome Labels — data model

```
@dataclass(frozen=True)
class CongestionPeak:
    scope: str        # "edge" | "node"
    object_id: str
    peak_occupancy: int

@dataclass(frozen=True)
class WorstHazardZone:
    zone_id: str
    hazard_score: float
    severity: HazardSeverity      # HazardSeverity.from_score(hazard_score) -- reused, not reinvented
    time: float                   # the tick at which this peak was observed

@dataclass(frozen=True)
class ScenarioOutcomeLabels:
    outcome_id: str                       # uuid4 -- see §10.1 for why this is NOT scenario_id
    scenario_id: str                       # ScenarioMetadata.scenario_id, verbatim
    recorded_at: str                       # ISO timestamp, provenance only (mirrors ScenarioMetadata)

    evaluation_time: float                 # §3 -- the window this label set actually covers

    total_occupants: int
    reachable_occupants: int
    unreachable_occupants: int             # subset with no route at all, ever
    evacuated_occupants: int               # arrival_time is not None and <= evaluation_time
    trapped_occupants: int                 # total - evacuated
    rescue_required: int                   # trapped - (fatalities or 0)
    fatalities: Optional[int]              # None <=> no CasualtyPolicy configured (§13) -- never 0
                                            #   as a default

    building_cleared: bool                 # trapped_occupants == 0

    max_node_congestion: Optional[CongestionPeak]   # from MultiAgentSimulationResult, §11
    max_edge_congestion: Optional[CongestionPeak]

    worst_hazard_zone: Optional[WorstHazardZone]     # None iff zero ticks ran (§3)

    simulation_end_reason: SimulationEndReason        # §12

    extra: Mapping[str, Any]               # open bag, mirrors ScenarioMetadata.extra's own precedent
```

### 10.1 Why a fresh `outcome_id`, and why this is not merged onto `Scenario`/`ScenarioMetadata`

A stored `Scenario` is immutable and content-hashed (`scenario_storage`, frozen) — and, critically, **the
same stored `Scenario` can legitimately be run through `SimulationRuntime` more than once**, with
different `dt`, different `end_time`, a different `AIDecisionEngine` tuning, or a different
`CasualtyPolicy` (§13), each producing a **different** `ScenarioOutcomeLabels`. Outcome Labels are
therefore **1:many** with a `scenario_id`, never 1:1 — writing them onto `ScenarioMetadata.extra` (which
would need to be mutated after the fact, on an object `scenario_storage.save_scenario()` already
persisted and content-hashed) is structurally wrong for the same reason two different `HazardSnapshot`s
at the same `timestamp` from different scenario branches still need distinguishable `snapshot_id`s
(`hazard/snapshot.py`'s own stated reasoning, reused here). `outcome_id` uses `uuid4()` deliberately,
**not** a content-hash — unlike `scenario_id`/`definition_id`, a Run's reproducibility is not fully
captured by serializable data (`decision_engine`, `perception_provider` are live Python objects, not
`Distribution`-shaped value objects) — flagged, not resolved, in §19.

## 11. Reachable / Evacuated / Trapped / Building Cleared / Maximum Congestion

Pure, deterministic post-hoc computation over `(context, movement_result, evaluation_time)` — no new
mechanism, every value already exists:

- `total_occupants = len(movement_result.occupants)`.
- `unreachable_occupants = len(movement_result.unreachable_occupant_ids)`.
- `reachable_occupants = total_occupants - unreachable_occupants`.
- `evacuated_occupants = count of OccupantTimeline where arrival_time is not None and arrival_time <=
  evaluation_time` — **not** `state == ARRIVED` read blindly (§3: that would leak past `evaluation_time`).
- `trapped_occupants = total_occupants - evacuated_occupants`.
- `building_cleared = trapped_occupants == 0`.
- `max_node_congestion` / `max_edge_congestion` — the single highest-valued entry of
  `movement_result.peak_node_occupancy` / `.peak_edge_occupancy` respectively (`None` if either mapping
  is empty, e.g. zero occupants). **Reused verbatim from `simulator/multi_agent_result.py` — this
  section computes nothing `MultiAgentSimulation` did not already compute.**

## 12. Simulation End Reason

```
class SimulationEndReason(Enum):
    NO_OCCUPANTS = auto()          # total_occupants == 0 (mirrors SimulationRuntime's own
                                    #   "starts finished" case, docs/architecture/simulation_runtime.md
                                    #   §9's end_time == 0.0 default)
    ALL_EVACUATED = auto()         # trapped_occupants == 0 at evaluation_time, regardless of
                                    #   whether current_time reached end_time
    END_TIME_REACHED = auto()      # current_time >= end_time and trapped_occupants > 0
    ABORTED = auto()               # caller-supplied only -- see below
```

`record_outcome()` infers `NO_OCCUPANTS`/`ALL_EVACUATED`/`END_TIME_REACHED` automatically — a pure
function of `(movement_result, evaluation_time, runtime.current_time, runtime.end_time)`, needing no
Runtime change. **`ABORTED` cannot be inferred** — a computation running *after* `SimulationRuntime.
run()` returned normally, by definition, never observed an exception. A caller that catches an exception
from `runtime.tick()`/`runtime.run()` (`docs/architecture/simulation_runtime.md` §15: fail-fast, no
silent recovery) may still call `record_outcome(..., end_reason_override=SimulationEndReason.ABORTED)`
with whatever `TickResult`s completed before the failure — producing a labeled, honest **partial**
outcome record rather than discarding the run's partial data silently. This does not require
`SimulationRuntime` to gain any new labeled-abort capability of its own (§9's "no modification" holds).

**Explicitly not designed here**: an early-stop capability that would make `ALL_EVACUATED` actually halt
`SimulationRuntime`'s own tick loop before `end_time`. That is the "early-stop predicates" extension
point `docs/architecture/simulation_runtime.md` §9/§21 already named and left undesigned — this document
does not reopen it; `ALL_EVACUATED` here is purely a **descriptive** classification of how a normal,
full-length run turned out, not a claim that the Runtime acted on it.

## 13. Fatalities and Rescue Required — the deliberately unresolved policy seam

**No fatality/casualty/tenability model exists anywhere in this codebase (§2), and this document does
not invent one.** Doing so would mean fixing, for the first time, an actual life-safety claim
(`HazardSeverity.from_score()`'s own cutoffs are explicitly documented as *not* validated for this
purpose) — a domain-expertise decision far outside an orchestration/architecture review's mandate, and
exactly the kind of unvalidated-model risk `HeatDetectorReading`'s deliberate omission of rate-of-rise
and `AIDecisionEngine`'s own "not a validated life-safety model" caveat on behaviour registry parameters
already show this codebase treats carefully rather than guesses at.

**Design**: an optional, injected `CasualtyPolicy` interface, structurally identical to
`PerceptionProvider`/`SimulationRuntime`'s `dataset_logger` — abstract only, no default implementation:

```
class CasualtyPolicy:
    def is_fatal(self, occupant_id, zone_id, hazard_snapshot, time) -> bool:
        raise NotImplementedError
```

`record_outcome(..., casualty_policy: Optional[CasualtyPolicy] = None)`:

- **No policy supplied** (the default): `fatalities = None` — never `0`. Reporting `0` would silently
  assert "confirmed zero deaths," precisely the "never fabricate absence as a safe/clear default"
  violation this codebase's own conventions exist to prevent everywhere else (`OccupancyObservation.
  occupant_count`'s `None`-means-no-reading, `BuildingObservation`'s `UNOBSERVED`, both already reused
  as direct precedent). `rescue_required = trapped_occupants` in this case — the conservative reading:
  absent better information, every trapped occupant is assumed to still need rescue.
- **Policy supplied**: for each trapped occupant, evaluated against the tick(s) during which they were
  present in a hazardous zone (cross-referencing `MovementTimelineOccupancyProvider`-style per-occupant
  zone resolution, `simulation_runtime/occupancy_bridge.py`, against each `TickResult.hazard_snapshot`) —
  `fatalities = count where is_fatal() is True`; `rescue_required = trapped_occupants - fatalities`.

This document takes **no position** on what a real `CasualtyPolicy` implementation should contain — that
is future domain-specific work, explicitly out of scope, flagged into §19.

## 14. Outcome computation — entry point and purity

```
def record_outcome(
    context: SimulationContext,
    movement_result: MultiAgentSimulationResult,
    tick_results: Tuple[TickResult, ...],
    evaluation_time: Optional[float] = None,       # defaults to the last tick_result's time, or 0.0
    casualty_policy: Optional[CasualtyPolicy] = None,
    end_reason_override: Optional[SimulationEndReason] = None,
) -> ScenarioOutcomeLabels:
    ...
```

A pure function of its inputs — no randomness, no file I/O, no simulation execution of its own (mirrors
`scenario_event_executor`/`simulation_runtime`'s own "reuse, never re-simulate" discipline). Callable
equally after a full `runtime.run()` or after a partial, manually-accumulated list of `tick()` results
following a caught exception (§12).

## 15. Where Outcome Labels are computed relative to Storage — no modification, confirmed

`record_outcome()` never calls into `scenario_storage` and never mutates a `Scenario`/`ScenarioMetadata`
— it only reads `context`/`movement_result`/`tick_results`, all already produced upstream by packages
this document does not touch. Persisting a `ScenarioOutcomeLabels` is a separate, additive concern (§16).

## 16. Integration with Storage — a new, parallel package, not an extension of the existing catalog

**`scenario_storage/catalog.py`'s `CATALOG_COLUMNS` is explicitly documented as appendable** ("new
columns may be appended to the end of this tuple in a future phase") — but appending Outcome Label
columns there would be **wrong**, not merely suboptimal: that catalog is one row per `scenario_id`
(§11 of `scenario_engine.md`: "one metadata-only CSV catalog"), and §10.1 already established Outcome
Labels are 1:many with `scenario_id`. Cramming a 1:many relationship into a 1:1 catalog's row shape
would mean either overwriting a prior run's labels (silent data loss) or violating the catalog's own
one-row-per-scenario invariant.

**Proposed instead**: a new, sibling top-level package — `scenario_outcome_storage/` — mirroring
`scenario_storage/`'s own two-artifact shape (one JSON file per record + one append-only CSV catalog),
keyed by `outcome_id`, with `scenario_id` as an ordinary (non-unique) column:

- One JSON file per `ScenarioOutcomeLabels`, sharded the same way `scenario_storage/paths.py` already
  shards Scenario JSON files (reusing that convention, not re-deriving a new one).
- One append-only `outcomes.csv` catalog, columns denormalized from `ScenarioOutcomeLabels` exactly as
  `scenario_storage/catalog.py::catalog_row_for()` denormalizes from `Scenario` — `outcome_id`,
  `scenario_id`, `recorded_at`, `evaluation_time`, `total_occupants`, `evacuated_occupants`,
  `trapped_occupants`, `fatalities` (empty string when `None`, matching `difficulty`'s own existing
  empty-when-absent convention in the current catalog), `rescue_required`, `building_cleared`,
  `simulation_end_reason` — nested fields (`worst_hazard_zone`, `max_node_congestion`,
  `max_edge_congestion`) live in the JSON file only, the same "CSV is a denormalized summary, JSON is
  the full record, never a second source of truth" split §11 of `scenario_engine.md` already commits to.

This package would import `scenario_outcome` (for `ScenarioOutcomeLabels`) and `serialization` (`
JsonWriter`/`JsonReader`, reused exactly as `scenario_storage` already does) — never `scenario_storage`
itself (no data dependency exists between the two catalogs; a caller wanting "every outcome for scenario
X" joins the two catalogs on `scenario_id` externally, the same way any two independent CSVs sharing a
foreign key would be joined, not through a code-level coupling between the packages).

## 17. Integration with the future Dataset Generation Pipeline

Every seam this document and its predecessors already designed chains, unmodified, into one straight
line — this section states the chain explicitly because the user brief asks this review to "integrate
with... the future Dataset Generation pipeline," not because any new mechanism is required beyond what
§4-§16 already specify:

```
DatasetIntent + Building + parameters + count
        │
        ▼
  resolve_intent()                                          §5-§8  (dataset_intent/, NEW)
        │
        ▼
  scenario_pipeline.run_batch_pipeline()                     UNCHANGED (scenario_pipeline/, frozen)
        │
        ▼
  scenario_storage.save_scenario()  (per accepted Scenario)  UNCHANGED (scenario_storage/, frozen)
        │
        ▼
  For each stored Scenario:
        scenario_runner.run(scenario, building)               UNCHANGED (scenario_runner/, frozen)
        behaviour_profile_resolver.register_occupants(context) UNCHANGED (behaviour_profile_resolver/,
                                                                            frozen)
        SimulationRuntime(context, decision_engine, dt, ...).run()  UNCHANGED (simulation_runtime/,
                                                                                 frozen, approved)
        │
        ▼
  record_outcome(context, movement_result, tick_results, ...)  §14  (scenario_outcome/, NEW)
        │
        ▼
  scenario_outcome_storage.save_outcome()                     §16  (scenario_outcome_storage/, NEW)
```

A not-yet-built Dataset Generation Pipeline orchestrates exactly this loop, plus whatever **curation**
step a specific intent wants over the resulting `(Scenario, ScenarioOutcomeLabels)` pairs — e.g.
`Rescue Training`'s pipeline run might keep only scenarios where `rescue_required > 0` (§9's own
compliance finding: this is curation over already-accepted, already-stored Scenarios, never a new
Validator rejection category). `RL Training`'s per-run step is additionally the one place a
`PerceptionProvider` and Dataset Logger (`docs/architecture/simulation_runtime.md` §13/§18, still
unbuilt) get attached to a given run — a Run Configuration decision made here, per §7's own note, not
inside `resolve_intent()`.

## 18. Dependency direction

- **`dataset_intent/`** may import: `scenario_definition` (its output shape), `models` (Building/Zone/
  Door/... enumeration, §4 point 2) — **the same, and only the same, packages `scenario_definition/`
  itself is permitted to import** (§4 point 3's own reasoning: Intent's output must uphold the identical
  purity guarantees its output type already commits to). Must **not** import `scenario_generator`,
  `scenario_validator`, `scenario_pipeline`, `scenario_storage`, `scenario_runner`, `simulation_runtime`,
  `scenario_outcome`, `navigation`, `pathfinding`, `simulator`, `behavior`, `behavior_library`,
  `hazard_evolution`, `ai_decision`, `perception`, `sandbox`, `designer`, or `random`.
- **`scenario_outcome/`** may import: `scenario` (`ScenarioMetadata` for `scenario_id` typing),
  `scenario_runner` (`SimulationContext`, one-way downstream dependency, mirroring
  `behaviour_profile_resolver`'s own precedent), `simulation_runtime` (`TickResult`), `simulator`
  (`MultiAgentSimulationResult`, `OccupantState`), `hazard` (`HazardSeverity`, for `WorstHazardZone`),
  `occupancy`. Must **not** import `scenario_generator`, `scenario_validator`, `scenario_pipeline`,
  `scenario_storage`, `dataset_intent`, `behavior`, `behavior_library`, `sandbox`, `designer`, or
  `random` — it never generates, validates, or orchestrates a retry loop, and computes nothing that
  requires randomness (§14).
- **`scenario_outcome_storage/`** may import: `scenario_outcome` (`ScenarioOutcomeLabels`),
  `serialization` (`JsonWriter`/`JsonReader`, reused exactly as `scenario_storage` already does). Must
  **not** import `scenario_storage` (§16's own "no code-level coupling between the two catalogs"
  finding), `scenario_generator`, `scenario_validator`, `scenario_pipeline`, `simulation_runtime`,
  `scenario_runner`, or `random`.

## 19. Open questions for a future review

- **Exact `WeightedOptions`/`UniformRange` parameter values per intent** (§7) — policy direction fixed,
  numeric tuning explicitly not decided here, mirroring Difficulty's own "placement decided, formula not"
  precedent.
- **A real `CasualtyPolicy` implementation** (§13) — deliberately out of scope; needs dedicated
  domain-expertise review, not an architecture pass.
- **Run reproducibility** (§10.1) — whether/how a `SimulationRuntime` run's full configuration
  (`decision_engine` tuning, `dt`, `casualty_policy`) should itself become a serializable, hashable
  value object (a "Run Configuration" type) so `outcome_id` could become content-derived rather than
  `uuid4()`-random. Flagged, not designed.
- **Whether `scenario_outcome_storage`'s CSV catalog needs a `content_hash`-style column** for cheap
  cross-run dedup, mirroring `scenario_storage.load_accepted_hashes()`'s own documented "a future phase
  could add a content_hash column" note — plausible, not decided here.
- **The Dataset Generation Pipeline's own curation/filtering DSL** (§17) — "keep runs where
  `rescue_required > 0`" is stated as an example, not a designed mechanism; a real implementation phase
  would need to decide how curation predicates are expressed and applied across a batch.
- **Whether Intent parameters should be validated/schema-checked per profile** (§4 point 4) — this
  document treats an unrecognized parameter key as silently ignored, matching `ScenarioMetadata.extra`'s
  precedent; a stricter, per-profile schema is a plausible future refinement, not designed here.

## 20. Status

Proposal only. Nothing in this document has been implemented; `scenario_definition`, `scenario_generator`,
`scenario_validator`, `scenario_pipeline`, `scenario_storage`, `scenario_runner`,
`behaviour_profile_resolver`, and `simulation_runtime` are all unmodified by it. Three new, additive
packages are proposed (`dataset_intent/`, `scenario_outcome/`, `scenario_outcome_storage/`), each
depending only downstream on already-frozen packages, none requiring a change to any of them. The design
resolves "how does a high-level intent become a `ScenarioDefinition`" (§4-§9) by strict translation, never
sampling, and resolves "how are outcome labels recorded" (§10-§17) by pure post-hoc computation over
artifacts the already-approved Simulation Runtime already produces — while explicitly refusing to
invent, in this pass, the one thing this codebase has never validated: a life-safety/fatality model
(§13). That refusal is a deliberate, flagged boundary, not an oversight.

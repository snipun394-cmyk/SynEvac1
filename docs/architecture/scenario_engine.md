# Scenario Engine — Architecture Proposal

Status: **proposal, largely approved, refinements pending implementation**. No code changes
accompany this document.

## 1. The pipeline

```
Scenario Definition
(distributions over every constrained field --
 declares what may be sampled, samples nothing)
        │
        ▼
    Generator
(samples ONE candidate Scenario by drawing from
 the Definition's distributions, field by field,
 via per-category, name-keyed rng streams fanned
 out from one scenario seed (§4.6) -- no repair,
 no constraint-awareness, no accept/reject logic)
        │
        ▼
    Validator
(checks the candidate against the Definition and
 Building/Nav Graph integrity -- sole gate)
        │
        ├── Accepted ──────────────────────────► Scenario (stored)
        │
        └── Rejected ──► Generate Again
                          (discard the candidate outright; Generator
                           draws a fresh sample from the next child
                           seed -- never patches or repairs the
                           rejected one)
```

**Two refinements in this revision**, both changing the Generator/Validator relationship and the
Definition's own vocabulary:

1. **No repair, anywhere.** The previous revision's "deterministically reopen exits until
   `min_open_exits` is met" step is removed. The Generator sample either already satisfies a joint
   constraint or it doesn't — if it doesn't, the Validator rejects it and the pipeline **generates
   again from scratch**, it does not adjust the rejected candidate into compliance. This is
   rejection sampling: a standard, well-understood way to sample from a constrained space (the
   product of the Definition's independent per-field distributions, intersected with the
   Definition's joint constraints) without the Generator ever having to reason about those joint
   constraints itself. See §4/§5 for the mechanics and §13 for the efficiency caveat this implies.

2. **Distributions, not ranges.** Every constrained field in `ScenarioDefinition` is now declared
   as a `Distribution`, not a bare range/probability/pin-list. A fixed value is simply the
   degenerate case of a distribution (`FixedValue`), which means fields that were previously
   *pairs* in the schema (`fixed_occupancy` + `occupancy_ranges`; `always_open_door_ids` +
   `always_closed_door_ids` + `open_probability`) collapse into **one** distribution-valued field
   each. See §3.

**This pass's refinement**: the diagram's `Generator` box above was previously a single opaque
step. It is now fully specified — §4 fixes its input/output contract, internal per-category
pipeline, randomness/seed ownership, and dependency surface, and is itself now frozen. Nothing
about this diagram's shape changes; §4 fills in what was previously left undesigned inside that one
box.

**Next pass's refinement**: the diagram's `Validator` box is now likewise fully specified — §5 fixes
its modular validation categories, its structured report shape, and the failure-category tagging
that lets orchestration compute retry diagnostics without the Validator itself coupling to the
Generator, and is itself now frozen. Nothing about this diagram's shape changes; §5 fills in what was
previously left undesigned inside that box.

## 2. Grounding in existing code

Unchanged from the previous revision: `FireGrowthModel.ignition_node_id` already equals `Zone.id`
for zone nodes; `Door.normally_open/locked/active`, `Exit.is_blocked`, `Obstacle.active` are the
existing state fields a Scenario's overrides set; `SandboxManager`'s unseeded `random`-module
placement should not be reused as-is; `Serializer` is `Project`-shaped and should not be
overloaded for `Scenario` storage; `navigation/validation.py::ValidationReport` is the report
shape the Validator reuses; `MultiAgentSimulation`'s own event heap is occupant-movement-only and
unrelated to `ScenarioEvent` (§6 of the prior revision, unchanged here).

One point now more load-bearing than before: `behavior_library/`'s `self.rng = rng or
random.Random()` injection convention (`decision_strategies.py:64`,
`pre_movement_strategies.py:33`, `route_choice_strategies.py:92`) is exactly the mechanism the
future Generator will use to drive sampling — **one** injected `random.Random` instance, threaded
through the whole sampling pass, not a `random.Random()` created per field. This revision's review
(§3) confirms that mechanism must live entirely on the Generator side — see §3's compliance
findings for why `Distribution` itself is corrected to carry no such method.

Grounding for §8 (Behaviour Profiles, this revision): `behavior/profile.py::BehaviorProfile` (note
the codebase's American spelling vs. this document's "Behaviour" — same concept, see §8) is a
per-occupant *trait bundle* (`occupant_id`, `walking_speed`, `familiarity`, `compliance_level`,
`role`, `group_id`, `traits: Dict[str, Any]`) that strategies read — not a named/registered preset.
`HumanBehaviorLayer.register(start_id, profile, decision_strategy, ...)`
(`behavior/orchestrator.py`) requires the *caller* to construct that `BehaviorProfile` plus explicit
strategy instances per occupant, per call — there is currently **no registry, factory, or named
preset anywhere in `behavior/` or `behavior_library/`** mapping a name like `"Adult_Default"` to a
concrete configuration; every existing usage (test code only, nothing production) constructs
strategies inline. `designer/widgets/occupant_generation_dialog.py:49-58` already has a disabled
`behavior_profile_combo` UI placeholder (single item `"Default"`, labeled "Behavior Profile
(future)", not wired to anything) — the Designer already anticipates this concept without it being
built. This confirms the profile-name-to-behavior-config registry §8 requires is a genuine,
currently-unbuilt gap that belongs entirely to the Behaviour Layer side, not the Scenario Engine.

## 3. Scenario Definition Engine — frozen

This section is the outcome of a dedicated compliance review against one governing principle: **a
Scenario Definition is a rulebook — it never creates scenarios, never runs simulations, never
contains randomness, never contains generated occupants/fire/hazards, and never contains
simulation, RL, AI, or Navigation state.** §3.1 records what the review found. §3.2 is the frozen
schema. Generation and simulation are explicitly out of scope for this section (and were not
reconsidered while writing it) — only what the Definition itself is and holds.

Package: `scenario_definition/` — self-validation only. Must not import `scenario_generator`,
`scenario_validator`, `scenario` (the resolved-output model package), `simulator`, `sandbox`,
`behavior`, `designer`, `navigation`, or `random`/`numpy`. **May** import `models` (Building/Zone/
Door/Exit/Obstacle/Camera/Detector/Staircase are engineering geometry, not simulation or navigation
state, and self-validation needs them to check that referenced ids actually exist).

### 3.1 Compliance findings from this review

Three real violations of the governing principle were found in the previous revision and are
corrected below; one gap against the requested category list was found and closed; one naming
mismatch against the user's own vocabulary was corrected.

1. **Violation — randomness inside the Definition.** The previous revision gave `Distribution` a
   `sample(self, rng: random.Random) -> T` method and explicitly allowed `scenario_definition/` to
   import `random`. That is exactly "a Scenario Definition contains randomness," regardless of the
   fact that only the Generator was ever *going to* call it. **Fix**: `Distribution` and its
   variants (`FixedValue`, `UniformRange`, `WeightedOptions`, and the reserved future kinds) become
   plain, behavior-free dataclasses in `scenario_definition/` — data describing an allowed set of
   values and, optionally, their relative likelihood, nothing more. No `sample()` method exists
   anywhere in this package; no `random` import exists anywhere in this package. The interpreter
   that knows how to turn a `Distribution` descriptor plus an `rng` into a concrete value is a
   Generator-side concern, out of scope for this document (per the constraint on this discussion),
   and is not designed here. **The distinction that resolves last revision's instruction ("describe
   probability distributions") against this revision's instruction ("never contains randomness")**:
   describing the *shape* of allowed values — including relative likelihood — is a rule, exactly
   like "Door D2: Random" or "Zone A: Minimum 20, Maximum 100" in the worked example. Randomness is
   the *act of drawing* a concrete value from that rule using a random-number generator. The
   Definition states rules; it does not, and structurally cannot, draw from them.

2. **Violation — a resolved-output type leaking into the Definition.** The previous revision's
   `fixed_events: List[ScenarioEvent]` field required importing `ScenarioEvent` from the `scenario`
   model package — a resolved-output type — directly contradicting the dependency rule this same
   document already stated (`scenario_definition/` must not import `scenario`). **Fix**: deleted.
   A "pinned" event is not a different kind of thing from a templated one — it is the degenerate
   case of the same `EventTemplate` mechanism (§3.2's Events row), exactly like `FixedValue` is the
   degenerate case of every other distribution-valued field. One mechanism, no shadow list sitting
   beside it — the same collapse already applied to occupancy and door state now applies uniformly
   to events too.

3. **Coupling — unnecessary reuse of a Navigation-package type.** The previous revision had
   self-validation "reuse `navigation/validation.py`'s `ValidationReport` shape." Reusing the
   *shape* was harmless in spirit, but importing it means `scenario_definition/` importing
   `navigation` — a package this document elsewhere insists Definitions must never contain state
   from. **Fix**: `scenario_definition/` defines its own minimal, structurally-identical
   `DefinitionValidationReport` (errors/warnings list, no behavior beyond accumulation), so the
   package has zero import surface into `navigation` at all. Any resemblance to
   `ValidationReport`'s shape is convention-following, not a dependency.

4. **Gap — Fire allow/forbid/floor rules were implicit.** The previous revision folded "allowed
   ignition zones," "forbidden zones," and "excluded floors" into the *implicit domain* of
   `ignition_zone_distribution`, rather than three explicit, separately-authored fields. The
   worked example in this prompt treats "Fire may start only on Floors 1–3" and "Fire may never
   start inside the Laboratory Zone" as two distinct, independently statable rules — collapsing
   them into an unstated "domain" hides exactly the kind of rule a user needs to see and edit
   directly. **Fix**: restored as explicit fields (§3.2's Fire row) — `allowed_ignition_zone_ids`,
   `forbidden_ignition_zone_ids`, `allowed_ignition_floor_ids` are first-class, human-authorable
   rules; an optional likelihood weighting *among* the allowed set remains available as a
   `Distribution` layered on top, but is no longer the only way to express the allow/forbid rule
   itself.

5. **Naming mismatch.** `EventRule` is renamed `EventTemplate` throughout, matching this prompt's
   own repeated, deliberate terminology ("these are NOT events. These are templates describing what
   events the Generator is allowed to create.").

Confirmed **already compliant**, no change needed: `min_open_exits` as a plain `int` (a joint rule
over the *result* of independent per-exit rules, not itself a value to draw — already correctly
not a `Distribution`); the occupancy field collapse (a forbidden/always-empty zone, e.g. "Zone C:
Always empty" in the worked example, is simply `FixedValue(0)` in `occupancy_distribution` — no
separate "forbidden occupancy zones" field needed, the mechanism already covers it); door/exit/
stair/obstacle/camera/detector state fields (already declarative rules, not generated state);
Occupant/Fire Profile fields (already optional, data-only distributions — §8/§9's "architecture
only" framing already matched this principle, only the `Distribution`-purity fix above applies to
them too).

### 3.2 Frozen schema

A `Distribution[T]` — **pure data, no methods, no `random` import** — describes an allowed set of
values for `T` and, optionally, the relative likelihood within that set:

- **`FixedValue(value)`** — exactly one allowed value.
- **`UniformRange(low, high, discrete=False)`** — every value in `[low, high]` allowed, no stated
  preference among them.
- **`WeightedOptions(weights: Dict[T, float])`** — a finite allowed set, each member with a
  relative weight (a boolean "open probability" is the two-key case:
  `WeightedOptions({OPEN: p, CLOSED: 1 - p})`).

Named and reserved, **not implemented**: `GaussianParameters(mean, stddev)`,
`TimeDependentSpec(...)` (parameters that vary along a declared time axis), `EmpiricalDatasetRef
(...)` (sample from a real recorded dataset instead of a parametric family). All three are future
*descriptions of an allowed-value rule*, same as the three committed kinds — none of them, now or
later, gain a method that draws a value.

| Category | Fields |
|---|---|
| Fire | `allowed_ignition_zone_ids: FrozenSet[zone_id]` (empty = no additional restriction beyond the other two fields), `forbidden_ignition_zone_ids: FrozenSet[zone_id]`, `allowed_ignition_floor_ids: FrozenSet[floor_id]` (empty = any floor), `ignition_zone_preference: Optional[Distribution[zone_id]]` (optional likelihood weighting *within* the allowed set — absence means no stated preference), `growth_parameter_distribution: Distribution[float]` ("fire parameter ranges"), `allowed_fire_profiles: FrozenSet[FireProfile]` (§9; "Electrical OR Flaming" in the worked example) |
| Exits | `exit_state_distribution: Dict[exit_id, Distribution[bool]]` (default rule applies to any exit id absent from the dict), `min_open_exits: int` — plain int, a joint rule over the *result* of every exit's independent rule, not itself a `Distribution` |
| Doors | `door_state_distribution: Dict[door_id, Distribution[DoorState]]`, `DoorState = OPEN \| CLOSED \| LOCKED` (maps onto `Door.normally_open`/`Door.locked` together) |
| Stairs | `stair_state_distribution: Dict[stair_id, Distribution[AvailabilityState]]`, `AvailabilityState = AVAILABLE \| CLOSED` |
| Obstacles | `obstacle_state_distribution: Dict[obstacle_id, Distribution[PresenceState]]`, `PresenceState = ACTIVE \| INACTIVE` |
| Cameras | `camera_state_distribution: Dict[camera_id, Distribution[AvailabilityState]]`, `AvailabilityState = AVAILABLE \| FAILED` |
| Detectors | `detector_state_distribution: Dict[detector_id, Distribution[AvailabilityState]]` |
| Occupants | `occupancy_distribution: Dict[zone_id, Distribution[int]]` (fixed/min-max/forbidden-i.e.-always-empty are all just different `Distribution` values in the same field — see §3.1 finding), `behaviour_profile_distribution: Dict[zone_id, Distribution[str]]` (§8 — an **opaque** identifier string, e.g. `"Adult_Default"`; this package never interprets it) |
| Events | `event_templates: List[EventTemplate]`, `EventTemplate(target_type, target_id, event_type, occurs: Distribution[bool], time: Distribution[float], parameters: dict)` — a pinned/always-happens event is `occurs=FixedValue(True)`, an exact time is `time=FixedValue(t)`; a description like "Camera C2 may fail any time after 90 seconds" is `occurs=WeightedOptions({True: p, False: 1-p})`, `time=UniformRange(90, scenario_horizon)` |
| Environmental | reserved extension point, not implemented — see §3.3 |
| Reproducibility | `seed: Optional[int]` — stored so a later Generator run is reproducible; **not used by anything inside this package** |

Difficulty remains absent from this table (§10, unchanged decision).

### 3.3 Environmental — reserved extension point

Not implemented, same treatment as Occupant/Fire Profiles (§8/§9): the category exists in the
schema as a named placeholder so future fields extend rather than restructure it. Named future
examples, per this prompt: HVAC state, ventilation, weather, utility failures. No field shapes are
committed — whether these end up as `Distribution`-valued per-system fields (mirroring
camera/detector availability) or something else is undecided.

### 3.4 Self-validation

`ScenarioDefinition.validate() -> DefinitionValidationReport` (§3.1 finding 3 — a locally-defined
report type, not `navigation`'s) checks structural well-formedness only, and **never samples
anything**:

- `UniformRange.low <= high`; `WeightedOptions` weights are non-negative and sum to a positive
  total.
- Every id referenced anywhere (`Dict[id, Distribution]` keys, `allowed_ignition_zone_ids`,
  `EventTemplate.target_id`, ...) actually exists on the referenced Building.
- `min_open_exits` does not exceed the number of exits whose `exit_state_distribution` entry could
  possibly resolve `OPEN` at all (an exit pinned `FixedValue(CLOSED)` is a hard commitment and
  never counts toward feasibility, even optimistically).
- `allowed_ignition_zone_ids` and `forbidden_ignition_zone_ids` do not share a zone id;
  `allowed_ignition_floor_ids`, if non-empty, actually contains at least one non-forbidden zone.
- `behaviour_profile_distribution` and any other `WeightedOptions`-shaped proportion field is
  internally consistent (weights sum positive — exact normalization to 1.0 is a Generator-side
  concern, not required here). **Deliberately not checked**: whether a given profile-id string is
  actually a registered Behaviour Profile — doing so would require importing `behavior`/
  `behavior_library`, which this package must never do (§8, §12). An unrecognized id is caught at
  simulation time by the Behaviour Layer, not here.

This is the entirety of what "validation" means for a Definition: checking that the rulebook is
internally consistent and refers to things that exist. It says nothing about whether any scenario
satisfying it is reachable, winnable, or "makes sense" as a fire scenario — that is what a
Scenario *Validator* checks, against a concrete sampled candidate, and remains completely out of
scope for this document.

## 4. Scenario Generator — frozen

This section is the outcome of a dedicated architecture review against one governing principle,
stated directly by this pass's brief: **the Scenario Generator is a constrained sampling engine —
not a randomizer, not a simulator, not an AI, and not a validator.** Its one responsibility is to
consume an already-frozen `ScenarioDefinition` (§3) plus a seed and produce one candidate `Scenario`
satisfying every constraint the Definition declares. §4.1–§4.13 record this pass's findings and the
frozen design that results. As with §3, generation/simulation *internals* (fire physics, smoke
physics, navigation pathfinding, behavioural decision-making, RL) are explicitly out of scope and
were not reconsidered while writing this section — only what the Generator itself is, receives, and
produces.

Package: `scenario_generator/`, split into a **construction module** (the sampler itself) and an
**orchestration module** (`pipeline.py`) — a split already established by the prior revision,
unchanged in kind here; the dependency surface for each is fixed in §4.2/§12.

### 4.1 Purpose — what the Generator is not

Four things the Generator is explicitly not, each with the concrete failure mode that rules it out:

- **Not a randomizer.** It never draws a value outside what a Definition field's `Distribution`
  allows — every draw is a draw *from* a Definition-declared rule, never free-floating randomness
  (§4.5/§4.6 fix *where* that randomness lives and how it's seeded).
- **Not a simulator.** It never runs `HazardEvolutionEngine`, never steps a `FireGrowthModel`, never
  runs `PathfindingEngine`, never executes a `MultiAgentSimulation` tick. It samples the *initial*
  state a simulation would later start from, and stops there.
- **Not an AI.** No learned policy, no adaptive behavior, no `ai_decision/` dependency (already
  excluded, §12) — every output is a direct, mechanical function of `(Definition, seed)`.
- **Not a validator.** It has no accept/reject branch anywhere in its own code (§1, unchanged) —
  acceptance is entirely `scenario_validator/`'s concern (§5).

### 4.2 Input contract

The Generator receives exactly four things, nothing else:

1. **`ScenarioDefinition`** (§3, frozen) — the rulebook.
2. **Building / engineering objects** — not a separate channel: every id a Definition references
   (`zone_id`, `door_id`, `exit_id`, ...) already resolves against a `models`-package `Building`, the
   same way §3.4's self-validation resolves them. The Generator reads this `Building` only to resolve
   ids to concrete geometry/state fields it must assign (e.g., a `Zone`'s polygon, to place a sampled
   occupant count inside it via `Zone.contains()`) — never to reason about connectivity, since
   reachability stays exclusively a Validator concern (§5, unchanged).
3. **A seed** — resolved per §4.6, either caller-supplied (single-scenario request) or derived from a
   batch's master seed and this scenario's index (§4.8).
4. **A `GenerationRequest`** — the minimal envelope carrying the above three plus a mode flag
   (single vs. batch-member); its shape is otherwise an implementation detail, not designed here.

**Compliance finding, this pass.** The prior revision's construction-module dependency list (§12)
included `navigation` and `fire_growth`, neither of which the Generator has any legitimate use for
once §4.1's non-goals are taken literally:

- `navigation` would only be needed for reachability/connectivity reasoning — already, explicitly,
  exclusively the Validator's job (§4/§5, unchanged since the previous revision: "no reachability
  awareness" for the Generator). Occupant placement needs only `Zone.contains(x, y)`
  (`models/zone.py:129`), pure geometry, no graph.
- `fire_growth` would only be needed to *construct* a `fire_growth.model.FireGrowthModel` instance —
  but that class directly implements `HazardSource` and plugs straight into
  `HazardEvolutionEngine` (simulation machinery, `fire_growth/model.py`). Constructing one inside the
  Generator would be preparing live simulation state — a direct violation of §4.1's "not a
  simulator." The Generator instead samples and stores **plain data** — `ignition_zone_id: str`,
  `growth_parameters` (sampled from `growth_parameter_distribution`), `fire_profile: FireProfile`
  (§4.7 fixes how the last one is chosen when more than one is allowed) — and it is the Simulator's
  job, at simulation start, to turn that data into a `FireGrowthModel`. This is the same
  "resolved data crosses the boundary, never a live object" pattern §8 already established for
  Behaviour Profiles.

**Fix**: the construction module's dependency list (§12) is corrected to `scenario_definition`,
`scenario`, `models` only — strictly narrower than the prior revision, consistent with §4.1's
non-goals actually being enforced rather than merely stated.

### 4.3 Output contract & reproducibility

The Generator produces a complete candidate `Scenario` (§7) such that regenerating from the same
recorded inputs reproduces it exactly — "nothing should depend on hidden state," per this pass's
brief, taken literally.

**Compliance finding — a real hidden-state gap.** §7 (prior revision) stores `definition_id` but not
the Definition's *content*. A `ScenarioDefinition` is an editable, Designer-authored object (§3) —
nothing stops a user from re-saving a Definition under the same id with different distributions. If
that happens, `(definition_id, seed)` no longer determines a unique Scenario: regenerating from the
same recorded `(definition_id, seed)` after the Definition was edited silently produces a *different*
result — exactly the hidden-state dependency this pass's brief rules out. **Fix**: `Scenario`
additionally stores `definition_content_hash: str`, a content hash of the exact `ScenarioDefinition`
used, computed the same way the Validator's uniqueness check already hashes candidates (§5,
unchanged mechanism, new target). Reproducing a `Scenario` now means: same `ScenarioDefinition`
content (verified against the stored hash, not just the id), same `seed`, same Generator version
(below).

**Second finding — generator-version drift.** §4.6 fixes the *property* the seed-derivation
function must satisfy, not the function itself (mirroring §4's existing restraint about the sampling
interpreter). If that derivation function ever changes between Generator releases, the same
`(definition_content_hash, seed)` pair can legitimately produce a different candidate under the new
version than the old one — not a bug, but a fact that must be recorded rather than silently assumed
away. **Fix**: `Scenario` stores `generation_version: str`, identifying the Generator release
(sampling interpreter + seed-derivation function) that produced it. Reproducibility is a claim about
`(definition_content_hash, seed, generation_version)` jointly, never `(definition_id, seed)` alone.

### 4.4 Internal generation pipeline

Reviewed against the prior revision's single-box "Generator" step in §1 — that box is now specified
as an ordered sequence of per-category sampling stages, each a thin wrapper running one (or a few
related) Definition field(s) through the sampling interpreter (§4, unchanged: "for every field the
Definition declares, it runs that field's `Distribution` descriptor through its own sampling
interpreter ... and assigns the result — nothing more"):

```
Initialize Random Generator        (§4.6: resolve this attempt's seed, fan out into
                                     per-category child streams)
        ↓
Generate Fire                      (ignition zone/floor/preference, growth params,
                                     fire profile — §4.7's default-sampling fix)
        ↓
Generate Occupants                 (per-zone count + in-polygon position, from
                                     occupancy_distribution)
        ↓
Assign Behaviour Profile IDs       (per-occupant behaviour_profile_id string, from
                                     behaviour_profile_distribution — §8, unchanged:
                                     opaque, never interpreted)
        ↓
Generate Door States
        ↓
Generate Exit States
        ↓
Generate Stair States
        ↓
Generate Obstacle States
        ↓
Generate Camera States
        ↓
Generate Detector States
        ↓
Generate Event Schedule            (EventTemplate.occurs/time — §6, unchanged)
        ↓
[reserved: Generate Environmental State — §3.3, not implemented]
        ↓
Assign Scenario Metadata           (§4.10 — derived summaries + provenance fields)
        ↓
Send to Validator                  (§5, unchanged: sole gate)
        ↓
Accepted → Serialize (§11)  /  Rejected → Generate Again (§4.7)
```

**This order carries no correctness weight.** Because every category draws from its own
independently name-keyed child stream (§4.6), reordering these stages — or inserting the reserved
Environmental stage once it exists — changes nothing about any *other* stage's sampled values. The
order shown is a fixed, documented convention for implementers to follow consistently, not a
dependency chain. This is the direct payoff of §4.6's keyed (not sequential) derivation: it is what
makes "changing door generation should not change occupant placement" true *by construction*, not
just by accidentally-preserved call order.

**Metadata timing.** "Assign Scenario Metadata" runs after every category because most metadata
fields are summaries *of* already-sampled fields (occupancy summary, open-exit list, ...) and cannot
exist earlier. One distinction made explicit here and expanded in §4.10: metadata splits into
**reproducibility-relevant** fields (`seed`, `definition_content_hash`, `generation_version` —
required to regenerate the candidate) and **provenance-only** fields (`created_at` timestamp,
human-readable summaries — describe the candidate but play no role in regenerating it). A wall-clock
timestamp is inherently non-deterministic and must never be treated as part of the reproducibility
contract.

### 4.5 Randomness ownership

Reviewed against this pass's brief: "every random decision must originate from the Generator."
Within the scope of *Scenario Generation* (this document's domain), this already held and is
reinforced, not changed, by this pass:

- The Definition never draws (§3.1 finding 1, frozen — no `random` import anywhere in
  `scenario_definition/`).
- The Validator never draws — it only checks an already-fully-sampled candidate (§5, unchanged); its
  dependency list (§12) is clarified this pass to state explicitly that `scenario_validator/` must
  not import `random` either, closing the same kind of gap-by-omission §3.1 finding 1 closed for the
  Definition.
- The Generator is the sole owner of every random draw that determines Scenario *content* — fire
  origin, occupant count/position, behaviour-profile-id assignment, every door/exit/stair/obstacle/
  camera/detector state, every event's `occurs`/`time`.

**One clarification this pass adds, to avoid an apparent conflict with already-grounded code**
(§2): `behavior_library/`'s `self.rng = rng or random.Random()` convention is real, existing, and
*not* touched by "the Behaviour Layer never does [own randomness]." That convention governs
**simulation-time behavioural randomness** — panic, route-choice jitter, decision latency — which
happens during a later, separate phase (actual simulation execution, driven by
`HumanBehaviorLayer`) after a `Scenario` already exists and has been handed off. It has no bearing
on what *is in* a Scenario, only on how an already-fixed occupant with an already-assigned
`behaviour_profile_id` (§8, unchanged) *behaves* once the simulation starts. "The Generator owns
randomness" is a claim about **Scenario Generation's randomness domain** specifically — it does not
conflict with the Behaviour Layer's own, independent randomness domain during simulation, which
remains entirely outside this document's scope (per this pass's own "do not think about the
simulator internals" instruction). Likewise, any runtime randomness the Simulator itself owns (e.g.
stochastic detector noise) is a *simulation*-reproducibility concern, orthogonal to and untouched by
*scenario*-reproducibility (§4.3), and out of scope here.

### 4.6 Seed architecture

Reviewed against the worked example (Master Seed → Scenario Seed → independent child generators) and
the stated goal ("changing door generation should not change occupant placement"). The example is
architecturally sound as a *goal*; this pass fixes the one property that actually delivers it, since
the worked example's own phrasing ("independent child generators," no further detail) is ambiguous
about *how* children are derived — and that ambiguity is exactly where a naive implementation would
break the goal.

**The property, stated precisely**: each category's child RNG stream must be a deterministic
function of `(attempt_seed, category_key)`, where `category_key` is a fixed string (`"fire"`,
`"occupant"`, `"behaviour_profile"`, `"door"`, `"exit"`, `"stair"`, `"obstacle"`, `"camera"`,
`"detector"`, `"event"`, reserved `"environmental"`) — **never** a function of call order, stage
index, or which other categories happen to exist in a given pipeline run. A naive "spawn the next
child stream from a running index" scheme (e.g., blindly calling `.spawn()` in pipeline order) fails
this property: inserting the reserved Environmental stage later, or reordering two stages for
readability, would silently shift every subsequent category's stream and invalidate every
previously-generated Scenario's reproducibility. Keying by a stable name instead of position is what
makes stage order genuinely inconsequential (§4.4's claim) rather than just usually-fine.

**The full chain**:

```
Master Seed  (one per batch/dataset run — §4.8)
   │
   ▼  index-keyed derivation, NOT sequential consumption (§4.8)
Scenario Seed  (one per scenario; = Master Seed for a single, non-batch request)
   │
   ▼  attempt-index-keyed derivation
Attempt Seed  (one per Generator/Validator retry attempt — §4.7)
   │
   ▼  category-key-keyed derivation (the property above)
Category child streams: Fire / Occupant / Behaviour Profile / Door / Exit /
Stair / Obstacle / Camera / Detector / Event / (Environmental, reserved)
```

Only the resolved **Scenario Seed** is stored on the accepted `Scenario` (§7, unchanged field).
Attempt seeds and category child streams are never persisted — they are cheaply and deterministically
re-derivable from `(scenario seed, generation_version)` (§4.3), so storing them would be redundant
bloat, not a reproducibility requirement. This document fixes the *property* the derivation function
must satisfy (name-keyed, order-independent, versioned) — it deliberately does not fix the
derivation function itself (HMAC? a stdlib `random.Random` seeded from a hashed tuple? a
`numpy.random.SeedSequence` keyed by string?), mirroring §4's existing restraint about the sampling
interpreter's own dispatch mechanics: that remains a construction-module implementation detail.

### 4.7 Constraint-aware generation and resampling

Reviewed against the worked pipeline ("generate legal exit states, NOT generate illegal states then
repair") — this restates, from a different angle, exactly the rejection-sampling design already
frozen by §1 finding 1 and §4/§5 (no repair, anywhere; joint constraints like `min_open_exits` are
the Validator's job, never the Generator's). That design already satisfies this pass's framing: the
Generator never produces a *known-illegal* candidate and then patches it — every joint-constraint
failure is handled by discarding and drawing a fresh, independently-sampled candidate. Confirmed,
unchanged.

**One real gap found and fixed — undefined sampling for allow-list-only fields.** §9 states a
candidate carries "the sampled [fire] profile," but §3.2's `allowed_fire_profiles:
FrozenSet[FireProfile]` has no paired `Distribution` (unlike `ignition_zone_preference`, which is
optional but at least *named*) — so when more than one profile is allowed (the worked example's
"Electrical OR Flaming"), nothing in the frozen Definition schema says how the Generator picks one.
The same gap exists more subtly for `ignition_zone_preference` itself: §3.2 says its absence "means
no stated preference," but doesn't say what the Generator *does* in that case. **Fix, entirely
Generator-side, no Definition schema change**: the sampling interpreter's dispatch (§4, "where, not
how") gains one fixed default policy — **any field expressed purely as an allow-list/`FrozenSet`
with no paired preference `Distribution` (or a `Distribution` left absent) is sampled uniformly at
random over its members, using that category's own child stream.** This resolves both gaps without
touching §3 (frozen): `allowed_fire_profiles` with 2 members and no preference field is
uniform-2-way; `ignition_zone_preference` absent means uniform over `allowed_ignition_zone_ids −
forbidden_ignition_zone_ids` (leaving the interaction question already flagged in §13 about a
*present* preference's support unresolved, unchanged).

**Optional optimization, not required by this freeze — category-scoped resampling.**
Full-candidate resampling on every Validator rejection (current, frozen behavior) is correct but
potentially wasteful at batch/dataset scale: a `min_open_exits` failure only ever involves the Exit
category's sampled fields, yet a full resample redraws Fire, Occupants, Doors, and everything else
too, for no reason. A narrower, still-repair-free alternative is architecturally sound and worth
recording as a *permitted* future refinement: if the Validator's rejection report can attribute a
failure to exactly one category (a Validator-side design question, out of scope today, flagged to
§13), the orchestration module *may* re-invoke only that category's sampling stage, drawing from a
further-nested `(attempt_seed, category_key, local_retry_index)` stream instead of a whole fresh
attempt — leaving every other already-sampled field on the candidate untouched. This is still
rejection sampling (a fresh, independent draw replaces the rejected one; nothing is patched into
compliance) and still lives entirely in the orchestration module, never in the construction module's
per-category sampling functions, which remain exactly as ignorant of "why" as before (§4, unchanged).
**This is explicitly not required for this freeze** — full-candidate resampling remains the frozen
baseline; category-scoped resampling is a documented, sanctioned option to adopt later without
touching the Scenario schema, the seed architecture, or the dependency-direction rules, precisely
because it only changes *which* stages of §4.4's pipeline get re-run, not what any stage does.

### 4.8 Batch generation

Reviewed for "single scenario → batch → millions" scaling. Architecture:

- A `BatchGenerationRequest` carries `(definition, master_seed, count, max_attempts_per_scenario)`
  and is conceptually "generate scenarios at indices `[0, count)`" — not "generate `count` scenarios,
  however that happens to unfold."
- **Index-keyed, not stream-position-keyed.** Scenario `i`'s seed is a deterministic function of
  `(master_seed, i)` (§4.6's same keyed-derivation property, one level up) — never "the `i`-th value
  drawn from one long-lived RNG fed sequentially through the whole batch." This is what makes batches
  **appendable and resumable**: generating scenarios `[1000, 2000)` in a second run, days later, on a
  different machine, produces byte-identical results to having generated `[0, 2000)` in one run —
  because scenario 1000's seed never depended on scenarios 0–999 having been generated first, only on
  its own index.
- **Embarrassingly parallel, with exactly one coupling point.** Generating scenario `i` (including
  its own internal Generator/Validator retry loop, §4.7) is fully independent of generating scenario
  `j`, `i ≠ j` — every input (seed, Definition, Building) is already fully determined before either
  starts. The **only** cross-scenario coupling in the entire pipeline is the uniqueness check's
  `accepted_hashes` set (§5, unchanged mechanism; §4.9 reviews its scaling). This means batch
  generation is trivially horizontally scalable (multiprocessing, multi-machine) *provided* that one
  shared set is coordinated correctly — everything else needs zero coordination.
- Single-scenario generation is the degenerate case `count=1`, `master_seed` unused, caller supplies
  `Scenario Seed` directly — no separate code path, consistent with §3.1's general "degenerate case
  of the same mechanism" pattern used throughout this document.

### 4.9 Non-repetition

Reviewed: is content-hash uniqueness (§5, unchanged mechanism) sufficient? For **exact**-duplicate
prevention, yes — a canonical-serialization hash catches any candidate identical in every sampled
field, and continuous fields (positions, growth parameters) make an accidental exact collision
astronomically unlikely on their own. Two gaps found, both about *scale*, neither about the hashing
mechanism itself:

1. **Cross-run persistence.** §4.8 establishes that batches must be resumable/appendable across
   separate process runs. An `accepted_hashes` set that only lives in one process's memory cannot
   dedupe against scenarios accepted in a *prior* run of the same Definition. **Fix**: at the start of
   any generation run, `accepted_hashes` must be seeded by reading the existing CSV catalog (§11) for
   that Definition, not assumed empty. This is a real requirement for §4.8's resumability claim to
   actually hold, not an optional nicety.
2. **Legitimate exhaustion vs. miscalibration.** A Definition built almost entirely from small
   `WeightedOptions`/`FixedValue` fields has a small *finite* legal-and-unique space (e.g., 2 doors ×
   2 states = 4 combinations); once that space is exhausted, acceptance rate correctly drops to
   zero — there are no more unique scenarios to find, which is correct behavior, not a bug. §5's
   existing retry-bound diagnostic ("this Definition rejected N/N attempts") cannot currently tell
   this apart from a poorly-calibrated joint constraint (§5's own example: `min_open_exits` too close
   to the exit count). Distinguishing the two is flagged, not designed, into §13 — it requires the
   Validator to report *why* each rejection happened (uniqueness collision vs. joint-constraint
   failure), which is Validator-internal design, out of scope for this Generator-only pass.

**Not required, flagged only**: whether uniqueness should mean more than exact-hash equality — e.g.,
a minimum-distance-from-nearest-accepted-scenario diversity guarantee, so "occupancy off by one
occupant" doesn't count as a meaningfully different scenario even though it hashes differently. This
is a dataset-quality/product question, not an architecture one; if ever needed, it would layer on top
of (not replace) content-hash dedup as an additional Validator check. Flagged into §13, not decided.

### 4.10 Scenario metadata

Reviewed against the requested field list (Scenario ID, Seed, Timestamp, Fire Origin, Occupancy
Summary, Open Exits, Closed Doors, Blocked Stairs, Obstacle Summary, Behaviour Profile Summary, Fire
Profile, Difficulty, Generation Version). These fields live in two places, per §4.4's
reproducibility-vs-provenance split: the full resolved state on `Scenario` itself (§7 — three fields
added this pass, `definition_content_hash`, `generation_version`, `rejected_attempt_count`, each
justified in §4.3/§4.9) and the CSV catalog's denormalized summary view for search (§11 — column
list updated this pass to match the requested list exactly, e.g. `behaviour_profile_summary` as
counts-per-id). Nothing here is new *data* — only a flattened view, computed once at serialization
time (§11) from fields already resolved onto `Scenario`; the catalog is never a second source of
truth. Nothing beyond the requested list plus these three additions was found missing.

### 4.11 Serialization and storage at scale

Reviewed: one-JSON-file-per-Scenario with the CSV strictly as a searchable catalog (§11, unchanged
mechanism) remains correct, and composes cleanly with §4.8's batch resumability — a new scenario is
one new file, disturbing nothing already written. Two scaling risks found, both folded into §11 as
conventions fixed now rather than retrofitted later: flat-directory file counts at
millions-of-scenarios scale (§11 now specifies id-prefix sharding), and full-file catalog rewrites
becoming the actual bottleneck (§11 now specifies append-only writes).

### 4.12 Performance — what to decide now vs. safely defer

Consolidating §4.6–§4.11's scattered findings into one list, since retrofitting the "decide now"
items after scenarios have already been generated at scale means every previously-generated
scenario's reproducibility guarantee breaks:

**Decide now (irreversible or expensive to change retroactively):**
- Name-keyed (not order/index-keyed) category child-stream derivation (§4.6).
- Index-keyed (not stream-position-keyed) per-scenario seed derivation for batches (§4.8).
- `definition_content_hash` and `generation_version` stored on every `Scenario` from the start
  (§4.3) — otherwise, scenarios generated before the field existed can never be retroactively
  verified reproducible.
- File-sharding convention for one-file-per-scenario storage (§11).
- Append-only catalog writes (§11).
- `accepted_hashes` seeded from the persisted catalog at run start, not assumed in-memory-only
  (§4.9).

**Safe to defer (additive, doesn't disturb anything already generated):**
- Category-scoped resampling (§4.7) — purely an orchestration-internal optimization; adding it later
  changes no stored data and no reproducibility contract.
- Exhaustion-vs-miscalibration retry diagnostics (§4.9) — a reporting improvement, not a schema or
  seed change.
- Near-duplicate diversity guarantees (§4.9) — additive on top of content-hash dedup, not a
  replacement.

### 4.13 What the Generator must never contain

Restating §4.1's non-goals as the enforcement mechanism, unchanged in kind from §3's approach (a
stated principle plus a dependency-direction test that makes violating it a build failure, not just
a convention): the construction module must never import `simulator`, `behavior`,
`behavior_library`, `ai_decision`, `sandbox`, `designer`, `scenario_validator` (already true, §12) —
and, per §4.2's finding, must no longer import `navigation` or `fire_growth` either. It never runs a
simulation, never contains Behaviour logic, Fire Physics, Smoke Physics, Navigation, RL, or Decision
Making — it only creates scenarios. The corrected full dependency list is folded into §12 directly,
not duplicated here.

## 5. Scenario Validator — frozen

This section is the outcome of a dedicated architecture review against one governing principle,
stated directly by this pass's brief: **the Validator is the sole authority deciding ACCEPT or
REJECT for a completed candidate Scenario — it never generates, never repairs, never modifies.**
§5.1–§5.9 record this pass's findings and the frozen design that results. As with §3/§4, simulation
execution, Behaviour logic, and RL are explicitly out of scope and were not reconsidered while
writing this section — only what the Validator itself checks, receives, and reports.

Package: `scenario_validator/` (depends on `scenario_definition`, the shared `scenario` model
package, `models`, `navigation`; must not import `scenario_generator`, `sandbox`, `designer`,
`simulator`, `behavior`, `behavior_library`, `ai_decision`, or `random` — unchanged, §12).

### 5.1 Purpose — what the Validator is not

Mirroring §4.1's structure for the Generator, four things the Validator is explicitly not:

- **Not a generator.** It never produces a candidate, never fills in a missing field, never invents
  a value — it only inspects one already-complete candidate handed to it.
- **Not a repair mechanism.** On any failure it discards the candidate outright (unchanged from the
  prior revision) — its return type is a verdict plus a report, never a "fixed" `Scenario`; there is
  no API surface that could return one.
- **Not a mutator.** The candidate object passed in is never written to. A rejected candidate is
  handed back to orchestration for disposal (§11: never serialized), not patched and resubmitted.
- **Not a simulator, a Behaviour Layer, or an RL agent.** It never steps time forward, never resolves
  a `behaviour_profile_id` (§8, unchanged — proportions only, never semantics), never learns or
  adapts. Every check is a direct, mechanical function of `(candidate, definition)` — this is what
  makes the Validator's output reproducible and audit-safe.

Its output is exactly `ACCEPT` or `REJECT`, backed by a structured `ScenarioValidationReport` (§5.4)
— never a bare boolean, per this pass's brief.

### 5.2 Input contract & statelessness

`ScenarioValidator.validate(candidate: Scenario, definition: ScenarioDefinition, accepted_hashes:
FrozenSet[str] = frozenset()) -> ScenarioValidationReport`. Three inputs, nothing else:

- `candidate` — one complete `Scenario` (§7) produced by the Generator (§4).
- `definition` — the `ScenarioDefinition` (§3) it must satisfy.
- `accepted_hashes` — externally supplied (§4.9/§11: seeded by orchestration from the persisted
  catalog), never accumulated or owned by the Validator itself.

**Frozen property, this pass**: the Validator is a **pure, stateless function** of these three
inputs. It carries no memory between calls — no attempt counter, no seed, no knowledge that this
candidate is a retry of a previous one. This is the literal mechanism behind "without coupling
itself to the Generator" (this pass's Retry Information brief, resolved in §5.8): the Validator
supplies one self-contained report per call; every notion of "attempt," "batch," or "retry count"
is assembled entirely by the orchestration module (§12, unchanged territory) from a *sequence* of
these reports, never by the Validator reasoning about sequence itself.

**Pipeline, confirmed unchanged (§1)**: Definition → Generator → Candidate Scenario → Validator →
Accepted → Serializer, or Rejected → Generator produces a new candidate. The Validator never
requests a repair and never edits the candidate — both already frozen invariants (§1 finding 1,
§4/§5 prior revisions), reconfirmed, not modified, by this pass.

### 5.3 Validation categories — modularized

Reviewed: should the Validator contain separate validation modules? **Yes** — the prior revision's
flat, uncategorized bullet list is replaced this pass with seven named modules, each producing
issues tagged with exactly one entry from the fixed category enum (§5.5). **Building Validation
is a precondition for the rest**: every other module's checks presuppose that the ids they operate
on actually resolve against the Building, so Building Validation runs first and a failure there
short-circuits the remaining modules (their checks would be meaningless against unresolved ids, not
merely redundant).

**1. Building Validation → `STRUCTURAL`.** New this pass — previously only implicit.
- Every id referenced anywhere on the candidate's resolved state (zone/door/exit/stair/obstacle/
  camera/detector ids, `ScenarioEvent.target_id`) actually exists on the referenced `Building`. This
  is a defense-in-depth re-check, not a redundant one: the Generator should only ever sample from
  dict keys the Definition already declared (themselves checked to exist by §3.4), so this check's
  job is to catch a **Generator bug**, not a Definition-authoring mistake — the two failure modes
  look identical from outside but have different root causes, which is exactly why this check exists
  independently of §3.4 rather than being assumed satisfied by it.
- "A valid Project exists" folds into this, not a separate check: `Project` (`models/project.py`) is
  a thin wrapper holding exactly one `Building` (`project.building: Building | None`); there is no
  Project-level validity concern beyond its `Building` being present and passing the check above.

**2. Occupant Validation → `OCCUPANCY` / `GEOMETRY`.**
- Every occupant belongs to exactly one zone (an explicit invariant, previously only implied by how
  placement works).
- Occupant position lies within its zone's polygon (`Zone.contains()`) — tagged `GEOMETRY`,
  distinct from the count/proportion checks below.
- Per-zone occupant count is consistent with its `occupancy_distribution` entry's support (e.g. a
  `UniformRange(3, 8)` draw must land in `[3, 8]`) — tagged `OCCUPANCY`.
- `behaviour_profile_id` is present (a non-empty string) and sampled proportions are statistically
  consistent with `behaviour_profile_distribution` (a *statistical* check only — the Validator never
  asks whether an id string is a real registered Behaviour Profile, for the same reason §3.4 doesn't;
  `scenario_validator/` must not import `behavior`/`behavior_library`, unchanged, §12) — tagged
  `OCCUPANCY`.

**3. Fire Validation → `FIRE`.**
- Sampled ignition zone id lies within `allowed_ignition_zone_ids`, not in
  `forbidden_ignition_zone_ids`, and on an allowed floor (§3.2's explicit Fire fields) — the
  Validator re-checking independently rather than trusting the Generator got the allow/forbid rules
  right.
- **New this pass**: sampled `fire_profile` is a member of `allowed_fire_profiles` — previously
  unchecked; a defense-in-depth re-check of §4.7's default-uniform-sampling policy, the same
  Generator-bug-vs-Definition-mistake reasoning as Building Validation.
- **New this pass**: sampled `growth_parameters` fall within `growth_parameter_distribution`'s
  support — previously the only support-checked field was occupancy count; growth parameters had no
  equivalent check.

**4. Door / Exit / Stair Validation → `STRUCTURAL`.**
- `min_open_exits` satisfied by the sampled open-exit count, taken exactly as drawn (no repair step
  exists to have adjusted it) — the clearest example of a joint constraint the Generator never
  touches: each exit's `Distribution[bool]` is sampled fully independently of every other exit, so a
  candidate failing this check is expected, ordinary, and handled purely by rejection — not a bug in
  the Generator.
- Pinned (`FixedValue`) door/exit/stair/obstacle/camera/detector states landed on their pinned value.
- **Deliberately scoped to state and count only.** Whether that state actually *enables* evacuation
  is exclusively Navigation Validation's concern (below) — the two modules are non-overlapping by
  design: one checks "is the building's egress state as declared," the other checks "does that state
  make evacuation actually possible."

**5. Event Validation → `EVENTS`.** Mostly new this pass — previously folded into a generic "user
constraints satisfied" bullet with no dedicated checks of its own.
- Event target ids exist (cross-references Building Validation's check, doesn't repeat it).
- Event timestamps are valid (non-negative, within the scenario horizon).
- **New this pass**: events are ordered correctly and no two events conflict — e.g. a sampled "door
  closes" event and a sampled "door opens" event targeting the same door at the same instant is
  rejected here; previously nothing checked cross-event consistency at all.
- Every `EventTemplate` with `occurs=FixedValue(True)` produced its event verbatim.
- No sampled event contradicts a `FixedValue`-pinned element (e.g. a sampled "door closes" event
  targeting a `FixedValue(OPEN)` door is rejected here).

**6. Navigation Validation → `NAVIGATION`.** The category this pass's brief calls "extremely
important" — expanded from two bullets into four explicit checks, reusing existing
`navigation/validation.py` connectivity machinery throughout (the Validator, unlike the Definition
and unlike the Generator, is allowed to depend on `navigation` — §12, unchanged):
- **At least one evacuation route exists** — every occupied zone can reach at least one open Exit
  through the Navigation Graph, given the candidate's fully-sampled door/obstacle/stair state.
- **Building is not partitioned into unreachable regions** — scoped to *occupied* zones only: an
  isolated region with zero sampled occupants this candidate doesn't threaten evacuation and is not
  rejected for it. (A stronger, occupancy-independent topology check would be a Building/Designer-time
  concern, not a per-candidate Scenario Validator concern — considered and deliberately not adopted
  here, since it would reject candidates for a property of the *Building*, not the *Scenario*.)
- **Fire origin does not create an impossible initial condition** — the general case of the above two
  (e.g. an ignition zone that is the sole path to every open exit), checked against the
  fully-resolved sampled state.
- **New finding, this pass — minimum reachable egress capacity.** The prior revision's
  `min_open_exits` check (Door/Exit/Stair Validation, above) counts how many exits are flagged
  `OPEN`, and the "at least one evacuation route" check above only requires that *some* open exit be
  reachable from *every* occupied zone — **neither check, nor their conjunction, guarantees that
  `min_open_exits` worth of *usable* egress capacity actually exists.** A candidate with
  `min_open_exits=2`, two exits sampled `OPEN`, but only one of them reachable from any occupied zone
  (the other blocked by sampled obstacle/door state) passes both existing checks independently while
  silently defeating the purpose `min_open_exits` was declared for — redundant, usable egress. **Fix**:
  Navigation Validation adds a check that at least `min_open_exits` *distinct* open exits are each
  individually reachable from at least one occupied zone — not merely that some single exit is
  reachable from every zone. This is the concrete meaning this pass assigns to "minimum evacuation
  connectivity preserved," distinguishing it from "at least one route exists" rather than treating the
  two phrases as redundant.

**7. Dataset Validation → `DATASET`.** Mostly new this pass — previously only uniqueness existed
under this heading.
- **Duplicate scenario** — the candidate's canonical serialized form is hashed and compared against
  `accepted_hashes`; a collision is a rejection like any other (unchanged mechanism, §4.9; tagged
  `code="DUPLICATE"` this pass, §5.8).
- **New this pass — metadata completeness.** Every reproducibility-relevant field from §7's canonical
  list (`scenario_id`, `definition_id`, `definition_content_hash`, `seed`, `generation_version`) is
  populated and non-empty. A candidate missing any of these would silently break §4.3's
  reproducibility contract if ever accepted — this check enforces that contract at the one point
  where enforcement is still possible, before serialization.
- **New this pass — serialization round-trip completeness.** The candidate survives
  `to_dict()` → `from_dict()` (§7) without data loss. A defensive check: catching a serialization bug
  here, before the candidate is written to disk (§11), is strictly better than discovering corrupted
  data on read-back later.
- **New this pass — schema/generation-version compatibility.** The candidate's `generation_version`
  (§4.3) is one the current Validator/tooling knows how to interpret. Relevant once §4.6's seed
  derivation function, or any other generation-version-tagged behavior, ever changes — guards against
  silently accepting a candidate produced by an incompatible Generator release.

### 5.4 Validation Report

Reviewed against the requested shape (`ScenarioValidationReport` → Accepted → Validation Messages →
Validation Categories → Failure Reasons → Warnings → Metadata). **One normalization applied**: rather
than four parallel lists (messages, categories, reasons, warnings) that would need external indexing
to know which message belongs to which category, this pass defines one flat, self-describing list of
typed issues — avoiding exactly the kind of indexing bug that four separate parallel lists invites.

```
ScenarioValidationIssue:
    category: str    # one of §5.5's fixed enum — "Validation Categories"
    severity: str     # ERROR | WARNING — "Warnings" (§5.6)
    code: str         # stable, machine-readable, e.g. "MIN_OPEN_EXITS_UNSATISFIED", "DUPLICATE"
                       #   — "Failure Reasons"
    message: str       # human-readable — "Validation Messages"
    object_id: str = ""

ScenarioValidationReport:
    issues: List[ScenarioValidationIssue]
    accepted: bool      # property: no issue in `issues` has severity == ERROR — "Accepted"
    metadata: dict       # attempt-agnostic diagnostic fields, e.g. candidate's own scenario_id —
                          #   "Metadata" (§5.8)
```

**Compliance finding, this pass**: the prior revision stated the Validator "reuses
`navigation/validation.py`'s shape" wholesale. That shape (`ValidationIssue`: `code`, `severity`,
`message`, `object_id`, `floor_id`) has no `category` field — the one piece this pass's Failure
Categories requirement needs and the prior shape structurally cannot provide without being
redefined. **Fix**: `scenario_validator/` defines its own `ScenarioValidationIssue`/
`ScenarioValidationReport`, structurally similar to `navigation.validation.ValidationReport`'s
convention (severity is `ERROR`/`WARNING`; `accepted`/`is_valid` is computed, never stored) but not a
subclass or re-export — the same resolution pattern §3.1 finding 3 already applied to the Definition
side (a locally-defined report type over cross-package reuse, once the shape needs to diverge), now
applied on the Validator side even though the Validator, unlike the Definition, *is* allowed to
import `navigation` — the dependency being permitted doesn't make the borrowed shape sufficient once
it's missing a field this pass requires.

**The Generator/orchestration must only consume this report at arm's length**, per this pass's
brief ("the Generator may use this report only to know whether to generate another candidate... it
should not inspect internal validation logic"): orchestration reads `report.accepted` to decide
accept-vs-resample, and reads `issues[*].category`/`code` only for the aggregate diagnostics in §5.8
— it never re-derives *why* a specific check passed or failed beyond what the issue's `code` already
states, and never reaches into a validation module's internals directly.

### 5.5 Failure categories

A fixed, closed enum this pass introduces: `STRUCTURAL`, `GEOMETRY`, `OCCUPANCY`, `FIRE`,
`NAVIGATION`, `EVENTS`, `DATASET` — exactly the seven named in this pass's brief. Mapping from §5.3's
seven validation modules is mostly 1:1, with two deliberate exceptions already called out where they
occur: Occupant Validation splits across `OCCUPANCY` (count/proportion) and `GEOMETRY` (polygon
containment), and Door/Exit/Stair Validation's checks land in `STRUCTURAL` alongside Building
Validation's (both are "is the declared/counted state correct," as opposed to `NAVIGATION`'s "does
that state work").

### 5.6 Warnings vs. errors

`severity` is exactly `ERROR` or `WARNING` (already the shape `navigation.validation.ValidationReport`
used, §5.4 keeps the convention). `ERROR` → the candidate is rejected (contributes to `accepted =
False`). `WARNING` → the candidate remains accepted; the issue is informational only.

**New this pass**: at least one `WARNING`-only check must exist so the distinction is exercised, not
merely declared with nothing ever using it. Adopting this pass's own worked example: Occupant
Validation gains a `WARNING`-severity check — sampled occupancy density for a zone exceeds a
soft, configurable threshold — that leaves `accepted` unaffected. The specific threshold is an
implementation detail, not decided here; that at least one such check exists, and that it never
downgrades `accepted`, is the frozen part.

### 5.7 Uniqueness — reviewed, not re-derived

This pass's brief re-asks whether content-hash uniqueness is sufficient and whether diversity metrics
should remain optional. Both were already reviewed in depth during the Generator pass (§4.9) — not
repeated here: exact-hash dedup is sufficient for exact-duplicate prevention; cross-run persistence of
`accepted_hashes` is orchestration's responsibility (§4.9/§11, unchanged); near-duplicate/diversity
guarantees beyond exact-hash equality remain a flagged, optional, additive extension, a product
question rather than an architecture one (§4.9/§13, unchanged). **The one addition this pass makes**:
a duplicate-collision rejection is tagged `category="DATASET", code="DUPLICATE"` (§5.3's Dataset
Validation module) — the specific tag that makes §5.8's exhaustion-vs-miscalibration diagnostic
possible without any further design work.

### 5.8 Retry information & generation diagnostics

This pass's brief asks for enough reporting to support generation statistics, acceptance rate,
rejection reasons, and retry counts, "without coupling itself to the Generator" — and resolves the two
questions §13 explicitly deferred to this Validator freeze pass:

1. **Category-scoped resampling's rejection-attribution requirement (§4.7, deferred here).**
   Resolved: a rejected `ScenarioValidationReport` is **single-category-attributable** exactly when
   every `ERROR`-severity issue in it shares one `category` value. Orchestration (§12) may use this
   to decide whether §4.7's optional category-scoped resample applies to a given rejection —
   re-invoking only that one category's sampling stage — falling back to a full resample whenever a
   rejection's `ERROR` issues span more than one category. No new field was needed beyond §5.4's
   `category` tag; the mechanism this pass's Failure Categories requirement already demanded turns
   out to be exactly the mechanism §4.7 needed and left undesigned.
2. **Exhaustion-vs-miscalibration diagnostics (§4.9, deferred here).** Resolved the same way:
   tabulating `(category, code)` across a batch's sequence of rejected reports tells orchestration
   whether rejections are dominated by `DATASET/DUPLICATE` (the Definition's finite legal-and-unique
   space is genuinely exhausted, §4.9) or by some other recurring `(category, code)` pair such as
   `STRUCTURAL/MIN_OPEN_EXITS_UNSATISFIED` or `NAVIGATION/INSUFFICIENT_REACHABLE_EGRESS` (a joint
   constraint is miscalibrated). Again, no new field — the same tagging already required by §5.5.

**The coupling boundary, stated precisely** (this pass's "without coupling itself to the Generator"):
the Validator (§5.2) is stateless and knows nothing about attempts, retries, seeds, or batches — it
returns one report for one candidate, full stop. Every aggregate quantity this pass's brief lists
(generation statistics, acceptance rate, rejection reasons *tabulated over many attempts*, retry
counts, dataset diagnostics) is computed entirely by the orchestration module (§12, unchanged
territory: orchestration already owns attempt-seed derivation, §4.6/§4.8) by collecting the sequence
of reports it receives across calls to `validate()` — never by the Validator reasoning about sequence,
history, or the Generator's internals itself.

### 5.9 What the Validator must never contain

Restating §5.1's non-goals as the enforcement mechanism, unchanged in kind from §3.1/§4.13's
approach: `scenario_validator/` must never import `scenario_generator`, `sandbox`, `designer`,
`simulator`, `behavior`, `behavior_library`, `ai_decision`, or `random` (§12, unchanged by this
pass — already correct). It never generates, modifies, or repairs a `Scenario`; never contains
simulation logic, Behaviour logic, RL, or decision-making. It only evaluates completed candidates and
returns a report.

**Time-varying validation scope, resolved this pass** (closing a question §13 previously carried
over unresolved): does Navigation Validation need to re-check reachability after every scheduled
event fires, not just at the candidate's initial (t=0) state? **No** — checking post-event
connectivity would require stepping the event schedule forward in time, which is simulation
execution, explicitly excluded from this pass's scope and from the Validator's architecture in
general (§5.1: "not a simulator"). Navigation Validation checks the candidate's resolved t=0 state
only. If a future need for post-event evacuation-safety guarantees emerges, it belongs to a
different mechanism entirely — e.g. a Definition-side constraint restricting which elements
`EventTemplate`s may target, or a Simulator-side safety check — not an expansion of the Validator's
own scope.

## 6. Scenario Events

`ScenarioEvent` (the *resolved*, sampled output — lives in the `scenario` model package, §7)
remains a Building-state trigger, not simulation physics; simulator integration seam still open —
see §13. The Definition-side declaration is `EventTemplate` (renamed from `EventRule` this
revision, §3.1 finding 5, and no longer has a separate `fixed_events` sibling — §3.1 finding 2):
`occurs`/`time` are `Distribution`s like every other field, sampled by the Generator, with
acceptance governed entirely by the Validator. An `EventTemplate` is, in the words of this
revision's prompt, a template describing what the Generator is *allowed* to create — never an
event itself, and the Definition holds only templates, never a schedule.

## 7. Scenario object (shared model package)

Package: `scenario/`. Canonical field list, updated this pass (§4.3/§4.9/§4.10 record why each
addition was needed — not repeated here):

- `scenario_id`, `definition_id`, `definition_content_hash` (new, §4.3 — a content hash of the exact
  `ScenarioDefinition` used, closing the hidden-state gap where an edited-in-place Definition could
  silently break `(definition_id, seed)` reproducibility).
- `seed` (the resolved Scenario Seed, §4.6 — the *only* seed stored; attempt/category child seeds are
  re-derived, never persisted).
- `generation_version` (new, §4.3 — identifies the Generator release whose sampling interpreter and
  seed-derivation function produced this candidate).
- Resolved fire state: `ignition_zone_id`, `growth_parameters`, `fire_profile` (§4.7's
  default-uniform-sampling fix makes this field always well-defined, even when `allowed_fire_profiles`
  has more than one member and no explicit preference).
- Resolved per-zone occupant records: position + `behaviour_profile_id: str` (§8, unchanged — the
  opaque label assigned by the Generator, and the **only** behavior-related field a `Scenario` ever
  stores).
- Resolved door/exit/stair/obstacle/camera/detector state.
- `events: List[ScenarioEvent]` (§6, unchanged).
- `difficulty: Optional[float]` (§10, unchanged — absent until post-validation scoring runs).
- `created_at` (provenance only, §4.4 — a wall-clock timestamp, explicitly *not* part of the
  reproducibility contract).
- `rejected_attempt_count: int` (new, §4.9/§4.10 — how many candidates were discarded before this one
  was accepted; a per-scenario echo of §5's retry-bound diagnostic).
- `to_dict()`/`from_dict()` (unchanged).

It never stores a `behavior.profile.BehaviorProfile` instance, a strategy choice, or any sampled
behavioral quantity (walking speed variation, panic level, ...) — those don't exist until the
Behaviour Layer resolves the id at simulation time (§8) — nor a constructed
`fire_growth.model.FireGrowthModel` instance (§4.2, new this pass — only the plain data a Simulator
would later use to build one).

## 8. Behaviour Profiles — Occupants vs. Behaviour *(architecture support only — no implementation)*

**Review, this revision**: does the architecture keep "who exists in a scenario" (population —
Scenario Generator's job) structurally separate from "how each occupant acts" (behaviour —
exclusively the Human Behavior Layer's job, `behavior/` + `behavior_library/`)? The prior revision's
`OccupantProfile` (a closed enum: `ADULT`, `CHILD`, `ELDERLY`, `WHEELCHAIR_USER`, `STAFF`,
`VISITOR`, `FIRE_WARDEN`) was already inert data and never leaked any behavioral quantity into the
Definition schema — no `walking_speed`, `panic`, `herding`, `pre_movement_delay`,
`route_choice`, `familiar_exit_preference`, `announcement_compliance`, `helping_behaviour`,
`stress_response`, or `decision_latency` field exists anywhere in §3.2, and none is added by this
revision either. But two things needed correcting, both about *identity*, not content:

1. **A closed enum implies the Scenario Engine owns the set of valid profiles.** It doesn't, and
   per the stated goal ("the same Scenario simulated using different Behaviour Models in future
   research without regenerating it") it structurally can't — the set of profile names is the
   Behaviour Layer's to define, extend, and reinterpret, entirely independent of any Scenario
   Definition or Generator release. **Fix**: `OccupantProfile` is replaced by a plain, open
   identifier string — by convention `<Category>_<Variant>` (`Adult_Default`, `Child_Default`,
   `Visitor_Default`, `Staff_Default`, `FireWarden_Default`, `Wheelchair_Default`, ...; the `_Default` suffix in
   every worked example is itself evidence this is meant to be an open, growing namespace, not a
   closed set of seven categories) — with **no enum, no closed set, defined nowhere in the Scenario
   Engine**.
2. **Nothing was actually checking that the Scenario Engine stays ignorant of what a profile
   *means*.** The Generator assigns a `behaviour_profile_id`; it must never resolve, interpret, or
   even validate that string beyond "it's a string" — doing so (e.g. checking it against a real
   registry) would require importing `behavior`/`behavior_library`, undoing the independence this
   whole revision asks for. This is now explicit at every layer that touches the field: §3.4
   (Definition self-validation doesn't check registry membership), §5 (Validator checks proportion
   statistics only, never semantics), §12 (dependency-direction tests forbid the import outright,
   not just by convention).

**The boundary, concretely**: a `Scenario`'s occupant record carries exactly one behavior-related
field, `behaviour_profile_id: str` (§7) — an opaque label. Interpreting it is entirely the
Behaviour Layer's responsibility, at simulation time, via a mechanism this document does not
design (per this prompt's scope — see §2's grounding: no such registry exists yet; `behavior/` and
`behavior_library/` today require every strategy/profile combination to be constructed inline by
the caller). What the *shape* of that resolution will eventually be — most likely something that
turns `"Adult_Default"` into the `(BehaviorProfile, DecisionStrategy, RouteChoiceStrategy,
PreMovementDelayStrategy)` bundle `HumanBehaviorLayer.register()` already expects per occupant — is
Behaviour Layer architecture, not Scenario Engine architecture, and is out of scope here exactly as
this prompt requires ("Do NOT think about ... implementation").

**Why this achieves the stated goal**: because a `Scenario` only ever stores the label, never the
behavioral values it resolves to, the *same* stored `Scenario` file can be handed to two different
versions (or entirely different implementations) of the Behaviour Layer's registry and simulated
twice, producing two different behavioral outcomes from *identical* population/fire/environment
conditions — without regenerating, revalidating, or even re-reading the `ScenarioDefinition`. The
independence isn't just "these are different packages" (already true structurally, §12) — it's that
the *data crossing the boundary* is a name, never a value.

**Re-audited, this pass**: re-checked against the same question with the worked examples restated
independently (`Visitor_Default`, `Adult_Default`, `Child_Default`, `Staff_Default`,
`FireWarden_Default`, `Wheelchair_Default`) and the same ten-item behavior list restated
independently (pre-movement delay, herding, panic, route choice, familiar exit preference,
announcement compliance, helping behaviour, stress response, walking speed variation, decision
latency). No new violation found — every item in both lists was already covered by the boundary
above. The only change from this pass is `Staff_Default` added to the identifier-convention example
list earlier in this section, which had cited five of the six category names.

## 9. Fire Profiles *(architecture support only — no implementation)*

Unchanged in intent; the worked example's "Fire profile = Electrical OR Flaming" is
`allowed_fire_profiles: FrozenSet[FireProfile]` (§3.2 — a restriction/allow-list, matching the
example's own phrasing exactly; no likelihood weighting is committed for this field, unlike
`ignition_zone_preference`, since nothing in the worked example calls for one — a future revision
can add a `Distribution[FireProfile]` preference layer the same way if a real use case needs it).
Still unconsumed by anything beyond echoing the sampled profile onto the `Scenario`; its effect on
`FireGrowthCurve` parameterization remains deferred. **Grounded this pass**: "the sampled profile"
was previously undefined when more than one member of `allowed_fire_profiles` is allowed and no
preference is stated — §4.7 closes this with a Generator-side default (uniform sampling over the
allowed set), entirely without adding a field here.

## 10. Difficulty

Unchanged from the previous revision: not a `ScenarioDefinition` field, computed only after the
Validator accepts a candidate, stored as `Scenario.difficulty` metadata. The rejection-sampling
change in this revision reinforces why timing matters here — a *rejected* candidate never reaches
difficulty scoring at all, since it never becomes a `Scenario`.

## 11. Scenario Storage

One serialized JSON file per accepted `Scenario`, one metadata-only CSV catalog — unchanged
mechanism from the prior revision, reviewed and confirmed correct this pass (§4.11), with two
conventions now fixed to avoid an expensive later migration at batch/dataset scale (§4.12):

- **File sharding.** Scenario files are sharded by a fixed prefix of `scenario_id` (e.g., a
  subdirectory per first two hex characters) rather than written flat into one directory — millions
  of files in one flat directory degrades on many filesystems well before "millions" is reached, and
  relocating already-written files later is exactly the kind of retroactive migration this pass's
  Performance review (§4.12) flags as worth avoiding. The specific scheme is an implementation
  detail; that sharding exists, keyed by `scenario_id` prefix, is fixed now.
- **Append-only catalog writes.** The CSV catalog is written by appending one row per newly accepted
  scenario, never by rewriting the file in full — a full rewrite per acceptance is O(n) and becomes
  the actual generation-throughput bottleneck at scale, not scenario sampling itself.

Catalog columns (updated this pass to match §4.10's reviewed metadata list): `scenario_id`,
`definition_id`, `seed`, `created_at`, `fire_origin_zone_id`, `fire_profile`, `occupancy_summary`,
`open_exit_ids`, `closed_door_ids`, `blocked_stair_ids`, `obstacle_summary`,
`behaviour_profile_summary` (counts per profile id), `difficulty`, `generation_version`,
`rejected_attempt_count`. Every column is a denormalized view of a field already resolved onto
`Scenario` (§7) — the catalog computes it once at serialization time and is never a second source of
truth for anything.

Built on `serialization/json_writer.py`/`json_reader.py`, not `Serializer` (unchanged). Rejected
candidates are never written anywhere (unchanged) — a Definition's empirical rejection rate (§5's
retry bound, §4.9's exhaustion-vs-miscalibration distinction) is a runtime diagnostic, not a storage
concern. **New this pass**: `accepted_hashes` (§5's uniqueness check) must be seeded by reading this
catalog at the start of any generation run targeting a given Definition, not assumed empty —
required for §4.8's batch resumability/appendability to actually hold across separate process runs,
not just within one.

## 12. Dependency direction

- `ScenarioDefinitionPackageDependencyDirectionTests` — `scenario_definition/` must not import
  `scenario_generator`, `scenario_validator`, `scenario` (the model package), `simulator`,
  `sandbox`, `behavior`, `behavior_library`, `designer`, `navigation`, or `random`/`numpy`
  (**reversed from an earlier revision** — §3.1 finding 1 removed the only reason `random` was ever
  needed here, and finding 3 removed the only reason `navigation` was; `behavior_library` is added
  explicitly this revision, §8). **May** import `models` only.
- `ScenarioGeneratorPackageDependencyDirectionTests` — the **construction module** (the sampler
  itself, distinct from the orchestration module below) may import `scenario_definition`,
  `scenario`, `models` **only** (**narrowed this pass, §4.2** — `navigation` and `fire_growth` are
  removed from the prior revision's list: neither has any legitimate use once §4.1's non-goals are
  enforced literally — reachability is exclusively the Validator's job, and fire state is stored as
  plain sampled data, never a constructed `FireGrowthModel`, §4.2); must not import
  `scenario_validator`, `sandbox`, `designer`, `simulator`, `behavior`, `behavior_library`,
  `ai_decision`, `navigation`, `fire_growth`. (This module is where the `Distribution`-sampling
  interpreter deferred in §4 lives — not designed in this document, except for the one fixed default
  policy §4.7 adds: uniform sampling over an allow-list field with no paired preference
  `Distribution`. It samples a `behaviour_profile_id` string exactly like any other field; it never
  imports `behavior_library` to do so, §8.)
- `ScenarioValidatorPackageDependencyDirectionTests` — `scenario_validator/` may import
  `scenario_definition`, `scenario`, `models`, `navigation`; must not import `scenario_generator`,
  `sandbox`, `designer`, `simulator`, `behavior`, `behavior_library`, `ai_decision`, or `random`
  (the Validator only ever checks an already-fully-sampled candidate; it has no legitimate reason to
  draw a random value, closing the same kind of gap-by-omission §3.1 finding 1 closed for the
  Definition). `ScenarioValidationReport`/`ScenarioValidationIssue` (§5.4) are locally defined in
  this package, not imported from `navigation` — permission to depend on `navigation` (for
  connectivity checks, §5.3's Navigation Validation module) doesn't extend to reusing its
  `ValidationReport` shape wholesale, since that shape has no `category` field (§5.4's compliance
  finding).
- **Orchestration** (`scenario_generator/pipeline.py`) is the one module allowed to depend on both
  the construction module and `scenario_validator` — it implements exactly the loop in §1's
  diagram (sample → validate → accept, or discard and resample with the next child seed, up to the
  max-attempts bound from §5). This module also owns every seed-derivation step above the
  per-category child streams — attempt-seed derivation (§4.6), batch index-seed derivation (§4.8),
  and loading `accepted_hashes` from the persisted catalog at run start (§4.9) — since these are
  cross-attempt/cross-scenario concerns the construction module's per-category sampling functions
  have no need to know about. The dependency-direction test targets the construction module
  specifically (or the package minus `pipeline.py`), so "the Generator never contains validation
  logic, and never repairs" stays true of the sampler itself, not just true "on average" of the
  package.

## 13. Open questions for review

- **Environmental constraints** — scope still undecided; §3.3 now at least names the reserved
  category and its expected future members (carried over, slightly narrowed).
- ~~**Event/time-varying constraint validation scope**~~ — **resolved, §5.9**: no, checking
  post-event connectivity would require stepping the event schedule forward in time, which is
  simulation execution, out of the Validator's architecture entirely. Navigation Validation checks
  the candidate's resolved t=0 state only.
- **Simulator integration seam for `ScenarioEvent`** — not a Scenario Engine-internal decision
  (carried over).
- **Fire Profile schema finalization** — naming/placement only, behavior deferred (carried over;
  Behaviour Profile's finalization is resolved this revision, §8 — Fire Profile remains a closed
  enum for now since nothing in this prompt asked it to become an open namespace too, but the same
  question could reasonably be raised for it later).
- **Behaviour Profile registry design** *(new, out of scope by this prompt's own instruction)* —
  where the id-to-config resolution lives (`behavior_library/`? a new sibling package?), whether
  it's a static dict or something more dynamic, and whether an *optional*, external
  authoring-time lint could warn "this Definition references a profile id with no matching
  registry entry" without the Scenario Engine itself ever depending on the registry — flagged, not
  designed, per §8.
- **Difficulty scoring formula** — placement/timing decided, formula not (carried over).
- **Rejection-sampling feasibility diagnostics** *(new)* — should `ScenarioDefinition.validate()`
  attempt to *estimate* an acceptance rate (e.g. analytically for simple constraints like
  `min_open_exits` against independent per-exit weights, since that specific case has a closed-form
  probability) and warn at Definition-authoring time, rather than only discovering a low acceptance
  rate empirically during a batch run? Not decided — flagged because §5's retry bound makes this a
  real operational question, not a hypothetical one.
- **Placement-utility extraction** touching `sandbox/manager.py` — carried over.
- **CSV index scope** (one running index vs. per-batch) — carried over.
- **`ignition_zone_preference` interaction with allow/forbid** *(new, from this revision's §3.1
  finding 4)* — if present, does its own internal support need to be a subset of
  `allowed_ignition_zone_ids − forbidden_ignition_zone_ids`, checked by self-validation (§3.4), or
  is it interpreted as "weighting only where it overlaps the allowed set, ignored elsewhere"?
  Not decided; §3.4's self-validation list does not yet check this field for that reason.
- **Environmental field shape** *(new)* — whether §3.3's reserved category ends up expressed in the
  same `Distribution`-valued vocabulary as everything else, or needs something new (e.g. a weather
  *time series* is arguably closer to the reserved `TimeDependentSpec` than to any committed kind)
  — deferred with the rest of §3.3.
- ~~**Category-scoped resampling's rejection-attribution requirement**~~ — **resolved, §5.8**: a
  rejected report is single-category-attributable exactly when every `ERROR`-severity issue shares
  one `category` value (§5.4/§5.5); multi-category rejections fall back to full resampling.
- ~~**Exhaustion-vs-miscalibration retry diagnostics**~~ — **resolved, §5.8**: tabulating
  `(category, code)` across a batch's rejected reports distinguishes `DATASET/DUPLICATE`-dominated
  rejection (space exhaustion) from any other recurring `(category, code)` pair (miscalibration) —
  computed by orchestration from the sequence of reports, not by the Validator itself.
- **Near-duplicate / diversity guarantees beyond exact-hash uniqueness** *(new, this pass, §4.9)* —
  whether dataset quality ever needs more than exact-content-hash dedup (e.g. a
  minimum-distance-from-nearest-accepted-scenario check) is a product question, not an architecture
  one. Flagged, not decided.
- **File-sharding scheme specifics** *(new, this pass, §11)* — that scenario files are sharded by a
  `scenario_id` prefix is now fixed; the prefix length and directory depth are implementation
  details, not decided here.
- **Seed-derivation function choice** *(new, this pass, §4.6)* — the *property* the function must
  satisfy (name/index-keyed, order-independent) is now fixed; the concrete function (HMAC, hashed-seed
  `random.Random`, `numpy.random.SeedSequence`, ...) is a construction-module implementation detail,
  deliberately not chosen here, mirroring how the `Distribution`-sampling interpreter's own dispatch
  is deliberately left undesigned by §4.
- **Occupancy-density warning threshold** *(new, this pass, §5.6)* — that at least one
  `WARNING`-severity check exists (occupancy density) is fixed; the specific density threshold and
  whether it's a fixed constant or a Definition-authorable field is not decided here.
- **Schema/generation-version compatibility policy** *(new, this pass, §5.3's Dataset Validation)* —
  that a compatibility check exists is fixed; which `generation_version` values count as
  "compatible" with a given Validator release (exact match only? a semver range? a migration path?)
  is not decided here.
- **`min_open_exits` reachable-egress-capacity check's exact reachability semantics** *(new, this
  pass, §5.3's Navigation Validation)* — "at least `min_open_exits` distinct open exits each
  reachable from at least one occupied zone" is fixed as the requirement; whether the same occupied
  zone may count toward satisfying multiple exits' reachability, or whether some notion of
  path-disjointness across occupied zones is also required, is not decided here — flagged as a
  refinement to consider if `min_open_exits`'s intent turns out to need more than "capacity exists
  somewhere reachable."

## 14. Status

Scenario Definition Engine (§3), Scenario Generator (§4), and **Scenario Validator (§5) are now all
three frozen.** §3 and §4 are unchanged by this pass except where §5's findings cascaded into their
wording (§1's diagram gained a pointer to §5; §12's Validator dependency bullet gained a note about
`ScenarioValidationReport` being locally defined) — no redesign of either, per this pass's own
instruction.

This pass adds the Validator's frozen boundary: purpose and non-goals restated as an explicit
"what it is not" list mirroring §4.1 (§5.1); a stateless, pure-function input contract — no memory of
attempts, seeds, or batches, ever (§5.2); seven modularized validation categories replacing the prior
flat bullet list, including a previously-absent Building Validation module and a previously-absent
Dataset Validation module (metadata completeness, serialization round-trip, schema-version
compatibility, §5.3); a real gap found and fixed in Navigation Validation — `min_open_exits` (a count)
and "at least one reachable route" (an existence check) could each pass independently while the
*combination* they're meant to jointly guarantee, redundant usable egress, silently failed; fixed with
an explicit minimum-reachable-egress-capacity check (§5.3); a structured `ScenarioValidationReport`
with category/severity/code-tagged issues, replacing the prior revision's bare reuse of
`navigation/validation.py`'s untagged shape (§5.4); a fixed seven-category failure taxonomy (§5.5); a
first real `WARNING`-severity check exercising the ERROR/WARNING distinction that was previously only
declared (§5.6); and — resolving the two questions the Generator pass explicitly deferred here
(§4.7/§4.9) plus one question carried over since an earlier revision — category-attributable
resampling and exhaustion-vs-miscalibration diagnostics are both now answered purely via the report's
existing category/code tags, and time-varying (post-event) connectivity checking is now explicitly
excluded from the Validator's scope as simulation execution (§5.8/§5.9).

Unchanged, still frozen from earlier revisions: **Scenario Generation remains independent of Human
Behaviour** (§8) — the Generator assigns an opaque `behaviour_profile_id` string per occupant and
samples nothing else behavioral, and the Validator checks its proportions only, never its semantics
(§5.3, reconfirmed this pass); the Generator's input/output/seed/pipeline contract (§4); the
Definition's frozen schema (§3).

Still **proposal, open for review, not touched by this pass**: Scenario Events' simulator-integration
seam (§6, unchanged), Fire Profiles' schema finalization (§9, unchanged), and the Difficulty scoring
formula (§10, unchanged) — none of these were in scope for a Validator-only pass. Nothing in this
document has been implemented; no existing package is modified by this revision.

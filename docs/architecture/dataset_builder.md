# Dataset Builder — Architecture Proposal

Status: **proposal, open for review**. No code changes accompany this document — architecture only,
per this pass's explicit instruction. This is the third document in this lineage: `docs/architecture/
simulation_runtime.md` (frozen, approved) designed the orchestrator that produces a completed
simulation run; `docs/architecture/dataset_intent.md` (frozen, approved) designed how that run's
outcome gets labeled. This document designs the package that sits after both — `dataset_builder/` —
and turns a completed run's artifacts into files an ML/RL training process can actually load.

## 1. Purpose

`dataset_builder/` transforms already-produced simulation artifacts — a `Scenario`, its
`ScenarioMetadata`, the `TickResult` history a `SimulationRuntime` run produced (including whichever
ticks carried a `BuildingObservation`), and a `ScenarioOutcomeLabels` record — into exportable dataset
files: CSV, Parquet, RL episode files, supervised-learning example files, analytics summary files. It
is **purely a data transformation and serialization layer** — it runs no simulation, generates no
scenarios, perceives nothing, and trains nothing.

The user brief states a constraint that looks, at first read, self-contradictory: consume `Scenario`,
`BuildingObservation` history, tick history, `ScenarioOutcomeLabels`, and `ScenarioMetadata`, while
remaining **completely independent** of Scenario Generation, Simulation, Perception, RL, and Computer
Vision — the very packages that define those types. §3 resolves this precisely; it is the single most
important design decision in this document and everything else follows from it.

## 2. Grounding in existing code

Confirmed this pass, directly from source, including one structural fact that turns out to be
load-bearing:

- **`TickResult`** (`simulation_runtime/result.py`, frozen, approved): `time: float`, `fired_events:
  Tuple[ScenarioEvent, ...]`, `hazard_snapshot: HazardSnapshot`, `occupancy_snapshot: OccupancySnapshot`,
  `decision: DecisionRecommendation`, `observation: Optional[BuildingObservation]`.
  `SimulationRuntime.run()` returns `Tuple[TickResult, ...]` — this **is** "tick history."
- **`BuildingObservation`** (`perception/models/building_observation.py`, frozen): `node_observations:
  Mapping[str, ObservedNodeState]`, `occupancy_observations: Mapping[str, ObservedOccupancy]`,
  `edge_observations: Mapping[str, ObservedEdgeState]`, `system_status: PerceptionSystemStatus`. Present
  on a `TickResult` only when a `PerceptionProvider` was configured for that run (`docs/architecture/
  simulation_runtime.md` §13) — "`BuildingObservation` history" is therefore inherently **sparse**: the
  subsequence of `TickResult`s whose `.observation` is not `None`.
- **`HazardSnapshot`/`OccupancySnapshot`/`DecisionRecommendation`/`BuildingObservation` have no
  `to_dict()`/`from_dict()` anywhere** — confirmed by reading every one of them this pass, and by a
  repository-wide search for `to_dict` across `hazard/`, `occupancy/`, `perception/`, `ai_decision/`,
  `simulation_runtime/` (zero matches). Unlike `Scenario`/`ScenarioMetadata`/`ScenarioEvent`
  (`scenario/`, all serializable), these are **runtime-only value objects** — designed for one
  in-process `SimulationRuntime` run, never for persistence. This is the concrete fact behind §3's
  finding: there is no existing seam that turns a `TickResult` into plain data. One has to be designed
  here, in this document, because nothing upstream already provides it.
- **`ScenarioOutcomeLabels`** (designed, not yet implemented, `docs/architecture/dataset_intent.md` §10):
  `outcome_id`, `scenario_id`, `recorded_at`, `evaluation_time`, `total/reachable/unreachable/evacuated/
  trapped_occupants`, `rescue_required`, `fatalities: Optional[int]`, `building_cleared: bool`,
  `max_node_congestion`/`max_edge_congestion: Optional[CongestionPeak]`, `worst_hazard_zone:
  Optional[WorstHazardZone]`, `simulation_end_reason: SimulationEndReason`, `extra`. Already a plain,
  flat-ish value type (§10.1 of that document: deliberately not merged onto `Scenario`).
- **`ScenarioMetadata`/`Scenario`** (`scenario/metadata.py`, `scenario/scenario.py`, frozen): both
  already have `to_dict()`. `scenario_storage/catalog.py::catalog_row_for()` already demonstrates the
  exact "denormalize a nested domain object into flat, primitive-typed fields for a tabular row"
  transformation this document needs to perform for `TickResult`/`BuildingObservation` too — reused as
  the precedent, not reinvented (§5).
- **`perception_layer_review_2.md` §7.2/§7.3** (`ObservationEncoder`, frozen): already established the
  precedent this document leans on most heavily — "a separate, downstream component whose only
  responsibility is translating an already-complete `BuildingObservation` into whatever input format a
  *specific* AI model requires... not part of the Perception Layer," with "a fixed node/edge ordering...
  fixed once at encoder-construction time" for array-shaped output. `dataset_builder/`'s tabular exports
  need the identical ordering discipline (§9); its RL Episode shape is the training-*dataset* analog of
  what `ObservationEncoder` does for one live observation (§8.3).

## 3. The most consequential finding: independence requires a boundary Intermediate Representation

**Reconciling "consume `BuildingObservation` history" with "independent of Perception."** A package
cannot read a field off a `BuildingObservation` instance without, somewhere, a line of code that knows
`BuildingObservation`'s shape — but "somewhere" does not have to be *inside* `dataset_builder/`. The
resolution used throughout this codebase for an analogous problem — `scenario_definition/
engineering_constraints.py` defines its **own** `DoorState`/`StairAvailability`/`PresenceState`/
`DeviceAvailability` enums, textually identical in spirit to `scenario/engineering_state.py`'s, but a
**separate Python type with no import relationship to it** — is applied here at package scale, not just
enum scale:

**`dataset_builder/` defines its own, plain, primitive-typed Intermediate Representation (IR) — §5 —
and consumes *only* the IR, never a live `Scenario`/`TickResult`/`BuildingObservation`/
`ScenarioOutcomeLabels` instance.** A separate, explicitly-named **Ingestion Adapter** (§6) — which does
import `scenario`, `simulation_runtime`, `perception`, `scenario_outcome` — is responsible for reading
those real objects and producing IR instances. The adapter is designed by this document (its field
mapping is a full, precise specification, §6) but is **not part of `dataset_builder/`** and is not
implemented here either (this remains an architecture-only pass).

This makes `dataset_builder/`'s independence a **structural, import-level fact**, not a promise kept by
convention: `dataset_builder/` has **zero** import-time dependency on any of the five forbidden domains
(§14), full stop — stricter than every other package designed this session, each of which had at least
one legitimate one-way downstream import (`behaviour_profile_resolver → scenario_runner`,
`scenario_event_executor → scenario_runner`, `simulation_runtime → six different packages`).
`dataset_builder/` is the first genuine **leaf** in this whole architecture: nothing points to it that
it needs to know exists.

## 4. Design principles

1. **Pure transformation, zero domain imports** — §3, §14.
2. **The IR is the entire contract.** `dataset_builder/`'s public surface is "what shape of plain data
   goes in" (§5) and "what dataset shapes/formats can come out" (§8/§9) — nothing else. A caller (the
   Ingestion Adapter, or the future Dataset Generation Pipeline `dataset_intent.md` §17 already
   anticipates) is responsible for getting real domain data into that shape.
3. **Deterministic, side-effect-free reshaping.** Same IR in ⇒ same dataset bytes out, always. No
   `random` import (mirrors every prior package's "no randomness where none is needed" rule), no
   `datetime.now()` inside export logic (timestamps are carried through from the IR, never invented
   here), no network I/O.
4. **Two independent axes, not one flat list of five formats.** §7 explains why "CSV / Parquet / RL
   Episodes / Supervised Learning / Analytics" is not a single enumeration — CSV and Parquet are
   *serialization formats*; RL Episodes, Supervised Learning, and Analytics are *dataset shapes*
   (what the rows/records **mean**). Treating them as one flat list would force a combinatorial
   explosion of exporters (5 × formats) or silently bake one arbitrary shape/format pairing in as "the"
   implementation; keeping the axes separate is what makes "support future extensions" (the brief's own
   closing requirement) actually cheap (§12).
5. **Additive-only extension, matching `ScenarioMetadata.extra`/`schema_version`'s own established
   convention.** Every IR record carries an `extra: Mapping[str, Any]` bag; new dataset shapes and
   formats are new registry entries (§10), never edits to an existing shape/format's own logic.

## 5. The Intermediate Representation (IR)

Every field below is a plain Python primitive, or a `Mapping`/`Tuple` of them — no enum, dataclass, or
type from `scenario`, `simulation_runtime`, `hazard`, `occupancy`, `perception`, or `ai_decision`
appears anywhere in this section. Node/edge/zone ids remain plain `str` keys (the same "opaque
identifier, never interpreted" convention `behaviour_profile_id` already established) — `dataset_builder/`
never resolves what a ZONE-shaped id "means," it only carries it.

```
@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    definition_id: str
    definition_content_hash: str
    generation_version: str
    seed: int
    created_at: str
    fire_profile: str
    fire_ignition_zone_id: str
    occupant_count: int
    difficulty: Optional[float]
    extra: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FiredEventRecord:
    event_id: str
    target_type: str
    target_id: str
    event_type: str
    time: float

@dataclass(frozen=True)
class ObservationRecord:
    time: float
    node_observed: Mapping[str, bool]
    node_alarm_active: Mapping[str, Optional[bool]]
    node_estimated_severity: Mapping[str, Optional[str]]     # PerceptionSeverity.name, plain string
    occupancy_estimated_count: Mapping[str, Optional[float]]
    edge_blocked_estimate: Mapping[str, Optional[bool]]
    extra: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TickRecord:
    time: float
    fired_events: Tuple[FiredEventRecord, ...]
    node_hazard_scores: Mapping[str, float]
    node_hazard_severities: Mapping[str, str]                # HazardSeverity.name, plain string
    edge_traversable: Mapping[str, Optional[bool]]
    node_occupancy_counts: Mapping[str, Optional[float]]
    unsafe_zone_ids: Tuple[str, ...]
    blocked_edge_ids: Tuple[str, ...]
    priority_zone_ids: Tuple[str, ...]                        # rank order preserved as tuple order
    observation: Optional[ObservationRecord] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OutcomeRecord:
    outcome_id: str
    evaluation_time: float
    total_occupants: int
    reachable_occupants: int
    unreachable_occupants: int
    evacuated_occupants: int
    trapped_occupants: int
    rescue_required: int
    fatalities: Optional[int]
    building_cleared: bool
    max_node_congestion_id: Optional[str]
    max_node_congestion_count: Optional[int]
    max_edge_congestion_id: Optional[str]
    max_edge_congestion_count: Optional[int]
    worst_hazard_zone_id: Optional[str]
    worst_hazard_score: Optional[float]
    worst_hazard_time: Optional[float]
    simulation_end_reason: str                                # SimulationEndReason.name, plain string
    extra: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SimulationRunRecord:
    run_id: str
    scenario: ScenarioRecord
    ticks: Tuple[TickRecord, ...]
    outcome: OutcomeRecord
```

**`SimulationRunRecord` is the one type every dataset shape (§8) and this package's own public API
consume — always as `Sequence[SimulationRunRecord]`, never a single one**, since a *dataset* is
inherently a collection of runs (possibly one, but never structurally assumed to be exactly one).

**`run_id` here is the same identifier as `ScenarioOutcomeLabels.outcome_id`** (`dataset_intent.md`
§10.1) — carried through, not re-minted; `dataset_builder/` never invents a new identity scheme for a
run it did not produce.

## 6. Populating the IR — the Ingestion Adapter (specification, not this package)

**Lives outside `dataset_builder/`** — in the future Dataset Generation Pipeline (`dataset_intent.md`
§17, the one place already permitted to know about every upstream package end to end), or a small,
dedicated `dataset_ingestion/` companion package if a tighter separation is preferred (§16, left open).
Either placement is a caller of `dataset_builder/`, never a dependency *of* it.

Exact field mapping this document specifies (so a future implementation phase has a precise contract to
build against, not just "convert somehow"):

| IR field | Source |
|---|---|
| `ScenarioRecord.*` | `Scenario`/`ScenarioMetadata` fields, verbatim — same set `scenario_storage/catalog.py::catalog_row_for()` already denormalizes, reused rather than re-derived |
| `TickRecord.time` | `TickResult.time` |
| `TickRecord.fired_events[*]` | `TickResult.fired_events[*]` — `ScenarioEvent.event_id/target_type/target_id/event_type/time`, verbatim |
| `TickRecord.node_hazard_scores[node_id]` | `TickResult.hazard_snapshot.node_states[node_id].hazard_score` — every id present in `hazard_snapshot.node_states`, never a synthesized id |
| `TickRecord.node_hazard_severities[node_id]` | `HazardSeverity.from_score(...).name` — reuses the existing, single classification authority (`hazard/severity.py`), never re-derived with new cutoffs |
| `TickRecord.edge_traversable[edge_id]` | `TickResult.hazard_snapshot.edge_states[edge_id].traversable` |
| `TickRecord.node_occupancy_counts[node_id]` | `TickResult.occupancy_snapshot.observations[node_id].occupant_count` |
| `TickRecord.unsafe_zone_ids` | `TickResult.decision.unsafe_zone_ids`, verbatim |
| `TickRecord.blocked_edge_ids` | `[route.edge_id for route in TickResult.decision.blocked_routes]` |
| `TickRecord.priority_zone_ids` | `[entry.zone_id for entry in TickResult.decision.priority_evacuation_order]`, rank order preserved |
| `TickRecord.observation` | `None` if `TickResult.observation is None`; else per `ObservationRecord` mapping below |
| `ObservationRecord.node_observed[node_id]` | `observation.node_observations[node_id].observation_state == ObservationState.OBSERVED` |
| `ObservationRecord.node_alarm_active[node_id]` | `.alarm_active` |
| `ObservationRecord.node_estimated_severity[node_id]` | `.estimated_severity.name if not None else None` |
| `ObservationRecord.occupancy_estimated_count[node_id]` | `observation.occupancy_observations[node_id].estimated_count` |
| `ObservationRecord.edge_blocked_estimate[edge_id]` | `observation.edge_observations[edge_id].blocked_estimate` |
| `OutcomeRecord.*` | `ScenarioOutcomeLabels` fields, verbatim (nested `CongestionPeak`/`WorstHazardZone` flattened to the `*_id`/`*_count`/`*_score`/`*_time` fields shown in §5) |
| `SimulationRunRecord.run_id` | `ScenarioOutcomeLabels.outcome_id` |

**Only ids actually present in a given tick's source mapping appear in that `TickRecord`'s mappings** —
an id absent from `hazard_snapshot.node_states` this tick is simply absent from `node_hazard_scores`
too, never backfilled with a fabricated `0.0`/`None` entry at ingestion time. (§9 covers how tabular
export handles that absence — differently, and later, at the export boundary, not here.)

## 7. Two independent axes: Dataset Shape × Serialization Format

**Dataset Shape** answers "what does one exported unit *mean*" — a row of aggregate statistics
(Analytics), a labeled (features, target) training pair (Supervised Learning), or a full trajectory
(RL Episode). **Serialization Format** answers "what bytes does that unit become on disk" — CSV,
Parquet, JSON/JSONL. A `DatasetShape` declares which `DatasetFormat`s it can honestly be written as
(§10) — some pairings are structurally awkward (an RL Episode's variable-length nested sequence does
not flatten cleanly into CSV columns) and this document does not pretend otherwise.

## 8. Dataset Shapes

### 8.1 Analytics

One row per **run** (default) or, when tick-level granularity is requested, one row per **(run, tick)**
pair with every `ScenarioRecord`/`OutcomeRecord` field denormalized onto every row (the exact
"CSV/catalog is a flat, denormalized summary" convention `scenario_storage`/`scenario_outcome_storage`
already use, §16 of `dataset_intent.md`). Purpose: "what fraction of `Rescue Training` runs had
`rescue_required > 0`," "how did `max_node_congestion` vary across a batch" — aggregate, cross-run
questions, not per-example ML training data.

- Run-level: `AnalyticsGranularity.RUN` → one row per `SimulationRunRecord`, from `scenario`+`outcome`
  fields only (no tick detail).
- Tick-level: `AnalyticsGranularity.TICK` → one row per tick, `scenario`+`outcome` fields repeated on
  every row, plus every scalar `TickRecord` field (`node_hazard_scores`/etc. flattened per §9).

### 8.2 Supervised Learning

One record per **(features, label)** pair — but *which* fields are features and which are the label is
an ML-task decision this package cannot make on a caller's behalf (a `TickRecord` could support
predicting `unsafe_zone_ids` from hazard/occupancy fields, or predicting `evacuated_occupants` from
early-tick conditions, or dozens of other framings). **Parametrized, not hardcoded**: the caller supplies
a `feature_extractor(TickRecord, ScenarioRecord) -> Mapping[str, float]` and a
`label_extractor(TickRecord, OutcomeRecord) -> Any`, and `dataset_builder/` applies them uniformly
across every tick of every run, producing one flat `SupervisedExample(features, label, provenance)`
per tick (`provenance` carries `run_id`/`time` for traceability back to the source tick — never silently
dropped). This mirrors `ai_decision.priority.EvacuationPriorityRule`/`hazard_evolution.
HazardMergeStrategy`'s own "the policy is injected, this module only applies it uniformly" pattern —
`dataset_builder/` owns the *iteration*, never the *feature engineering*.

### 8.3 RL Episodes

One **episode** per `SimulationRunRecord` — an ordered sequence of framework-agnostic `EpisodeStep`s,
each built directly from one `TickRecord` (`time`, the same flattened hazard/occupancy/observation
fields §5 already defines, `unsafe_zone_ids`, `is_terminal: bool` — `True` only for the run's last
tick), plus the run's `OutcomeRecord` attached once at the episode level (a terminal summary, not
repeated per step). **Explicitly not designed here**: reward values, action spaces, or any Gymnasium
(or other framework) shape — `dataset_builder/` produces the same kind of plain, framework-agnostic
record `BuildingObservation` itself is (`perception_layer_review_2.md`'s own "consumer-agnostic, not
RL-shaped" principle, §7.1 of that document, applied here to a whole *episode* instead of one
observation). Turning an `Episode` into actual reward-labeled, action-spaced training transitions is
exactly the same kind of downstream, out-of-package translation `ObservationEncoder` already performs
for a single `BuildingObservation` — a future, separate component (outside both `dataset_builder/` and
`perception/`), not designed further here.

## 9. Serialization Formats

### 9.1 CSV and Parquet — tabular formats require a stable, sorted id ordering

`TickRecord.node_hazard_scores`/`node_occupancy_counts`/etc. are `Mapping[str, ...]` — not
representable as a single tabular cell. Exporting to CSV/Parquet means **flattening each per-id mapping
into one column per id** (e.g. `hazard_score__zone-1`, `hazard_score__zone-2`, ...). This requires a
single, dataset-wide (not per-run, not per-tick) ordering of every id ever encountered, computed once,
up front, over the **whole `Sequence[SimulationRunRecord]`** being exported — **the exact same
discipline `ObservationEncoder` already commits to** (`perception_layer_review_2.md` §7.3: "a fixed
node/edge ordering... fixed once at encoder-construction time"), reused here rather than re-derived: the
column set is `sorted(union of every node_id/edge_id seen across every tick of every run in this
export)`, fixed once per export call. **An id absent from a specific tick's own mapping becomes an
empty/null cell for that row — never a fabricated `0.0`** (the same "never backfill absence" rule §6
already applies at ingestion time, reapplied here at the export boundary for the same reason).

Parquet's native support for nested/list-typed columns could, in principle, avoid this flattening
entirely for a Parquet-specific export — **not assumed or required by this design**; treated identically
to CSV here, with native-nested-column support flagged as a possible future optimization (§16), not a
requirement.

### 9.2 JSON / JSONL

No flattening needed — nested `Mapping`/`Tuple` fields serialize directly. The natural pairing for
**RL Episodes** (variable-length, deeply nested — awkward as flat columns) and for any Supervised
Learning export where a caller's `feature_extractor` returns nested structure rather than flat scalars.
One JSON file per export, or one JSON object per line (JSONL) for large episode/example counts where
streaming a growing file matters — both are the same underlying serialization, differing only in
framing.

### 9.3 Reuse of `serialization/`

`serialization/json_writer.py`/`json_reader.py` are domain-agnostic infrastructure — already reused
independently by `scenario_storage` and (proposed) `scenario_outcome_storage`, with no coupling to any
of the five forbidden domains (§3). `dataset_builder/`'s JSON/JSONL writer reuses them rather than
reimplementing file-writing, the same "don't duplicate logic" instruction this whole session has
followed throughout. CSV uses the stdlib `csv` module directly (`scenario_storage/catalog.py`'s own
precedent). **Parquet requires a third-party library** (e.g. `pyarrow`) — the first genuinely new,
non-stdlib runtime dependency anywhere in this codebase's history; flagged explicitly into §16 rather
than assumed.

## 10. Shape × Format compatibility and dispatch

Reuses the `{key: handler}` pluggable-registry pattern already established twice this session
(`scenario_event_executor.EVENT_HANDLERS`; `behaviour_profile_resolver.DEFAULT_PROFILE_REGISTRY`):

```
class DatasetShape(Enum):
    ANALYTICS = auto()
    SUPERVISED_LEARNING = auto()
    RL_EPISODES = auto()

class DatasetFormat(Enum):
    CSV = auto()
    PARQUET = auto()
    JSON = auto()
    JSONL = auto()

SHAPE_BUILDERS: Mapping[DatasetShape, Callable] = {...}          # -> shape-specific plain records
FORMAT_WRITERS: Mapping[DatasetFormat, Callable] = {...}          # shape-specific records -> bytes on disk

SHAPE_FORMAT_COMPATIBILITY: Mapping[DatasetShape, FrozenSet[DatasetFormat]] = {
    DatasetShape.ANALYTICS: frozenset({DatasetFormat.CSV, DatasetFormat.PARQUET, DatasetFormat.JSON}),
    DatasetShape.SUPERVISED_LEARNING: frozenset({
        DatasetFormat.CSV, DatasetFormat.PARQUET, DatasetFormat.JSON, DatasetFormat.JSONL,
    }),
    DatasetShape.RL_EPISODES: frozenset({DatasetFormat.JSON, DatasetFormat.JSONL}),
}

class IncompatibleDatasetShapeFormatError(ValueError): ...   # mirrors UnsupportedEventTargetTypeError's
                                                               # "raise, never silently coerce" precedent
```

A `(shape, format)` pairing outside `SHAPE_FORMAT_COMPATIBILITY` raises immediately, before any export
work begins — never silently degrades RL Episodes into a lossy flattened CSV a caller didn't ask for.

## 11. Export entry point

```
def export_dataset(
    records: Sequence[SimulationRunRecord],
    shape: DatasetShape,
    format: DatasetFormat,
    destination,
    **shape_kwargs,                    # e.g. feature_extractor/label_extractor for SUPERVISED_LEARNING,
                                        # granularity for ANALYTICS
) -> ExportResult:
    ...

@dataclass(frozen=True)
class ExportResult:
    shape: DatasetShape
    format: DatasetFormat
    destination: str
    record_count: int                  # rows / episodes / examples written
    run_count: int                     # how many SimulationRunRecords contributed
```

Mirrors `PipelineResult`/`BatchPipelineResult`'s own "always return a structured result describing what
happened," not a bare success/failure boolean.

## 12. Extensibility — proof this never touches Simulation Runtime or Scenario Engine

- **A new `DatasetFormat`** (e.g. a future binary/columnar format) — one new module under a `formats/`
  directory, one new `FORMAT_WRITERS` entry, one `SHAPE_FORMAT_COMPATIBILITY` update. Touches nothing
  outside `dataset_builder/`.
- **A new `DatasetShape`** — one new module under a `shapes/` directory, one new `SHAPE_BUILDERS` entry.
  Consumes the existing IR (§5) as-is; if a genuinely new *kind* of information is needed that the IR
  doesn't carry, the IR itself grows an **additive** field (mirrors `ScenarioMetadata.extra`/`schema_
  version`'s own precedent, §6 of `perception_layer_review_2.md`) and the Ingestion Adapter (§6, outside
  this package) is updated to populate it — `simulation_runtime`/`scenario`/`perception`/
  `scenario_outcome` themselves need no change unless the *source* data genuinely does not exist yet
  upstream, which is a different (and much larger) kind of change than "add a dataset export."
- **A new upstream producer of ticks** (e.g. a future non-`SimulationRuntime` source of `TickResult`-
  shaped data) — only requires a new or adjusted Ingestion Adapter; `dataset_builder/` neither knows nor
  cares which package produced the IR it was handed.

## 13. Error handling and determinism

Fail-fast, consistent with every prior package this session: an unrecognized `DatasetShape`/
`DatasetFormat` key raises `LookupError`-family errors at dispatch (§10); an incompatible `(shape,
format)` pairing raises `IncompatibleDatasetShapeFormatError` before any file I/O begins; a
`feature_extractor`/`label_extractor` raising for a given `TickRecord` is **not** caught or skipped
silently — it propagates, since silently dropping a training example is a data-integrity problem no
less serious than silently dropping a `ScenarioEvent` would be elsewhere in this codebase. No partial
files are an implicit contract of this design — an implementation phase should decide whether a failed
export leaves a partial file on disk or cleans up after itself, flagged into §16 as an implementation
detail this document does not fix.

Every export is a pure function of `(records, shape, format, shape_kwargs)` — the same input always
produces byte-identical output (module the ordinary non-determinism of filesystem timestamps, which this
document does not claim to control). No field of the IR is ever recomputed or re-aggregated beyond what
a specific `DatasetShape` builder explicitly defines (§8) — `dataset_builder/` reshapes and serializes;
it does not re-derive statistics `scenario_outcome`/`simulation_runtime` already computed.

## 14. Dependency direction

**`dataset_builder/` may import:** Python standard library only (`csv`, `json`, `dataclasses`, `typing`,
`enum`, `pathlib`), plus `serialization/` (domain-agnostic infrastructure, §9.3), plus — only if Parquet
support is actually implemented — a third-party Parquet library, isolated to `formats/parquet_writer.py`
alone so the rest of the package has no hard dependency on it even then.

**`dataset_builder/` must not import**, under any circumstance: `scenario`, `scenario_definition`,
`scenario_generator`, `scenario_validator`, `scenario_pipeline`, `scenario_storage`, `dataset_intent`
(Scenario Generation); `scenario_runner`, `scenario_event_executor`, `behaviour_profile_resolver`,
`behavior`, `behavior_library`, `simulator`, `hazard`, `hazard_evolution`, `occupancy`,
`simulation_runtime` (Simulation); `perception`, `sensors` (Perception); `ai_decision` — grouped with
Simulation/Perception here since it is itself a Simulation-Runtime-coordinated consumer of Ground Truth,
not a data-transformation concern; `scenario_outcome`, `scenario_outcome_storage` (produces the very
labels this package only ever sees pre-flattened via the IR, §5); any RL framework (Gymnasium, or
similar) or Computer Vision library (OpenCV, or similar) — the two domains the brief names explicitly and
which, not coincidentally, are also the two domains that today have **zero** concrete implementation
anywhere in this codebase to accidentally import from. `models`, `navigation`, `pathfinding`: not needed
either — every id `dataset_builder/` ever sees is already an opaque `str` by the time it reaches the IR.
`random`: no randomness anywhere in this package (§4 point 3). `sandbox`, `designer`: no UI/authoring
concern here.

This is the **only** package in this entire session's architecture with zero permitted domain imports —
stated explicitly because every other package designed so far had at least one legitimate downstream
import; `dataset_builder/`'s defining property, per the user's own brief, is having none.

## 15. Suggested package structure

Not a commitment to implement (this milestone is architecture-only):

```
dataset_builder/
    __init__.py
    records.py              # the IR -- ScenarioRecord, FiredEventRecord, ObservationRecord,
                             # TickRecord, OutcomeRecord, SimulationRunRecord (§5)
    shapes/
        analytics.py         # AnalyticsRow, build_analytics_rows(records, granularity) (§8.1)
        supervised.py         # SupervisedExample, build_supervised_examples(records,
                               #   feature_extractor, label_extractor) (§8.2)
        rl_episode.py          # EpisodeStep, Episode, build_episodes(records) (§8.3)
    formats/
        csv_writer.py          # write_csv(shaped_records, destination) (§9.1)
        parquet_writer.py       # write_parquet(shaped_records, destination) (§9.1) -- isolated
                                 #   third-party dependency, §14
        json_writer.py           # write_json / write_jsonl(shaped_records, destination) (§9.2/§9.3)
    exporter.py               # DatasetShape, DatasetFormat, SHAPE_BUILDERS, FORMAT_WRITERS,
                               # SHAPE_FORMAT_COMPATIBILITY, export_dataset(), ExportResult (§10/§11)
```

The Ingestion Adapter (§6) is explicitly **not** listed here — it is not part of this package. Where it
lives (a dedicated `dataset_ingestion/` package vs. folded into the future Dataset Generation Pipeline)
is left open, §16.

## 16. Open questions for a future review

- **Ingestion Adapter package placement** (§6/§15) — dedicated `dataset_ingestion/` package vs. folded
  into the Dataset Generation Pipeline `dataset_intent.md` §17 already anticipates. Both are consistent
  with this document; neither is fixed here.
- **Parquet library choice** (§9.1/§9.3/§14) — `pyarrow` vs. an alternative, and whether it becomes a
  hard project dependency or an optional extra (import-guarded, degrading to "Parquet export
  unavailable" rather than failing the whole package's import) — a project-dependency-management
  decision outside this document's architectural scope.
- **Whether Parquet should use native nested/list columns instead of the CSV-style flattening this
  document specifies as the baseline** (§9.1) — a possible optimization, not required.
- **Partial-file behavior on a mid-export failure** (§13) — leave-as-is vs. clean up — not fixed here.
- **A `DatasetShape` combining Analytics-style aggregation *within* an RL Episode or Supervised example**
  (e.g. rolling statistics as an engineered feature) — plausible future extension; this document's
  `feature_extractor` seam (§8.2) already accommodates a caller doing this themselves without any
  `dataset_builder/` change, but a *built-in* rolling-statistics helper is not designed here.
- **Streaming/incremental export** for very large batches (writing rows as runs are ingested rather than
  materializing the full `Sequence[SimulationRunRecord]` first) — this document specifies the
  simpler, whole-sequence-in-memory contract; a streaming variant is a plausible, backward-compatible
  future addition (an additional entry point, not a change to `export_dataset()`'s existing contract).

## 17. Status

Proposal only. Nothing in this document has been implemented; `scenario`, `simulation_runtime`,
`perception`, `ai_decision`, and the (also still-proposed) `scenario_outcome`/`scenario_outcome_storage`
are all unaffected by it — `dataset_builder/` imports none of them (§14), by design, not by omission.
The design resolves the brief's apparent contradiction ("consume X" vs. "independent of X's package")
via a boundary Intermediate Representation (§3/§5) populated by an explicitly-external Ingestion Adapter
(§6) whose field-level mapping is fully specified here even though its code is not. Two independent
axes — Dataset Shape and Serialization Format (§7-§10) — keep "support future extensions" (the brief's
own closing requirement) a matter of adding registry entries, never touching `simulation_runtime` or any
Scenario Engine package (§12), which is the property this whole document exists to guarantee.

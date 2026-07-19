# Perception Layer — Architecture Review Round 2

Status: **APPROVED, with one final refinement (this revision). Frozen — no code changes accompany
this document.** Builds directly on `docs/architecture/perception_layer.md` ("Round 1") — this
document refines Round 1's single `GroundTruthPerceptionLayer` into named, testable collaborators,
answers the eight questions raised for this round, and incorporates the final architectural change
below: `BuildingObservation` is decoupled from Gymnasium/any RL framework, and consumed by a fan-out of
equal peers rather than a two-way split between "Rule-Based Engine" and "RL Agent." Round 1 is not
superseded, only made more specific.

## Final refinement: BuildingObservation is not RL-shaped, it is consumer-agnostic

Earlier drafts of §7 mapped `BuildingObservation` directly onto `gymnasium.spaces.Dict`, which
implicitly treated Gymnasium/RL as `BuildingObservation`'s primary or defining consumer. That coupling
is removed. `BuildingObservation` has **no knowledge of, and no dependency on, any AI or RL framework**
— it is a plain, domain-shaped, immutable value object, exactly like `HazardSnapshot`/
`OccupancySnapshot` are today. RL is one of several equally-weighted consumers, reached only through a
new, separate, downstream translation component — the `ObservationEncoder` (§7) — which is explicitly
**not part of the Perception Layer**. See §7 and the revised diagram in §8.1.

---

## 1. BuildingObservation design

`BuildingObservation` is the terminal object of the pipeline in §8 — everything upstream of it
(Providers, Sensor Observations, Occupancy Estimation, Sensor Fusion) exists to produce it; everything
downstream of it (the Rule-Based Decision Engine, Dataset Generation, the Firefighter Dashboard,
Logging/Replay, and — only through a future `ObservationEncoder`, §7 — an RL Agent) consumes it and
nothing else, and none of those consumers can see past it back to Ground Truth. Every
field below is stated as **Direct Measurement** (a device reports this near-directly), **Estimated**
(a model/algorithm infers this from raw device output, with irreducible uncertainty), or **Derived**
(computed deterministically from other fields already in this object or upstream in Perception —
never from Ground Truth).

No field below carries a continuous, exact quantity that only Ground Truth could know (`hazard_score`,
exact `smoke_level`, exact temperature, exact position, exact fire-growth state). That is deliberate —
see the "explicitly excluded" list at the end of this section.

### 1.1 Top-level

| Field | Type | Class | Why it exists |
|---|---|---|---|
| `observation_id` | `str` (uuid) | Derived | Stable identity independent of timestamp/content — same role `HazardSnapshot.snapshot_id`/`OccupancySnapshot.snapshot_id`/`DecisionRecommendation.recommendation_id` already play, for replay/logging/RL dataset indexing. |
| `timestamp` | `float` | Derived | Simulation time (or real deployment time) this observation represents — consumed by `PerceptionProvider.observation_at(time)`, mirroring `HazardProvider`/`OccupancyProvider`. |
| `schema_version` | `int` | Derived | See §6. |
| `node_observations` | `Mapping[str, ObservedNodeState]` | — | Per-node hazard-side perception, keyed by the same `Node.id` strings `HazardSnapshot`/`OccupancySnapshot` already use. |
| `occupancy_observations` | `Mapping[str, ObservedOccupancy]` | — | Per-node occupancy-side perception. Kept as a **separate mapping from `node_observations`**, not folded into one combined per-node object — mirrors Round 1's reasoning for keeping `HazardSnapshot`/`OccupancySnapshot` structurally unrelated: hazard and occupancy are produced by different sensor classes (detectors vs. cameras/IoT) on different estimation pipelines (§8), and a consumer that only cares about one should not need to reach through a combined type to get it. |
| `edge_observations` | `Mapping[str, ObservedEdgeState]` | — | Per-edge perceived traversability, keyed by `Edge.id`. See 1.4. |
| `system_status` | `PerceptionSystemStatus` | — | Global, non-per-node metadata about the perception pipeline itself. See 1.5. |

### 1.2 `ObservedNodeState` (hazard side, per node)

| Field | Type | Class | Producing sensor(s) | Why it exists |
|---|---|---|---|---|
| `observation_state` | enum: `UNOBSERVED`, `OBSERVED` | Derived | — | The tri-state flag central to §5 — whether *any* device currently covers this node at all. Computed from whether any Provider contributed a reading for this node this cycle, never inferred from the absence/presence of a specific value (see §5's ambiguity warning). |
| `alarm_active` | `Optional[bool]` | Direct Measurement | `SmokeDetectorProvider`, `HeatDetectorProvider` | The single most reliable real-world signal — a Fire Alarm Control Panel reports exactly this: zone-in-alarm, yes/no. `None` iff `observation_state == UNOBSERVED`. |
| `alarm_source_types` | `List[str]` (subset of `Detector.DETECTOR_TYPES`) | Direct Measurement | `SmokeDetectorProvider`, `HeatDetectorProvider`, future `FlameDetectorProvider`/`GasDetectorProvider` | Which *kind(s)* of detector tripped — a real FACP reports device type alongside zone, and smoke-vs-heat-vs-gas carries different tactical meaning for a firefighter or an RL policy. Empty list, not `None`, when observed-but-no-alarm (mirrors `BlockedRoute.stranded_zone_ids`'s "empty means none, not absent" convention from `ai_decision/recommendation.py`). |
| `visibility_estimate` | `Optional[str]` (coarse category: e.g. `"clear"`, `"reduced"`, `"heavy"`) | Estimated | `CameraProvider` (via a future vision model) | A camera-derived, deliberately coarse read on smoke logging in frame — real CV visibility estimation is categorical/confidence-banded long before it is a precise meter figure, and encoding it as a small enum (rather than a float) is itself part of "don't expose more precision than a real sensor could give." |
| `estimated_severity` | `Optional[HazardSeverity]` | Derived | — | Fuses `alarm_active`, `alarm_source_types`, and `visibility_estimate` into the one ordinal classification every other consumer already reads (`HazardSeverity`, reused as-is — no second severity scale invented). Computed by a Perception-side classifier (not `HazardSeverity.from_score()`, which is defined in terms of a continuous ground-truth score Perception must never see) — see 1.6. `None` iff `UNOBSERVED`. |
| `last_observed_time` | `Optional[float]` | Derived | — | Timestamp of the most recent contributing reading at this node. Lets a consumer distinguish "confirmed clear a moment ago" from "confirmed clear ten minutes ago, no update since" — real coverage is not always continuous (a panned camera, a comms dropout). `None` iff never observed. |

### 1.3 `ObservedOccupancy` (per node)

| Field | Type | Class | Producing sensor(s) | Why it exists |
|---|---|---|---|---|
| `estimated_count` | `Optional[float]` | Estimated | `CameraProvider` (people-counting model), future `FutureIoTProvider` (BLE/UWB/WiFi localization) | Reuses `OccupancyObservation.estimated_count`'s exact convention from Round 1 (`None` means no reading, never zero). This is the field the codebase's own `OccupancyObservation` docstring already anticipated feeding an "AI Decision Engine" from. |
| `confidence` | `Optional[float]` in `[0.0, 1.0]` | Estimated (metadata) | same as above | Per-observation confidence — see §4. `None` iff `estimated_count` is `None`. |

### 1.4 `ObservedEdgeState` (per edge)

| Field | Type | Class | Producing sensor(s) | Why it exists |
|---|---|---|---|---|
| `blocked_estimate` | `Optional[bool]` | Derived | Derived from the `ObservedNodeState` of the edge's endpoints, and/or a `CameraProvider` positioned to view a doorway/corridor directly | Ground Truth already has an edge-level override (`HazardEdgeState.traversable`) that `HazardAwareCostModel` consumes — a Rule-Based Engine or RL agent reasoning about routing needs the *perceived* equivalent, not the ground-truth one. Deliberately **not** a direct sensor reading — there is no dedicated "edge sensor" concept in the Building Model (Cameras/Detectors are placed at points, not bound to edges), so this is always inferred, never measured, and must be documented as such rather than dressed up as a direct reading. `None` iff neither endpoint is `OBSERVED` and no camera happens to cover the connector directly. |

### 1.5 `PerceptionSystemStatus` (global, not per-node)

| Field | Type | Class | Producing sensor(s) | Why it exists |
|---|---|---|---|---|
| `panel_communication_ok` | `bool` | Direct Measurement | `FutureIoTProvider` / FACP integration | Models a real, important failure mode: the perception pipeline itself can go dark (a downed panel, a lost comms link) independent of any single zone's conditions. A building where the whole detection system just dropped offline is a categorically different, arguably higher-priority situation than "no alarms currently active," and collapsing the two would hide that. Also directly relevant to an RL agent learning to distrust a currently-unreliable feed. |
| `active_camera_count` / `active_detector_count` | `int` | Derived | — | Coarse, run-level coverage summary — a convenience for dashboards/logging, computed by counting `active=True` devices that reported this cycle. Not a substitute for per-node `observation_state` (§5) — never used by the Rule-Based Engine/RL to infer any single zone's status. |

### 1.6 Explicitly excluded (must never appear on `BuildingObservation`)

Stated explicitly because it is easy to accidentally reintroduce Ground Truth under a plausible-looking
field name:

- Exact occupant positions or per-occupant identity (only aggregate, per-node `estimated_count`).
- `hazard_score`, `smoke_level`, `temperature`, or any other `HazardNodeState` field, in any form —
  including "just for debugging." If a debug view of Ground Truth is needed, it belongs to the
  Designer/Sandbox path (Round 1 §5), never anywhere near `BuildingObservation`.
- Fire growth curve state, ignition node/time, or anything else that only `FireGrowthModel`/
  `SmokePropagationModel` (as `HazardSource`s) know about themselves.
- A continuous `hazard_score`-shaped float anywhere — `estimated_severity` is the ordinal replacement,
  deliberately lossier than Ground Truth's own severity derivation.

---

## 2. Perception update model

### Option A — Simulator Tick → Perception Tick → BuildingObservation

Purely periodic, pull-based, lockstep with the simulation clock. Matches the shape every existing
Ground Truth producer already uses: `HazardEvolutionEngine.evolve(snapshot, time, dt)`,
`EvolutionBackedHazardProvider`'s dt-stepped, cached, forward-only timeline. Simple, fully
deterministic, trivially replayable.

Weakness: a detector crossing its alarm threshold one instant after a tick boundary is only "seen" at
the *next* tick — understates real alarm responsiveness (real FACPs react in seconds, not on a
simulation's dt grid), which matters for RL training realism and for firefighter-facing latency.

### Option B — Event-driven (camera update → observation update → fusion)

Push-based, reactive — matches how real devices actually behave (an RTSP frame arrives, a detector
interrupt fires) and matches `MultiAgentSimulation`'s own internal architecture (`heapq`-based discrete
event scheduling — the simulator is *already* event-driven internally, worth noting).

Weakness: purely event-driven fusion would make `PerceptionProvider` the one subsystem in this
codebase whose *external* contract is push/reactive rather than the pull-based
`snapshot_at(time)`/`observation_at(time)` shape every other provider interface shares
(`HazardProvider`, `OccupancyProvider`). `AIDecisionEngine`/a future RL training loop calling
`observation_at(time)` on demand is architecturally simpler and more testable than either polling an
event stream or being pushed into. Pure event-driven also complicates deterministic replay for RL
training, where reproducing an exact training episode matters.

### Option C — Hybrid: periodic fusion, event-triggered sensor updates

Recommended. The key realization: this is not really "periodic OR event-driven" as a single global
choice — it is "pull-based *external* contract, event-capable *internal* implementation," and those
are independent axes.

- **External contract stays exactly `PerceptionProvider.observation_at(time)`** — a pull, snapshot-at-
  time interface, identical in shape to `HazardProvider`/`OccupancyProvider`, so `AIDecisionEngine`/RL
  never need to know or care whether the observation they received was produced on a scheduled tick or
  early because a detector tripped.
- **Internal fusion cadence is periodic by default** (a `dt`-stepped grid, same shape as
  `EvolutionBackedHazardProvider`) — this is what keeps simulation runs deterministic and replayable.
- **Individual Provider readings can carry their own event timestamp** (the instant a detector crossed
  its threshold), and a materially significant reading (an alarm transitioning off→on) triggers an
  out-of-schedule fusion pass *ahead* of the next periodic tick, rather than waiting for it — modeling
  realistic alarm latency without abandoning the periodic grid for routine (non-alarm) updates.

**Why this generalizes to real-world deployment without changing the interface**: a real
`PerceptionProvider` implementation is free to maintain an internally event-driven buffer (a live
RTSP/detector-bus listener continuously updating internal state) while still answering
`observation_at(time)` calls from the identical RL agent loop used in simulation — pull-based on the
outside, event-capable on the inside. This is exactly what "the RL interface must remain identical
between simulation and deployment" (Round 1's stated requirement) requires: the *contract* cannot
depend on which side of the simulation/real-building boundary it's running on, only the
*implementation* behind it can.

---

## 3. Provider interfaces

These are **abstract interfaces only** — no implementation. They live one level above Round 1's
`sensors.provider.CameraProvider`/`DetectorProvider` (which stay exactly as designed: the raw,
device-agnostic, opaque-payload contract) and specialize per sensor *class*, returning typed —
but still per-device, still unfused, still not node-keyed — "Sensor Observation" objects. This is the
"Providers → Sensor Observations" stage of the §8 pipeline.

Each interface below is presumed to live in a new `perception/providers.py`.

### `CameraProvider` (extends `sensors.provider.CameraProvider`)

- **Required method**: `frame_observation_at(camera_id: str, time: float) -> CameraFrameObservation`
- **Expected output**: `CameraFrameObservation` — `camera_id`, `timestamp`,
  `estimated_occupant_count: Optional[float]`, `visibility_estimate: Optional[str]`,
  `confidence: Optional[float]`. Deliberately **device-scoped, not node-scoped** — resolving which
  `Node.id`(s) a given camera's frame corresponds to is the Occupancy Estimation stage's job (via
  `Camera.coverage_polygon()`), not this interface's.
- **Ownership**: one instance serves the whole fleet of cameras in a scenario/deployment, parameterized
  by `camera_id` per call — mirrors `HazardProvider`/`OccupancyProvider` being single, building-wide
  instances rather than one instance per Zone.
- **Lifetime**: constructed once per simulation run or once per real deployment session; lives exactly
  as long as the owning `PerceptionFusionEngine` (§8). A simulated implementation can be pure/stateless
  (same as `ManualHazardProvider`); a real implementation may hold live connections internally, but
  that is invisible at this interface.

### `SmokeDetectorProvider` (extends `sensors.provider.DetectorProvider`)

- **Required method**: `alarm_states_at(time: float) -> List[SmokeDetectorReading]`
- **Expected output**: `SmokeDetectorReading` — `detector_id`, `timestamp`, `alarm_active: bool`,
  `confidence: Optional[float]`. Never a continuous smoke value — a real smoke detector's native output
  *is* a threshold-crossing bit, so this is not a lossy simplification of a richer measurement, it is
  the honest shape of the device.
- **Ownership**: one instance for every `Detector` in the building with `detector_type == "Smoke"`.
- **Lifetime**: same pattern as `CameraProvider`.

### `HeatDetectorProvider` (extends `sensors.provider.DetectorProvider`)

- **Required method**: `alarm_states_at(time: float) -> List[HeatDetectorReading]`
- **Expected output**: `HeatDetectorReading` — `detector_id`, `timestamp`, `alarm_active: bool`,
  `confidence: Optional[float]`.
- **Scope note (revised)**: SynEvac V1's supported perception devices are CCTV Cameras, Smoke
  Detectors, and **Fixed Temperature** Heat Detectors only. Rate-of-Rise heat detectors are
  deliberately out of scope for V1 — `HeatDetectorReading` therefore carries no rate-of-rise field.
  An earlier draft of this section specified a `rate_of_rise_triggered: Optional[bool]` field on the
  reasoning that real heat detectors often report both trip conditions; that field has been removed
  rather than left as a dormant, always-`None` placeholder. If Rate-of-Rise support is added later, it
  should arrive as a new detector type (e.g. a `RateOfRiseDetectorProvider`) or an explicit, versioned
  extension of `HeatDetectorReading` — never as an unused field sitting in V1 waiting to be filled in.
- **Ownership / lifetime**: same pattern, scoped to `detector_type == "Heat"`.

### `FutureIoTProvider` (extends `sensors.provider.SensorProvider` directly — *not*
`CameraProvider`/`DetectorProvider`)

- **Required method**: `readings_at(time: float) -> List[SensorReading]` — deliberately left at Round
  1's opaque-payload level.
- **Expected output**: intentionally unspecified beyond `SensorReading`'s existing opaque `payload`.
  This is the one interface Round 2 should **not** over-specify: "Future IoT" exists precisely to name
  the category of device whose data shape is not yet known (BLE/UWB tags, WiFi localization,
  environmental/gas sensors, occupant wearables). Mirrors why `sensors/reading.py` keeps `payload`
  opaque in the first place. When a *specific* IoT integration is actually built, it gets its own new,
  sibling typed interface next to `CameraProvider`/`SmokeDetectorProvider`/`HeatDetectorProvider` at
  that point — `FutureIoTProvider` itself should never be stretched to fit a now-known shape.
- **Ownership**: potentially *plural* instances — unlike Camera/Smoke/Heat (each a single,
  well-understood protocol), "IoT" is a category, not one protocol, so a deployment may register
  several `FutureIoTProvider` instances side by side.
- **Lifetime**: same run-scoped pattern as the others.

### Where these are owned

A `PerceptionFusionEngine` (§8) holds injected lists of each provider type — `camera_providers`,
`smoke_detector_providers`, `heat_detector_providers`, `iot_providers` — constructed once and never
reconstructed mid-run, exactly mirroring `HazardEvolutionEngine.sources: List[HazardSource]`. No
provider is ever looked up ad hoc; all are registered up front.

---

## 4. Confidence model

**Recommendation: Option A — each observation owns its own confidence**, with one caveat below that
distinguishes this from a naive Option C.

Rejecting Option B (global-only): a single building-wide confidence figure destroys exactly the
information this milestone exists to preserve. A building can simultaneously have high confidence in
Zone A (camera + two detectors covering it) and zero confidence in Zone B (no coverage at all) — the
entire point of §5's UNOBSERVED distinction is that this must stay resolvable *per zone*. Collapsing it
to one number would make it impossible for the Rule-Based Engine or RL agent to reason about *which*
parts of the building it should distrust.

Choosing Option A: this already has a precedent in Round 1 — `OccupancyObservation.confidence` exists
today specifically as "optional provenance metadata a future AI Decision Engine can use to weigh
disagreeing sources against each other." Extending the same per-observation pattern to the hazard side
(`ObservedNodeState`, indirectly, via how `estimated_severity` is derived) is the consistent choice,
not a new one.

**The caveat**: `BuildingObservation`/`PerceptionSystemStatus` may expose a derived, read-only
*summary* (e.g., `active_camera_count`, or a computed "fraction of nodes currently OBSERVED") purely as
a convenience for dashboards/logging. This is **not** a second, independently-authored confidence value
sitting alongside the per-node ones — it is computed *from* them, the same "derived, never stored
redundantly" convention `HazardNodeState.severity`/`Node`'s engineering properties already enforce
elsewhere in this codebase. A naive Option C (both levels independently *own* confidence) would create
two sources of truth that can silently drift apart — the summary must always be a pure function of the
per-node values, never a value someone (or something) sets on its own.

---

## 5. Unknown vs. safe

**RL and the Rule-Based Engine must always receive UNKNOWN for an unmonitored zone, never `0`
occupants or a "clear" severity.** This is the single most important behavioral property of the entire
Perception Layer, and it must hold at every stage of the pipeline, not just at the final
`BuildingObservation` boundary.

**Why "0 occupants" would be actively dangerous, not just imprecise**:

- An RL agent trained against occupancy data where unmonitored zones silently read as `0` would learn a
  policy that systematically ignores blind spots — worse, in a deployment context it could learn that
  expanding sensor coverage provides no signal benefit, because "uncovered" already looks
  indistinguishable from "confirmed empty" in its training data. That bias would only surface in a real
  incident, in exactly the zone it was never trained to worry about.
- Feeding a false "clear" severity into `AIDecisionEngine._zone_recommendation`'s `is_unsafe` check
  would invert the actual risk ordering: an unmonitored zone during an active incident is arguably the
  *highest*-priority zone for firefighter verification, not the lowest — it is the one place the
  system has no idea what's happening.

**How UNKNOWN must propagate, concretely, at each layer**:

1. **Provider layer**: a provider simply does not emit a reading for a device/zone it has no data for —
   it never emits a manufactured zero/clear reading to "fill a gap."
2. **`ObservedOccupancy.estimated_count`**: stays `Optional[float] = None` for "no reading," exactly
   matching `OccupancyObservation`'s own documented convention ("`None` means no reading available,
   never zero people") — `BuildingObservation` must not introduce a different convention than the type
   Round 1 already established this pattern on.
3. **`ObservedNodeState.observation_state`**: must be an **explicit tri-state field**
   (`UNOBSERVED`/`OBSERVED`), never inferred from `alarm_active is None`. This is a concrete
   implementation guardrail: if "unknown" is only implied by a `None` in some other field, a future
   consumer can trivially misinterpret `alarm_active is None` as "observed and silent" instead of
   "never observed" — making the state its own explicit field removes that ambiguity at the type level
   rather than relying on every future reader to remember the convention.
4. **Fusion stage**: `PerceptionFusionEngine`/`OccupancyEstimationEngine` must never coalesce a missing
   contribution to a default value the way `HazardEvolutionEngine`'s carry-forward does for Ground
   Truth (§8) — Ground Truth's "absent means clear" convention is correct for a simulation that
   genuinely has no event at a node; Perception's "absent means never observed" is a different claim
   and must not reuse the same collapsing logic.
5. **Rule-Based Engine**: an UNKNOWN zone should surface as its own distinguishable status (e.g. "zone
   status unknown — verify"), not be silently folded into either `is_unsafe=True` or `is_unsafe=False`.
   Which exact policy `AIDecisionEngine` should apply to an UNKNOWN zone (treat as elevated-priority for
   verification vs. a third top-level bucket) is a decision-logic question, not an architecture
   question — flagged here as a consequence of this design, deliberately **not** decided in this
   document.
6. **`ObservationEncoder` output (§7)**: when a future `ObservationEncoder` flattens `BuildingObservation`
   into a Gymnasium space (or any other framework's tensor/graph shape), UNKNOWN must still be encoded
   as its own explicit channel (a per-zone observed mask), never as an in-band sentinel value inside the
   same channel as real measurements. A `-1` placed inside a `[0, N]` occupancy channel to mean "unknown"
   is exactly the failure mode this whole section argues against: it silently corrupts the value
   distribution the agent learns over, using the same collapsing-to-a-number mistake the rest of this
   section is designed to prevent, just moved one layer later, and moved into a component
   (`ObservationEncoder`) that sits outside the Perception Layer's own guarantees — which is exactly why
   this rule must be stated as a requirement on the encoder, not assumed to be inherited for free.

---

## 6. BuildingObservation versioning

**Recommendation: a single `schema_version: int` field. No free-form `version` string, no
`feature_flags` field.**

**Existing precedent, and why it doesn't directly transfer**: `Project.version` (`models/project.py`,
currently `"1.0"`) already solves a schema-compatibility problem in this codebase — but for
occasionally hand-edited/migrated *save files*, read through `serialization/json_reader.py`. None of
the existing transient, in-memory, regenerated-every-run snapshot types (`HazardSnapshot`,
`OccupancySnapshot`, `DecisionRecommendation`) carry any version field, because none of them are ever
persisted and reloaded across a code change — they're rebuilt fresh every run.

`BuildingObservation` is different in kind from both: it is transient like `HazardSnapshot` today, but
once RL training begins it *will* be logged, persisted, and replayed — and a trained policy's weights
encode an implicit contract with a specific field layout. A code change that adds, removes, or
repurposes a field must be distinguishable to anything consuming a dataset of previously-logged
observations. This is `Project.version`'s problem, recurring in a new artifact.

- **`schema_version: int`, not a string** — simpler to gate logic on (`if observation.schema_version
  < 3: ...`) than a semantic-version string, and nothing about this object is hand-authored the way a
  save file occasionally is, so `Project.version`'s string format isn't buying anything here.
- **No separate `version` field** — would duplicate `schema_version`'s job for no added information;
  the same "derived, never stored redundantly" reasoning applies here as everywhere else in this
  review.
- **No `feature_flags` field on the object itself** — which sensor types were active during a given run
  is a property of *how Perception was configured for that run*, not a property of any individual
  timestep's output; it belongs on the run configuration / `PerceptionFusionEngine`'s own setup, not
  replicated onto every `BuildingObservation` instance. A consumer that needs to know whether a
  particular field is meaningfully populated already has the answer, per field, via `UNOBSERVED`/`None`
  (§5) — no separate flag registry is needed to answer the same question twice.
- **Long-term compatibility discipline**: follow the additive-optional-field convention this codebase
  already documents on `DecisionContext` ("designed to grow additively... without breaking existing
  strategy implementations that don't look at those new fields"). New fields should default to
  `None`/`UNOBSERVED`-equivalent, so a policy trained against an older `schema_version` simply sees
  "unknown" for a field it predates, when replayed against newer logs. `schema_version` should only
  increment when a field is *removed* or *repurposed* (its meaning changes) — not for purely additive
  changes — so most evolution stays non-breaking by construction, and version bumps are reserved for
  the genuinely breaking case, exactly when a trained policy's actual input contract changes shape.

---

## 7. BuildingObservation consumers, and the future `ObservationEncoder`

### 7.1 BuildingObservation is a consumer-agnostic fan-out point, not an RL type

`BuildingObservation` is consumed by several independent, equally-weighted downstream components —
none privileged over any other, none requiring the others to exist:

- **Rule-Based Decision Engine** — today's `AIDecisionEngine`, reading `BuildingObservation` the same
  way it reads `HazardSnapshot`/`OccupancySnapshot` today (Round 1 §5).
- **Dataset Generation** — logging `BuildingObservation` instances (with their `observation_id`/
  `timestamp`/`schema_version`, §1.1/§6) to build a training corpus, independent of any specific model
  or framework that will later train against it.
- **Firefighter Dashboard** — a human-facing view rendering `BuildingObservation` directly (per-node
  `estimated_severity`, `alarm_active`, `estimated_count`, and critically, `observation_state ==
  UNOBSERVED` zones, §5) — a dashboard has exactly the same "never show 0/clear for an unmonitored
  zone" obligation as the Rule-Based Engine or an RL agent, so it belongs on this list as a first-class
  consumer, not an afterthought.
- **Logging / Replay** — persisting a timeline of `BuildingObservation`s for later deterministic replay
  (mirrors `EvolutionBackedHazardProvider`'s own timeline-caching pattern, one layer up), independent of
  training or decision-making.
- **`ObservationEncoder` (future)** — the *only* consumer that produces an RL-framework-specific shape,
  detailed in 7.2. Everything else on this list consumes `BuildingObservation` directly, in its native
  domain-shaped form.

None of these consumers changes what `BuildingObservation` contains. Adding a consumer never means
adding an RL-shaped field to `BuildingObservation` — it means adding a new component downstream of it.

### 7.2 `ObservationEncoder` — definition

**The `ObservationEncoder` is not part of the Perception Layer.** It is a separate, downstream
component whose only responsibility is translating an already-complete `BuildingObservation` into
whatever input format a *specific* AI model requires — a Gymnasium `Dict` space, a flat tensor, a graph
representation for a GNN, or anything else a future model architecture needs. It is closer in spirit to
a serializer than to a sensor or a fusion engine.

- **Input**: exactly one `BuildingObservation` (plus, where needed, static configuration fixed at
  construction time — e.g. a chosen `node_id`/`edge_id` ordering for array-shaped output, per 7.3).
  Never Ground Truth, never a raw `SensorReading`, never anything from `hazard`/`occupancy`/
  `hazard_evolution` — the encoder sits entirely on the consumer side of the `BuildingObservation`
  boundary, with the same access `AIDecisionEngine` or the Firefighter Dashboard has and no more.
- **Output**: model-specific — a `gymnasium.spaces.Dict`-shaped sample for a Gym-based RL agent, a
  `torch.Tensor`/`numpy.ndarray` for a different training stack, a graph object (nodes/edges/features)
  for a GNN-based model — the shape is whatever the *target model* dictates, not something
  `BuildingObservation` or the Perception Layer has any opinion about.
- **Ownership**: owned by whichever RL/ML training or inference component needs it — analogous to how
  `serialization/json_writer.py` is owned by the Designer's save/load flow, not by the Building Model
  itself. Multiple `ObservationEncoder` implementations can coexist (one per model architecture being
  experimented with) without any of them touching `BuildingObservation`'s definition.
- **Lifetime**: constructed once per training run/model, alongside whatever fixed configuration that
  run needs (e.g. the node ordering from §7.3) — same construction-time-fixed pattern
  `PerceptionFusionEngine`'s provider lists already use.

### 7.3 What moves into `ObservationEncoder` (previously drafted as part of §7)

Everything below was drafted in the prior revision of this section as if it were `BuildingObservation`'s
own shape. It remains architecturally correct, but all of it now belongs to `ObservationEncoder`, not to
`BuildingObservation` or anything in `perception/`:

- **Fixed node/edge ordering** for array-shaped output — the sorted `node_id`/`edge_id` list for a given
  Building/scenario, fixed once at encoder-construction time.
- **Observed-mask channel** (`MultiBinary(n_nodes)`, 1 = `OBSERVED`, 0 = `UNOBSERVED`) — the numeric
  encoding of §5's tri-state, produced by the encoder from `ObservedNodeState.observation_state`, kept
  as its own channel for the reasons §5 already states.
- **Severity, alarm, occupancy, and staleness channels** — derived by the encoder from
  `estimated_severity`, `alarm_active`/`alarm_source_types`, `estimated_count`, and
  `last_observed_time` respectively.
- **Edge and global sub-spaces** — derived by the encoder from `ObservedEdgeState`/`PerceptionSystemStatus`.

This section remains deliberately silent on reward shaping and action space design, per the review's
scope — those belong to the RL Agent itself, further downstream of `ObservationEncoder`.

### 7.4 Confirmation: BuildingObservation remains completely AI-framework agnostic

- `perception/` contains no import of, or reference to, Gymnasium, PyTorch, NumPy-as-a-training-dep, or
  any other ML/RL library. Its only dependencies are Ground Truth (`hazard`, `occupancy`, restricted per
  §8.2) and the Building Model's `Camera`/`Detector` geometry.
- `BuildingObservation` and every type it's built from (`ObservedNodeState`, `ObservedOccupancy`,
  `ObservedEdgeState`, `PerceptionSystemStatus`) use only plain Python types (`str`, `float`, `bool`,
  `Optional`, `Mapping`, `List`, the existing `HazardSeverity` enum) — the same primitive vocabulary
  `HazardSnapshot`/`OccupancySnapshot` already use, chosen for the same reason: any consumer, RL-based or
  not, can read it with zero framework dependency.
- The one and only place Gymnasium (or any other framework) may be imported is inside a specific
  `ObservationEncoder` implementation, which lives outside `perception/` entirely (e.g. a future
  `rl/` or `training/` package) — never inside `perception/`, never inside `ai_decision/`. A
  dependency-direction test analogous to the ones in §8.2 should forbid `perception/` from importing any
  RL/ML framework, the same mechanism already used to keep every other layer boundary in this codebase
  real rather than aspirational.

---

## 8. Final architecture

### 8.1 Dependency diagram

Two arms feed the same downstream pipeline — this is what makes "Ground Truth can never bypass
Perception" a structural property rather than a convention to remember:

```
 SIMULATION ARM                              REAL-BUILDING ARM (future)
 ───────────────                             ───────────────────────────
 Simulation (fire_growth, smoke_propagation,   Real CCTV / Real Smoke &
 hazard_evolution, occupancy)                  Heat Detectors / Real IoT
        │                                              │
        ▼                                              │
 Ground Truth                                           │
 (HazardSnapshot, OccupancySnapshot)                    │
        │                                              │
        ▼                                              ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Providers                                                    │
 │  CameraProvider / SmokeDetectorProvider /                     │
 │  HeatDetectorProvider / FutureIoTProvider   (perception/)     │
 │  — the ONLY components allowed to read Ground Truth AND       │
 │    the ONLY components a real device integration replaces —   │
 │    everything below this line is identical on both arms.      │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 Sensor Observations
 (CameraFrameObservation, SmokeDetectorReading, HeatDetectorReading,
  raw SensorReading for FutureIoTProvider)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Occupancy Estimation             Sensor Fusion
 (OccupancyEstimationEngine)      (PerceptionFusionEngine +
        │                          PerceptionMergeStrategy,
        │                          mirrors HazardEvolutionEngine +
        │                          HazardMergeStrategy)
        │                              │
        └──────────────┬───────────────┘
                        ▼
              Building Observation
              (perception/observation.py — §1)
              AI-framework agnostic — plain Python types only (§7.4)
                        │
                        ▼
              PerceptionProvider.observation_at(time)
                        │
        ┌──────────┬──────────┬──────────┬───────────────┐
        ▼          ▼          ▼          ▼               ▼
   Rule-Based   Dataset   Firefighter  Logging /   ObservationEncoder (future)
   Decision     Genera-   Dashboard    Replay      (NOT part of Perception —
   Engine       tion                              lives outside perception/)
   (AIDecision-                                              │
   Engine,                                                   ▼
   unchanged)                                       Gymnasium Observation
                                                      (or tensor / graph rep.,
                                                       per target model)
                                                               │
                                                               ▼
                                                          RL Agent
```

All five consumers under `BuildingObservation` are peers — none is privileged, none depends on the
others existing. Only `ObservationEncoder` performs any RL-framework-specific translation, and it does
so strictly downstream of `BuildingObservation`, never inside `perception/` itself.

### 8.2 Verifying Ground Truth cannot bypass Perception

This is enforced the same way every other layer boundary in this codebase already is — by a
regex-based dependency-direction test, not by convention alone (see
`SensorsPackageDependencyDirectionTests`, `OccupancyPackageIndependenceTests`,
`AIDecisionPackageDependencyDirectionTests` in the existing test suite).

- **Only `perception/providers.py` (and Round 1's `sensors/replay.py`, for the simulated-replay
  bridge) may import `hazard`, `hazard_evolution`, or `occupancy`.** Every other module in `perception/`
  — `observation.py`, `fusion.py`, `occupancy_estimation.py`, `merge_strategy.py` — must be forbidden
  from importing them, exactly mirroring how `sensors/provider.py`'s own dependency test already
  forbids the *generic* raw layer from reaching into Hazard/Occupancy.
- **`ai_decision/` must stop being allowed to import `hazard`/`occupancy` directly.** Round 1 flagged
  that today's `AIDecisionPackageDependencyDirectionTests` *permits* this (only
  `hazard_evolution`/`sensors`/`simulator`/`behavior`/`models`/`designer` are currently forbidden) —
  this review makes that tightening concrete: once `AIDecisionEngine.decide()` takes a
  `BuildingObservation`, `hazard` and `occupancy` must move onto `ai_decision/`'s forbidden-import list,
  with `perception` added to its allowed set. This single test change is what makes "the Rule-Based
  Engine cannot see Ground Truth" a CI-enforced fact rather than a design intention.
- **A future RL package (and the `ObservationEncoder` within it) must never import `hazard`,
  `hazard_evolution`, or `occupancy`, only `perception`** — the same rule as `ai_decision/`, applied
  before any RL code exists, not retrofitted after. Its only legitimate input is a `BuildingObservation`
  already produced by `perception/`.
- **`perception/` must never import Gymnasium or any other RL/ML framework** (§7.4) — this is the
  converse of the rule above, and equally enforceable by the same regex-based mechanism: just as Ground
  Truth cannot leak downstream past Providers, RL-framework specifics cannot leak upstream into
  `BuildingObservation`'s own definition.
- **Dataset Generation, the Firefighter Dashboard, and Logging/Replay** are held to the same
  Ground-Truth-import restriction as `ai_decision/` and the future RL package — they consume
  `BuildingObservation` only, never `hazard`/`occupancy` directly, for the same reason: a dashboard or a
  logged dataset that quietly reached past Perception to Ground Truth would reintroduce exactly the leak
  this whole design exists to close, just through a different door.
- **Designer/Sandbox visualization is explicitly outside this constraint** (Round 1 §5) — it is allowed
  to keep reading Ground Truth directly, because it is human-facing debugging/authoring tooling for the
  *current, real* state of a scenario under design, not a path to any of `BuildingObservation`'s
  consumers. The dependency-direction tests above apply to `ai_decision/`, Dataset Generation, the
  Firefighter Dashboard, Logging/Replay, and the future RL package specifically, not to `designer/`.

With those import rules enforced, there is exactly one path from Ground Truth to any of
`BuildingObservation`'s consumers — through Providers, and only through Providers — and every consumer
in the §8.1 fan-out, RL included, is structurally incapable of receiving anything but a
`BuildingObservation`, or of leaking framework-specific shape back upstream into it.

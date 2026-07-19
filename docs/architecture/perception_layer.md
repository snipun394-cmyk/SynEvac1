# Perception Layer — Architecture Review

Status: **proposal, not implemented**. No code changes accompany this document.

## 1. The mandate, restated precisely

Pipeline today:

```
Simulation --> Ground Truth --> AIDecisionEngine
```

Pipeline required going forward:

```
Simulation --> Ground Truth --> Perception Layer --> Building Observation --> AIDecisionEngine / future RL Agent
```

`Building Observation` must be the *only* thing the decision-making seam ever sees. Ground Truth
(`HazardSnapshot`, `OccupancySnapshot`) must stop being visible past the Perception Layer's boundary —
not just "discouraged", but structurally prevented, the same way this codebase already prevents
`sensors/` from importing `hazard_evolution/` (see `tests/test_sensors.py`,
`SensorsPackageDependencyDirectionTests`). Perception is an abstraction layer, not a CV/YOLO/OpenCV
implementation — those are future concrete producers behind the interfaces described below.

## 2. What already exists (this is the important part)

This is not a greenfield milestone. Three earlier decisions already anticipated it:

**`sensors/` package** (`sensors/reading.py`, `sensors/provider.py`, `sensors/replay.py`) —
`SensorReading` is an intentionally opaque, protocol-agnostic payload. Its own docstring says the
quiet part out loud:

> "Turning a payload into a HazardContribution or an OccupancyObservation is a job for a future
> translation layer built on top of this contract, not for this class."

That "future translation layer" is what this milestone is asking for. `CameraProvider` /
`DetectorProvider` / `NullSensorProvider` already exist as the raw-data seam; nothing consumes them
in production yet (only tests). `SimulatedReplaySensorProvider` is the one deliberate exception —
comments call it out explicitly as "the one deliberate bridge" allowed to touch
`hazard_evolution`/`occupancy` directly, everything else in `sensors/` is enforced independent of
them by regex-based dependency tests.

**`occupancy/` package** (`occupancy/observation.py`, `occupancy/provider.py`, `occupancy/snapshot.py`) —
`OccupancyProvider.snapshot_at(time) -> OccupancySnapshot` mirrors `HazardProvider` exactly.
`ManualOccupancyProvider` is V1's only real implementation. **Nothing currently produces a ground-truth
`OccupancySnapshot` from `MultiAgentSimulationResult` or `SandboxManager`** — occupant positions exist
(`simulator/multi_agent_result.py: OccupantTimeline`, `sandbox/occupant.py: SandboxOccupant`), but no
adapter turns them into `OccupancySnapshot`. This is a gap independent of Perception, worth noting but
out of scope here.

**`ai_decision/engine.py`** — `AIDecisionEngine.decide(hazard_snapshot, occupancy_snapshot, time)` is
the exact seam the new pipeline must intercept. Today it is called only from tests, with hand-built
Ground Truth snapshots — it has no production call site yet, which makes this the ideal moment to
change its input shape before anything downstream depends on the current one.
`tests/test_ai_decision.py::AIDecisionPackageDependencyDirectionTests` currently *allows*
`ai_decision/` to depend on `hazard`/`occupancy` (only `hazard_evolution`, `sensors`, `simulator`,
`behavior`, `models`, `designer` are forbidden). That allow-list is exactly what needs to tighten.

**`models/camera.py` / `models/detector.py`** — pure design-time engineering objects: position,
rotation, `horizontal_fov`/`max_range` (Camera) or `coverage_radius` (Detector), `active` flag, and a
derived-never-stored `coverage_polygon()`/`coverage_circle()`. They carry **zero** runtime/perception
behavior today — they are geometry the Designer draws and serializes, nothing more. `Zone.contains()`
is the only existing "is this point inside this shape" helper in the codebase (used by
`SandboxManager._find_zone`), and is the right analogue for coverage-vs-node resolution.

**Conclusion**: the Perception Layer is not a new idea bolted on — it's the thing `sensors/` was
always going to need on top of it, arriving earlier than a real device protocol did. That changes the
build order (see §6).

## 3. Design conventions already established (Perception must follow these, not invent new ones)

Every existing subsystem in this codebase — `hazard/`, `hazard_evolution/`, `occupancy/`, `sensors/`,
`ai_decision/` — follows the same shape:

1. **One interface, raises `NotImplementedError`, plus a trivial Null/Manual default**
   (`HazardProvider`/`ManualHazardProvider`, `OccupancyProvider`/`ManualOccupancyProvider`,
   `SensorProvider`/`NullSensorProvider`, `HazardSource`/`NullHazardSource`,
   `DecisionEngine`/`AIDecisionEngine`).
2. **Producers return frozen, `MappingProxyType`-wrapped value objects**, keyed externally by
   `node_id`/`edge_id` — never stored on `Node`/`Edge` (which are rebuilt from the Building on every
   change and can't hold derived runtime state).
3. **Total accessors default absence to a defined, documented meaning** — `HazardSnapshot.node_state()`
   defaults absence to *clear*; `OccupancySnapshot.observation_at()` defaults absence to *no reading*.
   **Perception must break this convention on purpose** — see §4.2, this is the single most important
   design decision in this document.
4. **"Contribution" (sparse opinion) vs. "Snapshot" (definitive, point-in-time capture)** is a
   deliberate two-type split (`HazardContribution` vs. `HazardSnapshot`) so "no opinion" and "asserts
   clear" are never conflated. Perception has the same distinction to make.
5. **Dependency direction is enforced by tests that regex-scan source files**, not just by convention
   (`SensorsPackageDependencyDirectionTests`, `OccupancyPackageIndependenceTests`,
   `AIDecisionPackageDependencyDirectionTests`). A Perception package needs the same kind of test from
   day one.
6. **Structurally distinct concepts get structurally distinct types even when their shape overlaps** —
   `HazardSnapshot` and `OccupancySnapshot` deliberately share no base class, specifically so a future
   consumer combining them can't do so by accident through inheritance. `BuildingObservation` (§4.1)
   should follow the same rule relative to `HazardSnapshot`/`OccupancySnapshot`.

## 4. Proposed package: `perception/`

### 4.1 Core types

- **`perception/observation.py` — `BuildingObservation`**
  A new, frozen, `MappingProxyType`-wrapped, `snapshot_id`/`timestamp`-carrying value object — the
  same shape family as `HazardSnapshot`/`OccupancySnapshot`, but **not** a subclass or reuse of
  either. This is deliberate: it is the type-level guarantee that `AIDecisionEngine`/RL cannot receive
  Ground Truth by accident, the same way `HazardSnapshot` and `OccupancySnapshot` staying unrelated
  stops Hazard and Occupancy logic from accidentally cross-contaminating today.

  Holds per-node `ObservedNodeState` (perceived hazard indication) and `ObservedOccupancy` (perceived
  headcount), keyed by the same `node_id` strings `HazardSnapshot`/`OccupancySnapshot` already use.

- **`ObservedNodeState`** — deliberately *not* a copy of `HazardNodeState`. Real detectors and FACPs
  don't report a continuous `hazard_score`; they report discrete states (alarm / no alarm, or a
  detector-type-specific reading). Modeling this honestly means `ObservedNodeState` carries something
  closer to `alarm_active: Optional[bool]` plus a coarse `HazardSeverity`-shaped estimate, not a
  smuggled-through float. Reusing `HazardSeverity.from_score()`'s pattern of "one central classifier"
  is right; reusing `hazard_score` itself is not — it would let ground-truth precision leak through
  under a different name.

- **`ObservedOccupancy`** — closer to `OccupancyObservation` in shape (`estimated_count`,
  `confidence`), since headcount-from-CCTV is legitimately a continuous, uncertain estimate rather
  than a discrete alarm state.

### 4.2 The critical convention break: "absent" must mean "unobserved," never "clear"

`HazardSnapshot`/`OccupancySnapshot` both default a missing entry to "nothing wrong here" — correct
for Ground Truth, where the simulation genuinely has no opinion at an untouched node. It would be
**wrong and dangerous** for `BuildingObservation`: a node with no camera or detector coverage isn't
safe, it's *unknown*, and treating unknown as clear is exactly the failure mode a real building's
blind spots produce. `BuildingObservation`'s total accessor needs a third state — `UNOBSERVED` —
distinguishable from "observed and clear." This is the one place Perception must not copy the
"absent-means-default" pattern verbatim, precisely because copying it verbatim would silently defeat
the entire point of the milestone.

### 4.3 `PerceptionProvider` — the seam

```
perception/provider.py

class PerceptionProvider:
    def observation_at(self, time: float) -> BuildingObservation:
        raise NotImplementedError

class NullPerceptionProvider(PerceptionProvider):
    # returns a BuildingObservation that is UNOBSERVED everywhere
```

This is the one method `AIDecisionEngine`/future RL is allowed to depend on — mirrors
`HazardProvider.snapshot_at()`/`OccupancyProvider.snapshot_at()` exactly, and is the swap point
between "simulated Ground Truth behind Perception" and "real CCTV/detectors behind Perception" that
keeps the RL interface identical across both deployments, per the stated requirement.

### 4.4 `GroundTruthPerceptionLayer` — V1's one concrete implementation

The single class allowed to import **both** Ground Truth packages (`hazard`, `occupancy`) **and** the
Building Model's `Camera`/`Detector` geometry — the Perception-package equivalent of
`sensors/replay.py`'s documented role as "the one deliberate bridge." Everything else in `perception/`
stays independent of `models`/`designer`, mirroring how `sensors/provider.py` stays independent of
`hazard`/`occupancy`.

For each step it:
1. Reads the Building's placed `Camera`/`Detector` objects (`active=True` only).
2. Resolves which nodes/zones fall inside each device's `coverage_polygon()`/`coverage_circle()` —
   reusing that existing geometry rather than inventing new coverage math, and reusing
   `Zone.contains()`-style point-in-shape logic rather than a new one.
3. For covered nodes, delegates to a per-device-type **`ObservationModel`** strategy to turn the
   Ground Truth value at that node into an `ObservedNodeState`/`ObservedOccupancy` — this is the
   swappable seam (mirrors `CostModel`/`CapacityModel`/`HazardMergeStrategy`/`RouteChoiceStrategy`)
   where thresholding, false-negative rates, latency, or (much later) an actual CV/YOLO model plugs
   in, without `GroundTruthPerceptionLayer` itself changing.
4. Leaves every uncovered node `UNOBSERVED`.

This does **not** round-trip through `SensorReading`/`SensorProvider` for V1. Reasoning in §6.

## 5. Integration points — what changes, what deliberately doesn't

| Consumer | Today | After Perception Layer |
|---|---|---|
| `AIDecisionEngine.decide()` | Takes `(hazard_snapshot, occupancy_snapshot, time)` — Ground Truth directly | Takes `(observation: BuildingObservation, time)` — Perception-mediated only |
| Designer / Sandbox panels | Visualize Ground Truth directly | **Unchanged.** A human designer/operator debugging a scenario is *supposed* to see the real state — Perception gates the AI/RL path, not human tooling. |
| `behavior/context.py` (`DecisionContext.hazard_snapshot`) | Occupant behavior strategies read Ground Truth `HazardSnapshot` directly for WAIT/EVACUATE decisions | **Open question, not decided here** — see §7. An occupant sensing their own immediate surroundings is arguably a different problem than a building-wide sensor network's limited knowledge; do not conflate them without a deliberate decision. |
| `hazard_evolution/`, `occupancy/` | Produce Ground Truth | **Unchanged** — Perception consumes their output, never modifies how they produce it. |
| `sensors/` | Scaffolding, test-only | **Unchanged for V1** — reserved for when a real device protocol or sensor-shaped replay is wired in (§6). |

## 6. Relationship to `sensors/` — two valid paths, recommend starting with the direct one

The `sensors/` package's own design already points at two possible ways to build Perception:

**(a) Direct**: `GroundTruthPerceptionLayer` reads `HazardSnapshot`/`OccupancySnapshot` +
`Camera`/`Detector` geometry straight into `BuildingObservation`, no `SensorReading` in between.

**(b) Sensor-mediated**: Ground Truth first becomes `SensorReading` objects (a simulated camera
"emits" a reading), and a translation layer turns those opaque payloads into `BuildingObservation` —
the literal "future translation layer" `sensors/reading.py` already names.

Recommend **(a) for this milestone**. Coverage-geometry-based perception from Ground Truth doesn't
need an opaque payload round-trip to be honest about partial observability — the payload
opacity in `SensorReading` exists to decouple *raw device protocols* from interpretation, and there is
no real device protocol yet. Forcing path (b) now would add a layer of indirection with nothing on the
other side of it.

This is not a rejection of `sensors/` — design `PerceptionProvider` so a **second** implementation,
`SensorBackedPerceptionProvider`, can be added later that consumes real or replayed `SensorReading`s
via `CameraProvider`/`DetectorProvider` and produces the *same* `BuildingObservation` type. That is
exactly the swap `SimulatedReplaySensorProvider` already proves is safe one layer down (a
sensor-shaped source substituting for a simulated `HazardSource`/`OccupancyProvider` with zero engine
changes) — Perception should preserve the same substitutability one layer up.

## 7. Open questions (need your decision before implementation starts)

1. **Behavior layer's `DecisionContext.hazard_snapshot`** — should individual occupants' WAIT/EVACUATE
   decisions also be perception-limited (e.g., an occupant only reacts to hazard they could plausibly
   sense nearby), or is that a deliberately different problem from the building-wide sensor/RL
   pipeline and out of scope for this milestone?
2. **`BuildingObservation` granularity** — one merged snapshot across all device types (as designed
   above), or should Camera-derived and Detector-derived observations stay in separate typed
   sub-objects even within one `BuildingObservation`, so a future consumer can tell *which kind* of
   sensor produced a given reading?
3. **Migration of `AIDecisionEngine.decide()`'s signature** — hard cutover to
   `BuildingObservation`-only, or a transitional period where it accepts both shapes so the (currently
   test-only) call sites don't need simultaneous rewriting? Given there is no production call site
   yet, a hard cutover is likely cheaper and avoids a two-signature transitional API that would just
   need deleting shortly after — flagging so you can confirm rather than deciding unilaterally.
4. **Occupancy Ground Truth gap** — noted in §2: nothing currently produces a Ground Truth
   `OccupancySnapshot` from `MultiAgentSimulationResult`/`SandboxManager`. Perception can be built and
   tested against hand-authored `OccupancySnapshot`s in the interim, but real end-to-end perception of
   *simulated* occupants needs this gap closed first. Should that adapter be scoped into this
   milestone, or tracked separately?

## 8. Suggested build order (still planning — no implementation without separate approval)

1. `perception/` package skeleton: `BuildingObservation`, `ObservedNodeState`, `ObservedOccupancy`,
   `PerceptionProvider` + `NullPerceptionProvider`, plus a dependency-direction test mirroring
   `SensorsPackageDependencyDirectionTests`/`OccupancyPackageIndependenceTests`.
2. Coverage-resolution helpers built on `Camera.coverage_polygon()`/`Detector.coverage_circle()`
   against Building/Zone geometry.
3. `GroundTruthPerceptionLayer` with swappable `ObservationModel` strategies per device type
   (threshold-based detector alarms, partial/noisy camera counts).
4. Tests proving the §4.2 convention: uncovered nodes stay `UNOBSERVED`, never default to clear.
5. Only then, migrate `AIDecisionEngine.decide()` to `BuildingObservation`, and tighten
   `AIDecisionPackageDependencyDirectionTests`'s forbidden-import list to include `hazard`/`occupancy`
   directly (mirroring how `sensors/`'s own tests forbid its core contracts from reaching into
   downstream layers).
6. Future RL agent is built against `PerceptionProvider.observation_at()`/`BuildingObservation` from
   its first line of code — never against `HazardSnapshot`/`OccupancySnapshot` — so the simulation-time
   and real-deployment agents share one interface by construction, not by later discipline.

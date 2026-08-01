# Capacity Architecture Investigation V1

## Scope and constraints

This is an architectural investigation only. No SynEvac source file was modified — not the stair
model, not the capacity model, not any simulation code. One small, disclosed investigation script
(`scripts/capacity_architecture_isolated_edge_experiment.py`) was added to obtain one specific,
controlled piece of evidence (§4.2); it calls only already-existing, unmodified production code.

**Motivating evidence** (from the two prior Published Scenario Validation campaigns): a 10-story
NIST office building recreation ran ~3× slower than the real drill; an 18-story recreation ran
~16× slower; a falsification attempt traced the single largest bottleneck to a plain `Door`, not a
`Staircase`, refining the working hypothesis from "the stair model is wrong" to "the shared
admission-control/capacity architecture used by every traversable edge is the dominant factor."
This document tests that refined hypothesis by reading the actual implementation end to end.

---

## 1. The Complete Execution Path

Every class and function in the path below was read directly this session (file paths given so
each can be re-opened). The path is a **discrete-event simulation**: `MultiAgentSimulation`
(`simulator/coordinator.py`) owns one `heapq`-based priority queue of `(time, sequence, kind,
occupant_id, generation)` tuples and drains it via `run()` → `_process_next_event()` until empty.
There are exactly two event kinds: `TRY_ENTER_EDGE` and `ARRIVE_AT_NODE`.

```
Occupant wants to enter an edge
        |
        v
_handle_try_enter_edge(time, occupant)          [simulator/coordinator.py]
        |
        |-- capacity = self.capacity_model.capacity(edge)      [simulator/capacity.py]
        |-- current_on_edge = len(self._edge_occupancy.get(edge.id, ()))
        |
        +-- current_on_edge < capacity? ----------------------------+
        |                                                           |
       NO                                                          YES
        |                                                           |
        v                                                           v
Queue creation                                              _admit_onto_edge(time, occupant, edge)
_edge_queues.setdefault(edge.id, deque()).append(occupant)          |
_queue_join_time[occupant] = time                                   |
occupant.state = QUEUED                                             |
        |                                                           |
        | (occupant now waits -- no event is scheduled              |
        |  for them; they are re-activated only when                |
        |  another occupant's ARRIVE_AT_NODE on THIS                |
        |  edge pops them from the deque)                           |
        |                                                           v
        |                                              Admission (edge_occupancy.add,
        |                                              speed_factor via congestion_model,
        |                                              duration = distance / effective_speed,
        |                                              schedule ARRIVE_AT_NODE at time+duration)
        |                                                           |
        +<----------------------- (later, on a departure) ----------+
        |
        v
_handle_arrive_at_node(time, occupant)          [simulator/coordinator.py]
        |
        |-- edge_occupants.discard(occupant)              <- RELEASE (frees one capacity slot)
        |-- queue = self._edge_queues.get(edge.id)
        |-- if queue: next_id = queue.popleft(); schedule TRY_ENTER_EDGE(next_id) at `time`
        |-- self._add_node_occupant(to_node, occupant)
        |-- occupant.current_edge_index += 1
        |
        +-- more edges left in route? ---- NO --> occupant.state = ARRIVED (done)
        |
       YES
        |
        v
occupant.state = AT_NODE
schedule TRY_ENTER_EDGE(occupant) at `time`     --> back to the top, for the NEXT edge
```

### Classes and functions involved

| Component | File | Role |
|---|---|---|
| `MultiAgentSimulation` | `simulator/coordinator.py` | Owns the event heap, all per-edge/per-node occupancy and queue state, and every transition above. The *only* place admission/queueing decisions are made. |
| `CapacityModel` / `DefaultCapacityModel` / `StairCapacityModel` | `simulator/capacity.py` | Answers "how many occupants may be on this edge at once" — a pure function of the edge's own `width` (and, for stairs, `walking_distance`), called fresh on every `TRY_ENTER_EDGE`. |
| `CongestionModel` / `DefaultCongestionModel` / `StairAwareCongestionModel` | `simulator/congestion.py` | Answers "how much should this edge's current crowding slow this one occupant down," called once, at the moment of admission. |
| `Edge` | `navigation/edge.py` | A stateless, read-only view over a `Door`/`Exit`/`Staircase`; exposes `width`, `capacity` (Exit-only), `traversal_cost` (a fixed, pre-computed distance), and `traversable`. Carries no dynamic occupancy/queue state of its own. |
| `Occupant` / `OccupantState` | `simulator/occupant.py` | One occupant's mutable runtime record (`current_edge_index`, `state`) — `PENDING → AT_NODE → QUEUED → TRAVERSING → ARRIVED` (or `UNREACHABLE`/`STATIONARY`). |
| `OccupantTimelineStep` / `OccupantTimeline` / `MultiAgentSimulationResult` | `simulator/multi_agent_result.py` | The read-only, post-hoc record of what actually happened — `queue_wait_time`, `start_time`, `end_time` per hop; `peak_edge_occupancy`/`peak_node_occupancy` across the whole run. |

---

## 2. Per-Step Documentation

### 2.1 Capacity check — `_handle_try_enter_edge`

- **Purpose:** decide, at the instant an occupant wants to cross one specific edge, whether they may do so immediately.
- **Assumptions:** capacity is a single integer, valid for the *entire* edge, computed identically regardless of how long the edge physically is or how many people are already mid-transit at different points along it (an edge has no internal spatial structure — it is a black box between two nodes).
- **Mathematical model:** `admit ⟺ len(edge_occupancy[edge.id]) < capacity_model.capacity(edge)`. `DefaultCapacityModel.capacity(edge) = max(int(edge.width × 1.5), 1)` (or `edge.capacity` verbatim, for the one edge type — `Exit` — that carries an authored capacity field independent of width). `StairCapacityModel` narrows this further for `Stair` edges only: `max(int(width × 1.2 − ⌊vertical_travel_distance / 4.0⌋), 1)`.
- **Queue ownership:** none yet at this step — this step only *reads* `self._edge_occupancy`, a `dict[edge_id -> set[occupant_id]]` owned entirely by `MultiAgentSimulation`.
- **Interaction with congestion:** none at this step — congestion (`speed_factor`) is only computed *after* a successful admission (§2.3), never used to decide *whether* to admit.
- **Interaction with movement:** none yet — movement time is only computed on admission.

### 2.2 Queue creation

- **Purpose:** hold an occupant who cannot yet enter their next edge.
- **Assumptions:** the queue for edge `E` is a plain `collections.deque`, keyed *only* by `edge.id`, created lazily on first use (`self._edge_queues.setdefault(edge.id, deque())`). There is **no upper bound on queue length** (an occupant can always join, regardless of how many are already waiting) and **no queue exists anywhere except keyed by a single edge's own id** — there is no per-node queue, no per-stairwell queue, no building-wide queue.
- **Mathematical model:** FIFO — `deque.append()` / (later) `deque.popleft()`. Position in the deque, not any priority/fairness score, determines admission order.
- **Queue ownership:** `MultiAgentSimulation._edge_queues[edge.id]`, a private attribute of the coordinator. `Node`, `Edge`, `Door`, `Staircase` never hold or know about this state — fully consistent with the codebase's own stated design ("Node and Edge deliberately carry no blocked/smoke/fire/congestion fields... dynamic state instead belongs to whichever layer produces it").
- **Admission logic (from the queue's perspective):** an occupant sits in this deque with **no event scheduled for them at all** — they are inert until some *other* occupant's departure (`_handle_arrive_at_node` on the *same* edge) explicitly pops them out. If nobody currently on the edge ever leaves, a queued occupant would wait forever (never independently re-checked, never timed-out).
- **Interaction with congestion/movement:** none — a queued occupant accrues simulated time (`queue_wait_time = admission_time − join_time`, computed once they are eventually admitted) but experiences no congestion model and no movement while queued.

### 2.3 Admission — `_admit_onto_edge`

- **Purpose:** formally place an occupant onto an edge and schedule exactly when they will arrive at the far end.
- **Assumptions:** once admitted, an occupant's arrival time is fully determined at that instant — nothing that happens on the edge *afterward* (more people joining, someone else leaving) revises an already-scheduled arrival. Congestion is a one-time snapshot taken at entry, not a continuously-integrated effect.
- **Mathematical model:**
  - `other_occupants = len(edge_occupants) − 1` (everyone else already on the edge, not counting the occupant being admitted).
  - `opposing_occupants` — counted only for `Stair` edges, by scanning every other occupant currently on the same edge and checking whether their own `from_node`/`to_node` are reversed relative to the admitted occupant's.
  - `speed_factor = congestion_model.speed_factor(edge, other_occupants, capacity, opposing_occupants)`. For `DefaultCongestionModel`: `factor = 1 − min(other_occupants/capacity, 1.0) × (1 − 0.3)`, floored at 0.3. `StairAwareCongestionModel` additionally subtracts `0.15 × opposing_occupants` (floored at the same 0.3) for `Stair` edges only.
  - `effective_speed = occupant.walking_speed × speed_factor`.
  - `duration = edge.traversal_cost / effective_speed`, where `edge.traversal_cost` is a **fixed, pre-computed distance** (`walking_distance`, derived once at navigation-graph build time from geometry/floor heights — see `navigation/edge.py`), never adjusted for congestion itself (only the *speed* dividing it is).
  - `start_time = time`; `end_time = time + duration`; an `ARRIVE_AT_NODE` event is scheduled at `end_time`.
- **Queue ownership / release relationship:** this step is triggered either directly from `_handle_try_enter_edge` (capacity was free) or from a queue pop in `_handle_arrive_at_node` (§2.4) — the *same* function handles both origins identically; admission logic has no notion of "was this occupant queued."
- **Interaction with congestion:** as above — computed once, from a snapshot of who else is currently on the edge, at the exact moment of admission.
- **Interaction with movement:** this *is* where movement time is computed — there is no separate "movement" step; admission and the scheduling of the eventual arrival are the same event.

### 2.4 Release — `_handle_arrive_at_node`

- **Purpose:** free the edge slot the arriving occupant was occupying, advance them to the next edge (or mark them arrived), and admit the next-in-line occupant for the edge just vacated.
- **Assumptions:** exactly **one** admission is triggered per **one** departure (`if queue: next_occupant_id = queue.popleft()`), regardless of how much spare capacity might already exist (e.g., if capacity is 10 and only 3 are currently on the edge, a departure still only pops exactly one waiting occupant — though that occupant's own subsequent `TRY_ENTER_EDGE` check will re-evaluate freely against the *live* capacity, so multiple concurrent occupants up to capacity is still reachable over several such cycles, just never in a single batch release).
- **Mathematical model:** `edge_occupancy[edge.id].discard(occupant)` (capacity freed); `queue.popleft()` (FIFO, exactly one); the departing occupant's own `current_edge_index` is incremented and, if their route isn't finished, they immediately (same simulated `time`) re-enter `TRY_ENTER_EDGE` for their *own* next edge.
- **Queue ownership:** identical to §2.2 — only the queue for the edge *just vacated* is ever touched. There is no mechanism anywhere that looks at, notifies, or otherwise couples this edge's release to any *other* edge's queue, including an adjacent edge on the very same physical staircase.
- **Interaction with congestion:** none directly — releasing a slot doesn't retroactively speed up anyone already mid-transit on the edge; it only makes room for the *next* admission, whose own speed will be computed fresh (§2.3) against the occupancy *at that later moment*.
- **Interaction with movement:** the departing occupant's own onward movement (to their next edge) is scheduled in the very same function call, at the same simulated time — there is no intermediate "at the landing, deciding what's next" delay beyond whatever the next `TRY_ENTER_EDGE`/capacity check itself produces.

### 2.5 Next edge

- Identical to §2.1 — `_handle_try_enter_edge` is called again for the same occupant, now referencing `occupant.route.edges[occupant.current_edge_index]` (already incremented). Every one of the properties and assumptions in §2.1–2.4 applies again, completely independently, with **zero carried-over state** from the edge just left (no memory of how congested it was, how long the occupant waited there, or how many others are still queued behind them there).

---

## 3. A or B? Independent Queues, or Continuous Flow?

**Conclusion: (A) — SynEvac's admission-control architecture fundamentally models independent
queues on every edge, not continuous pedestrian flow through connected infrastructure.**

Evidence, directly from the implementation:

1. **Queue state is keyed per-edge, with no aggregating structure above it.** `self._edge_queues: Dict[str, Deque[str]]` — one deque per `edge.id`. There is no `Stairwell`, `Corridor`, or any other object that groups multiple `Edge`s' queues into one shared resource. (`models/staircase.py`'s own code comment confirms this was a known, anticipated, but never-built gap: *"A future 'Stairwell' grouping object (multiple flights belonging to one physical stairwell spanning >2 floors) can be layered on top of this without changing what a Staircase is."*)
2. **Admission is decided purely from local edge occupancy vs. local edge capacity** (§2.1) — no edge's admission logic ever reads any *other* edge's occupancy, queue length, or congestion state. A stair flight two floors down from a queued occupant has no way to signal "I am backed up" to the flight above it.
3. **No spillback/backpressure exists.** In real continuous-flow or coupled-mesoscopic models (e.g., cell-transmission traffic models), a downstream link at capacity actively reduces the *admission rate* of the link upstream of it. Here, an occupant denied entry to edge N simply joins edge N's own unbounded queue at the node between N−1 and N — edge N−1's own throughput and admission decisions are completely unaffected by how long that queue grows.
4. **Release triggers exactly one, edge-local admission**, never a batch reassessment of everyone waiting across the route, and never anything that resembles "flow" (a continuous rate) — it is a discrete, one-for-one hand-off, structurally identical to a single-server (or bulk-server, for capacity > 1) queueing-theory model.
5. **A door merging several independent upstream sources is treated with the exact same, undifferentiated logic as a single stair flight** (confirmed empirically in Validation V2: `door-7-lobby`, a plain `Door`, produced the single worst bottleneck of the entire campaign, using literally the same `DefaultCapacityModel`/queue mechanism every other edge uses) — there is no "merge point" concept that behaves any differently from an ordinary single-source edge.

This is not a defect of implementation quality — every function above is small, clear, well-tested, and internally consistent with its own stated design intent (`MultiAgentSimulation`'s own docstring: *"state that belongs entirely to this coordinator, never to Node/Edge themselves"*). It is a **deliberate, coherent architectural choice**: a discrete-event, per-edge queueing-network model. §5 situates this choice in the wider literature.

---

## 4. Architectural Assumptions and Their Behavioral Effects

*(Analysis only — no fixes proposed, per this milestone's own constraint.)*

- **Per-edge admission.** Every capacity/admission decision is scoped to exactly one edge, evaluated independently every time. *Effect:* a long route through many capacity-constrained edges accumulates one independent local queue delay per edge, with no mechanism for the system to "know" the true end-to-end bottleneck is a single downstream point — every edge along the way behaves as if it were the only constraint that matters.
- **Edge-local queues.** `self._edge_queues[edge.id]` is the *only* place waiting state exists. *Effect:* queueing at edge N has zero visibility to edge N−1; the two are, from a modelling standpoint, entirely separate queueing systems that happen to be visited sequentially by the same occupants.
- **No queue propagation (no spillback).** Confirmed absent in §3.3. *Effect:* a severely backed-up downstream edge (like V2's lobby door) cannot cause upstream edges to "fill up" or slow their own admission rate — occupants are instead absorbed into an unbounded virtual queue at the node between the two edges, which is invisible to anything upstream of *that* node.
- **No shared stairwell state.** A `Staircase` object represents exactly one flight; a real 9-, 13-, or 17-flight stairwell is therefore modelled as that many *fully independent* queueing systems chained in series, each with its own capacity, its own queue, and its own congestion snapshot, unaware of the others. *Effect:* directly explains Validation V1/V2's own repeated-queueing signature — each flight recreates a fresh local bottleneck rather than the whole stairwell sharing one continuous occupancy/flow state.
- **Door merge behaviour.** A `Door` receiving traffic from several upstream sources (e.g., several stair flights merging into one lobby doorway) is admitted/queued with *exactly* the same single-edge, single-FIFO-queue logic as a door with only one possible source. *Effect:* the door has no way to represent "multiple streams are converging here" as a distinct phenomenon — from the queue's own perspective, it is simply receiving requests from several independent callers in whatever order their own upstream processing happens to produce them, which (as V2 showed) can produce dramatically worse aggregate queueing than any single contributing stream would alone.
- **Occupancy release timing.** Exactly one admission per one departure (§2.4). *Effect:* even a large jump in nominal capacity does not automatically translate into proportionally faster *aggregate* throughput once a queue has already formed and multiple upstream sources keep feeding it — release is always mediated one event at a time, and the isolated single-edge experiment (§4.2 below) shows the resulting throughput-vs-capacity relationship is real but **sub-linear**, not a clean multiplier.
- **Capacity calculation.** `width × constant`, no allowance for an edge's own physical *length* (a longer stair flight can, in reality, physically hold more people queued along its own run than a short one of the same width) and no allowance for multiple concurrent streams sharing one merge point. *Effect:* two edges of very different physical scale (a 1-meter door vs. a 30-meter corridor of the same width) are assigned identical capacity by this formula, even though the corridor could physically buffer far more people in transit at once.

### 4.2 Controlled evidence: does capacity matter at all, in isolation?

A natural question raised by V1/V2's own "raising stair capacity did not help" diagnostic is whether
capacity matters to this architecture *at all*. `scripts/capacity_architecture_isolated_edge_experiment.py`
tests this directly on one single, unchained edge (one room, one door, one exit — no merging, no
chaining):

| Door width | Computed capacity | n occupants | Total evacuation time | Seconds/occupant |
|---|---|---|---|---|
| 0.5 m | 1 | 50 | 528.4 s | 10.57 |
| 5.0 m | 7 | 50 | 201.3 s | 4.03 |
| 20.0 m | 30 | 50 | 63.3 s | 1.27 |

**Capacity clearly does matter for a single, isolated edge** — a 7× capacity increase produced a
~2.6× throughput improvement; a 30× increase produced a further large improvement. The relationship
is real but **sub-linear** (not a clean 1:1 multiplier), because `DefaultCongestionModel`'s own
`speed_factor` is a *ratio* (`other_occupants / capacity`) that saturates at the same 0.3 floor once
demand meets or exceeds capacity, **regardless of capacity's raw magnitude** — so a higher-capacity
edge under sufficiently high demand still degrades every individual occupant's own speed to the same
worst-case floor, even though more occupants can be concurrently in transit.

This directly explains why Validation V1/V2's own *chained, multi-source* diagnostic (raising every
stair's capacity uniformly) showed **no aggregate improvement**, while this *isolated* single-edge
test shows a real one: in a chain, raising one edge's capacity does not remove the true system
bottleneck if a *different*, unchanged edge downstream (a lobby door, in V2's case) or the same
saturating-ratio ceiling (in V1's pure-stair case) remains the binding constraint. Capacity is not
architecturally inert — but in a chained, per-edge-independent queueing network, raising it
piecemeal does not reliably move the *system's* throughput, only that one edge's own local one.

---

## 5. Literature Comparison

`[LIT]` = drawn from published/vendor documentation found this session. This section compares
modelling *philosophy* only — it does not claim any tool is "better," per this milestone's own
instruction.

The pedestrian-simulation literature commonly classifies models into three tiers `[LIT]`:

- **Microscopic** — individual pedestrians occupy continuous (or fine-grained) space and interact
  directly with each other and geometry.
- **Macroscopic** — pedestrians are treated as a continuum (a fluid/gas), with no individual
  interaction modelled at all.
- **Mesoscopic** — pedestrians are not individually tracked in continuous space; flow is computed
  over a network of nodes (rooms) and links (corridors/doors/stairs), typically via **discrete-event
  simulation and queueing networks** `[LIT]`.

**SynEvac's `MultiAgentSimulation` is, by this classification, a mesoscopic, discrete-event,
node-and-link queueing-network model** — individual occupants are tracked (an agent identity exists
per occupant), but their position *within* an edge is never modelled; only *which* edge they occupy
and *when* they will finish crossing it.

Compared with four established, named evacuation-simulation tools:

- **Pathfinder** (Thunderhead Engineering) `[LIT]` — a microscopic tool using continuous steering
  behaviour (or an SFPE-guideline mode with density-dependent speed/flow limits) over a 3D
  triangulated mesh, explicitly contrasted by its own vendor against grid/cell-based tools for
  avoiding "artificially constrained occupant movement." Individuals occupy continuous 2D/3D space
  and interact with each other's actual positions, not with an abstract per-edge counter.
- **FDS+Evac** `[LIT]` — a continuum crowd model built on Helbing's **social force model**: motion
  is governed by a literal equation of motion with repulsive/attractive force terms between
  individuals and geometry, including an explicit counterflow model. Congestion here is an emergent
  consequence of many individual force interactions, not a single ratio computed once per edge.
- **buildingEXODUS** `[LIT]` — grid/cellular-automaton based: floor space is discretised into small
  (commonly ~40 cm) tiles, one occupant per tile, with per-timestep, neighbour-dependent transition
  probabilities. This is the closest of the four to a "discrete" model, but the discretisation is
  spatial (fine-grained tiles covering real, continuous floor area) rather than topological
  (SynEvac's coarse "which edge, not where on it" abstraction).
- **MassMotion** (Oasys) `[LIT]` — microscopic, full 3D continuous-space agent simulation, whose own
  documentation explicitly states agents provide *"a window on the entire and continuous pedestrian
  experience... they don't just disappear from one area and pop up in another – you can follow them
  up the stairs."* This is a direct, named contrast with SynEvac's own architecture, in which an
  occupant genuinely has no represented position between entering and leaving an edge, and no
  continuous experience connecting one edge's queue to the next.

**The modelling-philosophy contrast, stated plainly:** all four named tools represent pedestrian
crowding as an emergent property of many individuals occupying and moving through *shared,
continuous (or finely-discretised) physical space* — a queue forming at a door, in these models, is
visible as people physically packed into the real space in front of it, which is *the same space*
a moment ago used by whoever just walked through, and *the same space* the corridor leading to it
occupies. SynEvac's mesoscopic queueing-network approach instead represents crowding as an abstract
counter (`len(edge_occupancy)` vs. `capacity`) attached to a topological link with no represented
interior space at all — a legitimate, established, and computationally far cheaper modelling choice
`[LIT]` (mesoscopic models are noted in the literature specifically for their computational-speed
advantage over microscopic ones), but one that, by construction, cannot represent a queue's own
physical footprint spilling backward into an upstream space, nor a merge point's spatial crowding as
anything other than an ordinary single-source queue.

---

## FINAL REPORT

**1. How does SynEvac's admission-control architecture actually work?**
A discrete-event simulation (`MultiAgentSimulation`) processes two event kinds — `TRY_ENTER_EDGE`
and `ARRIVE_AT_NODE` — via a single global time-ordered heap. Each edge has its own independent
occupancy set, its own independent FIFO queue, and its own capacity value (from `CapacityModel`,
width-derived) checked fresh on every entry attempt. Admission computes a congestion-adjusted speed
(from `CongestionModel`, a local occupancy-ratio snapshot) once, at the moment of entry, and
schedules a single future arrival event. Departure frees exactly one capacity slot and admits
exactly one waiting occupant from that same edge's own queue, then immediately attempts the
departing occupant's own next edge.

**2. What assumptions is it making?**
That an edge is an adequate physical abstraction on its own (no internal space, no length-based
capacity, only width); that capacity/congestion decisions can be made purely locally, per edge, with
no coupling to neighbouring edges; that a multi-flight stairwell is correctly represented as N fully
independent single-flight queueing systems; that a merge point (multiple upstream sources into one
edge) needs no different treatment from a single-source edge; and that releasing one capacity slot
should trigger exactly one new admission, never a batch reassessment.

**3. Which assumptions are supported by the validation campaigns?**
The *isolated*, single-edge capacity-vs-throughput relationship (§4.2) is genuinely real and
non-trivial — capacity is not architecturally inert, and the per-edge queueing mechanism, in
isolation, behaves reasonably (FIFO order was itself verified correct by this codebase's own
existing unit tests, e.g. `QueueFormationTests`).

**4. Which assumptions are contradicted by the validation campaigns?**
"A multi-flight stairwell is adequately modelled as independent per-flight queues" is directly
contradicted — V1's repeated per-flight ~250s waits and V2's escalation to 16× overprediction are
exactly the signature this assumption predicts and nothing else in the architecture prevents. "A
merge point needs no different treatment" is also directly contradicted — V2's `door-7-lobby`
finding (91.6% of one occupant's total time at a single non-stair edge) shows a merge point can be
*worse* than a simple series chain, which an undifferentiated single-source queueing model has no
way to anticipate or represent as a distinct case.

**5. Is the observed discrepancy primarily architectural, mathematical, or implementation-related?**
`[INF]` **Primarily architectural.** The `capacity`/`congestion` *formulas* themselves are simple
but internally consistent, and every function reads exactly as its own docstring describes — this
is not sloppy or buggy code. The discrepancy instead follows directly and predictably from a
deliberate architectural choice (a mesoscopic, per-edge-independent queueing network with no
inter-edge coupling, no shared multi-flight state, and no merge-aware queueing) applied to scenarios
(tall buildings, many chained flights, converging streams) that specifically stress exactly the
dimension that choice does not represent. A different capacity *number* would not resolve this (§4.2
already shows raising the constant does not fix the chained/merged case); only a different
*architecture* for how adjacent/converging edges relate to one another would.

**6. If an architectural redesign is eventually required, which subsystem should be redesigned first?**
`[INF]` `simulator/capacity.py` and `simulator/coordinator.py`'s shared per-edge admission-control
mechanism — specifically, the lack of any concept linking multiple `Edge`s that represent one
continuous physical structure (a multi-flight stairwell) or one converging merge point (a door fed
by several upstream sources). This is a broader target than "the stair model" (per V2's own
falsification finding) and a narrower, more precisely-evidenced target than "redesign the whole
simulator." `models/staircase.py`'s own pre-existing code comment, anticipating a not-yet-built
"Stairwell" grouping object, already points at approximately the right place to begin such an
investigation — though this document takes no position on what that redesign should look like, only
on where the evidence points.

---

*No SynEvac source file was modified in the course of this investigation. New artifacts:
`scripts/capacity_architecture_isolated_edge_experiment.py` (reuses only existing, unmodified
production code) and this document.*

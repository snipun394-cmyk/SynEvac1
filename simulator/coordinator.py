import heapq
import itertools

from collections import deque

from navigation.edge import Edge
from navigation.flow_region import FlowRegion

from simulator.capacity import DefaultCapacityModel
from simulator.congestion import DefaultCongestionModel
from simulator.decision import BehaviorDecision
from simulator.engine import OccupantSimulator
from simulator.multi_agent_result import (
    MultiAgentSimulationResult,
    OccupantTimeline,
    OccupantTimelineStep,
)
from simulator.occupant import Occupant, OccupantState


class MultiAgentSimulation:

    # A discrete-event coordination layer over multiple occupants,
    # each executing a Route that OccupantSimulator (Single Occupant
    # Simulation V1, frozen, unmodified) obtained from PathfindingEngine
    # -- this class never plans a route itself, and never changes one
    # once assigned (no dynamic rerouting). It only decides *when* each
    # occupant crosses each hop of their already-fixed route, subject
    # to shared edge capacity, queueing, and congestion -- state that
    # belongs entirely to this coordinator, never to Node/Edge
    # themselves (which stay pure, stateless, rebuildable views, same
    # reasoning that kept dynamic_state off them during the Engineering
    # Navigation Graph milestone).

    TRY_ENTER_EDGE = "try_enter_edge"
    ARRIVE_AT_NODE = "arrive_at_node"

    # Admission Control V4/V7. A third event kind, keyed by
    # admission_key (not occupant_id/generation like the two above) --
    # see _schedule_retry()/_handle_retry_admission() below. Only ever
    # scheduled when discharge_model or buffer_model is supplied; never
    # reachable when both are None, which is exactly what keeps every
    # existing caller's event ordering byte-identical to before either
    # milestone.
    RETRY_ADMISSION = "retry_admission"

    def __init__(
        self,
        engine,
        occupant_simulator=None,
        capacity_model=None,
        congestion_model=None,
        flow_region_map=None,
        discharge_model=None,
        buffer_model=None,
    ):

        self.engine = engine
        self.graph = engine.graph

        self.occupant_simulator = occupant_simulator or OccupantSimulator(engine)
        self.capacity_model = capacity_model or DefaultCapacityModel()
        self.congestion_model = congestion_model or DefaultCongestionModel()

        # Admission Control V4 -- Dual Constraint Architecture. The new
        # THROUGHPUT constraint, entirely independent of and additive to
        # capacity_model's own STORAGE constraint above -- see
        # simulator/discharge.py's own module docstring for the full
        # architectural rationale (the root problem CapacityModel.
        # capacity() alone conflated: one integer standing in for both
        # "how many can be here at once" and "how fast can they leave").
        # None (every existing caller today, and every caller through
        # Calibration Studio/Automatic Calibration Engine/calibration_
        # benchmark, none of which pass this parameter) means admission
        # control is governed by capacity_model ALONE, structurally
        # identical to this coordinator before this milestone -- see
        # _can_admit() below, which never even calls discharge_model
        # when this is None.
        self.discharge_model = discharge_model

        # Admission Control V7 -- Hybrid Buffer-Service Architecture.
        # The new BUFFER constraint -- independent of, and additive to,
        # BOTH capacity_model (service-zone storage) and discharge_model
        # (service-zone throughput). See simulator/buffer.py's own
        # module docstring for the full architectural rationale (the
        # missing quantity Admission Control V5/V6 identified: queueing
        # theory's own `K`, waiting-room/buffer size, gated at LANDING
        # granularity -- a Node -- never at Edge/FlowRegion granularity).
        # None (every existing caller today, and every caller through
        # Calibration Studio/Automatic Calibration Engine/calibration_
        # benchmark, none of which pass this parameter) means admission
        # control is governed by capacity_model/discharge_model ALONE,
        # structurally identical to this coordinator before this
        # milestone -- see _can_admit() below, which never even calls
        # buffer_model when this is None.
        self.buffer_model = buffer_model

        # Hybrid Flow Regions (Option D), Milestone 3 -- optional,
        # edge.id -> FlowRegion (see navigation/flow_region.py), the
        # exact shape NavigationGraph.flow_regions already carries.
        # None (every existing caller today) means throughput stays
        # keyed by each edge's own id, structurally identical to this
        # coordinator before Milestone 3 -- see _resolve_throughput()
        # below.
        #
        # Admission Control V10 -- Storage-Throughput Separation. As of
        # V10, `flow_region_map` no longer changes STORAGE admission at
        # all -- storage is always local, per edge (see
        # _resolve_admission()/_can_admit() below), regardless of this
        # map. What it now governs is solely THROUGHPUT: when a mapped
        # edge belongs to a CHAIN/MERGE region, discharge gating applies
        # only at that region's own identified bottleneck member
        # edge(s) (FlowRegionCapacityModelV2.bottleneck_edges()), not at
        # every member edge. `capacity_model` no longer needs to be
        # Flow-Region-aware for storage purposes (every existing
        # CapacityModel already dual-accepts a plain Edge correctly);
        # it only needs bottleneck_edges() if THROUGHPUT gating should
        # ever narrow below "every member edge" -- see
        # _get_bottleneck_edges()'s own documented fail-safe when it
        # doesn't.
        self.flow_region_map = flow_region_map

        self._occupants = {}
        self._timelines = {}
        self._arrival_time = {}
        self._unreachable_ids = []

        # Bumped every time an occupant_id is (re-)registered, via
        # add_occupant() or submit_decision(). Events carry the
        # generation they were scheduled under; _process_next_event()
        # ignores any event whose generation no longer matches --
        # this is what makes a later BehaviorDecision for the same
        # occupant_id supersede an earlier one cleanly, without heap
        # surgery. See _begin_registration().
        self._generation = {}

        self._event_heap = []
        self._counter = itertools.count()

        # Per-edge occupancy -- unchanged in meaning since before this
        # milestone. Kept purely for peak_edge_occupancy reporting
        # (Ground Truth and every other existing per-edge consumer),
        # never for the admission-control decision itself once a
        # FlowRegion is mapped. See _admission_occupancy below.
        self._edge_occupancy = {}
        self._node_occupancy = {}

        # Admission Control V10 -- the actual STORAGE admission-control
        # state: occupancy and FIFO queue, always keyed by the edge's
        # own id (see _resolve_admission()) -- no two edges ever share
        # an entry here, regardless of flow_region_map. Pre-V10, a
        # FlowRegion's own id could appear here instead, letting
        # multiple member edges pool one shared entry; V10's Design
        # Review found that pooling was the root mechanism behind both
        # FlowRegionCapacityModelV2's and DefaultDischargeModel's
        # otherwise-unrelated failures, so it is retired here in favor
        # of always-local storage. THROUGHPUT's own shared-clock state
        # (_last_admission_time, immediately below) is what still
        # legitimately groups multiple member edges together, at a
        # region's own identified bottleneck only -- see
        # _resolve_throughput().
        self._admission_occupancy = {}
        self._admission_queues = {}

        # Admission Control V4 -- Dual Constraint Architecture. The
        # discharge-side admission-control state, both keyed by the
        # same admission_key as the two dicts above. _last_admission_time
        # records when an admission_key last actually admitted someone
        # (never touched when discharge_model is None); _pending_retry
        # tracks which admission_keys already have a RETRY_ADMISSION
        # event scheduled, so a discharge-blocked queue never
        # accumulates redundant duplicate timers.
        self._last_admission_time = {}
        self._pending_retry = set()

        # Admission Control V7 -- Hybrid Buffer-Service Architecture.
        # occupant_ids currently "in flight" between being released from
        # the FRONT of a queue (via the peek-then-defer pattern below)
        # and their own deferred re-verification actually firing.
        # Needed because buffer state (a DESTINATION node's occupancy)
        # can change in that narrow window due to a completely
        # UNRELATED occupant arriving there -- unlike storage/discharge,
        # which only ever change via this SAME admission_key's own
        # activity, and therefore could never flip between the peek and
        # the re-check (Admission Control V4's own established
        # invariant). If a requeued occupant's deferred check fails, it
        # must return to the FRONT of the queue, not the back -- see
        # _handle_try_enter_edge()'s own comment.
        self._requeue_at_front = set()

        # Admission Control V7 -- Hybrid Buffer-Service Architecture.
        # node_id -> set of admission_keys currently blocked because
        # THAT node's own buffer is full. Populated only by
        # _maybe_schedule_retry() (never touched when buffer_model is
        # None); drained by _on_node_occupancy_decreased() whenever
        # that node's own occupancy drops, which is the only event that
        # can ever make a full buffer un-full again.
        self._buffer_waiters = {}

        self._queue_join_time = {}

        self._peak_edge_occupancy = {}
        self._peak_node_occupancy = {}
        self._total_queue_events = 0

        # Admission Control V10 -- Storage-Throughput Separation.
        # region.id -> frozenset of member edge ids identified as that
        # region's own throughput bottleneck (see FlowRegionCapacityModelV2.
        # bottleneck_edges(), simulator/flow_region_capacity.py). Computed
        # once per region, lazily, the first time any of its member
        # edges is entered -- the underlying min-cut computation is
        # already documented as cheap at real building region sizes,
        # but there is no reason to repeat it on every single admission
        # attempt for the same region's own fixed topology.
        self._bottleneck_edges_cache = {}

    # =====================================================
    # Registration
    # =====================================================

    def add_occupant(
        self,
        start_id,
        goal_id=None,
        walking_speed=None,
        occupant_id=None,
        depart_time=0.0,
        route=None,
    ):

        occupant_id = occupant_id or f"occupant-{len(self._occupants) + 1}"
        walking_speed = walking_speed or Edge.ASSUMED_WALK_SPEED_M_PER_S

        # Route planning delegated entirely to OccupantSimulator --
        # the one and only place a route is ever computed -- unless
        # the caller already has one (e.g. the Human Behavior Layer
        # chose a specific alternative route, not just a goal). Fixed
        # for the rest of this occupant's life in the simulation.
        if route is not None:

            planned_route = route
            reached_goal = True

        elif goal_id is not None:

            result = self.occupant_simulator.simulate_to_goal(
                start_id, goal_id, occupant_id=occupant_id,
            )
            planned_route = result.route
            reached_goal = result.reached_goal

        else:

            result = self.occupant_simulator.evacuate(
                start_id, occupant_id=occupant_id,
            )
            planned_route = result.route
            reached_goal = result.reached_goal

        return self._register(
            occupant_id, planned_route, reached_goal, walking_speed, depart_time,
        )

    # =====================================================

    def submit_decision(self, decision: BehaviorDecision):

        # The Human Behavior Layer's one entry point into Simulation.
        # BehaviorDecision is immutable -- an occupant "changing their
        # mind" is expressed by submitting a *new* BehaviorDecision for
        # the same occupant_id, never by mutating a previous one (which
        # the type itself makes impossible). Whichever decision was
        # submitted most recently always governs what actually happens:
        # any events still pending under a superseded decision are
        # silently ignored (see _begin_registration()/
        # _process_next_event()). Simulation does not interpret
        # `decision.action_type` at all -- movement vs. no movement vs.
        # unreachable is determined purely by goal_id/route/
        # route_unavailable, keeping this method exactly as decoupled
        # from behavioral vocabulary as the rest of Simulation is from
        # engineering models.

        if decision.route_unavailable:
            # Movement was required but no route to any exit exists --
            # a structural disconnection, not a choice to stay put.
            # Registered the same way add_occupant() registers any
            # other unreachable occupant (reached_goal=False), not as
            # stationary. See docs/validation/technical_report.md §6.
            return self._register(
                decision.occupant_id,
                route=None,
                reached_goal=False,
                walking_speed=decision.walking_speed or Edge.ASSUMED_WALK_SPEED_M_PER_S,
                depart_time=(
                    decision.depart_time if decision.depart_time is not None else 0.0
                ),
            )

        if decision.goal_id is None and decision.route is None:
            return self._register_stationary(decision.occupant_id)

        return self.add_occupant(
            start_id=decision.start_id,
            goal_id=decision.goal_id,
            walking_speed=decision.walking_speed,
            occupant_id=decision.occupant_id,
            depart_time=(
                decision.depart_time if decision.depart_time is not None else 0.0
            ),
            route=decision.route,
        )

    # =====================================================

    def _register(self, occupant_id, route, reached_goal, walking_speed, depart_time):

        self._begin_registration(occupant_id)

        occupant = Occupant(
            occupant_id=occupant_id,
            walking_speed=walking_speed,
            route=route,
            depart_time=depart_time,
        )
        self._occupants[occupant_id] = occupant

        if not reached_goal:

            occupant.state = OccupantState.UNREACHABLE
            self._unreachable_ids.append(occupant_id)

            return occupant_id

        if not route.edges:

            # Trivial route: already at their goal, nothing to walk.
            occupant.state = OccupantState.ARRIVED
            self._arrival_time[occupant_id] = depart_time

            return occupant_id

        occupant.state = OccupantState.AT_NODE

        start_node = route.nodes[0]
        self._add_node_occupant(start_node.id, occupant_id)

        self._schedule(depart_time, self.TRY_ENTER_EDGE, occupant_id)

        return occupant_id

    # =====================================================

    def _register_stationary(self, occupant_id):

        self._begin_registration(occupant_id)

        self._occupants[occupant_id] = Occupant(
            occupant_id=occupant_id,
            walking_speed=0.0,
            route=None,
            depart_time=0.0,
            state=OccupantState.STATIONARY,
        )

        return occupant_id

    # =====================================================

    def _begin_registration(self, occupant_id):

        # Bumps the generation counter for occupant_id and clears
        # every piece of bookkeeping a *previous* registration might
        # have left behind, so a re-registration (whether from
        # add_occupant() or submit_decision()) always starts from a
        # clean slate and never leaves stale state (a lingering queue
        # entry, a stale arrival time, a leftover UNREACHABLE flag)
        # visible in the final result.

        self._generation[occupant_id] = self._generation.get(occupant_id, -1) + 1

        self._arrival_time.pop(occupant_id, None)
        self._timelines[occupant_id] = []
        self._queue_join_time.pop(occupant_id, None)

        if occupant_id in self._unreachable_ids:
            self._unreachable_ids.remove(occupant_id)

        for queue in self._admission_queues.values():
            if occupant_id in queue:
                queue.remove(occupant_id)

    # =====================================================

    def _schedule(self, time, kind, occupant_id):

        heapq.heappush(
            self._event_heap,
            (time, next(self._counter), kind, occupant_id, self._generation[occupant_id]),
        )

    # =====================================================

    def _schedule_retry(self, time, admission_key):

        # Admission Control V4 -- Dual Constraint Architecture. Not
        # occupant-keyed (an admission_key may outlive, or be shared
        # across, any one occupant) -- generation is a literal None
        # sentinel, never checked the way an occupant event's is (see
        # _process_next_event(), which dispatches on `kind` before ever
        # looking at generation).
        heapq.heappush(
            self._event_heap,
            (time, next(self._counter), self.RETRY_ADMISSION, admission_key, None),
        )

    # =====================================================
    # Execution
    # =====================================================

    def run(self):

        # Decomposed as a loop over _process_next_event() (rather than
        # one large while-block) so a future live-stepping consumer
        # could drive this one event at a time without restructuring.
        while self._event_heap:
            self._process_next_event()

        return self._build_result()

    # =====================================================

    def _process_next_event(self):

        time, _, kind, payload, generation = heapq.heappop(self._event_heap)

        if kind == self.RETRY_ADMISSION:
            # Admission Control V4/V7. Not occupant-keyed -- dispatched
            # before the occupant-generation check below, which only
            # ever applies to TRY_ENTER_EDGE/ARRIVE_AT_NODE events. Only
            # ever reached when discharge_model or buffer_model is not
            # None (see _maybe_schedule_retry()); dead code otherwise.
            self._handle_retry_admission(time, payload)
            return

        occupant_id = payload

        if generation != self._generation.get(occupant_id):
            # Stale -- this occupant_id was re-registered (a later
            # BehaviorDecision superseded whatever scheduled this)
            # since this event was put on the heap. Silently dropped,
            # never processed.
            return

        occupant = self._occupants[occupant_id]

        if kind == self.TRY_ENTER_EDGE:
            self._handle_try_enter_edge(time, occupant)
        else:
            self._handle_arrive_at_node(time, occupant)

    # =====================================================

    def _resolve_admission(self, edge):

        # Admission Control V10 -- Storage-Throughput Separation.
        # STORAGE resolution only: storage is always LOCAL, per edge,
        # regardless of whether a flow_region_map is supplied and
        # regardless of that edge's own region_kind (Design Review
        # correction #1, mandatory). This is a literal (edge, edge.id)
        # passthrough unconditionally -- no dict lookup, ever -- which
        # is also exactly what every pre-V10 caller with
        # flow_region_map=None already required, so the zero-behavior-
        # change guarantee for those callers remains structural, not
        # merely coincidental. See _resolve_throughput() below for the
        # separate, genuinely region-aware resolution throughput needs.

        return edge, edge.id

    def _resolve_throughput(self, edge):

        # Admission Control V10 -- Storage-Throughput Separation.
        # Returns (throughput_object, applies): throughput_object is
        # what discharge_model.discharge_rate() is called with (always
        # has its own .id, used as the shared-clock key in
        # _last_admission_time) -- an Edge for the legacy/SINGLE-region
        # case, a FlowRegion for a CHAIN/MERGE region. `applies` is
        # True only when THIS edge-crossing should be gated/tracked by
        # the throughput clock at all.
        #
        # Legacy (flow_region_map=None) and SINGLE-kind regions: no
        # region grouping exists, so throughput -- when a discharge_model
        # is configured -- always applies directly to the edge itself,
        # exactly Admission Control V4's own original behavior,
        # unconditionally preserved.
        #
        # CHAIN/MERGE regions: `applies` is True only when this specific
        # edge is one of the region's own identified bottleneck member
        # edges (Design Review correction #2) -- an occupant's internal
        # continuation across any OTHER member edge of the same region
        # never touches the throughput clock at all, eliminating the
        # self-throttling failure mode (an occupant rate-limited against
        # their own earlier admission timestamp) the Admission Semantics
        # Investigation traced this whole redesign to.

        if self.flow_region_map is None:
            return edge, True

        region = self.flow_region_map.get(edge.id)

        if region is None or region.region_kind == FlowRegion.SINGLE:
            return edge, True

        bottleneck_ids = self._get_bottleneck_edges(region)

        return region, edge.id in bottleneck_ids

    def _get_bottleneck_edges(self, region):

        cached = self._bottleneck_edges_cache.get(region.id)

        if cached is not None:
            return cached

        if hasattr(self.capacity_model, "bottleneck_edges"):
            # FlowRegionCapacityModelV2 (or any future capacity model
            # that chooses to expose the same method) -- see
            # simulator/flow_region_capacity.py.
            result = frozenset(self.capacity_model.bottleneck_edges(region))
        else:
            # Design Review correction #4's own fail-safe: a capacity
            # model that cannot identify a bottleneck (FlowRegionCapacityModel
            # V1, or a plain DefaultCapacityModel/StairCapacityModel
            # somehow paired with a flow_region_map) means no member
            # edge of this region is ever treated as ITS throughput
            # bottleneck -- admission for every member edge reduces to
            # storage-only (per edge) + buffer. Never crashes, never
            # deadlocks, never silently guesses.
            result = frozenset()

        self._bottleneck_edges_cache[region.id] = result

        return result

    # =====================================================

    def _handle_try_enter_edge(self, time, occupant):

        edge = occupant.route.edges[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        # Admission Control V7. True only for an occupant just released
        # from the FRONT of this same queue (see _handle_arrive_at_node()'s
        # own peek-then-defer comment) whose deferred re-verification,
        # here, is about to run -- never true for a genuinely fresh
        # admission attempt.
        was_released_from_queue_front = occupant.occupant_id in self._requeue_at_front
        self._requeue_at_front.discard(occupant.occupant_id)

        if self._can_admit(edge, to_node, time):

            self._admit_onto_edge(time, occupant, edge)

        else:

            # Admission Control V10 -- storage is always local, so the
            # queue key is always this edge's own id (never a region's).
            queue = self._admission_queues.setdefault(edge.id, deque())

            if was_released_from_queue_front:
                # A transient race, not a fresh arrival -- buffer state
                # at to_node changed between the peek and this deferred
                # re-check (an unrelated occupant arrived there in the
                # interim). This occupant was already rightfully at the
                # front; it must stay there, never be pushed behind
                # occupants who were never released at all.
                queue.appendleft(occupant.occupant_id)
            else:
                queue.append(occupant.occupant_id)

            # setdefault, not unconditional assignment -- a requeued
            # occupant's own original queue_join_time (from when they
            # FIRST joined) must be preserved, never reset by a
            # transient re-verification failure.
            self._queue_join_time.setdefault(occupant.occupant_id, time)

            occupant.state = OccupantState.QUEUED

            if not was_released_from_queue_front:
                self._total_queue_events += 1

            self._maybe_schedule_retry(edge, to_node, time)

    # =====================================================

    def _can_admit(self, edge, to_node, time):

        # Admission Control V10 -- Storage-Throughput Separation
        # (Design Review correction #3, mandatory). STORAGE, THROUGHPUT,
        # and BUFFER are three fully independent checks combined into
        # ONE ordered decision on ONE queue per edge -- never two
        # separately-queued gates -- which is exactly what preserves
        # FIFO: whichever occupant is at the front of THIS edge's own
        # queue is always the one re-evaluated first, regardless of
        # which of the three constraints currently blocks them.
        #
        # STORAGE is always LOCAL: capacity_model.capacity(edge) and
        # _admission_occupancy[edge.id] are both always evaluated
        # against the plain edge itself, never a FlowRegion, regardless
        # of flow_region_map (Design Review correction #1). Every
        # existing CapacityModel (DefaultCapacityModel,
        # StairCapacityModel, FlowRegionCapacityModel V1, and
        # FlowRegionCapacityModelV2) already dual-accepts a plain Edge
        # and delegates to its own base_model in that case -- calling
        # capacity(edge) unconditionally here requires no change to any
        # of them and is correct regardless of which one is configured.

        capacity = self.capacity_model.capacity(edge)
        current_admitted = len(self._admission_occupancy.get(edge.id, ()))

        if current_admitted >= capacity:
            return False

        if self.buffer_model is not None:

            # Admission Control V7 -- Hybrid Buffer-Service Architecture.
            # Unchanged by V10: already correctly node-scoped, gating
            # entry onto THIS edge by whether its own DESTINATION
            # landing has room. See this method's own pre-V10 comment
            # history for the full disclosed-approximation rationale
            # (never blocks mid-traversal arrival, only admission).
            buffer_capacity = self.buffer_model.buffer_capacity(to_node)

            if buffer_capacity is not None and buffer_capacity > 0:

                current_buffer_occupancy = len(self._node_occupancy.get(to_node.id, ()))

                if current_buffer_occupancy >= buffer_capacity:
                    return False

        if self.discharge_model is None:
            return True

        # THROUGHPUT is evaluated ONLY when this specific edge is (or
        # stands in for, in the legacy/SINGLE case) a genuine bottleneck
        # -- Design Review correction #2. An internal continuation
        # across a non-bottleneck member edge of a CHAIN/MERGE region
        # never reaches the discharge-rate check at all.
        throughput_object, throughput_applies = self._resolve_throughput(edge)

        if not throughput_applies:
            return True

        discharge_rate = self.discharge_model.discharge_rate(throughput_object)

        if not discharge_rate or discharge_rate <= 0:
            # No genuine rate constraint derivable (None, zero, or
            # negative) -- fails OPEN, storage alone governs, exactly
            # like DefaultCapacityModel/StairCapacityModel's own "None
            # means not derivable, fall back to a safe default"
            # convention. A misconfigured/undefined discharge rate must
            # never silently deadlock a queue.
            return True

        throughput_key = throughput_object.id
        last_time = self._last_admission_time.get(throughput_key)

        if last_time is None:
            # This throughput key has never admitted anyone yet -- no
            # prior admission to measure a gap from.
            return True

        return (time - last_time) >= (1.0 / discharge_rate)

    # =====================================================

    def _maybe_schedule_retry(self, edge, to_node, time):

        # Admission Control V4/V7, updated for V10's always-per-edge
        # storage keying. Called whenever an admission attempt is
        # blocked. If the block is STORAGE-driven (the edge is
        # genuinely full), no explicit retry is needed at all -- the
        # next occupant to LEAVE will free a slot and re-trigger
        # admission on its own, exactly as this coordinator already
        # guarantees for every existing (unconstrained) caller.
        #
        # A genuinely BUFFER-driven block (storage has room, but the
        # destination landing does not) is resolved by registering this
        # edge's own admission_key as a waiter on to_node --
        # _on_node_occupancy_decreased() retries it the moment that
        # specific node's own occupancy drops, which is the only event
        # that can ever make a full buffer un-full again.
        #
        # A genuinely THROUGHPUT-driven block (storage and buffer both
        # have room, but this edge IS an identified region bottleneck
        # and not enough time has elapsed since that bottleneck's last
        # admission) is resolved exactly as Admission Control V4
        # already does, with an explicit RETRY_ADMISSION timer -- keyed
        # by this edge's own admission_key (edge.id) even though the
        # TIMING is governed by the region's shared throughput clock,
        # so the retry always lands on the correct per-edge queue.
        #
        # _pending_retry gates the WHOLE method, not just one reason --
        # once ANY retry mechanism is pending for this edge's own
        # admission_key (a buffer wait, a timer, or both), this returns
        # immediately rather than registering redundant duplicates;
        # _handle_retry_admission() clears it before re-evaluating from
        # scratch.

        admission_key = edge.id

        if admission_key in self._pending_retry:
            return

        capacity = self.capacity_model.capacity(edge)
        current_admitted = len(self._admission_occupancy.get(admission_key, ()))

        if current_admitted >= capacity:
            return

        scheduled_something = False

        if self.buffer_model is not None:

            buffer_capacity = self.buffer_model.buffer_capacity(to_node)

            if buffer_capacity is not None and buffer_capacity > 0:

                current_buffer_occupancy = len(self._node_occupancy.get(to_node.id, ()))

                if current_buffer_occupancy >= buffer_capacity:
                    self._buffer_waiters.setdefault(to_node.id, set()).add(admission_key)
                    scheduled_something = True

        if self.discharge_model is not None:

            throughput_object, throughput_applies = self._resolve_throughput(edge)

            if throughput_applies:

                discharge_rate = self.discharge_model.discharge_rate(throughput_object)

                if discharge_rate and discharge_rate > 0:

                    last_time = self._last_admission_time.get(throughput_object.id)

                    if last_time is not None:

                        retry_time = max(time, last_time + (1.0 / discharge_rate))

                        # Admission Control V7. Strictly-future check.
                        # Under V4 alone this branch was only ever
                        # reached when discharge itself was the reason
                        # _can_admit() failed -- which made retry_time >
                        # time a structural guarantee. V7's buffer check
                        # sits BEFORE the discharge check, so this
                        # branch can also be reached while discharge's
                        # own gate is already satisfied (buffer alone is
                        # what's blocking) -- in that case retry_time
                        # collapses to exactly `time`, and scheduling it
                        # unconditionally created a same-timestamp
                        # RETRY_ADMISSION that rescheduled itself
                        # forever (confirmed via instrumentation on the
                        # NIST 10-story building, pre-V10). A retry
                        # timer must only ever fire strictly in the
                        # future; when discharge is already satisfied,
                        # the buffer-waiter registration above is the
                        # only mechanism that can ever legitimately
                        # unblock this admission_key.
                        if retry_time > time:
                            self._schedule_retry(retry_time, admission_key)
                            scheduled_something = True

        if scheduled_something:
            self._pending_retry.add(admission_key)

    # =====================================================

    def _handle_retry_admission(self, time, admission_key):

        # Admission Control V4/V7. Only ever reached via a
        # RETRY_ADMISSION event, which is only ever scheduled by
        # _maybe_schedule_retry() -- a no-op when both discharge_model
        # and buffer_model are None, so this method is unreachable
        # (dead code) in that case, preserving every existing caller's
        # behavior exactly.
        #
        # Admission Control V10 -- admission_key is always an edge id
        # now (storage is always local), so the queue this retry
        # concerns is unambiguously that ONE edge's own queue; every
        # occupant in it is, by construction, waiting to enter that
        # SAME edge, and therefore shares the exact same to_node -- no
        # per-occupant re-derivation needed (the pre-V10 comment about
        # "a shared FlowRegion's own branches" leading to different
        # destinations no longer applies, since storage never pools
        # multiple edges together anymore).

        self._pending_retry.discard(admission_key)

        queue = self._admission_queues.get(admission_key)

        if not queue:
            return

        occupant_id = queue[0]
        occupant = self._occupants[occupant_id]
        edge = occupant.route.edges[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        if self._can_admit(edge, to_node, time):

            queue.popleft()
            self._admit_onto_edge(time, occupant, edge)

            if queue:
                self._maybe_schedule_retry(edge, to_node, time)

        else:

            self._maybe_schedule_retry(edge, to_node, time)

    # =====================================================

    def _admit_onto_edge(self, time, occupant, edge):

        from_node = occupant.route.nodes[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        self._remove_node_occupant(from_node.id, occupant.occupant_id, time)

        # Per-edge occupancy -- reporting only (peak_edge_occupancy),
        # never consulted for the admission decision below. See this
        # dict's own comment in __init__.
        edge_occupants = self._edge_occupancy.setdefault(edge.id, set())
        edge_occupants.add(occupant.occupant_id)
        self._track_peak(self._peak_edge_occupancy, edge.id, len(edge_occupants))

        occupant.state = OccupantState.TRAVERSING

        # Admission Control V10 -- storage admission pool is always
        # edge-keyed (local), regardless of flow_region_map.
        admission_occupants = self._admission_occupancy.setdefault(edge.id, set())
        admission_occupants.add(occupant.occupant_id)

        # Admission Control V10 -- throughput's own bookkeeping
        # (updating the shared clock) happens here, at the single point
        # every admission -- fresh or queue-released -- already passes
        # through, rather than being duplicated across both callers of
        # this method as it was pre-V10. Only touches the clock when
        # this edge is genuinely a region's identified bottleneck (or
        # stands in for the edge itself in the legacy/SINGLE case) --
        # never on an internal continuation across a non-bottleneck
        # member edge.
        if self.discharge_model is not None:

            throughput_object, throughput_applies = self._resolve_throughput(edge)

            if throughput_applies:
                self._last_admission_time[throughput_object.id] = time

        capacity = self.capacity_model.capacity(edge)
        other_occupants = len(admission_occupants) - 1

        opposing_occupants = 0

        if edge.edge_type == Edge.STAIR:
            opposing_occupants = self._count_opposing_occupants(
                edge, admission_occupants, from_node, to_node, occupant.occupant_id,
            )

        # Admission Control V10 -- Design Review's own "explicitly
        # resolve congestion granularity: per-edge occupancy, per-edge
        # capacity" requirement. `edge` (never a FlowRegion) is passed
        # unconditionally now; `admission_occupants`/`capacity` above
        # are already edge-local by construction, so this alone
        # completes the same locality storage already established --
        # no separate congestion-specific resolution is needed.
        # FlowRegionCongestionModel remains fully usable (backward
        # compatible, never modified) but, since it is now never handed
        # a FlowRegion by this coordinator, always takes its own plain-
        # edge delegation path -- behaviorally identical to its own
        # base_model for every V10 admission decision.
        speed_factor = self.congestion_model.speed_factor(
            edge, other_occupants, capacity, opposing_occupants=opposing_occupants,
        )
        effective_speed = occupant.walking_speed * speed_factor

        # Stair Simulation Reliability & Multi-Floor Reachability Audit
        # milestone, Phase 17 -- `edge.walking_distance or 0.0` silently
        # coerced BOTH a genuinely unknown distance (None) and a
        # genuinely computed zero to the same 0.0, producing an
        # instantaneous (start_time == end_time) traversal whenever
        # walking_distance was None -- exactly the historical zero-
        # duration Stair bug's own simulation-level symptom, and the one
        # mechanism that would have silently neutralized navigation.
        # graph_builder.NavigationGraphGenerator._add_stair_edges()'s own
        # new None-on-degenerate-distance guard (see that method's own
        # comment). `edge.traversal_cost` already implements exactly the
        # right fallback (walking_distance when known, Edge.DEFAULT_
        # TRAVERSAL_COST otherwise -- see navigation/edge.py) and is
        # numerically IDENTICAL to the old expression whenever
        # walking_distance was already known (including a genuine 0.0),
        # so this changes behavior ONLY for the previously-broken
        # None case.
        distance = edge.traversal_cost
        duration = distance / effective_speed

        start_time = time
        end_time = time + duration

        join_time = self._queue_join_time.pop(occupant.occupant_id, None)
        queue_wait_time = (time - join_time) if join_time is not None else 0.0

        self._timelines[occupant.occupant_id].append(
            OccupantTimelineStep(
                index=occupant.current_edge_index,
                from_node=from_node,
                to_node=to_node,
                edge=edge,
                queue_wait_time=queue_wait_time,
                start_time=start_time,
                end_time=end_time,
            )
        )

        self._schedule(end_time, self.ARRIVE_AT_NODE, occupant.occupant_id)

    # =====================================================

    def _count_opposing_occupants(self, edge, candidate_occupants, from_node, to_node, occupant_id):

        # Every V1 edge is bidirectional (see navigation/edge.py), so
        # two occupants can legitimately be on the same Stair edge at
        # once travelling opposite ways -- one's from_node/to_node is
        # the other's to_node/from_node. Reuses each other occupant's
        # already-tracked route/current_edge_index; no new tracking
        # state is added for this.
        #
        # Admission Control V10 -- `candidate_occupants` is always this
        # edge's own local admission pool now (never a whole FlowRegion's
        # shared occupants, see _resolve_admission()), so counterflow is
        # scanned across exactly the same edge-local pool the storage/
        # congestion decision itself uses. Pre-V10, when a FlowRegion
        # was mapped, this pool could include occupants on OTHER member
        # edges of the same region; the node-pair reversal test below
        # already excluded those as false positives (they could never
        # match this edge's own exact from/to node pair), so this is a
        # narrower search space now, not a behavior change -- see the
        # Admission Semantics Investigation's own Phase 2 finding that
        # counterflow counting was never actually incorrect pre-V10,
        # only more expensive than it needed to be.

        count = 0

        for other_id in candidate_occupants:

            if other_id == occupant_id:
                continue

            other = self._occupants.get(other_id)

            if other is None:
                continue

            other_from = other.route.nodes[other.current_edge_index]
            other_to = other.route.nodes[other.current_edge_index + 1]

            if other_from.id == to_node.id and other_to.id == from_node.id:
                count += 1

        return count

    # =====================================================

    def _handle_arrive_at_node(self, time, occupant):

        edge = occupant.route.edges[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        edge_occupants = self._edge_occupancy.get(edge.id, set())
        edge_occupants.discard(occupant.occupant_id)

        # Admission Control V10 -- storage is always edge-keyed.
        admission_occupants = self._admission_occupancy.get(edge.id, set())
        admission_occupants.discard(occupant.occupant_id)

        queue = self._admission_queues.get(edge.id)

        if queue:

            # Admission Control V4/V7, simplified for V10. Peek (never
            # blindly pop) so a blocked queue head is never popped only
            # to be re-queued at the BACK by _handle_try_enter_edge()'s
            # own queueing branch, which would silently break FIFO
            # order for whoever is behind them. When discharge_model
            # and buffer_model are both None, _can_admit() reduces to
            # exactly "current_admitted < capacity", which is always
            # True here (exactly one slot just freed, by construction)
            # -- so this branch is taken unconditionally, preserving
            # the original pop-then-schedule behavior byte-for-byte.
            #
            # Admission Control V10 -- every occupant queued here is,
            # by construction, waiting to enter this SAME edge (storage
            # is always per-edge now, never pooled across a region's
            # member edges), so they all share this departing
            # occupant's own `to_node` exactly -- no per-occupant
            # re-derivation needed, unlike the pre-V10 shared-region
            # queue this replaces (where different member edges of one
            # region could lead to different destinations).
            if self._can_admit(edge, to_node, time):

                next_occupant_id = queue.popleft()

                # Admission Control V7 -- see _handle_try_enter_edge()'s
                # own comment: marks this occupant so that IF their
                # deferred re-verification (about to be scheduled below)
                # fails, they are returned to the FRONT of the queue,
                # never the back.
                self._requeue_at_front.add(next_occupant_id)

                self._schedule(time, self.TRY_ENTER_EDGE, next_occupant_id)

            else:

                self._maybe_schedule_retry(edge, to_node, time)

        self._add_node_occupant(to_node.id, occupant.occupant_id)

        occupant.current_edge_index += 1

        if occupant.current_edge_index >= len(occupant.route.edges):

            occupant.state = OccupantState.ARRIVED
            self._arrival_time[occupant.occupant_id] = time
            self._remove_node_occupant(to_node.id, occupant.occupant_id, time)

            return

        occupant.state = OccupantState.AT_NODE

        self._schedule(time, self.TRY_ENTER_EDGE, occupant.occupant_id)

    # =====================================================
    # Occupancy bookkeeping
    # =====================================================

    def _add_node_occupant(self, node_id, occupant_id):

        occupants = self._node_occupancy.setdefault(node_id, set())
        occupants.add(occupant_id)

        self._track_peak(self._peak_node_occupancy, node_id, len(occupants))

    # =====================================================

    def _remove_node_occupant(self, node_id, occupant_id, time):

        self._node_occupancy.get(node_id, set()).discard(occupant_id)

        self._on_node_occupancy_decreased(node_id, time)

    # =====================================================

    def _on_node_occupancy_decreased(self, node_id, time):

        # Admission Control V7 -- Hybrid Buffer-Service Architecture.
        # The only event that can ever make a full buffer un-full again
        # -- called from every place this coordinator ever reduces a
        # node's own occupancy. Always a cheap no-op when buffer_model
        # is None (_buffer_waiters is never populated in that case).

        waiters = self._buffer_waiters.pop(node_id, None)

        if not waiters:
            return

        for admission_key in waiters:
            self._schedule_retry(time, admission_key)

    # =====================================================

    def _track_peak(self, peak_dict, key, current_value):

        peak_dict[key] = max(peak_dict.get(key, 0), current_value)

    # =====================================================
    # Result assembly
    # =====================================================

    def _build_result(self):

        occupants = {
            occupant_id: OccupantTimeline(
                occupant_id=occupant_id,
                route=occupant.route,
                steps=self._timelines.get(occupant_id, []),
                state=occupant.state,
                depart_time=occupant.depart_time,
                arrival_time=self._arrival_time.get(occupant_id),
            )
            for occupant_id, occupant in self._occupants.items()
        }

        arrival_times = list(self._arrival_time.values())
        total_evacuation_time = max(arrival_times) if arrival_times else None

        return MultiAgentSimulationResult(
            occupants=occupants,
            total_evacuation_time=total_evacuation_time,
            unreachable_occupant_ids=list(self._unreachable_ids),
            peak_edge_occupancy=dict(self._peak_edge_occupancy),
            peak_node_occupancy=dict(self._peak_node_occupancy),
            total_queue_events=self._total_queue_events,
        )

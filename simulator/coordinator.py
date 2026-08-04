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
        # None (every existing caller today) means admission control
        # stays keyed by each edge's own id, structurally identical to
        # this coordinator before this milestone -- see
        # _resolve_admission() below, which never touches a dict
        # lookup at all in that case. When supplied, `capacity_model`/
        # `congestion_model` must themselves be Flow-Region-aware
        # (FlowRegionCapacityModel/FlowRegionCongestionModel, Milestone
        # 2) since every mapped edge resolves to a FlowRegion object,
        # not an Edge -- pairing a plain DefaultCapacityModel with a
        # non-None flow_region_map is a caller error, not something
        # this coordinator guards against.
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

        # Hybrid Flow Regions (Option D), Milestone 3 -- the actual
        # admission-control state: occupancy and FIFO queue keyed by
        # _resolve_admission()'s own admission_key (a FlowRegion's id
        # when one is mapped, otherwise the edge's own id -- so for
        # every existing caller, an admission_key IS an edge id, and
        # these two dicts behave exactly like the single per-edge
        # occupancy/queue dicts this coordinator has always had, just
        # renamed). Two or more edges that share a FlowRegion share one
        # entry in each of these dicts, which is the whole mechanism
        # behind merge/chain admission control.
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

        # Hybrid Flow Regions (Option D), Milestone 3 -- returns
        # (admission_object, admission_key): the object handed to
        # capacity_model/congestion_model (an Edge, or the FlowRegion
        # it belongs to -- both FlowRegionCapacityModel and
        # FlowRegionCongestionModel dispatch on isinstance(...,
        # FlowRegion) to pick their region formula vs. their plain
        # edge-delegating one) and the plain string key this
        # coordinator's own _admission_occupancy/_admission_queues
        # dicts are keyed by.
        #
        # When flow_region_map is None -- every existing caller today
        # -- this never touches a dict lookup at all: it is a literal
        # (edge, edge.id) passthrough, not "a map that happens to
        # default to identity", so the zero-behavior-change guarantee
        # for those callers is structural, not merely tested.

        if self.flow_region_map is None:
            return edge, edge.id

        region = self.flow_region_map.get(edge.id)

        if region is None or region.region_kind == FlowRegion.SINGLE:
            # A trivial, one-edge region carries no grouping
            # information beyond the edge itself -- FlowRegion's own
            # docstring promises these are "identical in effect to
            # today's per-edge behavior". Resolving straight to the
            # edge (not the FlowRegion wrapper) is what actually keeps
            # that promise all the way through capacity/congestion, not
            # just admission-pool bookkeeping: it preserves, for
            # example, a solo Stair's own counterflow penalty
            # (StairAwareCongestionModel), which
            # FlowRegionCongestionModel's region-shaped formula
            # deliberately does not apply (see that class's own
            # docstring).
            return edge, edge.id

        return region, region.id

    # =====================================================

    def _handle_try_enter_edge(self, time, occupant):

        edge = occupant.route.edges[occupant.current_edge_index]
        admission_object, admission_key = self._resolve_admission(edge)
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        # Admission Control V7. True only for an occupant just released
        # from the FRONT of this same queue (see _handle_arrive_at_node()'s
        # own peek-then-defer comment) whose deferred re-verification,
        # here, is about to run -- never true for a genuinely fresh
        # admission attempt.
        was_released_from_queue_front = occupant.occupant_id in self._requeue_at_front
        self._requeue_at_front.discard(occupant.occupant_id)

        if self._can_admit(admission_object, admission_key, to_node, time):

            self._admit_onto_edge(time, occupant, edge)

            if self.discharge_model is not None:
                self._last_admission_time[admission_key] = time

        else:

            queue = self._admission_queues.setdefault(admission_key, deque())

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

            self._maybe_schedule_retry(admission_object, admission_key, to_node, time)

    # =====================================================

    def _can_admit(self, admission_object, admission_key, to_node, time):

        # Admission Control V4/V7. STORAGE (capacity_model), THROUGHPUT
        # (discharge_model), and BUFFER (buffer_model) are three fully
        # independent checks -- all three must pass, none is ever
        # derived from another. When discharge_model and buffer_model
        # are both None, this reduces to exactly the single comparison
        # this coordinator has always made (current_admitted <
        # capacity), byte-identical to every existing caller's own
        # behavior.

        capacity = self.capacity_model.capacity(admission_object)
        current_admitted = len(self._admission_occupancy.get(admission_key, ()))

        if current_admitted >= capacity:
            return False

        if self.buffer_model is not None:

            # Admission Control V7 -- Hybrid Buffer-Service Architecture.
            # Gates entry onto THIS edge/region by whether its own
            # DESTINATION landing has room -- never by blocking arrival
            # mid-traversal (no such mechanism exists in this
            # coordinator's event model, and none is added here). This
            # is a deliberate, disclosed approximation of true spillback:
            # nobody is ever admitted onto an edge whose destination is
            # already known to be full at the moment of admission, so
            # nobody arrives at an already-full landing under ordinary
            # circumstances -- but several occupants admitted onto the
            # SAME edge in close succession, all converging on the same
            # destination, could in principle still arrive together and
            # briefly exceed it. Preferred over a much larger change
            # (pausing an occupant already mid-edge) for exactly the
            # reason this milestone asks for: the minimum new
            # abstraction that reuses the existing edge-entry admission
            # point, not a redesign of the event model itself.
            buffer_capacity = self.buffer_model.buffer_capacity(to_node)

            if buffer_capacity is not None and buffer_capacity > 0:

                current_buffer_occupancy = len(self._node_occupancy.get(to_node.id, ()))

                if current_buffer_occupancy >= buffer_capacity:
                    return False

        if self.discharge_model is None:
            return True

        discharge_rate = self.discharge_model.discharge_rate(admission_object)

        if not discharge_rate or discharge_rate <= 0:
            # No genuine rate constraint derivable for this edge/region
            # (None, zero, or negative) -- fails OPEN, storage alone
            # governs, exactly like DefaultCapacityModel/StairCapacityModel's
            # own "None means not derivable, fall back to a safe
            # default" convention. A misconfigured/undefined discharge
            # rate must never silently deadlock a queue.
            return True

        last_time = self._last_admission_time.get(admission_key)

        if last_time is None:
            # This admission_key has never admitted anyone yet -- no
            # prior admission to measure a gap from.
            return True

        return (time - last_time) >= (1.0 / discharge_rate)

    # =====================================================

    def _maybe_schedule_retry(self, admission_object, admission_key, to_node, time):

        # Admission Control V4/V7. Called whenever an admission attempt
        # is blocked. If the block is STORAGE-driven (the edge/region
        # is genuinely full), no explicit retry is needed at all -- the
        # next occupant to LEAVE will free a slot and re-trigger
        # admission on its own, exactly as this coordinator already
        # guarantees for every existing (unconstrained) caller.
        #
        # A genuinely BUFFER-driven block (storage has room, but the
        # destination landing does not) is resolved by registering this
        # admission_key as a waiter on to_node -- _on_node_occupancy_
        # decreased() retries it the moment that specific node's own
        # occupancy drops, which is the only event that can ever make a
        # full buffer un-full again.
        #
        # A genuinely DISCHARGE-driven block (storage and buffer both
        # have room, but not enough time has elapsed since the last
        # admission) is resolved exactly as Admission Control V4
        # already does, with an explicit RETRY_ADMISSION timer.
        #
        # _pending_retry gates the WHOLE method, not just one reason --
        # once ANY retry mechanism is pending for this admission_key
        # (a buffer wait, a timer, or both), this returns immediately
        # rather than registering redundant duplicates; _handle_retry_
        # admission() clears it before re-evaluating from scratch.

        if admission_key in self._pending_retry:
            return

        capacity = self.capacity_model.capacity(admission_object)
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

            discharge_rate = self.discharge_model.discharge_rate(admission_object)

            if discharge_rate and discharge_rate > 0:

                last_time = self._last_admission_time.get(admission_key)

                if last_time is not None:

                    retry_time = max(time, last_time + (1.0 / discharge_rate))

                    # Admission Control V7. Strictly-future check, new
                    # with this milestone. Under V4 alone this branch
                    # was only ever reached when discharge itself was
                    # the reason _can_admit() failed -- which made
                    # retry_time > time a structural guarantee (see
                    # _can_admit(): discharge fails only when `time -
                    # last_time < 1/discharge_rate`, so `last_time +
                    # 1/discharge_rate` is necessarily still ahead of
                    # `time`). V7's buffer check now sits BEFORE the
                    # discharge check, so this branch can also be
                    # reached while discharge's own gate is already
                    # satisfied (buffer alone is what's blocking) --
                    # in that case retry_time collapses to exactly
                    # `time`, and scheduling it unconditionally created
                    # a same-timestamp RETRY_ADMISSION that rescheduled
                    # itself forever, forever failing the same buffer
                    # check without buffer occupancy ever needing to
                    # change (confirmed via instrumentation: heap size
                    # and simulated time both frozen across 250,000+
                    # processed events on the NIST 10-story building).
                    # A retry timer must only ever fire strictly in the
                    # future; when discharge is already satisfied, the
                    # buffer-waiter registration above is the only
                    # mechanism that can ever legitimately unblock this
                    # admission_key.
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

        self._pending_retry.discard(admission_key)

        queue = self._admission_queues.get(admission_key)

        if not queue:
            return

        occupant_id = queue[0]
        occupant = self._occupants[occupant_id]
        edge = occupant.route.edges[occupant.current_edge_index]
        admission_object, _ = self._resolve_admission(edge)
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        if self._can_admit(admission_object, admission_key, to_node, time):

            queue.popleft()
            self._admit_onto_edge(time, occupant, edge)
            self._last_admission_time[admission_key] = time

            if queue:

                # The NEW queue head may be routed toward a different
                # destination than the occupant just admitted (e.g. a
                # shared FlowRegion's own branches) -- resolved fresh,
                # never assumed to match.
                new_head_occupant = self._occupants[queue[0]]
                new_head_to_node = new_head_occupant.route.nodes[new_head_occupant.current_edge_index + 1]

                self._maybe_schedule_retry(admission_object, admission_key, new_head_to_node, time)

        else:

            self._maybe_schedule_retry(admission_object, admission_key, to_node, time)

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

        admission_object, admission_key = self._resolve_admission(edge)

        admission_occupants = self._admission_occupancy.setdefault(admission_key, set())
        admission_occupants.add(occupant.occupant_id)

        capacity = self.capacity_model.capacity(admission_object)
        other_occupants = len(admission_occupants) - 1

        opposing_occupants = 0

        if edge.edge_type == Edge.STAIR:
            opposing_occupants = self._count_opposing_occupants(
                edge, admission_occupants, from_node, to_node, occupant.occupant_id,
            )

        speed_factor = self.congestion_model.speed_factor(
            admission_object, other_occupants, capacity, opposing_occupants=opposing_occupants,
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
        # Hybrid Flow Regions (Option D), Milestone 3 -- `candidate_occupants`
        # is the admission-level pool (a whole FlowRegion's shared
        # occupants when one is mapped, otherwise just this edge's, see
        # _resolve_admission()), so counterflow is scanned across the
        # same shared pool the capacity/congestion decision itself
        # uses. The exact node-pair reversal test below is unchanged --
        # it still only ever matches someone genuinely travelling this
        # same edge the opposite way; widening the pool does not change
        # that geometry, only where the search space itself is drawn
        # from.

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

        admission_object, admission_key = self._resolve_admission(edge)

        admission_occupants = self._admission_occupancy.get(admission_key, set())
        admission_occupants.discard(occupant.occupant_id)

        queue = self._admission_queues.get(admission_key)

        if queue:

            # Admission Control V4/V7. Peek (never blindly pop) so a
            # blocked queue head is never popped only to be re-queued
            # at the BACK by _handle_try_enter_edge()'s own queueing
            # branch, which would silently break FIFO order for whoever
            # is behind them. When discharge_model and buffer_model are
            # both None, _can_admit() reduces to exactly
            # "current_admitted < capacity", which is always True here
            # (exactly one slot just freed, by construction) -- so this
            # branch is taken unconditionally, preserving the original
            # pop-then-schedule behavior byte-for-byte.
            #
            # The queue head's OWN to_node is resolved fresh, not
            # assumed to match the departing occupant's edge -- a
            # shared FlowRegion's admission_key can group occupants
            # bound for different destinations.
            head_occupant = self._occupants[queue[0]]
            head_to_node = head_occupant.route.nodes[head_occupant.current_edge_index + 1]

            if self._can_admit(admission_object, admission_key, head_to_node, time):

                next_occupant_id = queue.popleft()

                # Admission Control V7 -- see _handle_try_enter_edge()'s
                # own comment: marks this occupant so that IF their
                # deferred re-verification (about to be scheduled below)
                # fails, they are returned to the FRONT of the queue,
                # never the back.
                self._requeue_at_front.add(next_occupant_id)

                self._schedule(time, self.TRY_ENTER_EDGE, next_occupant_id)

            else:

                self._maybe_schedule_retry(admission_object, admission_key, head_to_node, time)

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

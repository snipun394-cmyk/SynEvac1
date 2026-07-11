import heapq
import itertools

from collections import deque

from navigation.edge import Edge

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

    def __init__(
        self,
        engine,
        occupant_simulator=None,
        capacity_model=None,
        congestion_model=None,
    ):

        self.engine = engine
        self.graph = engine.graph

        self.occupant_simulator = occupant_simulator or OccupantSimulator(engine)
        self.capacity_model = capacity_model or DefaultCapacityModel()
        self.congestion_model = congestion_model or DefaultCongestionModel()

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

        self._edge_occupancy = {}
        self._node_occupancy = {}
        self._edge_queues = {}
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
        # `decision.action_type` at all -- movement vs. no movement is
        # determined purely by whether goal_id/route is set, keeping
        # this method exactly as decoupled from behavioral vocabulary
        # as the rest of Simulation is from engineering models.

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

        for queue in self._edge_queues.values():
            if occupant_id in queue:
                queue.remove(occupant_id)

    # =====================================================

    def _schedule(self, time, kind, occupant_id):

        heapq.heappush(
            self._event_heap,
            (time, next(self._counter), kind, occupant_id, self._generation[occupant_id]),
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

        time, _, kind, occupant_id, generation = heapq.heappop(self._event_heap)

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

    def _handle_try_enter_edge(self, time, occupant):

        edge = occupant.route.edges[occupant.current_edge_index]

        capacity = self.capacity_model.capacity(edge)
        current_on_edge = len(self._edge_occupancy.get(edge.id, ()))

        if current_on_edge < capacity:

            self._admit_onto_edge(time, occupant, edge)

        else:

            self._edge_queues.setdefault(edge.id, deque()).append(
                occupant.occupant_id
            )
            self._queue_join_time[occupant.occupant_id] = time

            occupant.state = OccupantState.QUEUED
            self._total_queue_events += 1

    # =====================================================

    def _admit_onto_edge(self, time, occupant, edge):

        from_node = occupant.route.nodes[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        self._remove_node_occupant(from_node.id, occupant.occupant_id)

        edge_occupants = self._edge_occupancy.setdefault(edge.id, set())
        edge_occupants.add(occupant.occupant_id)
        self._track_peak(self._peak_edge_occupancy, edge.id, len(edge_occupants))

        occupant.state = OccupantState.TRAVERSING

        capacity = self.capacity_model.capacity(edge)
        other_occupants = len(edge_occupants) - 1
        speed_factor = self.congestion_model.speed_factor(
            edge, other_occupants, capacity,
        )
        effective_speed = occupant.walking_speed * speed_factor

        distance = edge.walking_distance or 0.0
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

    def _handle_arrive_at_node(self, time, occupant):

        edge = occupant.route.edges[occupant.current_edge_index]
        to_node = occupant.route.nodes[occupant.current_edge_index + 1]

        edge_occupants = self._edge_occupancy.get(edge.id, set())
        edge_occupants.discard(occupant.occupant_id)

        queue = self._edge_queues.get(edge.id)

        if queue:

            next_occupant_id = queue.popleft()

            self._schedule(time, self.TRY_ENTER_EDGE, next_occupant_id)

        self._add_node_occupant(to_node.id, occupant.occupant_id)

        occupant.current_edge_index += 1

        if occupant.current_edge_index >= len(occupant.route.edges):

            occupant.state = OccupantState.ARRIVED
            self._arrival_time[occupant.occupant_id] = time
            self._remove_node_occupant(to_node.id, occupant.occupant_id)

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

    def _remove_node_occupant(self, node_id, occupant_id):

        self._node_occupancy.get(node_id, set()).discard(occupant_id)

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

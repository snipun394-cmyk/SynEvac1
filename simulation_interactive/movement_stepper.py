from typing import Callable, Optional, Tuple

from simulator.coordinator import MultiAgentSimulation
from simulator.multi_agent_result import MultiAgentSimulationResult
from simulator.occupant import OccupantState


class MovementStepper:

    # The only file in simulation_interactive/ that reaches into
    # MultiAgentSimulation's "private" (leading-underscore) instance
    # state. Centralizing every such read here keeps the boundary-
    # crossing auditable in one place, mirroring the precedent the
    # codebase's own test suite already sets (test_multi_agent_
    # simulation.py::MostRecentDecisionWinsTests reaches directly into
    # sim._total_queue_events).
    #
    # Never mutates simulator internals directly -- the only way this
    # class changes simulation state is via the coordinator's own public
    # submit_decision()/_process_next_event(), never by assigning into
    # _occupants/_edge_occupancy/_edge_queues/_node_occupancy/_generation.
    #
    # advance_to() is the seam simulator/coordinator.py's own comment
    # anticipates ("Decomposed as a loop over _process_next_event() ...
    # so a future live-stepping consumer could drive this one event at
    # a time without restructuring", coordinator.py:251-253): it pops
    # the event heap one entry at a time instead of draining it in
    # bulk, because _handle_arrive_at_node schedules an occupant's next
    # TRY_ENTER_EDGE at the *same* timestamp as their arrival -- a bulk
    # "advance to time T" would process both before any external code
    # got a chance to reroute the occupant in between.

    def __init__(self, simulation: MultiAgentSimulation):

        self._simulation = simulation

    # =====================================================

    @property
    def is_drained(self) -> bool:

        return not self._simulation._event_heap

    # =====================================================

    def peek_next_time(self) -> Optional[float]:

        heap = self._simulation._event_heap

        return heap[0][0] if heap else None

    # =====================================================

    def advance_to(
        self,
        target_time: float,
        on_arrive_at_node: Optional[Callable[[str, float], None]] = None,
    ) -> None:

        simulation = self._simulation

        while simulation._event_heap and simulation._event_heap[0][0] <= target_time:

            time, _, kind, occupant_id, generation = simulation._event_heap[0]

            simulation._process_next_event()

            if on_arrive_at_node is None:
                continue

            if kind != MultiAgentSimulation.ARRIVE_AT_NODE:
                continue

            if generation != simulation._generation.get(occupant_id):
                # Stale -- was superseded before this event was even
                # popped; _process_next_event() already dropped it.
                continue

            occupant = simulation._occupants.get(occupant_id)

            if occupant is not None and occupant.state == OccupantState.AT_NODE:
                on_arrive_at_node(occupant_id, time)

    # =====================================================
    # Read-only occupant introspection.
    # =====================================================

    def occupant_state(self, occupant_id: str) -> Optional[OccupantState]:

        occupant = self._simulation._occupants.get(occupant_id)

        return occupant.state if occupant is not None else None

    # =====================================================

    def occupant_current_node_id(self, occupant_id: str) -> Optional[str]:

        occupant = self._simulation._occupants.get(occupant_id)

        if occupant is None or occupant.route is None:
            return None

        if occupant.current_edge_index >= len(occupant.route.nodes):
            return None

        return occupant.route.nodes[occupant.current_edge_index].id

    # =====================================================

    def occupant_remaining_edge_ids(self, occupant_id: str) -> Tuple[str, ...]:

        occupant = self._simulation._occupants.get(occupant_id)

        if occupant is None or occupant.route is None:
            return ()

        return tuple(
            edge.id for edge in occupant.route.edges[occupant.current_edge_index:]
        )

    # =====================================================

    def occupant_remaining_node_ids(self, occupant_id: str) -> Tuple[str, ...]:

        occupant = self._simulation._occupants.get(occupant_id)

        if occupant is None or occupant.route is None:
            return ()

        return tuple(
            node.id for node in occupant.route.nodes[occupant.current_edge_index:]
        )

    # =====================================================
    # Read-only congestion introspection.
    # =====================================================

    def current_edge_occupancy(self, edge_id: str) -> int:

        return len(self._simulation._edge_occupancy.get(edge_id, ()))

    # =====================================================

    def current_queue_length(self, edge_id: str) -> int:

        return len(self._simulation._edge_queues.get(edge_id, ()))

    # =====================================================

    def current_node_occupancy(self, node_id: str) -> int:

        return len(self._simulation._node_occupancy.get(node_id, ()))

    # =====================================================

    def snapshot_result(self) -> MultiAgentSimulationResult:

        # _build_result() is a pure projection of already-mutated
        # instance dicts -- safe to call at any point, not just after
        # the event heap is drained.
        return self._simulation._build_result()

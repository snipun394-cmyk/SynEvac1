from dataclasses import dataclass, field
from typing import Dict, List, Optional

from navigation.edge import Edge
from navigation.node import Node
from pathfinding.route import Route

from simulator.occupant import OccupantState


@dataclass
class OccupantTimelineStep:

    # One realized hop -- distinct from simulator.result.SimulationStep
    # (Single Occupant Simulation V1, frozen, not reused here) because
    # this one carries congestion/queueing information that a
    # single-occupant, uncontested walk never has.

    index: int
    from_node: Node
    to_node: Node
    edge: Edge

    queue_wait_time: float
    start_time: float
    end_time: float

    # =====================================================

    @property
    def distance(self):

        return self.edge.walking_distance

    # =====================================================

    @property
    def is_floor_transition(self):

        # Same structural definition as SimulationStep -- deliberately
        # not "edge.edge_type == Edge.STAIR" -- see simulator/result.py.
        return (
            bool(self.from_node.floor_id)
            and bool(self.to_node.floor_id)
            and self.from_node.floor_id != self.to_node.floor_id
        )


@dataclass
class OccupantTimeline:

    # One occupant's full realized record for a MultiAgentSimulation
    # run -- their fixed Route (from OccupantSimulator, unchanged
    # throughout), plus the congestion/queue-aware timing the
    # coordinator computed for it.

    occupant_id: str

    route: Optional[Route]
    steps: List[OccupantTimelineStep]

    state: OccupantState

    depart_time: float
    arrival_time: Optional[float]

    # =====================================================

    @property
    def total_queue_wait_time(self):

        return sum(step.queue_wait_time for step in self.steps)

    # =====================================================

    @property
    def visited_node_ids(self):

        return self.route.node_ids if self.route is not None else []

    # =====================================================

    @property
    def traversed_edge_ids(self):

        return self.route.edge_ids if self.route is not None else []


@dataclass
class MultiAgentSimulationResult:

    occupants: Dict[str, OccupantTimeline]

    # max(arrival_time) across every ARRIVED occupant -- None if
    # nobody arrived (e.g. every occupant was unreachable).
    total_evacuation_time: Optional[float]

    unreachable_occupant_ids: List[str] = field(default_factory=list)

    peak_edge_occupancy: Dict[str, int] = field(default_factory=dict)
    peak_node_occupancy: Dict[str, int] = field(default_factory=dict)

    # How many times, across every occupant, an entry onto an edge
    # had to wait at all -- a single, simple congestion indicator for
    # the whole run.
    total_queue_events: int = 0

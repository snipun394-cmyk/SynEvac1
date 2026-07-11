from dataclasses import dataclass
from typing import List, Optional

from navigation.edge import Edge
from navigation.node import Node
from pathfinding.route import Route


@dataclass
class SimulationStep:

    # One discrete hop along the Route being executed -- from one
    # Node to the next, over one Edge. Anything derivable from
    # `edge`/`from_node`/`to_node` alone (distance, whether this
    # crosses a floor) is a property here, never a separately stored
    # value -- only start_time/end_time are genuinely new information,
    # since they depend on everything walked before this step.

    index: int
    from_node: Node
    to_node: Node
    edge: Edge

    start_time: Optional[float]
    end_time: Optional[float]

    # =====================================================

    @property
    def distance(self):

        return self.edge.walking_distance

    # =====================================================

    @property
    def is_floor_transition(self):

        # Deliberately structural (floor_id comparison) rather than
        # "edge.edge_type == Edge.STAIR", so it stays correct even if
        # a future edge type could also cross floors. Outside's
        # floor_id is "" (see navigation/node.py), so walking out an
        # Exit to Outside is correctly never counted as a floor
        # transition -- only a genuine Stair between two real floors
        # is.
        return (
            bool(self.from_node.floor_id)
            and bool(self.to_node.floor_id)
            and self.from_node.floor_id != self.to_node.floor_id
        )


@dataclass
class SimulationResult:

    # The structured record of one occupant's attempt to walk a
    # Route from start_node_id to goal_node_id. reached_goal is False
    # (with route=None, steps=[]) when PathfindingEngine could not
    # find a route at all -- Simulation never raises for an
    # unreachable goal, the same "no route, not an exception"
    # contract PathfindingEngine itself uses.
    #
    # visited_node_ids/traversed_edge_ids/floor_transitions are
    # properties derived from `route`/`steps`, not stored separately
    # -- `route` is already the single source of truth for the path
    # actually walked.

    occupant_id: str

    route: Optional[Route]
    steps: List[SimulationStep]

    start_node_id: Optional[str]
    goal_node_id: Optional[str]

    reached_goal: bool

    total_elapsed_time: Optional[float]
    total_distance: Optional[float]

    # =====================================================

    @property
    def visited_node_ids(self):

        return self.route.node_ids if self.route is not None else []

    # =====================================================

    @property
    def traversed_edge_ids(self):

        return self.route.edge_ids if self.route is not None else []

    # =====================================================

    @property
    def floor_transitions(self):

        return [
            step
            for step in self.steps
            if step.is_floor_transition
        ]

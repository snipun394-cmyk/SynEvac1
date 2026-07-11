from dataclasses import dataclass
from typing import List, Optional

from navigation.edge import Edge
from navigation.node import Node


@dataclass
class Route:

    # The result of a routing query -- an ordered walk through the
    # Navigation Graph from the query's start node to whichever node
    # it ended at (the goal, or the first node matching a predicate
    # for nearest_exit()/nearest_assembly_point()). len(edges) is
    # always len(nodes) - 1; a single-node Route (edges == []) means
    # start and goal were the same node.
    #
    # PathfindingEngine returns None instead of a Route when no path
    # exists -- there is no "empty"/"unreachable" Route value, only
    # "no Route".

    nodes: List[Node]
    edges: List[Edge]

    # Sum of cost_model.cost(edge) over `edges` -- what the search
    # actually optimized for, and the only value alternative_paths()
    # ranks candidates by.
    total_cost: float

    # Sum of edge.walking_distance over `edges`, or None if any edge
    # on the route has no derivable walking_distance. Kept separate
    # from total_cost because a custom CostModel (e.g. hazard-aware)
    # may optimize for something other than physical distance.
    total_distance: Optional[float]

    # =====================================================

    @property
    def node_ids(self):

        return [node.id for node in self.nodes]

    # =====================================================

    @property
    def edge_ids(self):

        return [edge.id for edge in self.edges]

    # =====================================================

    @property
    def start(self):

        return self.nodes[0] if self.nodes else None

    # =====================================================

    @property
    def goal(self):

        return self.nodes[-1] if self.nodes else None

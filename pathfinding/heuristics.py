class Heuristic:

    # Interface a future A* heuristic must implement. Deliberately
    # given the graph as well as both nodes (not just their ids), so
    # a heuristic can key into whatever future per-node data it needs
    # without PathfindingEngine ever needing to know what that is.

    def estimate(self, graph, node, goal_node):

        raise NotImplementedError


class ZeroHeuristic(Heuristic):

    # The only heuristic implemented in V1: always 0. Trivially
    # admissible and consistent, so A* using it is exactly as optimal
    # as Dijkstra -- it still exercises the real A* code path (the
    # f = g + h priority ordering in PathfindingEngine._search()),
    # just without requiring node coordinates, which Node does not
    # expose today (see navigation/node.py -- deliberately not
    # changed by this milestone). A future spatial heuristic
    # (straight-line distance) can be added as another Heuristic
    # implementation, with zero changes to PathfindingEngine, if/when
    # Node grows a position property.

    def estimate(self, graph, node, goal_node):

        return 0.0

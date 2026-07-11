import heapq
import math


# The only thing SmokePropagationModel reads off NavigationGraph:
# static topology (Node.id, Edge.edge_type, Edge.walking_distance),
# never Edge.traversable and never anything from navigation.cost or
# pathfinding -- this is a one-time structural read at construction,
# not a dependency on Navigation's routing behavior. Deliberately
# ignores Edge.traversable: a locked door or blocked exit stops a
# person, not smoke, which seeps through/under a closed opening
# regardless of its lock/active state -- traversal passability and
# smoke passability are different questions, and this module only
# ever answers the second one.
DEFAULT_HOP_DISTANCE = 1.0


def shortest_graph_distances(graph, source_node_id, edge_types):

    # Self-contained Dijkstra over graph.find_neighbors(), restricted
    # to edges whose edge_type is in edge_types -- deliberately not
    # PathfindingEngine.distances_from(): that method costs edges
    # through a CostModel and respects Edge.traversable, both of which
    # are routing concerns this module has no business depending on.
    # Returns {node_id: distance} for every node reachable from
    # source_node_id through only the given edge types (source_node_id
    # included, at distance 0.0); a node absent from the result is
    # simply not reachable that way, same "absent means unreachable"
    # convention PathfindingEngine.distances_from() already uses.

    if graph.find_node(source_node_id) is None:
        return {}

    distances = {source_node_id: 0.0}
    visited = set()

    frontier = [(0.0, source_node_id)]

    while frontier:

        distance, node_id = heapq.heappop(frontier)

        if node_id in visited:
            continue

        visited.add(node_id)

        node = graph.find_node(node_id)

        if node is None:
            continue

        for neighbor_node, edge in graph.find_neighbors(node):

            if edge.edge_type not in edge_types:
                continue

            if neighbor_node.id in visited:
                continue

            weight = (
                edge.walking_distance
                if edge.walking_distance is not None
                else DEFAULT_HOP_DISTANCE
            )

            candidate = distance + weight

            if candidate < distances.get(neighbor_node.id, math.inf):

                distances[neighbor_node.id] = candidate
                heapq.heappush(frontier, (candidate, neighbor_node.id))

    return distances

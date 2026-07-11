import heapq
import itertools
import math

from navigation.cost import DefaultCostModel
from navigation.node import Node

from pathfinding.heuristics import ZeroHeuristic
from pathfinding.route import Route


class PathfindingEngine:

    # The one routing entry point every future subsystem (Building
    # Analysis, Simulation, Scenario Generation, RL, Live AI Decision
    # Support) is meant to consume. It depends only on the Navigation
    # Graph's own public API (find_node/find_neighbors, Node.id/
    # node_type, Edge.id/traversable) and on the CostModel interface
    # -- never on Door, Exit, Staircase, Zone, or any other
    # engineering model. Swapping in a hazard-aware CostModel (see
    # navigation/cost.py) changes every routing outcome here with no
    # changes to this file.

    def __init__(self, graph, cost_model=None, heuristic=None):

        self.graph = graph

        self.cost_model = cost_model or DefaultCostModel()
        self.heuristic = heuristic or ZeroHeuristic()

    # =====================================================
    # Public queries
    # =====================================================

    def shortest_path(self, start_id, goal_id, algorithm="dijkstra"):

        if algorithm == "dijkstra":
            return self.dijkstra(start_id, goal_id)

        if algorithm == "a_star":
            return self.a_star(start_id, goal_id)

        raise ValueError(
            f"Unknown algorithm '{algorithm}' -- expected "
            f"'dijkstra' or 'a_star'."
        )

    # =====================================================

    def dijkstra(self, start_id, goal_id):

        goal_node = self.graph.find_node(goal_id)

        if goal_node is None:
            return None

        return self._search(
            start_id,
            is_goal=lambda node: node.id == goal_id,
            goal_node=goal_node,
            use_heuristic=False,
        )

    # =====================================================

    def a_star(self, start_id, goal_id):

        goal_node = self.graph.find_node(goal_id)

        if goal_node is None:
            return None

        return self._search(
            start_id,
            is_goal=lambda node: node.id == goal_id,
            goal_node=goal_node,
            use_heuristic=True,
        )

    # =====================================================

    def nearest_node(self, start_id, predicate):

        # Generic "closest node matching predicate" query -- Dijkstra
        # with early exit at the first matching node popped off the
        # frontier. Dijkstra always pops nodes in non-decreasing cost
        # order, so the first match popped is guaranteed nearest. No
        # heuristic: there is no single goal node to estimate a
        # distance to.

        return self._search(
            start_id,
            is_goal=predicate,
            goal_node=None,
            use_heuristic=False,
        )

    # =====================================================

    def nearest_exit(self, start_id):

        # Every Exit edge leads to the single shared Outside node
        # (see NavigationGraphGenerator._add_exit_edges), so "nearest
        # reachable Exit" and "nearest node of type Outside" are the
        # same query. The last edge of the returned Route is always
        # the specific Exit that was used to leave.

        return self.nearest_node(
            start_id,
            lambda node: node.node_type == Node.OUTSIDE,
        )

    # =====================================================

    def nearest_assembly_point(self, start_id):

        return self.nearest_node(
            start_id,
            lambda node: node.node_type == Node.ASSEMBLY_POINT,
        )

    # =====================================================

    def alternative_paths(self, start_id, goal_id, k=3):

        # Yen's algorithm: up to `k` distinct, loopless routes from
        # start to goal, in non-decreasing total_cost order -- index
        # 0 is always exactly what dijkstra() would return. Built
        # entirely on top of _search()'s exclusion parameters, so
        # every candidate is still costed only through
        # self.cost_model, same as a plain shortest_path() query.
        #
        # This is the "alternative-route generation" extension point:
        # a future Scenario Generation subsystem can ask for several
        # distinct evacuation routes, not just the single cheapest
        # one.

        if k < 1:
            return []

        shortest = self.dijkstra(start_id, goal_id)

        if shortest is None:
            return []

        accepted = [shortest]
        accepted_signatures = {tuple(shortest.edge_ids)}

        candidates = []
        counter = itertools.count()

        while len(accepted) < k:

            previous = accepted[-1]
            previous_node_ids = previous.node_ids

            for i in range(len(previous_node_ids) - 1):

                spur_node_id = previous_node_ids[i]
                root_node_ids = previous_node_ids[: i + 1]

                # Every already-accepted route that shares this same
                # root prefix must not be reproduced -- block the
                # edge each of them takes out of the spur node.
                excluded_edge_ids = {
                    route.edges[i].id
                    for route in accepted
                    if route.node_ids[: i + 1] == root_node_ids
                    and len(route.edges) > i
                }

                # Every root node except the spur node itself, so the
                # spur search can't loop back into the shared prefix.
                excluded_node_ids = set(root_node_ids[:-1])

                spur_route = self._search(
                    spur_node_id,
                    is_goal=lambda node: node.id == goal_id,
                    goal_node=self.graph.find_node(goal_id),
                    use_heuristic=False,
                    excluded_node_ids=excluded_node_ids,
                    excluded_edge_ids=excluded_edge_ids,
                )

                if spur_route is None:
                    continue

                root_edges = previous.edges[:i]

                root_cost = sum(
                    self.cost_model.cost(edge)
                    for edge in root_edges
                )

                candidate = Route(
                    nodes=previous.nodes[:i] + spur_route.nodes,
                    edges=root_edges + spur_route.edges,
                    total_cost=root_cost + spur_route.total_cost,
                    total_distance=self._total_distance(
                        root_edges + spur_route.edges
                    ),
                )

                signature = tuple(candidate.edge_ids)
                pending_signatures = {item[2] for item in candidates}

                if (
                    signature not in accepted_signatures
                    and signature not in pending_signatures
                ):

                    heapq.heappush(
                        candidates,
                        (
                            candidate.total_cost,
                            next(counter),
                            signature,
                            candidate,
                        ),
                    )

            if not candidates:
                break

            _, _, signature, next_route = heapq.heappop(candidates)

            accepted.append(next_route)
            accepted_signatures.add(signature)

        return accepted

    # =====================================================

    def distances_from(
        self,
        start_id,
        excluded_node_ids=frozenset(),
        excluded_edge_ids=frozenset(),
    ):

        # Single-source Dijkstra to *every* node start_id can reach,
        # in one pass -- the primitive whole-graph analyses (e.g.
        # Building Analysis's exit-service-area / max-travel-distance
        # reports) need, instead of running nearest_exit()/dijkstra()
        # once per node in the graph. distances_from(Node.
        # OUTSIDE_NODE_ID) gives every zone's distance to its nearest
        # exit in a single search.
        #
        # Returns {node_id: Route} for every reachable node (start_id
        # included, with a trivial zero-cost Route to itself).
        # Unreachable nodes are simply absent -- same "no Route"
        # contract as every other query on this engine. Never uses a
        # heuristic: there is no single goal node to estimate towards.

        _, best_cost, predecessor, predecessor_edge = self._relax(
            start_id,
            excluded_node_ids=excluded_node_ids,
            excluded_edge_ids=excluded_edge_ids,
        )

        return {
            node_id: self._reconstruct(
                node_id,
                start_id,
                predecessor,
                predecessor_edge,
                cost,
            )
            for node_id, cost in best_cost.items()
        }

    # =====================================================
    # Internal search core -- Dijkstra when use_heuristic is False,
    # A* when True. `is_goal` is a predicate over Node rather than a
    # fixed id so nearest_node()/nearest_exit()/nearest_assembly_
    # point() can share this exact code path with dijkstra()/
    # a_star(). excluded_node_ids/excluded_edge_ids exist solely for
    # alternative_paths()'s spur searches.
    # =====================================================

    def _search(
        self,
        start_id,
        is_goal,
        goal_node=None,
        use_heuristic=False,
        excluded_node_ids=frozenset(),
        excluded_edge_ids=frozenset(),
    ):

        goal_id, best_cost, predecessor, predecessor_edge = self._relax(
            start_id,
            is_goal=is_goal,
            goal_node=goal_node,
            use_heuristic=use_heuristic,
            excluded_node_ids=excluded_node_ids,
            excluded_edge_ids=excluded_edge_ids,
        )

        if goal_id is None:
            return None

        return self._reconstruct(
            goal_id,
            start_id,
            predecessor,
            predecessor_edge,
            best_cost[goal_id],
        )

    # =====================================================
    # Shared relaxation loop underneath _search() and distances_from().
    # With `is_goal` set, stops and returns the matching node's id the
    # moment it's popped (Dijkstra/A* pop in non-decreasing cost
    # order, so the first match is provably nearest/optimal). With
    # `is_goal` left None, runs to exhaustion, reaching every node
    # start_id can reach -- distances_from()'s single-source query.
    # =====================================================

    def _relax(
        self,
        start_id,
        is_goal=None,
        goal_node=None,
        use_heuristic=False,
        excluded_node_ids=frozenset(),
        excluded_edge_ids=frozenset(),
    ):

        start_node = self.graph.find_node(start_id)

        if start_node is None or start_id in excluded_node_ids:
            return None, {}, {}, {}

        counter = itertools.count()

        best_cost = {start_id: 0.0}
        predecessor = {}
        predecessor_edge = {}
        visited = set()

        frontier = [
            (
                self._heuristic(start_node, goal_node, use_heuristic),
                next(counter),
                start_id,
            )
        ]

        while frontier:

            _, _, current_id = heapq.heappop(frontier)

            if current_id in visited:
                continue

            visited.add(current_id)

            current_node = self.graph.find_node(current_id)

            if is_goal is not None and is_goal(current_node):
                return current_id, best_cost, predecessor, predecessor_edge

            for neighbor_node, edge in self.graph.find_neighbors(current_node):

                if neighbor_node.id in visited:
                    continue

                if neighbor_node.id in excluded_node_ids:
                    continue

                if edge.id in excluded_edge_ids:
                    continue

                if not edge.traversable:
                    continue

                tentative_cost = (
                    best_cost[current_id] + self.cost_model.cost(edge)
                )

                if tentative_cost < best_cost.get(neighbor_node.id, math.inf):

                    best_cost[neighbor_node.id] = tentative_cost
                    predecessor[neighbor_node.id] = current_id
                    predecessor_edge[neighbor_node.id] = edge

                    priority = tentative_cost + self._heuristic(
                        neighbor_node, goal_node, use_heuristic
                    )

                    heapq.heappush(
                        frontier,
                        (priority, next(counter), neighbor_node.id),
                    )

        return None, best_cost, predecessor, predecessor_edge

        return None

    # =====================================================

    def _heuristic(self, node, goal_node, use_heuristic):

        if not use_heuristic or goal_node is None:
            return 0.0

        return self.heuristic.estimate(self.graph, node, goal_node)

    # =====================================================

    def _reconstruct(
        self,
        goal_id,
        start_id,
        predecessor,
        predecessor_edge,
        total_cost,
    ):

        node_ids = [goal_id]
        edges = []

        current = goal_id

        while current != start_id:

            edges.append(predecessor_edge[current])
            current = predecessor[current]
            node_ids.append(current)

        node_ids.reverse()
        edges.reverse()

        nodes = [
            self.graph.find_node(node_id)
            for node_id in node_ids
        ]

        return Route(
            nodes=nodes,
            edges=edges,
            total_cost=total_cost,
            total_distance=self._total_distance(edges),
        )

    # =====================================================

    def _total_distance(self, edges):

        total = 0.0

        for edge in edges:

            if edge.walking_distance is None:
                return None

            total += edge.walking_distance

        return total

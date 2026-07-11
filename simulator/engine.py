from simulator.result import SimulationResult, SimulationStep


class OccupantSimulator:

    # Executes a single occupant's evacuation along a Route that
    # PathfindingEngine already computed -- this class never computes
    # a route itself (no Dijkstra/A*/graph search lives here). It
    # only walks node-by-node, edge-by-edge through what Pathfinding
    # Engine returns, accumulating elapsed time and distance from
    # Edge's own already-derived traversal_time/walking_distance. A
    # pure execution/proof-of-traversal layer, not a planning one --
    # every method below either delegates to `self.engine` for the
    # route or accepts one the caller already obtained.

    def __init__(self, engine):

        self.engine = engine
        self.graph = engine.graph

    # =====================================================

    def simulate_to_goal(self, start_id, goal_id, occupant_id="occupant-1"):

        route = self.engine.dijkstra(start_id, goal_id)

        return self._run(route, start_id, goal_id, occupant_id)

    # =====================================================

    def evacuate(self, start_id, occupant_id="occupant-1"):

        # Convenience for the most common single-occupant scenario:
        # walk to whichever Exit is nearest. Still just delegates to
        # PathfindingEngine.nearest_exit() for the route itself.

        route = self.engine.nearest_exit(start_id)

        goal_id = route.goal.id if route is not None else None

        return self._run(route, start_id, goal_id, occupant_id)

    # =====================================================

    def simulate_route(self, route, occupant_id="occupant-1"):

        # Executes a Route the caller already obtained from
        # PathfindingEngine by any means (dijkstra/a_star/
        # nearest_assembly_point/alternative_paths/...) -- the one
        # entry point that makes no routing decision of its own at
        # all, not even which PathfindingEngine method to call.

        if route is None or not route.nodes:
            return self._run(None, None, None, occupant_id)

        return self._run(route, route.start.id, route.goal.id, occupant_id)

    # =====================================================

    def _run(self, route, start_id, goal_id, occupant_id):

        if route is None or not route.nodes:

            return SimulationResult(
                occupant_id=occupant_id,
                route=None,
                steps=[],
                start_node_id=start_id,
                goal_node_id=goal_id,
                reached_goal=False,
                total_elapsed_time=None,
                total_distance=None,
            )

        steps = []

        elapsed = 0.0
        distance = 0.0

        time_known = True
        distance_known = True

        for index, edge in enumerate(route.edges):

            from_node = route.nodes[index]
            to_node = route.nodes[index + 1]

            start_time = elapsed if time_known else None
            step_time = edge.traversal_time

            if step_time is None:
                time_known = False
                end_time = None
            else:
                elapsed += step_time
                end_time = elapsed

            if edge.walking_distance is None:
                distance_known = False
            else:
                distance += edge.walking_distance

            steps.append(
                SimulationStep(
                    index=index,
                    from_node=from_node,
                    to_node=to_node,
                    edge=edge,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        return SimulationResult(
            occupant_id=occupant_id,
            route=route,
            steps=steps,
            start_node_id=start_id,
            goal_node_id=goal_id,
            reached_goal=True,
            total_elapsed_time=elapsed if time_known else None,
            total_distance=distance if distance_known else None,
        )

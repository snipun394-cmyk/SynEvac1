from dataclasses import dataclass
from typing import Optional

from pathfinding.route import Route


@dataclass
class RouteChoice:

    # Navigation Stage's output -- ephemeral, internal to the
    # Behavior Layer's pipeline. HumanBehaviorLayer folds this into
    # the final BehaviorDecision handed to Simulation.

    goal_id: Optional[str]
    route: Optional[Route] = None


class RouteChoiceStrategy:

    # Navigation Stage interface: "given that movement is required,
    # which goal/route does this occupant actually take" -- consumes
    # PathfindingEngine (via context.engine), never computes a route
    # itself. Mirrors CostModel/Heuristic/CapacityModel/
    # CongestionModel's established pattern.

    def choose(self, context) -> RouteChoice:

        raise NotImplementedError


class ShortestRouteChoiceStrategy(RouteChoiceStrategy):

    # The trivial default -- proves the interface without implementing
    # any real behavior. Every occupant simply takes PathfindingEngine's
    # nearest exit, exactly what MultiAgentSimulation.evacuate() already
    # does today when nothing else is plugged in.

    def choose(self, context) -> RouteChoice:

        route = context.engine.nearest_exit(context.start_id)
        goal_id = route.goal.id if route is not None else None

        return RouteChoice(goal_id=goal_id, route=route)

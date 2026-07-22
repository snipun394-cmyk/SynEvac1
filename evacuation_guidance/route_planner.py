from typing import Optional, Tuple

from hazard.severity import HazardSeverity

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.engine import PathfindingEngine
from pathfinding.route import Route

from evacuation_guidance.models import GuidanceConfig, RouteStatus


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase
# 1/5/6 -- turns "Zone Z3 should use Exit E2" into an actual,
# graph-valid, ordered Route. Deliberately owns a SMALL, independent
# copy of the same hazard-exclusion logic trajectory_intelligence.
# route_progress.SafeRouteCalculator/evacuation_recommendation.ranking.
# SafeExitDistanceCalculator each already own -- this codebase's own
# established "sibling live-intelligence packages never import each
# other's engine internals" convention (see tests/
# test_evacuation_recommendation_architecture_guards.py's own identical
# allow-list discipline).
#
# UNLIKE evacuation_recommendation.ranking.SafeExitDistanceCalculator
# (which immediately collapses every Route to a plain float distance --
# see that module's own docstring, confirmed by direct investigation),
# this resolver KEEPS the full Route (nodes/edges) it computes, because
# the whole point of this package is the ordered path itself.
#
# Never fabricates a route: if the recommended exit's own edge cannot
# be found, is not traversable, or the occupant's zone cannot reach it
# without crossing a hazard-excluded zone, this returns (None, code) --
# never a fabricated/approximate path (Phase 4/10's own "do not
# fabricate a route" requirement).
# =====================================================


_SEVERITY_RANK = {
    HazardSeverity.NONE: 0, HazardSeverity.LOW: 1, HazardSeverity.MODERATE: 2,
    HazardSeverity.HIGH: 3, HazardSeverity.CRITICAL: 4,
}


def excluded_zone_ids(building_state, config: GuidanceConfig) -> frozenset:

    if building_state is None or building_state.hazard_summary is None:
        return frozenset()

    floor_rank = _SEVERITY_RANK[config.hazard_unsafe_severity_floor]

    return frozenset(
        zone_id
        for zone_id, severity in building_state.hazard_summary.zone_severities.items()
        if _SEVERITY_RANK[severity] >= floor_rank
    )


def _find_exit_edge(graph, exit_id: str) -> Optional[Edge]:

    for edge in graph.edges:

        if edge.edge_type == Edge.EXIT and edge.id == exit_id:
            return edge

    return None


def _exit_zone_id(edge: Edge) -> str:

    return edge.from_node if edge.from_node != Node.OUTSIDE_NODE_ID else edge.to_node


def _edge_distance(edge: Edge) -> float:

    return edge.walking_distance if edge.walking_distance is not None else edge.traversal_cost


def resolve_route(
    graph, zone_id: str, exit_id: str, building_state, config: Optional[GuidanceConfig] = None,
) -> Tuple[Optional[Route], str]:

    # Returns (Route, RouteStatus) or (None, RouteStatus) with an
    # explanatory status. Never returns a Route when RouteStatus isn't
    # ROUTE_AVAILABLE/ALREADY_AT_EXIT -- engine.py's own validation.py
    # pass re-confirms this before trusting it.

    config = config if config is not None else GuidanceConfig()

    if graph.find_node(zone_id) is None:
        return None, RouteStatus.ROUTE_UNCERTAIN

    exit_edge = _find_exit_edge(graph, exit_id)

    if exit_edge is None:
        return None, RouteStatus.ROUTE_UNCERTAIN

    if not exit_edge.traversable:
        return None, RouteStatus.ROUTE_UNAVAILABLE

    excluded = excluded_zone_ids(building_state, config)
    exit_zone = _exit_zone_id(exit_edge)

    if exit_zone in excluded:
        return None, RouteStatus.ROUTE_UNAVAILABLE

    # The interior-only search deliberately excludes Node.OUTSIDE_NODE_ID
    # (mirrors evacuation_recommendation.ranking.SafeExitDistanceCalculator's
    # own reasoning exactly): without this, a search could "cheat" by
    # walking out one Exit and back in through another, and would never
    # be forced to terminate specifically via THIS recommended exit --
    # Phase 22's own "Guidance Engine finds a valid safe path to E2,
    # never silently chooses E3" requirement, enforced structurally
    # rather than by convention.
    engine = PathfindingEngine(graph)
    interior_excluded = excluded | {Node.OUTSIDE_NODE_ID}

    route_map = engine.distances_from(zone_id, excluded_node_ids=interior_excluded)
    interior_route = route_map.get(exit_zone)

    if interior_route is None:
        return None, RouteStatus.ROUTE_UNAVAILABLE

    outside_node = graph.find_node(Node.OUTSIDE_NODE_ID)

    full_route = Route(
        nodes=list(interior_route.nodes) + [outside_node],
        edges=list(interior_route.edges) + [exit_edge],
        total_cost=interior_route.total_cost + exit_edge.traversal_cost,
        total_distance=(
            (interior_route.total_distance + _edge_distance(exit_edge))
            if interior_route.total_distance is not None else None
        ),
    )

    status = RouteStatus.ALREADY_AT_EXIT if not interior_route.edges else RouteStatus.ROUTE_AVAILABLE

    return full_route, status

from typing import Mapping, Optional

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.engine import PathfindingEngine
from pathfinding.route import Route

from trajectory_intelligence.history import OccupantTrajectoryState
from trajectory_intelligence.models import (
    RouteDistanceTrend, RouteProgressStatus, TrajectoryConfig, hazard_severity_at_least,
)


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 5/6/7/18 -- graph-aware route progress.
# Built entirely on pathfinding.engine.PathfindingEngine +
# navigation.graph.NavigationGraph (both already generic, decision_
# policy-free, and NOT wired into any live package before this
# milestone -- see this milestone's own investigation). Never falls
# back to Euclidean distance when the graph provides connectivity
# (Phase 5's own explicit "moving closer through a wall must not
# count" rule) -- route_distance_m below is always a graph-derived
# quantity (Route.total_distance, or Route.total_cost when a specific
# edge's physical distance genuinely isn't derivable -- still a
# consistent, non-Euclidean, connectivity-respecting ordering, never a
# straight-line shortcut).
# =====================================================


class SafeRouteCalculator:

    # Phase 6/29 -- the one safe-exit-distance-map calculator per
    # engine instance, cached per cycle/safety-state fingerprint exactly
    # like ai_decision.engine.AIDecisionEngine's own established
    # _route_cache/_route_cache_fingerprint pattern (that class's own
    # comment: recomputing this was "~72% of per-tick simulation time"
    # before caching) -- one Dijkstra run from the single shared Outside
    # node serves every occupant this cycle, instead of one shortest-
    # path search per occupant.
    #
    # Safety exclusion is a HARD node exclusion (excluded_node_ids),
    # never a soft cost penalty -- Phase 23's own "never recommend an
    # unsafe route" requirement is mechanically true here: an excluded
    # zone can never appear anywhere on any Route this calculator
    # returns, structurally, not just by a higher score.
    #
    # Only ZONE-LEVEL hazard (BuildingState.hazard_summary.
    # zone_severities) is available live downstream of BuildingStateEstimator
    # (confirmed by this milestone's own investigation -- HazardSnapshot's
    # own edge-level HazardEdgeState.traversable is consumed only inside
    # BuildingStateEstimator._summarize_hazard(), never retained on
    # BuildingState itself) -- structural blockage (locked doors, a
    # blocked Exit) is separately, always available live via Edge.
    # traversable itself (PathfindingEngine._relax() already refuses to
    # cross a non-traversable edge unconditionally, with no configuration
    # needed here).

    def __init__(self, graph, config: Optional[TrajectoryConfig] = None):

        self.graph = graph
        self.config = config if config is not None else TrajectoryConfig()

        self._cache_fingerprint = None
        self._distance_cache: Mapping[str, Route] = {}

    # =====================================================

    def compute(self, building_state) -> Mapping[str, Route]:

        excluded_zone_ids = self._excluded_zone_ids(building_state)

        fingerprint = (
            frozenset(excluded_zone_ids),
            frozenset(edge.id for edge in self.graph.edges if not edge.traversable),
        )

        if fingerprint != self._cache_fingerprint:

            engine = PathfindingEngine(self.graph)
            self._distance_cache = engine.distances_from(
                Node.OUTSIDE_NODE_ID, excluded_node_ids=excluded_zone_ids,
            )
            self._cache_fingerprint = fingerprint

        return self._distance_cache

    # =====================================================

    def _excluded_zone_ids(self, building_state) -> frozenset:

        if building_state is None or building_state.hazard_summary is None:
            return frozenset()

        return frozenset(
            zone_id
            for zone_id, severity in building_state.hazard_summary.zone_severities.items()
            if hazard_severity_at_least(severity, self.config.hazard_unsafe_severity_floor)
        )


# =====================================================


def _route_distance(route: Route) -> float:

    return route.total_distance if route.total_distance is not None else route.total_cost


def _nearest_safe_exit_id(route: Route) -> Optional[str]:

    # Route.nodes[0] is always Node.OUTSIDE_NODE_ID (this calculator
    # only ever runs distances_from(Node.OUTSIDE_NODE_ID, ...)) and
    # Outside connects to the graph exclusively via Exit edges (see
    # navigation.graph_builder._add_exit_edges) -- so edges[0] is
    # always the specific Exit this Route leaves through, mirroring
    # pathfinding.engine.PathfindingEngine.nearest_exit()'s own
    # documented "the last edge of a start->Outside Route is always the
    # Exit used" fact, reversed for a Outside->start Route.

    if not route.edges:
        return None

    return route.edges[0].id


def _floor_of(graph, zone_id: Optional[str]) -> Optional[str]:

    if zone_id is None:
        return None

    node = graph.find_node(zone_id)

    return node.floor_id if node is not None else None


def _has_direct_stair_edge(graph, zone_a_id: str, zone_b_id: str) -> bool:

    for edge in graph.edges:

        if edge.edge_type != Edge.STAIR:
            continue

        if {edge.from_node, edge.to_node} == {zone_a_id, zone_b_id}:
            return True

    return False


def _floor_transition_uncertain(occupant, graph) -> bool:

    # Phase 18 -- a floor change is only trusted if the Navigation Graph
    # itself contains a Stair edge directly connecting the previous and
    # current zone; otherwise this cycle's route is honestly marked
    # ROUTE_UNCERTAIN rather than silently treating an ungrounded floor
    # jump (e.g. a calibration/identity error) as a valid transition.
    # Reuses OccupantHistory.zone_transitions' own most recent record --
    # no separate "last known floor" storage needed.

    transitions = occupant.history.zone_transitions

    if not transitions:
        return False

    last = transitions[-1]

    if last.from_zone_id is None or last.to_zone_id is None:
        return False

    from_floor = _floor_of(graph, last.from_zone_id)
    to_floor = _floor_of(graph, last.to_zone_id)

    if from_floor is None or to_floor is None or from_floor == to_floor:
        return False

    return not _has_direct_stair_edge(graph, last.from_zone_id, last.to_zone_id)


def _trend(samples, config: TrajectoryConfig) -> str:

    if len(samples) < 2:
        return RouteDistanceTrend.UNKNOWN

    window = samples[-config.trend_window_samples:]
    reference = window[0].distance
    latest = window[-1].distance

    delta = latest - reference

    if delta <= -config.route_distance_tolerance_m:
        return RouteDistanceTrend.DECREASING

    if delta >= config.route_distance_tolerance_m:
        return RouteDistanceTrend.INCREASING

    return RouteDistanceTrend.STABLE


def compute_route_progress(
    occupant, time: float, distance_cache: Mapping[str, Route], graph,
    history_state: OccupantTrajectoryState, config: TrajectoryConfig,
):

    # Returns (status, route_distance_m, trend, nearest_safe_exit_id,
    # updated_history_state) -- the caller (engine.py) is responsible
    # for storing updated_history_state back into TrajectoryHistoryStore.

    zone_id = occupant.current_zone_id

    if zone_id is None:
        return RouteProgressStatus.ROUTE_UNCERTAIN, None, RouteDistanceTrend.UNKNOWN, None, history_state

    if _floor_transition_uncertain(occupant, graph):
        return RouteProgressStatus.ROUTE_UNCERTAIN, None, RouteDistanceTrend.UNKNOWN, None, history_state

    route = distance_cache.get(zone_id)

    if route is None:
        return RouteProgressStatus.NO_SAFE_ROUTE, None, RouteDistanceTrend.UNKNOWN, None, history_state

    distance = _route_distance(route)
    exit_id = _nearest_safe_exit_id(route)

    updated_state = history_state.with_route_distance_sample(time, distance)
    trend = _trend(updated_state.route_distance_samples, config)

    if distance <= config.exit_approaching_distance_m:
        return RouteProgressStatus.EXIT_APPROACHING, distance, trend, exit_id, updated_state

    status = {
        RouteDistanceTrend.DECREASING: RouteProgressStatus.PROGRESSING_TOWARD_EXIT,
        RouteDistanceTrend.INCREASING: RouteProgressStatus.MOVING_AWAY_FROM_EXIT,
        RouteDistanceTrend.STABLE: RouteProgressStatus.NOT_PROGRESSING,
        RouteDistanceTrend.UNKNOWN: RouteProgressStatus.ROUTE_UNCERTAIN,
    }[trend]

    return status, distance, trend, exit_id, updated_state

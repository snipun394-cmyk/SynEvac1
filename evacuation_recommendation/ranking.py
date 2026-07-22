from typing import Dict, Mapping, Optional

from hazard.severity import HazardSeverity

from navigation.edge import Edge
from navigation.node import Node

from pathfinding.engine import PathfindingEngine

from evacuation_recommendation.models import ExitCandidate, RecommendationConfig, RecommendationWeights
from evacuation_recommendation.scoring import score_candidate


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 3/4/6
# -- safe exit candidate generation and per-zone ranking.
#
# Deliberately does NOT import trajectory_intelligence.route_progress.
# SafeRouteCalculator (confirmed reusable in isolation, but this
# codebase's own established convention -- see trajectory_intelligence's
# own architecture guard, which forbids it from importing sibling live-
# intelligence engines and instead has it build its own small
# navigation/pathfinding-only calculator -- is that each live-
# intelligence package owns its OWN copy of this kind of graph logic,
# never importing a sibling's. navigation/pathfinding are the shared,
# decision_policy-free foundation every package is meant to build on
# (pathfinding.engine.PathfindingEngine's own docstring: "the one
# routing entry point every future subsystem is meant to consume").
#
# UNLIKE trajectory_intelligence's own SafeRouteCalculator (which finds
# the SINGLE nearest safe exit per zone via one Dijkstra seeded at the
# shared Outside node), this calculator needs every zone's distance to
# EVERY currently-safe exit individually, to support ranked
# alternatives (Phase 6). Outside is a single shared node all Exits
# connect to, so seeding from Outside collapses every exit choice into
# one best route. Instead: for each safe Exit edge, run ONE Dijkstra
# seeded at that exit's own zone with Node.OUTSIDE_NODE_ID itself
# excluded (so the search only ever explores IN-BUILDING connectivity
# -- Door/Stair edges -- and can never "cheat" by walking out one exit
# and back in through another), then add that exit edge's own walking
# distance as the final outdoor step. N safe exits -> N Dijkstra runs
# per cycle, each over the same graph -- cheap at the milestone's own
# named scale (Phase 15: 10 exits), and only re-run at all when the
# hazard/traversability fingerprint actually changes (mirrors ai_decision.
# engine.AIDecisionEngine's own established per-cycle route-cache
# pattern, and trajectory_intelligence.route_progress.SafeRouteCalculator's
# own identical fingerprint-caching discipline).
# =====================================================


def _severity_rank(severity):

    order = {
        HazardSeverity.NONE: 0, HazardSeverity.LOW: 1, HazardSeverity.MODERATE: 2,
        HazardSeverity.HIGH: 3, HazardSeverity.CRITICAL: 4,
    }
    return order[severity]


def _exit_zone_id(edge) -> str:

    return edge.from_node if edge.from_node != Node.OUTSIDE_NODE_ID else edge.to_node


def _edge_distance(edge) -> float:

    return edge.walking_distance if edge.walking_distance is not None else edge.traversal_cost


class SafeExitDistanceCalculator:

    def __init__(self, graph, config: Optional[RecommendationConfig] = None):

        self.graph = graph
        self.config = config if config is not None else RecommendationConfig()

        self._fingerprint = None
        self._distance_cache: Dict[str, Dict[str, float]] = {}
        self._exit_zone_map: Dict[str, str] = {}
        self._safe_exit_ids = ()

    # =====================================================

    def compute(self, building_state) -> Mapping[str, Mapping[str, float]]:

        # Returns {exit_id: {zone_id: route_distance_m}} -- one entry
        # per zone that can genuinely reach this exit without crossing
        # an excluded (hazardous) zone or a non-traversable edge.

        excluded_zone_ids = self._excluded_zone_ids(building_state)

        candidate_exits = [
            edge for edge in self.graph.edges
            if edge.edge_type == Edge.EXIT and edge.traversable and _exit_zone_id(edge) not in excluded_zone_ids
        ]

        fingerprint = (
            frozenset(excluded_zone_ids),
            tuple(sorted(edge.id for edge in candidate_exits)),
            frozenset(edge.id for edge in self.graph.edges if not edge.traversable),
        )

        if fingerprint == self._fingerprint:
            return self._distance_cache

        engine = PathfindingEngine(self.graph)
        distances: Dict[str, Dict[str, float]] = {}
        exit_zone_map: Dict[str, str] = {}

        interior_excluded = excluded_zone_ids | {Node.OUTSIDE_NODE_ID}

        for exit_edge in candidate_exits:

            zone_e = _exit_zone_id(exit_edge)
            exit_zone_map[exit_edge.id] = zone_e

            route_map = engine.distances_from(zone_e, excluded_node_ids=interior_excluded)

            per_zone = {}
            for zone_id, route in route_map.items():

                interior_distance = route.total_distance if route.total_distance is not None else route.total_cost
                per_zone[zone_id] = interior_distance + _edge_distance(exit_edge)

            distances[exit_edge.id] = per_zone

        self._distance_cache = distances
        self._exit_zone_map = exit_zone_map
        self._safe_exit_ids = tuple(sorted(exit_zone_map.keys()))
        self._fingerprint = fingerprint

        return self._distance_cache

    # =====================================================

    @property
    def exit_zone_map(self) -> Mapping[str, str]:

        return self._exit_zone_map

    # =====================================================

    @property
    def safe_exit_ids(self):

        return self._safe_exit_ids

    # =====================================================

    def _excluded_zone_ids(self, building_state) -> frozenset:

        if building_state is None or building_state.hazard_summary is None:
            return frozenset()

        floor_rank = _severity_rank(self.config.hazard_unsafe_severity_floor)

        return frozenset(
            zone_id
            for zone_id, severity in building_state.hazard_summary.zone_severities.items()
            if _severity_rank(severity) >= floor_rank
        )


# =====================================================


def rank_exits_for_zone(
    zone_id, distance_by_exit: Mapping[str, Mapping[str, float]], *, exit_zone_map: Mapping[str, str],
    crowd_snapshot=None, evacuation_progress_snapshot=None, trajectory_snapshot=None,
    emergency_response_snapshot=None, ai_bottleneck_probability=None,
    weights: RecommendationWeights, config: RecommendationConfig,
):

    reachable = {
        exit_id: per_zone[zone_id]
        for exit_id, per_zone in distance_by_exit.items()
        if zone_id in per_zone
    }

    if not reachable:
        return []

    min_distance = min(reachable.values())
    max_distance = max(reachable.values())

    trajectory_support_by_exit = _trajectory_support_by_exit(zone_id, trajectory_snapshot)

    candidates = []

    for exit_id, distance in reachable.items():

        congestion_level, queue_count = _crowd_evidence(exit_id, crowd_snapshot)
        throughput = _throughput_evidence(exit_id, evacuation_progress_snapshot)
        emergency_elevated = _emergency_response_elevated(exit_id, exit_zone_map, emergency_response_snapshot, config)

        score, reason_codes = score_candidate(
            route_distance_m=distance, min_distance=min_distance, max_distance=max_distance,
            congestion_level=congestion_level, queue_count=queue_count, throughput=throughput,
            trajectory_support=trajectory_support_by_exit.get(exit_id), emergency_response_elevated=emergency_elevated,
            ai_bottleneck_probability=ai_bottleneck_probability, weights=weights, config=config,
        )

        candidates.append(ExitCandidate(
            exit_id=exit_id, route_distance_m=distance, congestion_level=congestion_level,
            queue_candidate_count=queue_count, throughput_per_minute=throughput,
            trajectory_support=trajectory_support_by_exit.get(exit_id), emergency_response_elevated=emergency_elevated,
            score=score, reason_codes=reason_codes,
        ))

    # Deterministic ranking regardless of dict/iteration order: score
    # descending, then distance ascending, then exit_id -- a fixed,
    # disclosed tie-break (Phase 13's own "multiple zones deterministic"
    # requirement, mirroring emergency_response.engine._order_zones()'s
    # own tie-break discipline).
    candidates.sort(key=lambda c: (-c.score, c.route_distance_m, c.exit_id))

    return candidates


def _trajectory_support_by_exit(zone_id, trajectory_snapshot):

    if trajectory_snapshot is None:
        return {}

    votes: Dict[str, Dict[str, int]] = {}

    for result in trajectory_snapshot.occupants.values():

        if result.zone_id != zone_id or result.nearest_safe_exit_id is None:
            continue

        exit_votes = votes.setdefault(result.nearest_safe_exit_id, {"PROGRESSING": 0, "MOVING_AWAY": 0})

        if result.route_progress_status in ("PROGRESSING_TOWARD_EXIT", "EXIT_APPROACHING"):
            exit_votes["PROGRESSING"] += 1
        elif result.route_progress_status == "MOVING_AWAY_FROM_EXIT":
            exit_votes["MOVING_AWAY"] += 1

    support = {}
    for exit_id, tally in votes.items():

        if tally["PROGRESSING"] > tally["MOVING_AWAY"]:
            support[exit_id] = "PROGRESSING"
        elif tally["MOVING_AWAY"] > tally["PROGRESSING"]:
            support[exit_id] = "MOVING_AWAY"

    return support


def _crowd_evidence(exit_id, crowd_snapshot):

    if crowd_snapshot is None:
        return None, None

    metrics = crowd_snapshot.exit(exit_id)

    if metrics is None:
        return None, None

    congestion_level = metrics.congestion_level.name if metrics.congestion_level is not None else None

    return congestion_level, metrics.queue_candidate_count


def _throughput_evidence(exit_id, evacuation_progress_snapshot):

    if evacuation_progress_snapshot is None:
        return None

    flow = evacuation_progress_snapshot.exit(exit_id)

    return flow.recent_flow_per_minute if flow is not None else None


def _emergency_response_elevated(exit_id, exit_zone_map, emergency_response_snapshot, config) -> bool:

    if emergency_response_snapshot is None:
        return False

    zone_e = exit_zone_map.get(exit_id)

    if zone_e is None:
        return False

    priority = emergency_response_snapshot.zone(zone_e)

    return priority is not None and priority.priority_level in config.emergency_response_penalty_levels

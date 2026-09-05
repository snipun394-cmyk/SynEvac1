from typing import Dict, Iterable, Mapping, Optional, Tuple

from ai_decision.perception_adapter import hazard_snapshot_from_observation

from crowd_intelligence.capacity import exit_capacity, stair_capacity

from hazard.cost_model import HazardAwareCostModel

from models.building import Building

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator
from navigation.node import Node

from pathfinding.engine import PathfindingEngine
from pathfinding.route import Route

from perception.models.building_observation import BuildingObservation


# =====================================================
# Zone Recommendation Coordination milestone -- docs/architecture/
# multi_zone_joint_coordination_implementation_investigation.txt,
# Sections 6/7/8/10 (Model B, the narrowest first-milestone scope: single
# shared-target overload only, never a route-planning redesign, never a
# replacement for the RL policy).
#
# What this class is: a pure, stateless-per-tick translation step that
# sits between "the RL policy already chose ONE building-wide target this
# tick" and "RLActionDecisionEngine is told what to recommend per zone."
# It NEVER calls the RL model, NEVER re-infers an action, and NEVER
# invents a target no PathfindingEngine search actually found. Its ONLY
# output is a {zone_id: target_edge_id-or-None} mapping consumed by
# RLActionDecisionEngine.set_zone_recommendations() -- the existing,
# already-tested per-zone escape hatch (rl_training/production/
# decision_engine.py) this milestone deliberately reuses rather than
# redesigning DecisionRecommendation or the decision engine schema.
#
# Placement (governing investigation Section 3/K): this file lives in
# live_runtime/, mirroring live_runtime/live_decision_coordinator.py's own
# already-proven "thin, stateless translation layer downstream of an
# already-computed decision" placement. Two files in this codebase carry
# real, test-enforced import prohibitions this class must never violate:
#   - rl_training/production/decision_engine.py (tests/
#     test_rl_action_decision_engine.py::
#     NoPrivilegedOrRouteStateIntroducedTests) is forbidden from importing
#     pathfinding/navigation at all -- this class is NOT placed there, and
#     RLActionDecisionEngine itself remains completely unmodified by this
#     milestone.
#   - live_runtime/live_rl_decision_pipeline.py (tests/
#     test_live_rl_decision_pipeline.py::ArchitectureGuardTests) is
#     forbidden from importing pathfinding (and stair_congestion,
#     observable_assets, live_occupants, ...) directly -- this class is a
#     SEPARATE file that pipeline merely imports by name (a line that does
#     not itself start with "pathfinding"), the exact same pattern that
#     file's own existing "from live_runtime.live_decision_coordinator
#     import LiveDecisionCoordinator" line already uses today.
#
# Authority boundary (governing investigation Section 6, Model B -- the
# ONLY model this class implements): the RL-selected target is the
# default for EVERY zone. A zone's target is changed ONLY when this
# class can prove, from real repository data, that combined KNOWN
# occupant demand for the RL target exceeds that target's own real,
# disclosed-placeholder capacity number (crowd_intelligence.capacity),
# AND a real, hazard-aware-reachable alternate target exists for that
# specific zone. A zone with no honest alternate keeps the RL-selected
# target unchanged -- this class never fabricates a route, a capacity
# number, or a safety claim, and never leaves a zone with a target that
# no PathfindingEngine search actually produced.
#
# Explicitly NOT in scope for this milestone (governing investigation
# Section 7's own ruthless-scope cuts -- not silently expanded here):
#   - corridor/intermediate-edge overlap between zones with DIFFERENT
#     final targets;
#   - any dependency on CrowdIntelligenceEngine/LiveOccupantManager/
#     LiveOrchestrator;
#   - changing RLActionDecisionEngine.is_reachable's existing disclosed
#     always-True placeholder meaning;
#   - ProductionRLEnvironment/training-time integration.
# =====================================================


class ZoneRecommendationCoordinator:

    # Constructed ONCE per Building (mirrors rl_training.action_space.
    # ActionMapper's and rl_training.perception_observation_space.
    # PerceptionObservationEncoder's own "fixed shape per Building,
    # constructed once" discipline, already established in this exact
    # live pipeline) -- NOT per tick. Holds only a NavigationGraph and a
    # base (non-hazard-aware) PathfindingEngine derived from Building
    # alone; no LiveOccupantManager, no CrowdIntelligenceEngine, no
    # simulator/ground-truth reference of any kind.

    def __init__(self, building: Building):

        self._building = building

        graph = NavigationGraphGenerator().build(building)
        self._graph = graph

        self._base_engine = PathfindingEngine(graph)

        self._zone_nodes = [
            node for node in graph.nodes.values()
            if node.node_type == Node.ZONE
        ]

        self._exit_edges: Tuple[Edge, ...] = tuple(
            edge for edge in graph.edges if edge.edge_type == Edge.EXIT
        )
        self._stair_edges: Tuple[Edge, ...] = tuple(
            edge for edge in graph.edges if edge.edge_type == Edge.STAIR
        )

        # Every RECOMMEND_EXIT/RECOMMEND_STAIR target id this coordinator
        # can ever be asked about -- exactly the same two edge types
        # rl_training.action_space.ActionMapper's own action table is
        # built from (Exit/Stair only; NOOP/BROADCAST_*/DEPLOY_STAFF
        # entries never carry a recommended_edge_id at all, see
        # coordinate() below).
        self._edge_by_id: Dict[str, Edge] = {
            edge.id: edge
            for edge in self._exit_edges + self._stair_edges
        }

        # Stair edges are NOT a shared bottleneck node the way every Exit
        # shares Node.OUTSIDE_NODE_ID -- each Staircase has its own two
        # endpoint node ids, so a plain lookup (no search exclusion
        # needed) is enough to tell whether reaching a given node means
        # reaching a specific stair.
        self._stair_endpoint_to_edge: Dict[str, Edge] = {}
        for edge in self._stair_edges:
            self._stair_endpoint_to_edge[edge.from_node] = edge
            self._stair_endpoint_to_edge[edge.to_node] = edge

    # =====================================================

    def coordinate(
        self,
        recommended_edge_id: Optional[str],
        zone_ids: Iterable[str],
        observation: BuildingObservation,
        timestamp: float,
    ) -> Mapping[str, Optional[str]]:

        # The sole public entry point. Inputs mirror exactly what
        # RLActionDecisionEngine.set_action() already receives today
        # (recommended_edge_id, zone_ids) plus the same BuildingObservation/
        # timestamp this pipeline already threads through Step A/D
        # (live_runtime/live_rl_decision_pipeline.py) -- no new perception
        # input is introduced.

        zone_ids = tuple(zone_ids)

        # NOOP (or any broadcast/staff-deployment action, which already
        # decodes to recommended_edge_id=None upstream, see
        # rl_training/production/live_inference.py::_decode_action()) --
        # nothing to group, sum, or redistribute. Every zone maps to
        # None, unchanged. This class never fabricates an evacuation
        # target when the RL policy chose none.
        if recommended_edge_id is None:
            return {zone_id: None for zone_id in zone_ids}

        target_edge = self._edge_by_id.get(recommended_edge_id)

        # A target id this coordinator's own Building-derived graph does
        # not recognize (should not happen in production -- the RL policy
        # and this coordinator are built from the same Building -- but
        # honestly handled rather than assumed impossible): abstain,
        # pass the broadcast through unchanged rather than silently
        # dropping or fabricating a recommendation.
        if target_edge is None:
            return {zone_id: recommended_edge_id for zone_id in zone_ids}

        # Honest demand accounting (governing investigation Section 4/9):
        # sum only zones with a real, non-None occupancy estimate. A zone
        # with no reading this cycle contributes nothing to the KNOWN
        # total -- it is never treated as a confirmed zero, and it is
        # never treated as forcing a conflict either (see the capacity
        # comparison below: only KNOWN demand can prove an overload).
        known_demand = 0.0
        for zone_id in zone_ids:
            occupancy = observation.occupancy_observation(zone_id)
            if occupancy.estimated_count is not None:
                known_demand += occupancy.estimated_count

        capacity = self._capacity_for(target_edge)

        assignments: Dict[str, Optional[str]] = {
            zone_id: recommended_edge_id for zone_id in zone_ids
        }

        # No proven conflict -- either capacity itself is not honestly
        # derivable for this target (never invented), or the KNOWN
        # portion of demand alone does not exceed it. Every zone keeps
        # the RL-selected target, byte-identical to today's uniform
        # broadcast. This is the common case and must remain the common
        # case: nothing below runs unless a real overload is already
        # provable from known evidence alone.
        if capacity is None or known_demand <= capacity:
            return assignments

        # Proven conflict: known_demand > capacity. Redistribute only the
        # provable overflow, using a hazard-aware engine built fresh for
        # THIS tick from THIS observation -- the same perception-honest
        # construction ai_decision.engine.AIDecisionEngine already uses in
        # production (never a ground-truth HazardSnapshot read
        # independently of perception).
        engine = self._hazard_aware_engine(observation, timestamp)

        distance_to_target: Dict[str, Optional[float]] = {}
        for zone_id in zone_ids:
            route = self._route_via_target(engine, zone_id, target_edge)
            distance_to_target[zone_id] = route.total_cost if route is not None else None

        # Deterministic ordering (governing investigation Section 8/9):
        # farthest-from-target first, zone_id as the final tie-break --
        # the SAME "...then zone_id for full determinism when tied"
        # convention ai_decision.priority.SeverityOccupancyPriorityRule
        # already establishes, reused rather than a second one invented
        # here. A zone with no honest route to the target at all
        # (distance None) is excluded from this ordered list entirely --
        # it was never a provable redistribution candidate over something
        # it cannot even honestly reach, and it keeps the original target
        # unchanged like every other zone this pass does not touch.
        candidates = sorted(
            (zone_id for zone_id in zone_ids if distance_to_target[zone_id] is not None),
            key=lambda zone_id: (-distance_to_target[zone_id], zone_id),
        )

        running_known_demand = known_demand

        for zone_id in candidates:

            if running_known_demand <= capacity:
                # Enough provable overflow has already been relieved --
                # every remaining zone (including this one) keeps its
                # current assignment.
                break

            alternative = self._best_alternative(engine, zone_id, target_edge)

            if alternative is None:
                # No honest alternate target exists for this specific
                # zone (every other Exit/Stair is unreachable given
                # current perceived hazard state) -- abstain for this
                # zone exactly as RecommendationAwareRouteChoiceStrategy's
                # own existing "fall back rather than fail" discipline
                # already does (simulation_interactive/replanning.py) --
                # it keeps the original, possibly over-capacity target,
                # never None, never a fabricated id. Continue to the next
                # farthest candidate rather than stopping outright.
                continue

            assignments[zone_id] = alternative.id

            occupancy = observation.occupancy_observation(zone_id)
            if occupancy.estimated_count is not None:
                running_known_demand -= occupancy.estimated_count

        return assignments

    # =====================================================

    def _hazard_aware_engine(self, observation: BuildingObservation, timestamp: float) -> PathfindingEngine:

        # Identical construction to ai_decision.engine.AIDecisionEngine.
        # _hazard_aware_engine() -- reused pattern, not reused code,
        # since that method is private to a different class this
        # coordinator does not import (and must not, per this milestone's
        # own scope: no dependency on ai_decision.engine.AIDecisionEngine
        # itself, only on the perception-honest adapter function it is
        # itself built from).
        hazard_snapshot = hazard_snapshot_from_observation(
            observation, self._zone_nodes, self._graph.edges, timestamp,
        )
        cost_model = HazardAwareCostModel(self._base_engine.cost_model, hazard_snapshot)

        return PathfindingEngine(self._graph, cost_model=cost_model, heuristic=self._base_engine.heuristic)

    # =====================================================

    def _capacity_for(self, edge: Edge) -> Optional[int]:

        # The repository's own real capacity source -- crowd_intelligence.
        # capacity's already-vetted, already-tested wrapper around
        # simulator.capacity.DefaultCapacityModel/StairCapacityModel
        # (governing investigation Section 5). A documented, disclosed
        # engineering-estimate PLACEHOLDER ("not a validated life-safety
        # flow-rate model, just a reasonable default" -- crowd_intelligence/
        # capacity.py's own module docstring) -- never presented by this
        # class as a validated real-world limit. No capacity constant of
        # any kind is invented here.
        if edge.edge_type == Edge.EXIT:
            return exit_capacity(edge.reference)

        if edge.edge_type == Edge.STAIR:
            return stair_capacity(edge.reference, self._building)

        return None

    # =====================================================

    def _route_via_target(self, engine: PathfindingEngine, zone_id: str, target_edge: Edge) -> Optional[Route]:

        # "Can this zone honestly reach THIS SPECIFIC target" -- not
        # merely "can it reach some exit/stair." For an Exit, every exit
        # in the building leads to the SAME single shared Outside node
        # (navigation/node.py: Node.OUTSIDE_NODE_ID, "the whole graph
        # shares exactly one Outside node"), so a plain search to Outside
        # would report whichever exit happens to be globally cheapest --
        # not necessarily this one. Excluding every OTHER Exit edge
        # (never Doors, never Stairs -- a multi-floor route to THIS exit
        # may legitimately need to cross an unrelated stair) forces the
        # only possible way to reach Outside to be through target_edge
        # itself, if a route exists at all.
        if target_edge.edge_type == Edge.EXIT:

            other_exit_ids = frozenset(
                edge.id for edge in self._exit_edges if edge.id != target_edge.id
            )
            reachable = engine.distances_from(zone_id, excluded_edge_ids=other_exit_ids)

            return reachable.get(Node.OUTSIDE_NODE_ID)

        # Stair: no shared-node ambiguity (each Staircase owns its own
        # two endpoint node ids), so no exclusion is needed -- reaching
        # either of this stair's own endpoints honestly means reaching
        # this specific stair. If both endpoints happen to be reachable
        # (e.g. a loop), the cheaper of the two is used.
        reachable = engine.distances_from(zone_id)

        candidates = [
            reachable[node_id]
            for node_id in (target_edge.from_node, target_edge.to_node)
            if node_id in reachable
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda route: route.total_cost)

    # =====================================================

    def _best_alternative(self, engine: PathfindingEngine, zone_id: str, exclude_edge: Edge) -> Optional[Edge]:

        # The single cheapest real, hazard-aware-reachable Exit OR Stair
        # target for this zone, other than exclude_edge -- deliberately
        # NOT restricted to the same edge_type as exclude_edge (governing
        # investigation's own explicit "must not accidentally bias the
        # system toward stairs" requirement): a stair-overloaded zone may
        # honestly be redirected to a nearer exit, and vice versa, purely
        # on which real alternative this specific zone can actually reach
        # most cheaply. Only exclude_edge itself is excluded from the
        # search -- every OTHER exit/stair, including any this zone's
        # route happens to pass through on the way, remains a legitimate
        # part of the graph to route across.
        reachable = engine.distances_from(zone_id, excluded_edge_ids=frozenset({exclude_edge.id}))

        best_edge: Optional[Edge] = None
        best_cost: Optional[float] = None

        for node_id, route in reachable.items():

            candidate_edge: Optional[Edge] = None

            if node_id == Node.OUTSIDE_NODE_ID and route.edges:
                last_edge = route.edges[-1]
                if last_edge.edge_type == Edge.EXIT:
                    candidate_edge = last_edge

            elif node_id in self._stair_endpoint_to_edge:
                candidate_edge = self._stair_endpoint_to_edge[node_id]

            if candidate_edge is None or candidate_edge.id == exclude_edge.id:
                continue

            if best_cost is None or route.total_cost < best_cost:
                best_cost = route.total_cost
                best_edge = candidate_edge

        return best_edge

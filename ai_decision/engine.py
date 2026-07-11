from typing import Optional

from navigation.node import Node

from pathfinding.engine import PathfindingEngine

from hazard.cost_model import HazardAwareCostModel
from hazard.severity import HazardSeverity

from analysis.engine import BuildingAnalysisEngine

from ai_decision.announcements import DefaultVoiceAnnouncementTemplate, VoiceAnnouncementTemplate
from ai_decision.priority import SEVERITY_RANK, EvacuationPriorityRule, SeverityOccupancyPriorityRule
from ai_decision.recommendation import (
    BlockedRoute,
    DecisionRecommendation,
    FirefighterPriority,
    VoiceAnnouncement,
    ZoneRecommendation,
)


class DecisionEngine:

    # The swap point: a future RL-based engine implements this exact
    # method and nothing else -- every downstream consumer (a future
    # dashboard, firefighter UI, Voice Evacuation System) depends only
    # on DecisionEngine/DecisionRecommendation, never on
    # AIDecisionEngine specifically. Mirrors HazardSource/
    # OccupancyProducer's role as a whole-subsystem swap point.

    def decide(self, hazard_snapshot, occupancy_snapshot, time: float) -> DecisionRecommendation:

        raise NotImplementedError


class AIDecisionEngine(DecisionEngine):

    # The single consumer of HazardSnapshot and OccupancySnapshot, and
    # single producer of DecisionRecommendation -- consumes only
    # already-frozen interfaces (PathfindingEngine, HazardAwareCostModel,
    # BuildingAnalysisEngine, HazardSeverity) and never imports
    # simulator/behavior/models/designer/hazard_evolution/sensors: this
    # engine sits entirely downstream of the snapshots those layers
    # already produce, it never reaches back into how they were made.
    #
    # base_engine is whatever (non-hazard-aware) PathfindingEngine the
    # deployment already uses -- constructed once. Each decide() call
    # wraps its cost_model fresh in a HazardAwareCostModel for that
    # specific hazard_snapshot and builds a throwaway hazard-aware
    # PathfindingEngine from it, so nearest_exit() automatically routes
    # around hazard-blocked/penalized edges with zero new graph
    # algorithms here.
    #
    # Structural criticality (which edge, if it fails, strands which
    # zones) is a fact about the building's topology alone -- it does
    # not change from one hazard_snapshot to the next, so it is computed
    # exactly once here, via BuildingAnalysisEngine.critical_connectors()
    # on the *base* (hazard-unaware) engine. This is deliberate, not an
    # oversight: running critical_connectors() on an already
    # hazard-aware engine instead would be self-defeating for an edge
    # that is *currently* hazard-blocked -- its own baseline reachable
    # set would already exclude everything only it serves, so removing
    # it again could never appear to "strand" anything. Precomputing
    # structural criticality once, up front, from the unaffected
    # topology is what lets _blocked_routes() correctly answer "what
    # would reopening this specific blocked edge restore" later.

    DEFAULT_UNSAFE_SEVERITY_THRESHOLD = HazardSeverity.HIGH

    def __init__(
        self,
        base_engine: PathfindingEngine,
        unsafe_severity_threshold: HazardSeverity = DEFAULT_UNSAFE_SEVERITY_THRESHOLD,
        priority_rule: Optional[EvacuationPriorityRule] = None,
        announcement_template: Optional[VoiceAnnouncementTemplate] = None,
    ):

        self.base_engine = base_engine
        self.graph = base_engine.graph

        self.unsafe_severity_threshold = unsafe_severity_threshold
        self.priority_rule = priority_rule or SeverityOccupancyPriorityRule()
        self.announcement_template = announcement_template or DefaultVoiceAnnouncementTemplate()

        self._structural_stranded_zone_ids_by_edge_id = {
            finding.edge_id: [
                node_id
                for node_id in finding.stranded_node_ids
                if self._is_zone(node_id)
            ]
            for finding in BuildingAnalysisEngine(base_engine).critical_connectors()
        }

    # =====================================================

    def decide(self, hazard_snapshot, occupancy_snapshot, time: float) -> DecisionRecommendation:

        hazard_engine = self._hazard_aware_engine(hazard_snapshot)

        zone_recommendations = {
            node.id: self._zone_recommendation(
                node, hazard_engine, hazard_snapshot, occupancy_snapshot,
            )
            for node in self._zone_nodes()
        }

        unsafe_zone_ids = [
            zone_id
            for zone_id, recommendation in zone_recommendations.items()
            if recommendation.is_unsafe
        ]

        blocked_routes = self._blocked_routes(hazard_snapshot)

        priority_candidates = [
            recommendation
            for recommendation in zone_recommendations.values()
            if recommendation.is_unsafe or (recommendation.occupant_count or 0.0) > 0.0
        ]

        return DecisionRecommendation(
            timestamp=time,
            zone_recommendations=zone_recommendations,
            priority_evacuation_order=self.priority_rule.rank(priority_candidates),
            unsafe_zone_ids=unsafe_zone_ids,
            blocked_routes=blocked_routes,
            voice_announcements=self._voice_announcements(zone_recommendations, unsafe_zone_ids),
            firefighter_priorities=self._firefighter_priorities(
                zone_recommendations, unsafe_zone_ids, blocked_routes,
            ),
        )

    # =====================================================

    def _hazard_aware_engine(self, hazard_snapshot) -> PathfindingEngine:

        hazard_cost_model = HazardAwareCostModel(self.base_engine.cost_model, hazard_snapshot)

        return PathfindingEngine(
            self.graph, cost_model=hazard_cost_model, heuristic=self.base_engine.heuristic,
        )

    # =====================================================

    def _zone_nodes(self):

        return [
            node
            for node in self.graph.nodes.values()
            if node.node_type == Node.ZONE
        ]

    # =====================================================

    def _is_zone(self, node_id) -> bool:

        node = self.graph.find_node(node_id)
        return node is not None and node.node_type == Node.ZONE

    # =====================================================

    def _zone_recommendation(
        self, node, hazard_engine, hazard_snapshot, occupancy_snapshot,
    ) -> ZoneRecommendation:

        severity = hazard_snapshot.node_state(node.id).severity
        occupant_count = occupancy_snapshot.observation_at(node.id).occupant_count

        route = hazard_engine.nearest_exit(node.id)
        is_reachable = route is not None

        # A zone with no safe route to any exit is unsafe regardless of
        # its own hazard reading -- being cut off is itself the
        # emergency, even at a zone the hazard layer has no opinion
        # about (severity NONE). Without this, a zone stranded by a
        # hazard-blocked corridor elsewhere in the building would
        # silently get no unsafe flag, no announcement, and no
        # priority/firefighter entry.
        is_unsafe = (
            SEVERITY_RANK[severity] >= SEVERITY_RANK[self.unsafe_severity_threshold]
            or not is_reachable
        )

        return ZoneRecommendation(
            zone_id=node.id,
            severity=severity,
            occupant_count=occupant_count,
            is_unsafe=is_unsafe,
            is_reachable=is_reachable,
            recommended_exit_edge_id=(
                route.edges[-1].id if route is not None and route.edges else None
            ),
            recommended_route=route,
        )

    # =====================================================

    def _blocked_routes(self, hazard_snapshot):

        return [
            BlockedRoute(
                edge_id=edge.id,
                edge_type=edge.edge_type,
                stranded_zone_ids=list(
                    self._structural_stranded_zone_ids_by_edge_id.get(edge.id, [])
                ),
            )
            for edge in self.graph.edges
            if hazard_snapshot.edge_state(edge.id).traversable is False
        ]

    # =====================================================

    def _voice_announcements(self, zone_recommendations, unsafe_zone_ids):

        announcements = []

        for zone_id in unsafe_zone_ids:

            recommendation = zone_recommendations[zone_id]

            node = self.graph.find_node(zone_id)
            zone_name = node.name if node is not None else zone_id
            exit_name = self._exit_name(recommendation.recommended_exit_edge_id)

            message = self.announcement_template.announce(recommendation, zone_name, exit_name)

            if message is not None:

                announcements.append(
                    VoiceAnnouncement(
                        zone_id=zone_id, message=message, severity=recommendation.severity,
                    )
                )

        return announcements

    # =====================================================

    def _exit_name(self, exit_edge_id) -> Optional[str]:

        if exit_edge_id is None:
            return None

        edge = next(
            (candidate for candidate in self.graph.edges if candidate.id == exit_edge_id), None,
        )

        if edge is None:
            return None

        return getattr(edge.reference, "name", None) or edge.id

    # =====================================================

    def _firefighter_priorities(self, zone_recommendations, unsafe_zone_ids, blocked_routes):

        priorities = []
        rank = 1

        ranked_zones = sorted(
            (zone_recommendations[zone_id] for zone_id in unsafe_zone_ids),
            key=lambda recommendation: (
                -SEVERITY_RANK[recommendation.severity],
                -(recommendation.occupant_count or 0.0),
                recommendation.zone_id,
            ),
        )

        for recommendation in ranked_zones:

            reachability_clause = (
                "" if recommendation.is_reachable
                else "; no safe route out -- rescue access required"
            )

            priorities.append(
                FirefighterPriority(
                    rank=rank,
                    target_type="zone",
                    target_id=recommendation.zone_id,
                    reason=f"{recommendation.severity.name} hazard{reachability_clause}",
                )
            )
            rank += 1

        critical_blocked = sorted(
            (blocked for blocked in blocked_routes if blocked.stranded_zone_ids),
            key=lambda blocked: (-len(blocked.stranded_zone_ids), blocked.edge_id),
        )

        for blocked in critical_blocked:

            priorities.append(
                FirefighterPriority(
                    rank=rank,
                    target_type="connector",
                    target_id=blocked.edge_id,
                    reason=(
                        f"Reopening this {blocked.edge_type} would restore access "
                        f"for {len(blocked.stranded_zone_ids)} zone(s)"
                    ),
                )
            )
            rank += 1

        return priorities

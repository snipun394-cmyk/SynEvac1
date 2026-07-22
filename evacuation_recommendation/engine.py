from typing import Optional

from evacuation_recommendation.evidence import ai_bottleneck_probability, hazard_evidence_available, zone_coverage_fraction
from evacuation_recommendation.explanation import explain_recommendation
from evacuation_recommendation.models import (
    EvacuationRecommendationSnapshot, RecommendationConfig, RecommendationReason, RecommendationStatus,
    RecommendationWeights, ZoneEvacuationRecommendation,
)
from evacuation_recommendation.ranking import SafeExitDistanceCalculator, rank_exits_for_zone
from evacuation_recommendation.scoring import compute_confidence


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 2 --
# the ONE object LiveRuntime owns for this package, mirroring every
# prior live-intelligence engine's own "exactly one shared instance per
# live session" role. compute(time, building_state, crowd_snapshot,
# evacuation_progress_snapshot, trajectory_snapshot,
# emergency_response_snapshot, ai_prediction_snapshot) is the single
# entry point.
#
# LiveOccupantManager remains the canonical persistent occupant owner --
# this engine only ever reads live_occupant_manager.active_occupants()
# each cycle and owns no occupant identity/lifecycle state itself. It
# owns exactly one small piece of its own cross-cycle state (the
# SafeExitDistanceCalculator's own per-cycle Dijkstra cache).
#
# SITS AFTER deterministic safety evaluation, never replaces or
# modifies it -- decision_policy/ is never imported here (see this
# package's own architecture guard test), and an unsafe exit can never
# appear as a candidate at all (SafeExitDistanceCalculator's own hard
# node-exclusion, ranking.py's own module docstring). Never makes an
# evacuation decision with execution authority, never broadcasts, never
# executes a control -- purely a deterministic, explainable ranking/
# reporting layer.
# =====================================================


class EvacuationRecommendationEngine:

    def __init__(
        self, building, navigation_graph, live_occupant_manager,
        config: Optional[RecommendationConfig] = None,
        weights: Optional[RecommendationWeights] = None,
    ):

        self.building = building
        self.graph = navigation_graph
        self.live_occupant_manager = live_occupant_manager

        self.config = config if config is not None else RecommendationConfig()
        self.weights = weights if weights is not None else RecommendationWeights()

        self._distance_calculator = SafeExitDistanceCalculator(navigation_graph, self.config)
        self._zones = [(zone, floor.id) for floor in building.ordered_floors() for zone in floor.zones]

    # =====================================================

    def compute(
        self, time: float, building_state=None, crowd_snapshot=None, evacuation_progress_snapshot=None,
        trajectory_snapshot=None, emergency_response_snapshot=None, ai_prediction_snapshot=None,
    ) -> EvacuationRecommendationSnapshot:

        distance_by_exit = self._distance_calculator.compute(building_state)
        exit_zone_map = self._distance_calculator.exit_zone_map
        safe_exit_ids = self._distance_calculator.safe_exit_ids

        occupants = self.live_occupant_manager.active_occupants()

        occupants_by_zone = {}
        for occupant in occupants:
            if occupant.current_zone_id is not None:
                occupants_by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

        bottleneck_probability = ai_bottleneck_probability(ai_prediction_snapshot)
        hazard_available = hazard_evidence_available(building_state)

        zones = {}

        for zone, floor_id in self._zones:

            occupant_list = occupants_by_zone.get(zone.id, ())

            if not occupant_list:
                # Phase 8 -- unoccupied zones produce no evacuation
                # recommendation at all, never a fabricated empty one.
                continue

            zones[zone.id] = self._compute_zone_recommendation(
                zone_id=zone.id, floor_id=floor_id, occupant_count=len(occupant_list),
                distance_by_exit=distance_by_exit, exit_zone_map=exit_zone_map,
                crowd_snapshot=crowd_snapshot, evacuation_progress_snapshot=evacuation_progress_snapshot,
                trajectory_snapshot=trajectory_snapshot, emergency_response_snapshot=emergency_response_snapshot,
                bottleneck_probability=bottleneck_probability, hazard_available=hazard_available, time=time,
            )

        return EvacuationRecommendationSnapshot(timestamp=time, zones=zones, safe_exit_ids=safe_exit_ids)

    # =====================================================

    def _compute_zone_recommendation(
        self, *, zone_id, floor_id, occupant_count, distance_by_exit, exit_zone_map,
        crowd_snapshot, evacuation_progress_snapshot, trajectory_snapshot, emergency_response_snapshot,
        bottleneck_probability, hazard_available, time,
    ) -> ZoneEvacuationRecommendation:

        candidates = rank_exits_for_zone(
            zone_id, distance_by_exit, exit_zone_map=exit_zone_map,
            crowd_snapshot=crowd_snapshot, evacuation_progress_snapshot=evacuation_progress_snapshot,
            trajectory_snapshot=trajectory_snapshot, emergency_response_snapshot=emergency_response_snapshot,
            ai_bottleneck_probability=bottleneck_probability, weights=self.weights, config=self.config,
        )

        coverage_fraction = zone_coverage_fraction(zone_id, crowd_snapshot)

        if not candidates:

            reason_codes = (RecommendationReason.NO_SAFE_EXIT_REACHABLE,)

            return ZoneEvacuationRecommendation(
                zone_id=zone_id, floor_id=floor_id, status=RecommendationStatus.NO_SAFE_EXIT_AVAILABLE,
                reason_codes=reason_codes,
                explanation=explain_recommendation(zone_id, RecommendationStatus.NO_SAFE_EXIT_AVAILABLE, None, reason_codes, coverage_fraction),
                confidence=compute_confidence(
                    position_coverage_fraction=coverage_fraction, route_distance_derived=False,
                    hazard_evidence_available=hazard_available,
                ),
                coverage_fraction=coverage_fraction, occupant_count=occupant_count, timestamp=time,
            )

        top = candidates[0]
        ranked_ids = tuple(candidate.exit_id for candidate in candidates)

        return ZoneEvacuationRecommendation(
            zone_id=zone_id, floor_id=floor_id, status=RecommendationStatus.RECOMMENDED,
            recommended_exit_id=top.exit_id, ranked_exit_ids=ranked_ids, alternative_exit_ids=ranked_ids[1:],
            candidates=tuple(candidates), reason_codes=top.reason_codes,
            explanation=explain_recommendation(zone_id, RecommendationStatus.RECOMMENDED, top.exit_id, top.reason_codes, coverage_fraction),
            confidence=compute_confidence(
                position_coverage_fraction=coverage_fraction, route_distance_derived=True,
                hazard_evidence_available=hazard_available,
            ),
            coverage_fraction=coverage_fraction, occupant_count=occupant_count, timestamp=time,
        )

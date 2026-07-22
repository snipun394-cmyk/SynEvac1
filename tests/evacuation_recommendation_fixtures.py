from crowd_intelligence.models import AssetApproachMetrics, CrowdIntelligenceSnapshot, IntensityLevel

from evacuation_progress.models import EvacuationProgressSnapshot, ExitFlow

from emergency_response.models import EmergencyResponseSnapshot, ResponsePriorityLevel, ZoneResponsePriority

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from building_state.models import BuildingState, HazardSummary

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.models import RecommendationConfig, RecommendationWeights

from tests.trajectory_intelligence_fixtures import make_trajectory_building, make_trajectory_graph


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone -- reuses the
# SAME shared building topology tests.trajectory_intelligence_fixtures
# already established (z1/Lobby/EXIT-1 -- DOOR-1 -- z2/Hall -- DOOR-2 --
# z4/Annex/EXIT-2, plus z3/Room2F on floor f2 via STAIR-1), so both
# milestones' own tests reason about identical route distances.
# =====================================================


def make_building_state(hazard_severity_by_zone=None):

    hazard_summary = HazardSummary(zone_severities=dict(hazard_severity_by_zone or {}))

    return BuildingState(hazard_summary=hazard_summary)


def make_engine(building=None, graph=None, config=None, weights=None, manager=None, event_bus=None):

    building = building if building is not None else make_trajectory_building()
    graph = graph if graph is not None else make_trajectory_graph(building)
    event_bus = event_bus if event_bus is not None else EventBus()
    manager = manager if manager is not None else LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=100000.0)
    config = config if config is not None else RecommendationConfig()

    engine = EvacuationRecommendationEngine(building, graph, manager, config=config, weights=weights)

    return engine, manager


def make_crowd_snapshot(exit_metrics=None):

    return CrowdIntelligenceSnapshot(exit_metrics=dict(exit_metrics or {}))


def make_exit_metrics(exit_id, congestion_level=None, queue_candidate_count=0):

    return AssetApproachMetrics(
        asset_id=exit_id, asset_type="Exit", position_available=True,
        congestion_level=congestion_level, queue_candidate_count=queue_candidate_count,
    )


def make_evacuation_progress_snapshot(exit_flows=None):

    return EvacuationProgressSnapshot(exits=dict(exit_flows or {}))


def make_exit_flow(exit_id, recent_flow_per_minute=None):

    return ExitFlow(exit_id=exit_id, recent_flow_per_minute=recent_flow_per_minute)


def make_emergency_response_snapshot(zone_priorities=None):

    return EmergencyResponseSnapshot(zones=dict(zone_priorities or {}))


def make_zone_priority(zone_id, priority_level=ResponsePriorityLevel.LOW):

    return ZoneResponsePriority(zone_id=zone_id, priority_level=priority_level)


class _FakeBottleneck:

    def __init__(self, probability):
        self.probability = probability


class FakeAIPredictionSnapshot:

    def __init__(self, probability):
        self.bottleneck = _FakeBottleneck(probability)

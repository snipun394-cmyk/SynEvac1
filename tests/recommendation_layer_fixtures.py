from crowd_intelligence.models import BuildingCrowdSummary, CrowdIntelligenceSnapshot

from emergency_response.models import EmergencyResponseSnapshot, ResponsePriorityLevel, ZoneResponsePriority

from evacuation_guidance.models import EvacuationGuidancePlan, EvacuationGuidanceSnapshot

from evacuation_recommendation.models import (
    EvacuationRecommendationSnapshot, ExitCandidate, RecommendationStatus, ZoneEvacuationRecommendation,
)

from advisory_system.recommendation_models import (
    AdvisoryReport, BuildingRecommendation, FirefighterIntelligenceReport, IncidentCommanderDashboard,
)

from tests.evacuation_recommendation_fixtures import FakeAIPredictionSnapshot  # noqa: F401 -- re-exported


# =====================================================
# recommendation_layer/ -- shared fixture builders, no real engine
# dependency. Mirrors tests/evacuation_recommendation_fixtures.py's own
# "plain builder functions, sensible defaults" convention.
# =====================================================


def make_zone_recommendation(
    zone_id="zone-1", floor_id="floor-1", status=RecommendationStatus.RECOMMENDED,
    recommended_exit_id="exit-1", alternative_exit_ids=(), candidates=(),
    reason_codes=(), confidence=0.9, coverage_fraction=0.8, occupant_count=1, timestamp=1.0,
):

    ranked_exit_ids = (recommended_exit_id,) + tuple(alternative_exit_ids) if recommended_exit_id else ()

    return ZoneEvacuationRecommendation(
        zone_id=zone_id, floor_id=floor_id, status=status, recommended_exit_id=recommended_exit_id,
        ranked_exit_ids=ranked_exit_ids, alternative_exit_ids=tuple(alternative_exit_ids), candidates=tuple(candidates),
        reason_codes=tuple(reason_codes), explanation=f"explanation for {zone_id}", confidence=confidence,
        coverage_fraction=coverage_fraction, occupant_count=occupant_count, timestamp=timestamp,
    )


def make_exit_candidate(exit_id="exit-1", route_distance_m=10.0, congestion_level=None, reason_codes=(), **kwargs):

    return ExitCandidate(
        exit_id=exit_id, route_distance_m=route_distance_m, congestion_level=congestion_level,
        reason_codes=tuple(reason_codes), **kwargs,
    )


def make_recommendation_snapshot(zones=(), safe_exit_ids=(), timestamp=1.0):

    return EvacuationRecommendationSnapshot(
        timestamp=timestamp, zones={zone.zone_id: zone for zone in zones}, safe_exit_ids=tuple(safe_exit_ids),
    )


def make_zone_response_priority(zone_id="zone-1", priority_level=ResponsePriorityLevel.LOW, **kwargs):

    return ZoneResponsePriority(zone_id=zone_id, priority_level=priority_level, **kwargs)


def make_emergency_response_snapshot(zones=(), timestamp=1.0):

    return EmergencyResponseSnapshot(timestamp=timestamp, zones={zone.zone_id: zone for zone in zones})


def make_crowd_snapshot(congested_exits=(), congested_doors=(), congested_stairs=(), timestamp=1.0):

    return CrowdIntelligenceSnapshot(
        timestamp=timestamp,
        building_summary=BuildingCrowdSummary(
            congested_exits=tuple(congested_exits), congested_doors=tuple(congested_doors),
            congested_stairs=tuple(congested_stairs),
        ),
    )


def make_guidance_plan(zone_id="zone-1", inconsistencies=(), confidence=0.9):

    return EvacuationGuidancePlan(zone_id=zone_id, inconsistencies=tuple(inconsistencies), confidence=confidence)


def make_guidance_snapshot(plans=(), timestamp=1.0):

    return EvacuationGuidanceSnapshot(timestamp=timestamp, zones={plan.zone_id: plan for plan in plans})


def make_building_recommendation(action="Monitor Congestion at Exit exit-1", target_type="exit", target_id="exit-1", reason="congestion", confidence=0.7):

    return BuildingRecommendation(
        action=action, target_type=target_type, target_id=target_id, reason=reason, confidence=confidence,
        expected_engineering_benefit="reduced congestion",
    )


def make_commander_dashboard(critical_zones=()):

    return IncidentCommanderDashboard(
        safe_zones=(), critical_zones=tuple(critical_zones), warning_zones=(),
        occupants_remaining=0, occupants_by_zone={}, predicted_bottlenecks=(), blocked_routes=(),
        available_exits=(), building_system_status={}, predicted_rset_seconds=None,
        highest_priority_rescue_areas=(), occupancy_confidence=None, recommendation_confidence=None,
        overall_incident_severity="UNKNOWN",
    )


def make_firefighter_intelligence(live_priority_zone_ids=(), live_possible_assistance_zone_ids=()):

    return FirefighterIntelligenceReport(
        building_status="UNKNOWN", occupants_remaining=0, occupants_by_zone={}, children_count=0,
        wheelchair_users_count=0, possible_injury_count=0, fallen_count=0, helping_groups_count=0,
        hazard_severity_by_zone={}, smoke_conditions_by_zone={}, available_routes=(), blocked_routes=(),
        building_system_status={}, predicted_rset_seconds=None, confidence=None, rescue_priority_areas=(),
        suggested_access_routes=(), live_priority_zone_ids=tuple(live_priority_zone_ids),
        live_possible_assistance_zone_ids=tuple(live_possible_assistance_zone_ids),
    )


def make_advisory_report(
    building_recommendations=(), critical_zones=(), live_priority_zone_ids=(), live_possible_assistance_zone_ids=(),
    scenario_id="s", simulation_time=1.0,
):

    return AdvisoryReport(
        scenario_id=scenario_id, simulation_time=simulation_time, civilian_announcements=(),
        firefighter_intelligence=make_firefighter_intelligence(live_priority_zone_ids, live_possible_assistance_zone_ids),
        building_recommendations=tuple(building_recommendations),
        commander_dashboard=make_commander_dashboard(critical_zones),
        recommendation_history=(),
    )

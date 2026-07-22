import dataclasses
from typing import Dict, Optional, Tuple

from evacuation_guidance.instruction_builder import build_instructions, build_navigation_steps
from evacuation_guidance.message_planner import build_voice_plan, speaker_coverage_for_zone
from evacuation_guidance.models import (
    DeliveryStatus, EvacuationGuidancePlan, EvacuationGuidanceSnapshot, GuidanceConfig, GuidanceInconsistency,
    RouteStatus,
)
from evacuation_guidance.route_planner import excluded_zone_ids, resolve_route
from evacuation_guidance.validation import validate_route

from navigation.node import Node
from navigation.edge import Edge

from evacuation_recommendation.models import RecommendationStatus


# =====================================================
# Live Evacuation Guidance & Zoned Message Planning milestone, Phase 2
# -- the ONE object LiveRuntime owns for this package, mirroring every
# prior live-intelligence engine's own "exactly one shared instance per
# live session" role. compute(time, recommendation_snapshot, building_state,
# speaker_manager) is the single entry point.
#
# Converts EvacuationRecommendationSnapshot ("which exit") into
# EvacuationGuidanceSnapshot ("how to reach it" + "what to say,
# planned only"). NEVER reranks or second-guesses which exit was
# recommended (Phase 22's own "Guidance Engine finds a valid safe path
# to E2, never silently chooses E3 behind Recommendation Engine's
# back" requirement) -- if the recommended exit is unreachable, this
# reports RECOMMENDED_EXIT_UNREACHABLE, it never substitutes another
# exit.
#
# ZERO execution authority: never calls VoiceEvacuationController,
# never imports voice_evacuation/speaker_manager/command_center/
# building_control, never broadcasts (mechanically enforced -- see
# tests/test_evacuation_guidance_architecture_guards.py).
# =====================================================


class EvacuationGuidanceEngine:

    def __init__(self, building, navigation_graph, config: Optional[GuidanceConfig] = None):

        self.building = building
        self.graph = navigation_graph
        self.config = config if config is not None else GuidanceConfig()

        # Phase 12 -- the ONE piece of cross-cycle state this engine
        # owns: {zone_id: (fingerprint, revision)}. Mirrors trajectory_
        # intelligence.history.TrajectoryHistoryStore's own "small,
        # independently-owned, mutable owning registry" convention --
        # NOT a second occupant/zone registry, just a revision counter.
        self._revision_store: Dict[str, Tuple[tuple, int]] = {}

        # THIS-CYCLE-ONLY scratch space: the full Route objects computed
        # while building each zone's EvacuationGuidancePlan, so compute()
        # can hand the same Route straight to message_planner.build_voice_plan()
        # without re-running pathfinding a second time. Rebuilt fresh at
        # the start of every compute() call -- never read across cycles.
        self._route_cache: Dict[str, Optional[object]] = {}

    # =====================================================

    def compute(
        self, time: float, recommendation_snapshot=None, building_state=None, speaker_manager=None,
    ) -> EvacuationGuidanceSnapshot:

        if recommendation_snapshot is None:

            self._revision_store = {}
            return EvacuationGuidanceSnapshot(timestamp=time, zones={}, voice_plans={})

        excluded = excluded_zone_ids(building_state, self.config)

        self._route_cache = {}
        zones = {}
        voice_plans = {}

        for zone_id, recommendation in recommendation_snapshot.zones.items():

            plan = self._plan_for_zone(zone_id, recommendation, excluded, building_state, recommendation_snapshot.timestamp, time)
            zones[zone_id] = plan

            if plan.is_valid():

                coverage = speaker_coverage_for_zone(zone_id, speaker_manager)

                # Phase 17 -- missing speaker coverage never invalidates
                # the route; it becomes its own honest inconsistency
                # code on the PLAN (visible), never silently dropped,
                # and the voice plan is still generated (with an honest
                # NO_SPEAKER_COVERAGE delivery_status) since Phase 25
                # test 23/24 both require the route to stay valid AND
                # the gap to be reported.
                if coverage.delivery_status == DeliveryStatus.NO_SPEAKER_COVERAGE and GuidanceInconsistency.NO_SPEAKER_COVERAGE not in plan.inconsistencies:
                    zones[zone_id] = dataclasses.replace(
                        plan, inconsistencies=plan.inconsistencies + (GuidanceInconsistency.NO_SPEAKER_COVERAGE,),
                    )

                voice_plan = build_voice_plan(
                    zone_id, self._route_cache.get(zone_id), plan.route_status, plan.recommended_exit_id,
                    plan.revision, coverage, time,
                )

                if voice_plan is not None:
                    voice_plans[zone_id] = voice_plan

        self._revision_store = {zid: v for zid, v in self._revision_store.items() if zid in recommendation_snapshot.zones}

        return EvacuationGuidanceSnapshot(timestamp=time, zones=zones, voice_plans=voice_plans)

    # =====================================================

    def _plan_for_zone(
        self, zone_id, recommendation, excluded, building_state, recommendation_timestamp, time,
    ) -> EvacuationGuidancePlan:

        stale = (time - recommendation_timestamp) > self.config.staleness_seconds
        stale_codes = (GuidanceInconsistency.STALE_GUIDANCE,) if stale else ()

        if recommendation.status != RecommendationStatus.RECOMMENDED or recommendation.recommended_exit_id is None:

            self._route_cache[zone_id] = None
            fingerprint = (None, RouteStatus.NO_SAFE_EXIT)
            revision = self._next_revision(zone_id, fingerprint)

            return EvacuationGuidancePlan(
                zone_id=zone_id, floor_id=recommendation.floor_id, recommended_exit_id=None,
                route_status=RouteStatus.NO_SAFE_EXIT, inconsistencies=stale_codes,
                revision=revision, confidence=recommendation.confidence, evidence_timestamp=recommendation_timestamp,
            )

        exit_id = recommendation.recommended_exit_id

        route, status = resolve_route(self.graph, zone_id, exit_id, building_state, self.config)

        inconsistencies = list(stale_codes)

        if route is None:

            if status == RouteStatus.ROUTE_UNCERTAIN:
                inconsistencies.append(GuidanceInconsistency.RECOMMENDATION_ROUTE_MISMATCH)
            else:
                inconsistencies.append(GuidanceInconsistency.RECOMMENDED_EXIT_UNREACHABLE)

            self._route_cache[zone_id] = None
            fingerprint = (exit_id, status)
            revision = self._next_revision(zone_id, fingerprint)

            return EvacuationGuidancePlan(
                zone_id=zone_id, floor_id=recommendation.floor_id, recommended_exit_id=exit_id,
                route_status=status, inconsistencies=tuple(inconsistencies),
                revision=revision, confidence=recommendation.confidence, evidence_timestamp=recommendation_timestamp,
            )

        is_valid, validation_codes = validate_route(route, zone_id, exit_id, excluded)
        inconsistencies.extend(validation_codes)

        if not is_valid:

            self._route_cache[zone_id] = None
            fingerprint = (exit_id, RouteStatus.ROUTE_UNAVAILABLE)
            revision = self._next_revision(zone_id, fingerprint)

            return EvacuationGuidancePlan(
                zone_id=zone_id, floor_id=recommendation.floor_id, recommended_exit_id=exit_id,
                route_status=RouteStatus.ROUTE_UNAVAILABLE, inconsistencies=tuple(inconsistencies),
                revision=revision, confidence=recommendation.confidence, evidence_timestamp=recommendation_timestamp,
            )

        ordered_zone_ids = tuple(node.id for node in route.nodes if node.node_type == Node.ZONE)
        ordered_door_ids = tuple(edge.id for edge in route.edges if edge.edge_type == Edge.DOOR)
        ordered_stair_ids = tuple(edge.id for edge in route.edges if edge.edge_type == Edge.STAIR)

        navigation_steps = build_navigation_steps(route)
        instructions = build_instructions(route, self.building)

        self._route_cache[zone_id] = route

        fingerprint = (exit_id, ordered_door_ids, ordered_stair_ids, status)
        revision = self._next_revision(zone_id, fingerprint)

        return EvacuationGuidancePlan(
            zone_id=zone_id, floor_id=recommendation.floor_id, recommended_exit_id=exit_id,
            route_status=status, route_distance_m=route.total_distance,
            ordered_zone_ids=ordered_zone_ids, ordered_door_ids=ordered_door_ids, ordered_stair_ids=ordered_stair_ids,
            ordered_navigation_steps=navigation_steps, instructions=instructions,
            inconsistencies=tuple(inconsistencies), revision=revision,
            confidence=recommendation.confidence, evidence_timestamp=recommendation_timestamp,
        )

    # =====================================================

    def _next_revision(self, zone_id: str, fingerprint) -> int:

        # Phase 12/25 tests 14/15 -- deterministic, monotonic, per-zone.
        # An identical fingerprint (same recommended exit + same ordered
        # doors/stairs + same status) keeps the SAME revision; any
        # genuine change increments it. Never a random UUID.

        previous = self._revision_store.get(zone_id)

        if previous is not None and previous[0] == fingerprint:
            return previous[1]

        new_revision = (previous[1] + 1) if previous is not None else 1
        self._revision_store[zone_id] = (fingerprint, new_revision)

        return new_revision

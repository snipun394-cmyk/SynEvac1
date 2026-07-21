from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from advisory_system.ai_evidence import AIDecisionEvidence, UNAVAILABLE_AI_DECISION_EVIDENCE, evidence_from_bottleneck_prediction
from advisory_system.crowd_evidence import (
    CrowdAssetDetail, CrowdDecisionEvidence, CrowdZoneDetail, UNAVAILABLE_CROWD_DECISION_EVIDENCE,
)
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs, AdvisoryReport

from crowd_intelligence.models import CrowdIntelligenceSnapshot, TrendDirection

from live_system.live_ai_gateway import LiveAIPredictionSnapshot


# =====================================================
# AI-Augmented Decision Policy & Advisory Integration milestone -- the
# seam LiveOrchestrator uses to reach advisory_system, mirroring
# live_system.live_ai_gateway's own established "thin Protocol + real
# adapter, LiveOrchestrator never constructs the underlying package's
# internals itself" shape exactly. This is the ONE module in
# live_system allowed to import advisory_system (enforced by
# tests/test_live_system.py, mirroring the identical rule
# live_ai_gateway.py already established for ai_registry).
#
# THE HONEST LIMIT THIS MODULE DOES NOT PAPER OVER: Phase 1/2's own
# investigation (docs/architecture/ai_augmented_advisory_integration.md
# §1) confirmed decision_policy.generate_policy()'s six rule modules
# read exclusively from GroundTruth's own post-hoc, completed-
# simulation analytics (zone_risk_scores, stair_risk_scores,
# zone_route_stats, hazard_spread_order, exits/stairs_exceeding_
# capacity) and Scenario's own designed occupant/exit-state data --
# NONE of which has a building_state.models.BuildingState equivalent.
# This module therefore does NOT, and architecturally cannot, build an
# AdvisoryReport from BuildingState alone -- it requires a caller to
# supply an already-valid, already-real Building/Scenario/GroundTruth
# (exactly the artifacts a completed campaign, or a Replay session
# reading one, already has -- see command_center/incident_data.py's
# own _build_advisory_reports() for the precedent this class's
# decision_policy_provider callable directly mirrors: recomputing
# zone_policy/exit_policy/stair_policy/announcement_policy fresh each
# cycle from a fixed Building/Scenario/GroundTruth). It never
# fabricates a GroundTruth-shaped stand-in from BuildingState.
#
# What IS genuinely live here is the AI evidence layered on top --
# ai_decision_evidence_from_prediction_snapshot() converts a live_ai_
# gateway.LiveAIPredictionSnapshot (built fresh from a real live/replay
# BuildingState every cycle) into advisory_system's own plain-value
# AIDecisionEvidence, letting the SAME already-existing, real Building/
# Scenario/GroundTruth-based AdvisoryReport pipeline additionally
# reflect a genuinely current AI signal each cycle.
# =====================================================


def ai_decision_evidence_from_prediction_snapshot(
    snapshot: Optional[LiveAIPredictionSnapshot],
) -> AIDecisionEvidence:

    if snapshot is None or snapshot.bottleneck is None:
        return UNAVAILABLE_AI_DECISION_EVIDENCE

    bottleneck = snapshot.bottleneck

    return evidence_from_bottleneck_prediction(
        probability=bottleneck.probability,
        predicted_occurrence=bottleneck.predicted_occurrence,
        threshold=bottleneck.threshold,
        model_id=bottleneck.model_id,
        model_version=bottleneck.model_version,
        prediction_timestamp=snapshot.timestamp,
        building_state_timestamp=snapshot.building_state_timestamp,
        feature_schema_version=snapshot.feature_schema_version,
    )


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone
# -- the crowd-evidence counterpart to
# ai_decision_evidence_from_prediction_snapshot() immediately above,
# same placement/role: this module is the ONE place in live_system
# allowed to import BOTH advisory_system AND crowd_intelligence (each
# already independently permitted here -- see this module's own
# docstring for advisory_system; crowd_intelligence is already a
# live_system dependency via live_system.crowd_intelligence_gateway),
# so the reduction from a rich CrowdIntelligenceSnapshot down to
# advisory_system's own plain-value CrowdDecisionEvidence happens here,
# keeping advisory_system itself free of any crowd_intelligence import
# (see advisory_system.crowd_evidence's own docstring).
#
# Only ALREADY-FLAGGED zones/assets (from CrowdIntelligenceSnapshot.
# building_summary's own congested_doors/congested_exits/congested_
# stairs/zones_above_configured_density_threshold, plus whichever
# additionally show a RISING/FALLING trend or a nonzero queue) get a
# CrowdZoneDetail/CrowdAssetDetail entry -- never one entry per zone/
# asset in the building (Phase 2's own "do not duplicate the entire
# snapshot unnecessarily").
# =====================================================


def crowd_decision_evidence_from_snapshot(
    snapshot: Optional[CrowdIntelligenceSnapshot],
) -> CrowdDecisionEvidence:

    if snapshot is None:
        return UNAVAILABLE_CROWD_DECISION_EVIDENCE

    summary = snapshot.building_summary

    asset_lookup = (
        [("Door", asset_id, metrics) for asset_id, metrics in snapshot.door_metrics.items()]
        + [("Exit", asset_id, metrics) for asset_id, metrics in snapshot.exit_metrics.items()]
        + [("Stair", asset_id, metrics) for asset_id, metrics in snapshot.stair_metrics.items()]
    )

    queue_detected_ids = tuple(sorted(
        asset_id for _type, asset_id, metrics in asset_lookup if metrics.queue_candidate_count > 0
    ))
    rising_ids = tuple(sorted(
        asset_id for _type, asset_id, metrics in asset_lookup if metrics.trend == TrendDirection.RISING
    ))
    clearing_ids = tuple(sorted(
        asset_id for _type, asset_id, metrics in asset_lookup if metrics.trend == TrendDirection.FALLING
    ))
    position_unavailable_ids = tuple(sorted(
        asset_id for _type, asset_id, metrics in asset_lookup if not metrics.position_available
    ))

    flagged_asset_ids = (
        set(summary.congested_doors) | set(summary.congested_exits) | set(summary.congested_stairs)
        | set(queue_detected_ids) | set(rising_ids) | set(clearing_ids)
    )

    asset_details = {
        asset_id: CrowdAssetDetail(
            asset_type=asset_type,
            congestion_level=metrics.congestion_level.name if metrics.congestion_level is not None else None,
            trend=metrics.trend.name if metrics.trend != TrendDirection.UNKNOWN else None,
            queue_candidate_count=metrics.queue_candidate_count,
            approaching_count=metrics.approaching_count,
            position_available=metrics.position_available,
        )
        for asset_type, asset_id, metrics in asset_lookup
        if asset_id in flagged_asset_ids
    }

    most_congested_level = None
    if summary.most_congested_asset_id is not None:

        matching = next(
            (metrics for _type, asset_id, metrics in asset_lookup if asset_id == summary.most_congested_asset_id), None,
        )
        if matching is not None and matching.congestion_level is not None:
            most_congested_level = matching.congestion_level.name

    highest_density_level = None
    zone_ids_needing_detail = set(summary.zones_above_configured_density_threshold)

    if summary.highest_density_zone is not None:

        zone_ids_needing_detail.add(summary.highest_density_zone)
        zone_metrics = snapshot.zone(summary.highest_density_zone)

        if zone_metrics is not None and zone_metrics.density_classification is not None:
            highest_density_level = zone_metrics.density_classification.name

    zone_details = {}
    for zone_id in zone_ids_needing_detail:

        zone_metrics = snapshot.zone(zone_id)
        if zone_metrics is None:
            continue

        zone_details[zone_id] = CrowdZoneDetail(
            density_classification=(
                zone_metrics.density_classification.name if zone_metrics.density_classification is not None else None
            ),
            density_people_per_m2=zone_metrics.density_people_per_m2,
            trend=zone_metrics.trend.name if zone_metrics.trend != TrendDirection.UNKNOWN else None,
            position_coverage_fraction=zone_metrics.position_coverage_fraction,
        )

    return CrowdDecisionEvidence(
        available=True,
        timestamp=snapshot.timestamp,
        highest_density_zone_id=summary.highest_density_zone,
        highest_density_level=highest_density_level,
        most_congested_asset_id=summary.most_congested_asset_id,
        most_congested_asset_type=summary.most_congested_asset_type,
        most_congested_level=most_congested_level,
        congested_exit_ids=tuple(sorted(summary.congested_exits)),
        congested_stair_ids=tuple(sorted(summary.congested_stairs)),
        congested_door_ids=tuple(sorted(summary.congested_doors)),
        queue_detected_asset_ids=queue_detected_ids,
        rising_congestion_asset_ids=rising_ids,
        clearing_congestion_asset_ids=clearing_ids,
        position_unavailable_asset_ids=position_unavailable_ids,
        position_coverage_fraction=summary.position_coverage_fraction,
        zones_above_density_threshold=tuple(sorted(summary.zones_above_configured_density_threshold)),
        zone_details=zone_details,
        asset_details=asset_details,
    )


# =====================================================


class LiveAdvisoryGateway(Protocol):

    # Returning None is this Protocol's own documented "no update this
    # cycle" signal -- covers BOTH "not enough information yet" (e.g.
    # decision_policy_provider itself returned None) and any caught
    # internal failure (see ReplayCompatibleAdvisoryGateway.generate()'s
    # own try/except) -- LiveOrchestrator never distinguishes the two,
    # it only ever leaves the previous AdvisoryReport in StateManager
    # untouched under its own honest, non-bumped component_timestamps
    # entry, the identical staleness-detection mechanism live_ai_gateway
    # already established.
    #
    # crowd_evidence defaults to None -- Live Crowd Intelligence ->
    # Operational Advisory Integration milestone, additive. A caller
    # that never supplies one (every existing test/deployment) keeps
    # working unchanged; LiveOrchestrator.run_cycle() always passes one
    # (UNAVAILABLE_CROWD_DECISION_EVIDENCE when no crowd_intelligence_
    # gateway is configured, or this cycle's real evidence otherwise).

    def generate(
        self, ai_evidence: Optional[AIDecisionEvidence], time: float,
        crowd_evidence: Optional[CrowdDecisionEvidence] = None,
    ) -> Optional[AdvisoryReport]: ...


DecisionPolicyProvider = Callable[[float], Optional[object]]  # -> decision_policy.policy.DecisionPolicy or None
HumanObservationsProvider = Callable[[float], Mapping[str, object]]
BuildingSystemStateProvider = Callable[[float], Mapping[str, Mapping[str, str]]]
TimelineRowsProvider = Callable[[float], Tuple[Dict[str, object], ...]]


class ReplayCompatibleAdvisoryGateway:

    # The real adapter. Named "ReplayCompatible" deliberately, not
    # "Live" -- it is honest about requiring the same Building/Scenario/
    # GroundTruth artifacts a Replay/completed-campaign session already
    # has, per this module's own docstring above. decision_policy_
    # provider is called fresh every cycle (mirroring command_center/
    # incident_data.py's own per-frame zone_policy/exit_policy/
    # stair_policy/announcement_policy recomputation) and may itself
    # return None to mean "not enough information yet this cycle" --
    # the same documented convention live_system.integration.
    # DecisionInputsBuilder already established for an analogous gap.

    def __init__(
        self,
        *,
        building,
        scenario,
        ground_truth,
        decision_policy_provider: DecisionPolicyProvider,
        human_observations_provider: Optional[HumanObservationsProvider] = None,
        building_system_state_provider: Optional[BuildingSystemStateProvider] = None,
        timeline_rows_provider: Optional[TimelineRowsProvider] = None,
        orchestrator: Optional[AdvisoryOrchestrator] = None,
    ):

        self._building = building
        self._scenario = scenario
        self._ground_truth = ground_truth
        self._decision_policy_provider = decision_policy_provider
        self._human_observations_provider = human_observations_provider
        self._building_system_state_provider = building_system_state_provider
        self._timeline_rows_provider = timeline_rows_provider
        self._orchestrator = orchestrator if orchestrator is not None else AdvisoryOrchestrator()

    # =====================================================

    def generate(
        self, ai_evidence: Optional[AIDecisionEvidence], time: float,
        crowd_evidence: Optional[CrowdDecisionEvidence] = None,
    ) -> Optional[AdvisoryReport]:

        try:

            decision_policy = self._decision_policy_provider(time)

            if decision_policy is None:
                return None

            inputs = AdvisoryInputs(
                building=self._building,
                scenario=self._scenario,
                ground_truth=self._ground_truth,
                decision_policy=decision_policy,
                human_observations=(
                    self._human_observations_provider(time) if self._human_observations_provider is not None else {}
                ),
                building_system_state=(
                    self._building_system_state_provider(time)
                    if self._building_system_state_provider is not None else {}
                ),
                timeline_rows=(
                    self._timeline_rows_provider(time) if self._timeline_rows_provider is not None else ()
                ),
                simulation_time=time,
                ai_decision_evidence=ai_evidence,
                crowd_decision_evidence=crowd_evidence,
            )

            return self._orchestrator.generate_report(inputs)

        except Exception:  # noqa: BLE001 -- Advisory failure must never crash the live cycle (Phase 7)

            return None

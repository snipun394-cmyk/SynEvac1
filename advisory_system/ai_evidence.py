from dataclasses import dataclass
from typing import Any, Dict, Optional


# =====================================================
# AI-Augmented Decision Policy & Advisory Integration milestone --
# Phase 3's own required "AI Decision Evidence" structure, placed here
# in advisory_system (not decision_policy) as a direct, evidence-based
# consequence of Phase 1/2's own investigation findings:
#
#   - decision_policy.generate_policy()'s six rule modules (zone_policy,
#     exit_policy, stair_policy, rescue_policy, announcement_policy,
#     human_priority_policy) read EXCLUSIVELY from GroundTruth's own
#     post-hoc, completed-simulation analytics (zone_risk_scores,
#     stair_risk_scores, zone_route_stats, hazard_spread_order,
#     exits_exceeding_capacity, stairs_exceeding_capacity, ...) and
#     Scenario's own designed occupant/exit-state data -- confirmed,
#     field by field, to have NO building_state.models.BuildingState
#     equivalent. decision_policy cannot run from BuildingState alone,
#     and this milestone does not fabricate GroundTruth to force it to.
#   - decision_policy.policy.DecisionPolicy itself carries no
#     confidence/explanation field of any kind -- there is structurally
#     nowhere on it for "AI evidence" to attach even if it could run
#     live.
#   - advisory_system already has a real, working, purpose-built
#     extension point for exactly this (AdvisoryInputs.ai_predictions/
#     .ai_confidence, advisory_engine._ai_signal_for()/_confidence_source(),
#     confidence_engine.recommendation_confidence()) -- and the
#     established ai_inference.recommendation.py precedent already
#     proves the discipline this milestone needs: AI only ever
#     ANNOTATES an already-decided DecisionPolicy output, never mutates
#     or replaces one.
#
# Net effect: AI Decision Evidence flows into advisory_system, not
# decision_policy. decision_policy is not imported by this module and
# is not modified anywhere in this milestone -- the strongest possible
# safety guarantee available (AI cannot influence AVOID/CLOSE/SHELTER
# decisions because it never reaches the code that makes them, not
# merely because a rule says it must not).
#
# NO localization field (no suspected_bottleneck_asset_ids, no
# "likely_zone_id") exists on this type. Phase 2's own investigation
# found NO live-compatible signal anywhere in this codebase capable of
# identifying which specific stair/door/exit/zone is congested --
# ground_truth/bottleneck.py's per-asset findings require a completed
# simulation (GROUND_TRUTH_ONLY); simulation_interactive's per-edge
# queue/occupancy state is the *interactive simulator's own internal
# event-heap bookkeeping* (SIMULATION_ONLY, never perception-derived);
# and building_state/multi_camera_fusion have no door/edge-indexed
# occupancy concept at all (the Navigation Graph's Node type is
# Zone-only; Door/Exit/Stair are Edge-only, and BuildingState.
# zone_occupancy is Node-keyed). A future milestone that builds a real
# per-door congestion aggregation layer (e.g. from FusedTrack.history.
# zone_transitions, the closest already-live-estimable building block
# -- see docs/architecture/ai_augmented_advisory_integration.md §4)
# could add such a field, sourced from THAT subsystem -- never
# synthesized from this building-wide occurrence model.
# =====================================================


@dataclass(frozen=True)
class AIDecisionEvidence:

    # `available=False` (every other field left at its None default) is
    # the honest "no AI evidence this cycle" value -- UNAVAILABLE_
    # AI_DECISION_EVIDENCE below is the one canonical instance of it,
    # never fabricated as an all-zero/all-optimistic placeholder.

    available: bool

    bottleneck_occurrence_probability: Optional[float] = None
    bottleneck_predicted: Optional[bool] = None
    threshold: Optional[float] = None

    model_id: Optional[str] = None
    model_version: Optional[str] = None

    # Always "PRODUCTION_CANDIDATE" for the one model this milestone is
    # authorized to treat as operational (BottleneckOccurrenceModel_
    # LiveCompatible) -- carried explicitly rather than assumed, so a
    # future second AI evidence source (e.g. an eventual, still-
    # EXPERIMENTAL evacuation-time signal) could never silently be
    # treated as equally authoritative just by existing on this type.
    model_status: Optional[str] = None

    prediction_timestamp: Optional[float] = None
    building_state_timestamp: Optional[float] = None
    feature_schema_version: Optional[str] = None

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "available": self.available,
            "bottleneck_occurrence_probability": self.bottleneck_occurrence_probability,
            "bottleneck_predicted": self.bottleneck_predicted,
            "threshold": self.threshold,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "prediction_timestamp": self.prediction_timestamp,
            "building_state_timestamp": self.building_state_timestamp,
            "feature_schema_version": self.feature_schema_version,
        }


UNAVAILABLE_AI_DECISION_EVIDENCE = AIDecisionEvidence(available=False)


# =====================================================


def evidence_from_bottleneck_prediction(
    *,
    probability: float,
    predicted_occurrence: bool,
    threshold: float,
    model_id: str,
    model_version: str,
    prediction_timestamp: float,
    building_state_timestamp: Optional[float],
    feature_schema_version: str,
) -> AIDecisionEvidence:

    # The one constructor this milestone's live wiring calls -- kept
    # free of any import of live_system/ai_registry/ai_inference types
    # (plain floats/strings/bools only) so advisory_system itself gains
    # no new package dependency; the adapter that extracts these plain
    # values out of a live_system.live_ai_gateway.LiveAIPredictionSnapshot
    # lives in live_system, not here (see live_system/live_advisory_
    # gateway.py) -- the same dependency-direction discipline every
    # earlier milestone's own gateway module already established.

    return AIDecisionEvidence(
        available=True,
        bottleneck_occurrence_probability=probability,
        bottleneck_predicted=predicted_occurrence,
        threshold=threshold,
        model_id=model_id,
        model_version=model_version,
        model_status="PRODUCTION_CANDIDATE",
        prediction_timestamp=prediction_timestamp,
        building_state_timestamp=building_state_timestamp,
        feature_schema_version=feature_schema_version,
    )

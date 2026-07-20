from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from advisory_system.ai_evidence import AIDecisionEvidence, UNAVAILABLE_AI_DECISION_EVIDENCE, evidence_from_bottleneck_prediction
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs, AdvisoryReport

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

    def generate(self, ai_evidence: Optional[AIDecisionEvidence], time: float) -> Optional[AdvisoryReport]: ...


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

    def generate(self, ai_evidence: Optional[AIDecisionEvidence], time: float) -> Optional[AdvisoryReport]:

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
            )

            return self._orchestrator.generate_report(inputs)

        except Exception:  # noqa: BLE001 -- Advisory failure must never crash the live cycle (Phase 7)

            return None

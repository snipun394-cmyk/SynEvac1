from typing import FrozenSet, Iterable, Mapping, Sequence, Tuple

from models.building import Building

from ai_decision.recommendation import DecisionRecommendation

from live_perception.building_observation_adapter import BuildingObservationAdapter

from live_runtime.live_decision_coordinator import LiveDecisionCoordinator
from live_runtime.zone_recommendation_coordinator import ZoneRecommendationCoordinator

from rl_training.observation_space import BROADCAST_NONE, RECOMMENDATION_NONE
from rl_training.production.decision_engine import RLActionDecisionEngine
from rl_training.production.live_inference import LiveRLInferenceEngine

from sensor_fusion.observation import FusedObservation


# =====================================================
# Live RL Decision Pipeline milestone -- docs/architecture/
# live_rl_decision_pipeline_investigation.txt, Sections A-I. The narrowest
# honest end-to-end live decision cycle this codebase has:
#
#     FusedObservation tuples
#       -> BuildingObservationAdapter -> BuildingObservation
#       -> LiveRLInferenceEngine.infer() -> InferredRLAction
#       -> ZoneRecommendationCoordinator.coordinate(...) -> per-zone mapping
#       -> RLActionDecisionEngine.set_zone_recommendations(...)
#       -> LiveDecisionCoordinator.tick(...) -> DecisionRecommendation
#
# This class is a PURE COMPOSITION layer -- it contains no RL observation
# encoding, no model prediction, no action decoding, no route/capacity
# reasoning, and no congestion prediction algorithm of its own. Every one
# of those already exists, already tested, in the five components this
# class owns and calls through their existing public interfaces only.
# Nothing here duplicates StairCongestionPredictor/ObservableAssetSnapshot
# logic -- LiveDecisionCoordinator.tick() already owns that responsibility
# entirely, unchanged, called verbatim. Nothing here duplicates
# PathfindingEngine/capacity reasoning either -- ZoneRecommendationCoordinator
# already owns that responsibility entirely, unchanged, called verbatim
# (Zone Recommendation Coordination milestone, docs/architecture/
# multi_zone_joint_coordination_implementation_investigation.txt).
#
# Construction model (deliberate, not an oversight -- see the governing
# investigation's Section 10): this class constructs ALL FIVE of
# LiveRLInferenceEngine/RLActionDecisionEngine/LiveDecisionCoordinator/
# BuildingObservationAdapter/ZoneRecommendationCoordinator INTERNALLY. It
# never accepts a pre-built instance of any of them. This guarantees every
# LiveRLDecisionPipeline instance goes through LiveRLInferenceEngine's own
# existing IncompatibleModelArtifactError check (Section 10) and starts
# with completely fresh decision/congestion state (Section 6/7) --
# "construct a new pipeline" IS the entire session-reset mechanism; no
# separate reset() method exists or is needed, mirroring
# StairCongestionRuntimeBridge's own already-established "a fresh instance
# means a fresh history" precedent.
#
# Ordering contract (Section 4 of the governing investigation, restated
# here as the reason this file's own tick() is written in exactly this
# order, not merely as an incidental consequence of the current code): the
# RL action is selected via LiveRLInferenceEngine.infer() BEFORE
# ZoneRecommendationCoordinator ever runs, and coordination itself
# (Zone Recommendation Coordination milestone) runs BEFORE stair
# congestion state is ever computed or published. Coordination NEVER
# re-invokes the RL model and NEVER changes what the policy itself chose
# this tick -- it only decides, per zone, whether that single choice is
# passed through unchanged or (only when a real, provable capacity
# conflict exists) redirected to a real, reachable alternate for that
# specific zone; it consumes recommended_edge_id/zone_ids exactly as
# set_action() used to. Stair congestion remains purely informational and
# additive to the DECISION TEXT only (RLActionDecisionEngine.
# _stair_congestion_note(), unmodified) -- it has no mechanism to reach
# back and change an action or a coordination decision already recorded in
# a prior step, and this pipeline's own structure makes that true by
# construction, not by coincidence of independent setters happening to
# commute. See tests/test_live_rl_decision_pipeline.py::
# CongestionCannotInfluenceActionSelectionTests for the direct regression
# proof (unmodified by this milestone).
#
# Standalone by design: does not import or construct LiveOrchestrator,
# advisory_system, decision_policy, or ProductionRLEnvironment anywhere.
# Nothing automatically invokes this class -- a caller must still call
# tick() itself, supplying already-computed fused observations and stair
# occupancy/coverage evidence (exactly as every component below it already
# requires).
# =====================================================


class LiveRLDecisionPipeline:

    def __init__(
        self,
        model_directory: str,
        building: Building,
        congestion_threshold: float,
        max_time: float,
    ):

        # All five components constructed internally -- see this module's
        # own docstring for why a caller-supplied, pre-built instance of
        # any of them is deliberately not accepted.
        self._rl_engine = LiveRLInferenceEngine(model_directory, building)
        self._decision_engine = RLActionDecisionEngine()
        self._decision_coordinator = LiveDecisionCoordinator(self._decision_engine, congestion_threshold)
        self._observation_adapter = BuildingObservationAdapter()
        self._zone_coordinator = ZoneRecommendationCoordinator(building)

        self._max_time = max_time

        # Self-issued bookkeeping ONLY -- never perception, never
        # privileged, mirroring rl_training.production.environment.
        # ProductionRLEnvironment's own identical fields (self._active_
        # recommendation_target/_active_recommendation_label), reproduced
        # here in miniature because this pipeline must not import or
        # construct ProductionRLEnvironment itself. `active_broadcast` is
        # always BROADCAST_NONE, exactly matching ProductionRLEnvironment.
        # _encode()'s own current hardcoded value -- no broadcast action
        # seam exists in production to track anything else. `steps_since_
        # recommendation_change` is tracked as a plain local counter, not
        # via rl_training.reward_function.ActionState (that class is
        # reward-training machinery this pipeline has no reason to import
        # for a value it can track directly from the label alone).
        self._active_recommendation_label = RECOMMENDATION_NONE
        self._active_recommendation_target = None
        self._steps_since_recommendation_change = 0

    # =====================================================

    def tick(
        self,
        fused_observations: Sequence[FusedObservation],
        known_zone_ids: Iterable[str],
        stair_ids: Iterable[str],
        occupant_ids_by_stair: Mapping[str, Tuple[str, ...]],
        covered_stair_ids: FrozenSet[str],
        timestamp: float,
    ) -> DecisionRecommendation:

        # Step A -- build the RL observation. known_zone_ids is the
        # caller's own honest claim of which zones have real FACP/detector
        # coverage this cycle -- never defaulted to "every zone this
        # pipeline knows about" (that would fabricate coverage the caller
        # never confirmed). BuildingObservationAdapter's own existing
        # honesty semantics (no smoke/heat/fire, no fabricated confirmed-
        # zero occupancy, no route information) are unchanged and used
        # verbatim.
        observation = self._observation_adapter.to_building_observation(
            fused_observations, timestamp, known_zone_ids=known_zone_ids,
        )

        # Step B -- RL inference. active_recommendation_by_zone reflects
        # whatever this pipeline decided LAST tick (RECOMMENDATION_NONE on
        # the very first tick) -- built fresh here, before this tick's own
        # decision updates it below, exactly matching ProductionRLEnvironment.
        # _encode()'s own call-order convention.
        active_recommendation_by_zone = {
            zone_id: self._active_recommendation_label for zone_id in self._rl_engine.zone_ids
        }

        inferred = self._rl_engine.infer(
            observation,
            active_recommendation_by_zone,
            BROADCAST_NONE,
            elapsed_time=timestamp,
            max_time=self._max_time,
            active_recommendation_target=self._active_recommendation_target,
            steps_since_recommendation_change=self._steps_since_recommendation_change,
        )

        # Step C -- Zone Recommendation Coordination milestone: the RL
        # policy's single chosen target (still fully target-agnostic --
        # inferred.recommended_edge_id is None for NOOP, an exit's own id
        # for RECOMMEND_EXIT, or a stair's own id for RECOMMEND_STAIR, and
        # ZoneRecommendationCoordinator.coordinate() itself has no branch
        # on which of these it is) is handed to the coordinator, which
        # returns a per-zone mapping -- the RL-selected target unchanged
        # for every zone UNLESS a real, provable capacity conflict was
        # found for that specific zone (see live_runtime/
        # zone_recommendation_coordinator.py's own docstring for the full
        # authority boundary). This REPLACES the previous direct
        # set_action(...) call with set_zone_recommendations(...) --
        # RLActionDecisionEngine.decide() branches on whichever setter was
        # called identically either way (rl_training/production/
        # decision_engine.py), so no change to that engine was needed.
        zone_assignments = self._zone_coordinator.coordinate(
            inferred.recommended_edge_id, self._rl_engine.zone_ids, observation, timestamp,
        )
        self._decision_engine.set_zone_recommendations(zone_assignments)

        if inferred.recommendation_label != self._active_recommendation_label:
            self._steps_since_recommendation_change = 0
        else:
            self._steps_since_recommendation_change += 1

        self._active_recommendation_label = inferred.recommendation_label

        if inferred.recommended_edge_id is not None:
            self._active_recommendation_target = inferred.recommended_edge_id

        # Step D -- congestion monitoring and decision assembly. Delegated
        # entirely to LiveDecisionCoordinator.tick(), unmodified -- this
        # pipeline never calls StairCongestionPredictor or
        # compute_asset_occupancy_snapshot()/ObservableAssetSnapshot
        # itself. The SAME observation built in Step A (not a fresh empty
        # default) and the SAME timestamp are passed through, so decide()
        # sees exactly the perception evidence the RL policy and the
        # coordinator itself just acted on.
        return self._decision_coordinator.tick(
            stair_ids, occupant_ids_by_stair, covered_stair_ids, timestamp,
            building_observation=observation,
        )

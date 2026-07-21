from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from perception.models.building_observation import (
    BuildingObservation,
    ObservedEdgeState,
    ObservedNodeState,
    ObservedOccupancy,
)
from perception.models.human_observation import HumanObservation

from ai_inference.predictor import Prediction
from ai_inference.recommendation import Recommendation

from decision_policy.policy import DecisionPolicy

from advisory_system.recommendation_models import AdvisoryReport

from building_state.models import BuildingState

from crowd_intelligence.models import CrowdIntelligenceSnapshot

from evacuation_progress.models import EvacuationProgressSnapshot

from live_system.live_ai_gateway import LiveAIPredictionSnapshot


@dataclass(frozen=True)
class LiveBuildingSnapshot:

    # The one object the Update Loop assembles once per cycle and the
    # rest of live_system (Command Center notification, tests,
    # operator tooling) is meant to read instead of reaching into any
    # individual upstream package's own output type directly -- same
    # "one canonical object downstream consumers read" role
    # BuildingObservation/GroundTruth/DecisionPolicy each already play
    # for their own layer. Immutable and replace-not-mutate, the same
    # convention every other snapshot type in this codebase (
    # HazardSnapshot, OccupancySnapshot, BuildingObservation) already
    # follows -- StateManager below never edits a field in place, only
    # ever constructs a new LiveBuildingSnapshot from the previous one.
    #
    # building_state (Canonical Live BuildingState Runtime Assembly
    # milestone) is the CANONICAL current-state field going forward --
    # see building_state.models.BuildingState's own docstring: the
    # single authoritative snapshot of "what is true right now",
    # assembled each cycle by whatever BuildingStateGateway this
    # orchestrator was configured with (live_system.building_state_
    # gateway.EstimatorBuildingStateGateway composes it from
    # BuildingStateEstimator, exactly as designer/building_state_debug_
    # runner.py already does for the Designer). None until a
    # building_state_gateway is configured and has run at least once --
    # never a fabricated empty-but-present BuildingState.
    #
    # building_observation (Live Perception's own already-fused shape,
    # via SensorFusionPerceptionGateway) is kept, unchanged, as a
    # SEPARATE, pre-existing compatibility field -- it predates this
    # milestone, existing callers/tests still populate and read it, and
    # it is deliberately NOT folded into or replaced by building_state:
    # BuildingState's own docstring is explicit that it never carries
    # an AI/decision-policy field, whereas ai_predictions/decision_
    # policy/recommendations below remain this snapshot's own,
    # unconnected to BuildingState by this milestone's own explicit
    # scope. occupancy_for()/hazard_for()/edge_for() below stay total
    # accessors over building_observation specifically, exactly as
    # before -- they are not redefined against building_state.
    #
    # engineering_state has no live source composed into this phase
    # yet (Phase 2 explicitly scopes CCTV/Smoke/Heat/FACP only, not a
    # door/exit/stair control integration) -- it defaults to empty
    # rather than a fabricated "all normal" reading, and is populated
    # only if/when a caller's own integration supplies one via
    # StateManager.update_engineering_state(). (BuildingState.control_
    # status, reachable through building_state above once a control_
    # snapshot_provider is configured, is the richer, typed equivalent
    # for whatever a caller no longer needs this untyped mapping for.)

    timestamp: float = 0.0

    building_observation: Optional[BuildingObservation] = None
    building_state: Optional[BuildingState] = None
    engineering_state: Mapping[str, Any] = field(default_factory=dict)

    ai_predictions: Mapping[str, Prediction] = field(default_factory=dict)

    # Live AI Inference Runtime Integration milestone -- the canonical
    # live_system.live_ai_gateway.LiveAIPredictionSnapshot field, kept
    # deliberately distinct from ai_predictions above (that field
    # belongs to the original, pre-existing, still-unimplemented-in-
    # production AIInferenceGateway/feature_row_builder seam operating
    # on this whole LiveBuildingSnapshot -- see live_system.integration
    # -- and is untouched by this milestone). None until a live_ai_
    # gateway is configured and has produced at least one snapshot --
    # never a fabricated "AVAILABLE" placeholder.
    ai_prediction_snapshot: Optional[LiveAIPredictionSnapshot] = None

    # AI-Augmented Decision Policy & Advisory Integration milestone --
    # the canonical live_system.live_advisory_gateway output. Kept
    # deliberately distinct from decision_policy/recommendations below
    # (the original, pre-existing, still-unimplemented-in-production
    # decision_policy_gateway/recommendation_builder seam -- see
    # live_system.integration -- untouched by this milestone). None
    # until a live_advisory_gateway is configured and has produced at
    # least one report -- never a fabricated placeholder.
    advisory_report: Optional[AdvisoryReport] = None

    decision_policy: Optional[DecisionPolicy] = None
    recommendations: Tuple[Recommendation, ...] = field(default_factory=tuple)

    # Live Occupancy, Crowd Density & Congestion Intelligence milestone --
    # the canonical live_system.crowd_intelligence_gateway output, kept
    # as its own sibling field exactly like ai_prediction_snapshot/
    # advisory_report above (investigated and deliberately NOT folded
    # into building_state -- see live_system.crowd_intelligence_gateway's
    # own module docstring for why). None until a crowd_intelligence_
    # gateway is configured and has produced at least one snapshot --
    # never a fabricated empty-but-present snapshot.
    crowd_intelligence: Optional[CrowdIntelligenceSnapshot] = None

    # Live Evacuation Progress, Flow & Clearance Intelligence milestone --
    # the canonical live_system.evacuation_progress_gateway output, kept
    # as its own sibling field exactly like crowd_intelligence/
    # ai_prediction_snapshot/advisory_report above. None until an
    # evacuation_progress_gateway is configured and has produced at
    # least one snapshot -- never a fabricated empty-but-present snapshot.
    evacuation_progress: Optional[EvacuationProgressSnapshot] = None

    # component -> the timestamp its own field was last actually
    # updated -- distinct from `timestamp` (this snapshot's own
    # as-of time) because a cycle in which, say, AI Inference is not
    # configured leaves ai_predictions/its component_timestamps entry
    # unchanged rather than stamped with a time nothing happened at.
    component_timestamps: Mapping[str, float] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(
            self, "engineering_state", MappingProxyType(dict(self.engineering_state)),
        )
        object.__setattr__(
            self, "ai_predictions", MappingProxyType(dict(self.ai_predictions)),
        )
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(
            self, "component_timestamps", MappingProxyType(dict(self.component_timestamps)),
        )

    # =====================================================
    # Total accessors -- delegate to BuildingObservation's own, or fall
    # back to the same "never observed" default it would if no
    # observation exists yet at all.
    # =====================================================

    def occupancy_for(self, zone_id: str) -> ObservedOccupancy:

        if self.building_observation is None:
            return ObservedOccupancy()

        return self.building_observation.occupancy_observation(zone_id)

    # =====================================================

    def hazard_for(self, node_id: str) -> ObservedNodeState:

        if self.building_observation is None:
            return ObservedNodeState()

        return self.building_observation.node_observation(node_id)

    # =====================================================

    def edge_for(self, edge_id: str) -> ObservedEdgeState:

        if self.building_observation is None:
            return ObservedEdgeState()

        return self.building_observation.edge_observation(edge_id)

    # =====================================================

    def human_observation_for(self, person_id: str) -> Optional[HumanObservation]:

        # Human Perception Integration -- additive. Unlike occupancy_for()/
        # hazard_for()/edge_for() above, there is no default
        # HumanObservation to fabricate for a person_id not currently
        # observed (see BuildingObservation.human_observation()'s own
        # docstring for why) -- this mirrors that same None-returning
        # total accessor, one layer up.

        if self.building_observation is None:
            return None

        return self.building_observation.human_observation(person_id)

    # =====================================================

    def human_observations(self) -> Mapping[str, HumanObservation]:

        if self.building_observation is None:
            return {}

        return self.building_observation.human_observations

    # =====================================================

    def prediction_for(self, prediction_type: str) -> Optional[Prediction]:

        return self.ai_predictions.get(prediction_type)

    # =====================================================

    def replace(self, **changes) -> "LiveBuildingSnapshot":

        # The one place a new snapshot is ever constructed from the
        # previous one -- every StateManager.update_*() method below
        # goes through this, so "carry every field forward except the
        # ones this update actually changed" is expressed exactly once,
        # not re-typed at each call site.

        current = {
            "timestamp": self.timestamp,
            "building_observation": self.building_observation,
            "building_state": self.building_state,
            "engineering_state": self.engineering_state,
            "ai_predictions": self.ai_predictions,
            "ai_prediction_snapshot": self.ai_prediction_snapshot,
            "advisory_report": self.advisory_report,
            "decision_policy": self.decision_policy,
            "recommendations": self.recommendations,
            "crowd_intelligence": self.crowd_intelligence,
            "evacuation_progress": self.evacuation_progress,
            "component_timestamps": self.component_timestamps,
        }
        current.update(changes)

        return LiveBuildingSnapshot(**current)


# =====================================================


class StateManager:

    # Owns the single "latest LiveBuildingSnapshot" of record -- Phase
    # 5's own scope: maintain it, never compute it. Every update_*()
    # method here only ever *receives* an already-computed value
    # (from Live Perception, AI Inference, Decision Policy, or a
    # caller's own engineering-state integration) and folds it into a
    # freshly constructed snapshot; none of them call into any other
    # package themselves -- that composition lives in
    # live_system.integration/orchestrator, not here.

    def __init__(self, initial: Optional[LiveBuildingSnapshot] = None):

        self._snapshot = initial or LiveBuildingSnapshot()

    # =====================================================

    def current(self) -> LiveBuildingSnapshot:

        return self._snapshot

    # =====================================================

    def latest_building_state(self) -> Optional[BuildingState]:

        # The canonical-state accessor this milestone adds -- a thin
        # convenience over current().building_state, mirroring current()
        # itself rather than replacing it (current() remains the "full
        # LiveBuildingSnapshot, including whatever compatibility/AI/
        # decision-policy fields are also populated" accessor; this one
        # is for a caller that only ever wants the canonical BuildingState
        # and should not need to know LiveBuildingSnapshot's own shape).

        return self._snapshot.building_state

    # =====================================================

    def update_building_state(
        self, building_state: BuildingState, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            building_state=building_state,
            component_timestamps=self._stamp("building_state", time),
        )

    # =====================================================

    def latest_ai_prediction(self) -> Optional[LiveAIPredictionSnapshot]:

        # Mirrors latest_building_state()'s own convenience-accessor
        # role, one field over.

        return self._snapshot.ai_prediction_snapshot

    # =====================================================

    def update_ai_prediction(
        self, ai_prediction_snapshot: LiveAIPredictionSnapshot, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            ai_prediction_snapshot=ai_prediction_snapshot,
            component_timestamps=self._stamp("ai_prediction_snapshot", time),
        )

    # =====================================================

    def latest_advisory_report(self) -> Optional[AdvisoryReport]:

        # Mirrors latest_building_state()/latest_ai_prediction()'s own
        # convenience-accessor role, one field over.

        return self._snapshot.advisory_report

    # =====================================================

    def update_advisory_report(
        self, advisory_report: AdvisoryReport, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            advisory_report=advisory_report,
            component_timestamps=self._stamp("advisory_report", time),
        )

    # =====================================================

    def update_perception(
        self, building_observation: BuildingObservation, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            building_observation=building_observation,
            component_timestamps=self._stamp("perception", time),
        )

    # =====================================================

    def update_engineering_state(
        self, engineering_state: Mapping[str, Any], time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            engineering_state=engineering_state,
            component_timestamps=self._stamp("engineering_state", time),
        )

    # =====================================================

    def update_ai_predictions(
        self, ai_predictions: Mapping[str, Prediction], time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            ai_predictions=ai_predictions,
            component_timestamps=self._stamp("ai_predictions", time),
        )

    # =====================================================

    def update_decision_policy(
        self, decision_policy: Optional[DecisionPolicy], time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            decision_policy=decision_policy,
            component_timestamps=self._stamp("decision_policy", time),
        )

    # =====================================================

    def update_recommendations(
        self, recommendations: Tuple[Recommendation, ...], time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            recommendations=recommendations,
            component_timestamps=self._stamp("recommendations", time),
        )

    # =====================================================

    def latest_crowd_intelligence(self) -> Optional[CrowdIntelligenceSnapshot]:

        # Mirrors latest_building_state()/latest_ai_prediction()/
        # latest_advisory_report()'s own convenience-accessor role, one
        # field over.

        return self._snapshot.crowd_intelligence

    # =====================================================

    def update_crowd_intelligence(
        self, crowd_intelligence: CrowdIntelligenceSnapshot, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            crowd_intelligence=crowd_intelligence,
            component_timestamps=self._stamp("crowd_intelligence", time),
        )

    # =====================================================

    def latest_evacuation_progress(self) -> Optional[EvacuationProgressSnapshot]:

        # Mirrors latest_crowd_intelligence()'s own convenience-accessor
        # role, one field over.

        return self._snapshot.evacuation_progress

    # =====================================================

    def update_evacuation_progress(
        self, evacuation_progress: EvacuationProgressSnapshot, time: float,
    ) -> LiveBuildingSnapshot:

        return self._replace(
            timestamp=time,
            evacuation_progress=evacuation_progress,
            component_timestamps=self._stamp("evacuation_progress", time),
        )

    # =====================================================

    def _replace(self, **changes) -> LiveBuildingSnapshot:

        self._snapshot = self._snapshot.replace(**changes)

        return self._snapshot

    # =====================================================

    def _stamp(self, component: str, time: float) -> Mapping[str, float]:

        stamped = dict(self._snapshot.component_timestamps)
        stamped[component] = time

        return stamped

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
    # occupancy/hazard both live inside building_observation (Live
    # Perception's own already-fused, already-canonical shape) rather
    # than being restated as separate fields here -- there is exactly
    # one honest source for "what does Live Perception currently
    # believe," and duplicating it into parallel occupancy_state/
    # hazard_state fields would only invite the two to drift out of
    # sync. occupancy_for()/hazard_for()/edge_for() below are this
    # type's own total-accessor convenience over that one field,
    # mirroring BuildingObservation's own node_observation()/
    # occupancy_observation()/edge_observation() total accessors.
    #
    # engineering_state has no live source composed into this phase
    # yet (Phase 2 explicitly scopes CCTV/Smoke/Heat/FACP only, not a
    # door/exit/stair control integration) -- it defaults to empty
    # rather than a fabricated "all normal" reading, and is populated
    # only if/when a caller's own integration supplies one via
    # StateManager.update_engineering_state().

    timestamp: float = 0.0

    building_observation: Optional[BuildingObservation] = None
    engineering_state: Mapping[str, Any] = field(default_factory=dict)

    ai_predictions: Mapping[str, Prediction] = field(default_factory=dict)
    decision_policy: Optional[DecisionPolicy] = None
    recommendations: Tuple[Recommendation, ...] = field(default_factory=tuple)

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
            "engineering_state": self.engineering_state,
            "ai_predictions": self.ai_predictions,
            "decision_policy": self.decision_policy,
            "recommendations": self.recommendations,
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

    def _replace(self, **changes) -> LiveBuildingSnapshot:

        self._snapshot = self._snapshot.replace(**changes)

        return self._snapshot

    # =====================================================

    def _stamp(self, component: str, time: float) -> Mapping[str, float]:

        stamped = dict(self._snapshot.component_timestamps)
        stamped[component] = time

        return stamped

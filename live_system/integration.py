from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from perception.fusion.occupancy_estimation import EstimatedOccupancy, OccupancyEstimator
from perception.fusion.sensor_fusion import SensorFusion
from perception.models.building_observation import BuildingObservation
from perception.models.human_observation import HumanObservation
from perception.providers.human_observation_provider import HumanObservationProvider

from ai_inference.predictor import Prediction, Predictor
from ai_inference.recommendation import Recommendation

from decision_policy.policy import DecisionInputs, DecisionPolicy, generate_policy

from live_system.sensor_registry import SensorReadings
from live_system.state_manager import LiveBuildingSnapshot


# =====================================================
# Phase 7 -- Integration Layer. Wires live_perception (perception/) ->
# ai_inference -> decision_policy -> command_center *without modifying
# any of them*: every class below either (a) is a thin Protocol
# defining the one method live_system itself calls, or (b) is a real
# adapter that composes an already-existing package's own already-
# public API, translating only where the two sides' shapes genuinely
# differ. None of these adapters reimplement Sensor Fusion, AI
# Inference, or Decision Policy's own logic -- every actual decision
# is still made by that package's own code.
#
# Every gateway is optional and independently swappable
# (dependency-injected into live_system.orchestrator.LiveOrchestrator,
# never hardcoded) -- exactly the "interfaces and dependency injection
# only" rule this phase is scoped to. A test that wants a deterministic
# double simply implements the same Protocol; it never has to subclass
# a real adapter or monkeypatch a real package.
# =====================================================


class PerceptionGateway(Protocol):

    def collect(self, readings: SensorReadings, time: float) -> BuildingObservation: ...


class AIInferenceGateway(Protocol):

    def predict(self, snapshot: LiveBuildingSnapshot) -> Mapping[str, Prediction]: ...


class DecisionPolicyGateway(Protocol):

    def evaluate(self, snapshot: LiveBuildingSnapshot) -> Optional[DecisionPolicy]: ...


class CommandCenterGateway(Protocol):

    def notify(self, snapshot: LiveBuildingSnapshot) -> None: ...


# =====================================================
# Live Perception -- perception.fusion.sensor_fusion.SensorFusion is
# the real, only place a BuildingObservation is ever assembled (see
# that class's own docstring); this adapter's entire job is handing it
# this cycle's raw readings in the shape it already expects.
# occupancy_estimator is optional: Phase 2 scopes CCTV/Smoke/Heat/FACP
# as abstract interfaces only, and a deployment with no people-counting
# camera pipeline configured yet simply gets empty occupancy_estimates
# -- SensorFusion already treats that as "no occupancy observed
# anywhere," never a fabricated zero (see
# SensorFusion._build_occupancy_observations()).
#
# human_observation_provider is the Human Perception Integration seam
# (perception.providers.human_observation_provider.HumanObservationProvider)
# -- also optional, also additive: a deployment with no per-person
# tracking source configured yet (the only case this platform can
# actually exercise today, per that phase's own "no YOLO, no vision
# model" rule) simply gets an empty human_observations mapping, exactly
# the same "absent stage contributes nothing" pattern every other
# optional collaborator in this gateway already follows. Unlike the
# four raw-reading lists SensorReadings carries, a HumanObservation is
# not a per-device raw reading this gateway would fuse itself -- it is
# already a complete, per-person record by the time
# HumanObservationProvider.observations_at() returns it (see that
# interface's own docstring), so it is queried independently of
# SensorRegistry/SensorReadings and handed straight through to
# SensorFusion.fuse()'s own human_observations parameter.
# =====================================================


class SensorFusionPerceptionGateway:

    def __init__(
        self,
        sensor_fusion: SensorFusion,
        occupancy_estimator: Optional[OccupancyEstimator] = None,
        human_observation_provider: Optional[HumanObservationProvider] = None,
    ):

        self._sensor_fusion = sensor_fusion
        self._occupancy_estimator = occupancy_estimator
        self._human_observation_provider = human_observation_provider

    # =====================================================

    def collect(self, readings: SensorReadings, time: float) -> BuildingObservation:

        occupancy_estimates: Mapping[str, EstimatedOccupancy] = (
            self._occupancy_estimator.estimate(readings.camera_observations)
            if self._occupancy_estimator is not None else {}
        )

        human_observations: Mapping[str, HumanObservation] = (
            self._human_observation_provider.observations_at(time)
            if self._human_observation_provider is not None else {}
        )

        return self._sensor_fusion.fuse(
            timestamp=time,
            occupancy_estimates=occupancy_estimates,
            human_observations=human_observations,
            camera_observations=readings.camera_observations,
            smoke_detector_readings=readings.smoke_detector_readings,
            heat_detector_readings=readings.heat_detector_readings,
        )


# =====================================================
# AI Inference -- ai_inference.Predictor is the real, only place a
# trained model is ever invoked. feature_row_builder is the one
# translation this adapter cannot do itself: turning a
# LiveBuildingSnapshot into the Dict[str, Any] feature-row shape a
# specific trained model expects is a feature-engineering decision
# (which columns, which fill-ins for missing live data) that belongs
# to whoever trained and is deploying that model, not to this
# orchestration layer -- supplied here exactly the way
# decision_inputs_builder is supplied to GeneratePolicyDecisionPolicy
# Gateway below, for the same reason.
# =====================================================


FeatureRowBuilder = Callable[[LiveBuildingSnapshot], Dict[str, Any]]


class PredictorAIInferenceGateway:

    def __init__(self, predictor: Predictor, feature_row_builder: FeatureRowBuilder):

        self._predictor = predictor
        self._feature_row_builder = feature_row_builder

    # =====================================================

    def predict(self, snapshot: LiveBuildingSnapshot) -> Mapping[str, Prediction]:

        X_row = self._feature_row_builder(snapshot)

        return self._predictor.predict_all(X_row)


# =====================================================
# Decision Policy -- decision_policy.generate_policy is the real,
# only place a DecisionPolicy is ever produced. decision_inputs_builder
# is this adapter's one required translation seam, for the same reason
# feature_row_builder is required above: DecisionInputs structurally
# requires a Building, a Scenario, and a GroundTruth (§ decision_policy/
# policy.py) -- all three are offline-simulation-shaped artifacts a
# live incident does not intrinsically have. Producing them from live
# sensor data (e.g. synthesizing a GroundTruth-compatible view from
# BuildingObservation) is a deployment-specific decision this
# orchestration layer must not make unilaterally; a builder that
# returns None (rather than raising) is this adapter's documented way
# of saying "not enough information yet this cycle" -- evaluate() then
# returns None rather than calling generate_policy() on fabricated
# inputs.
# =====================================================


DecisionInputsBuilder = Callable[[LiveBuildingSnapshot], Optional[DecisionInputs]]


class GeneratePolicyDecisionPolicyGateway:

    def __init__(self, decision_inputs_builder: DecisionInputsBuilder):

        self._decision_inputs_builder = decision_inputs_builder

    # =====================================================

    def evaluate(self, snapshot: LiveBuildingSnapshot) -> Optional[DecisionPolicy]:

        inputs = self._decision_inputs_builder(snapshot)

        if inputs is None:
            return None

        return generate_policy(inputs)


# =====================================================
# Recommendations -- ai_inference.recommendation.build_recommendation
# is the real, only place a Recommendation is ever assembled from one
# AI prediction plus one Decision Policy. Like the two builders above,
# turning "the current snapshot" into the (loaded_model, X_row,
# location_type) triple that call needs is left to an injected
# callable rather than guessed at here.
# =====================================================


RecommendationBuilder = Callable[[LiveBuildingSnapshot], Tuple[Recommendation, ...]]


# =====================================================
# Command Center -- composes command_center.dashboard.Dashboard's own
# already-public seam exactly as MainWindow already uses it (see that
# class's own docstring: "MainWindow owns the playback clock... Dashboard
# owns nothing time-related itself beyond reacting to whatever frame
# index it is told to show"): set_incident() is called once, up front,
# to give the Dashboard a Building to render into; every live cycle
# after that calls show_frame() directly with a freshly built
# IncidentFrame, entirely bypassing IncidentData's dataset_builder
# timeline-row schema (a completed-simulation-replay concept this live
# system has no equivalent of). command_center itself is imported
# lazily, inside __init__, so importing live_system.integration never
# requires PyQt6 to be installed unless a caller actually constructs
# this one adapter.
#
# Only zone_occupancy is populated from live data below -- zone_smoke/
# zone_temperature/zone_visibility on command_center.incident_data.
# IncidentFrame expect the continuous physical values Ground Truth's
# HazardNodeState carries (smoke_level/temperature/visibility as
# floats); Live Perception's ObservedNodeState deliberately carries no
# such continuous reading (see that type's own docstring: "a real smoke
# detector does not measure and report a smoke level"), only a
# categorical visibility_estimate and a discrete alarm_active/
# estimated_severity. Forcing one shape into the other would fabricate
# precision no live sensor actually reports, so those three fields are
# left empty -- current_fire_zone_ids is populated instead, from zones
# Live Perception currently estimates HIGH/CRITICAL severity or active
# alarm, which is the honest live-sensor analogue of "currently
# hazardous," clearly distinct from (and never claimed to be) Ground
# Truth's own ignition-based fire zones.
# =====================================================


class DashboardCommandCenterGateway:

    def __init__(self, dashboard, building, scenario):

        from command_center.incident_data import IncidentData

        self._dashboard = dashboard

        self._dashboard.set_incident(
            IncidentData(building=building, scenario=scenario),
        )

        self._building = building

    # =====================================================

    def notify(self, snapshot: LiveBuildingSnapshot) -> None:

        self._dashboard.show_frame(self._build_frame(snapshot))

    # =====================================================

    def _build_frame(self, snapshot: LiveBuildingSnapshot):

        from perception.models.building_observation import ObservationState, PerceptionSeverity

        from command_center.incident_data import IncidentFrame

        observation = snapshot.building_observation

        zone_occupancy: Dict[str, Optional[float]] = {}
        hazardous_zone_ids = set()

        if observation is not None:

            for zone_id, occupancy in observation.occupancy_observations.items():
                zone_occupancy[zone_id] = occupancy.estimated_count

            for node_id, node_state in observation.node_observations.items():

                if node_state.observation_state != ObservationState.OBSERVED:
                    continue

                if node_state.alarm_active or node_state.estimated_severity in (
                    PerceptionSeverity.HIGH, PerceptionSeverity.CRITICAL,
                ):
                    hazardous_zone_ids.add(node_id)

        # door_states/exit_states/stair_states are left at IncidentFrame's
        # own empty default -- engineering_state's shape is deployment-
        # defined (see LiveBuildingSnapshot's own docstring), not
        # something this gateway can assume is door/exit/stair-keyed
        # without guessing at a schema nothing in this phase defines.
        #
        # human_observations, unlike zone_smoke/temperature/visibility
        # above, needs no translation at all -- BuildingObservation.
        # human_observations and IncidentFrame.human_observations are
        # both keyed by HumanObservation.person_id and hold the exact
        # same HumanObservation value, passed straight through.
        return IncidentFrame(
            time=snapshot.timestamp,
            current_fire_zone_ids=frozenset(hazardous_zone_ids),
            zone_occupancy=zone_occupancy,
            human_observations=(
                observation.human_observations if observation is not None else {}
            ),
        )

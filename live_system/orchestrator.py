from dataclasses import dataclass
from typing import Optional

from perception.models.building_observation import ObservationState

from building_state.models import BuildingState

from live_system.building_state_gateway import BuildingStateGateway
from live_system.event_bus import EventBus, EventType
from live_system.incident_manager import IncidentManager, IncidentState
from live_system.integration import (
    AIInferenceGateway,
    CommandCenterGateway,
    DecisionPolicyGateway,
    PerceptionGateway,
    RecommendationBuilder,
)
from live_system.sensor_registry import SensorRegistry
from live_system.state_manager import LiveBuildingSnapshot, StateManager
from live_system.update_loop import UpdateLoop


class LiveSystemAlreadyRunningError(Exception):

    pass


class LiveSystemNotRunningError(Exception):

    pass


@dataclass(frozen=True)
class OperatorAcknowledgement:

    operator_id: str
    time: float
    note: Optional[str] = None


class LiveOrchestrator:

    # Phase 1's central runtime controller -- the one object a real
    # deployment (or a test) constructs and owns. Coordinates Live
    # Perception, canonical BuildingState assembly (Canonical Live
    # BuildingState Runtime Assembly milestone), AI Inference, Decision
    # Policy, and Command Center (this class's own required scope) by
    # calling each one's live_system.integration/live_system.
    # building_state_gateway Gateway, never the underlying package
    # directly -- swapping a gateway (or leaving it None, meaning
    # "this stage isn't wired up yet in this deployment/test") is the
    # only way this class's behavior changes; it has no package-
    # specific logic of its own beyond sequencing and event
    # publication. It never constructs or owns a CameraManager,
    # SensorManager, MultiCameraFusionEngine, or SimulatedFACP itself --
    # a building_state_gateway's own providers are where a real
    # deployment composes those (see live_system.building_state_gateway's
    # own module docstring).
    #
    # sensor_registry/event_bus/state_manager/incident_manager and
    # every gateway are constructor-injected, defaulting to a fresh
    # instance of the concrete class only where that default is itself
    # a Phase 1-5 primitive with no external dependency (SensorRegistry,
    # EventBus, StateManager, IncidentManager) -- the four Gateways
    # default to None (that stage simply does not run this cycle),
    # never to a real adapter this class would otherwise have to import
    # unconditionally (e.g. a PyQt6 Dashboard).
    #
    # "Owns the system lifecycle" (Phase 1) means exactly what start()/
    # stop()/is_running below implement: run_cycle() (and therefore the
    # UpdateLoop that drives it) refuses to run while stopped, so a
    # caller can never advance a system that was never told to start,
    # or that has already been told to stop -- the same "must be
    # explicitly started" contract SimulationRuntime/InteractiveSimulation
    # apply to occupant movement, here applied to the live update cycle
    # itself.

    def __init__(
        self,
        sensor_registry: Optional[SensorRegistry] = None,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        incident_manager: Optional[IncidentManager] = None,
        perception_gateway: Optional[PerceptionGateway] = None,
        building_state_gateway: Optional[BuildingStateGateway] = None,
        ai_inference_gateway: Optional[AIInferenceGateway] = None,
        decision_policy_gateway: Optional[DecisionPolicyGateway] = None,
        command_center_gateway: Optional[CommandCenterGateway] = None,
        recommendation_builder: Optional[RecommendationBuilder] = None,
        interval_seconds: float = 1.0,
    ):

        self.sensor_registry = sensor_registry if sensor_registry is not None else SensorRegistry()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.state_manager = state_manager if state_manager is not None else StateManager()
        self.incident_manager = incident_manager if incident_manager is not None else IncidentManager()

        self.perception_gateway = perception_gateway
        self.building_state_gateway = building_state_gateway
        self.ai_inference_gateway = ai_inference_gateway
        self.decision_policy_gateway = decision_policy_gateway
        self.command_center_gateway = command_center_gateway
        self.recommendation_builder = recommendation_builder

        # Wired to this orchestrator's own run_cycle() -- the Update
        # Loop never has cycle logic of its own (Phase 4's scope is
        # purely "drive it repeatedly, at a configurable frequency").
        self.update_loop = UpdateLoop(self.run_cycle, interval_seconds=interval_seconds)

        self._is_running = False

    # =====================================================

    @property
    def is_running(self) -> bool:

        return self._is_running

    # =====================================================

    @property
    def latest_building_state(self) -> Optional[BuildingState]:

        # The canonical-state convenience accessor Phase 2 asks this
        # class to expose -- a thin forward onto state_manager's own
        # latest_building_state(), so a caller does not need to know
        # StateManager/LiveBuildingSnapshot's own shape just to read
        # "what is BuildingState right now" (mirrors is_running's own
        # forwarding-property style, one level up).

        return self.state_manager.latest_building_state()

    # =====================================================

    def start(self) -> None:

        if self._is_running:

            raise LiveSystemAlreadyRunningError(
                "LiveOrchestrator.start() called while already running -- call stop() first."
            )

        self._is_running = True

    # =====================================================

    def stop(self) -> None:

        self._is_running = False

    # =====================================================

    def run_cycle(self, time: float) -> LiveBuildingSnapshot:

        # Phase 4's exact cycle, extended (Canonical Live BuildingState
        # Runtime Assembly milestone) by exactly one new optional stage,
        # inserted after Live Perception and before AI Inference: read
        # sensors -> update snapshot -> [NEW] assemble canonical
        # BuildingState -> AI inference -> decision policy -> publish
        # recommendations -> notify command center. Each stage after
        # sensor reading only runs if its gateway is configured --
        # every stage's absence is a valid, working configuration (e.g.
        # a deployment with Live Perception wired up but no trained AI
        # model yet, or a BuildingState assembled with no cameras/
        # sensors/FACP/control provider configured yet), never an error.
        #
        # The new building_state_gateway stage is deliberately
        # independent of perception_gateway -- either, both, or neither
        # may be configured in a given deployment/test; BuildingState.
        # building_alarm_status is NOT (yet) wired into
        # _maybe_activate_alarm() below, which remains keyed on Live
        # Perception's own observation exactly as before this milestone
        # (BuildingState -> AI/Advisory/Command Center wiring, and any
        # change to incident-activation triggering, is explicitly a
        # later milestone's scope, not this one's).

        if not self._is_running:

            raise LiveSystemNotRunningError(
                "LiveOrchestrator.run_cycle() called before start() (or after stop())."
            )

        readings = self.sensor_registry.read_all(time)
        self.event_bus.emit(EventType.SENSOR_UPDATE, readings, time)

        snapshot = self.state_manager.current()

        if self.perception_gateway is not None:

            observation = self.perception_gateway.collect(readings, time)
            snapshot = self.state_manager.update_perception(observation, time)

            self.event_bus.emit(EventType.OCCUPANCY_UPDATED, observation, time)
            self.event_bus.emit(EventType.HAZARD_UPDATED, observation, time)

            self._maybe_activate_alarm(observation, time)

        if self.building_state_gateway is not None:

            building_state = self.building_state_gateway.collect(time)
            snapshot = self.state_manager.update_building_state(building_state, time)

            self.event_bus.emit(EventType.BUILDING_STATE_UPDATED, building_state, time)

        if self.ai_inference_gateway is not None:

            predictions = self.ai_inference_gateway.predict(snapshot)
            snapshot = self.state_manager.update_ai_predictions(predictions, time)

        if self.decision_policy_gateway is not None:

            policy = self.decision_policy_gateway.evaluate(snapshot)
            snapshot = self.state_manager.update_decision_policy(policy, time)

        if self.recommendation_builder is not None:

            recommendations = self.recommendation_builder(snapshot)
            snapshot = self.state_manager.update_recommendations(recommendations, time)

            self.event_bus.emit(EventType.RECOMMENDATION_UPDATED, recommendations, time)

        if self.command_center_gateway is not None:

            self.command_center_gateway.notify(snapshot)

        return snapshot

    # =====================================================

    def acknowledge(
        self, operator_id: str, time: float, note: Optional[str] = None,
    ) -> OperatorAcknowledgement:

        # An operator's own action on the live system (e.g. "I have
        # seen this alarm") -- recorded purely as a published event;
        # this class makes no judgment about what an acknowledgement
        # should cause to happen next (that policy belongs to whatever
        # subscribes to OPERATOR_ACKNOWLEDGEMENT, not to the
        # orchestrator itself).

        acknowledgement = OperatorAcknowledgement(operator_id=operator_id, time=time, note=note)
        self.event_bus.emit(EventType.OPERATOR_ACKNOWLEDGEMENT, acknowledgement, time)

        return acknowledgement

    # =====================================================

    def transition_incident(
        self, new_state: IncidentState, time: float, reason: Optional[str] = None,
    ):

        # The explicit, operator/procedure-driven counterpart to
        # _maybe_activate_alarm()'s automatic ALARM transition below --
        # every other lifecycle change (VERIFICATION, EVACUATION,
        # SUPPRESSION, RECOVERY, CLOSED, or a manual IDLE reset) is
        # driven by human/procedural judgment this class has no basis
        # to infer from sensor data alone, so it is never triggered
        # automatically the way ALARM is.

        transition = self.incident_manager.transition_to(new_state, time, reason=reason)

        if new_state == IncidentState.ALARM:
            self.event_bus.emit(EventType.ALARM_ACTIVATED, transition, time)

        return transition

    # =====================================================

    def _maybe_activate_alarm(self, observation, time: float) -> None:

        # The one automatic incident transition this class makes on
        # its own: an IDLE incident whose Live Perception this cycle
        # reports any OBSERVED zone in alarm moves to ALARM, mirroring
        # a real Fire Alarm Control Panel's own "first sensor trip
        # starts the incident" behavior. Every subsequent transition
        # (verification, evacuation, ...) is deliberately left to
        # transition_incident() above -- this class does not attempt to
        # infer procedure from sensor data beyond that first trip.

        if self.incident_manager.state != IncidentState.IDLE:
            return

        any_alarm = any(
            node_state.observation_state == ObservationState.OBSERVED and node_state.alarm_active
            for node_state in observation.node_observations.values()
        )

        if not any_alarm:
            return

        transition = self.incident_manager.transition_to(
            IncidentState.ALARM, time, reason="Live Perception reported an active alarm",
        )
        self.event_bus.emit(EventType.ALARM_ACTIVATED, transition, time)

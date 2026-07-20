import sys
import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario.metadata import ScenarioMetadata
from scenario.scenario import Scenario

from ground_truth.labels import GroundTruth

from decision_policy.policy import DecisionInputs

from perception.models.building_observation import (
    ObservationState,
    ObservedNodeState,
    ObservedOccupancy,
    PerceptionSeverity,
)
from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.fusion.occupancy_estimation import OccupancyEstimator
from perception.fusion.sensor_fusion import SensorFusion

from ai_inference.loader import LoadedModel, ModelProvenance
from ai_inference.predictor import Predictor

from live_system.event_bus import Event, EventBus, EventType
from live_system.incident_manager import (
    IncidentManager,
    IncidentState,
    InvalidIncidentTransition,
)
from live_system.integration import (
    DashboardCommandCenterGateway,
    GeneratePolicyDecisionPolicyGateway,
    PredictorAIInferenceGateway,
    SensorFusionPerceptionGateway,
)
from live_system.orchestrator import (
    LiveOrchestrator,
    LiveSystemAlreadyRunningError,
    LiveSystemNotRunningError,
)
from live_system.sensor_registry import (
    CCTVSensor,
    FACPReading,
    FireAlarmControlPanelSensor,
    HeatDetectorSensor,
    SensorKind,
    SensorRegistry,
    SmokeDetectorSensor,
    UnknownSensorKindError,
)
from live_system.state_manager import LiveBuildingSnapshot, StateManager
from live_system.update_loop import UpdateLoop


# =====================================================
# Shared fixtures
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-a", name="Zone A", x=0.0, y=0.0, width=4.0, height=4.0, floor_id="floor-1"),
            Zone(id="zone-b", name="Zone B", x=5.0, y=0.0, width=4.0, height=4.0, floor_id="floor-1"),
        ],
        exits=[Exit(id="exit-1", floor_id="floor-1", zone_id="zone-b")],
    )

    return Building(name="Live Test Building", id="building-1", floors=[floor])


def make_scenario(building):

    return Scenario(
        metadata=ScenarioMetadata(
            scenario_id="live-scn-1", definition_id="live-def-1",
            definition_content_hash="hash", generation_version="v1", seed=1,
            created_at="2026-07-15T00:00:00",
        ),
        occupants=(),
    )


class _FixedSensor:

    # A deterministic stand-in sensor: always returns the same reading
    # (or always None) regardless of `time` -- used across every test
    # below in place of a real CCTV/Smoke/Heat/FACP feed, matching
    # this phase's own "deterministic when driven by mocked sensor
    # inputs" requirement.

    def __init__(self, sensor_id, reading):
        self.sensor_id = sensor_id
        self._reading = reading

    def read(self, time):
        return self._reading


class _FixedCCTVSensor(_FixedSensor, CCTVSensor):
    # _FixedSensor first in MRO so its read() (not CCTVSensor's own
    # NotImplementedError stub) is what actually runs.
    def __init__(self, sensor_id, reading=None):
        _FixedSensor.__init__(self, sensor_id, reading)


class _FixedSmokeDetectorSensor(_FixedSensor, SmokeDetectorSensor):
    def __init__(self, sensor_id, reading=None):
        _FixedSensor.__init__(self, sensor_id, reading)


class _FixedHeatDetectorSensor(_FixedSensor, HeatDetectorSensor):
    def __init__(self, sensor_id, reading=None):
        _FixedSensor.__init__(self, sensor_id, reading)


class _FixedFACPSensor(_FixedSensor, FireAlarmControlPanelSensor):
    def __init__(self, sensor_id, reading=None):
        _FixedSensor.__init__(self, sensor_id, reading)


# =====================================================
# Phase 3 -- Event Bus
# =====================================================


class EventBusTests(unittest.TestCase):

    def test_subscribed_handler_receives_published_event(self):

        bus = EventBus()
        received = []
        bus.subscribe(EventType.SENSOR_UPDATE, received.append)

        bus.emit(EventType.SENSOR_UPDATE, {"n": 1}, time=5.0)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_type, EventType.SENSOR_UPDATE)
        self.assertEqual(received[0].payload, {"n": 1})
        self.assertEqual(received[0].timestamp, 5.0)

    def test_handler_for_a_different_event_type_is_not_called(self):

        bus = EventBus()
        received = []
        bus.subscribe(EventType.ALARM_ACTIVATED, received.append)

        bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)

        self.assertEqual(received, [])

    def test_multiple_handlers_are_called_in_subscription_order(self):

        bus = EventBus()
        order = []
        bus.subscribe(EventType.HAZARD_UPDATED, lambda e: order.append("first"))
        bus.subscribe(EventType.HAZARD_UPDATED, lambda e: order.append("second"))

        bus.emit(EventType.HAZARD_UPDATED, None, time=0.0)

        self.assertEqual(order, ["first", "second"])

    def test_wildcard_subscriber_receives_every_event_type(self):

        bus = EventBus()
        received = []
        bus.subscribe_all(received.append)

        bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)
        bus.emit(EventType.ALARM_ACTIVATED, None, time=1.0)

        self.assertEqual(len(received), 2)

    def test_unsubscribe_stops_further_delivery(self):

        bus = EventBus()
        received = []
        handler = received.append
        bus.subscribe(EventType.OCCUPANCY_UPDATED, handler)
        bus.unsubscribe(EventType.OCCUPANCY_UPDATED, handler)

        bus.emit(EventType.OCCUPANCY_UPDATED, None, time=0.0)

        self.assertEqual(received, [])

    def test_history_records_every_published_event_in_order(self):

        bus = EventBus()
        bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)
        bus.emit(EventType.ALARM_ACTIVATED, None, time=1.0)

        self.assertEqual(
            [event.event_type for event in bus.history],
            [EventType.SENSOR_UPDATE, EventType.ALARM_ACTIVATED],
        )

    def test_history_of_filters_to_one_event_type(self):

        bus = EventBus()
        bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)
        bus.emit(EventType.ALARM_ACTIVATED, None, time=1.0)
        bus.emit(EventType.SENSOR_UPDATE, None, time=2.0)

        self.assertEqual(len(bus.history_of(EventType.SENSOR_UPDATE)), 2)

    def test_event_ids_are_unique(self):

        bus = EventBus()
        first = bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)
        second = bus.emit(EventType.SENSOR_UPDATE, None, time=0.0)

        self.assertNotEqual(first.event_id, second.event_id)


# =====================================================
# Phase 2 -- Sensor Registry
# =====================================================


class SensorRegistryTests(unittest.TestCase):

    def test_registering_a_sensor_makes_it_discoverable_by_kind(self):

        registry = SensorRegistry()
        sensor = _FixedCCTVSensor("cam-1")
        registry.register(sensor)

        self.assertTrue(registry.is_registered("cam-1"))
        self.assertEqual(registry.sensors_of_kind(SensorKind.CCTV), [sensor])
        self.assertEqual(len(registry), 1)

    def test_registering_a_duplicate_sensor_id_raises(self):

        registry = SensorRegistry()
        registry.register(_FixedCCTVSensor("cam-1"))

        with self.assertRaises(ValueError):
            registry.register(_FixedCCTVSensor("cam-1"))

    def test_registering_an_unrecognized_sensor_type_raises(self):

        class NotASensor:
            sensor_id = "mystery-1"

        registry = SensorRegistry()

        with self.assertRaises(UnknownSensorKindError):
            registry.register(NotASensor())

    def test_unregister_removes_the_sensor(self):

        registry = SensorRegistry()
        registry.register(_FixedCCTVSensor("cam-1"))
        registry.unregister("cam-1")

        self.assertFalse(registry.is_registered("cam-1"))
        self.assertEqual(len(registry), 0)

    def test_unregister_of_an_unknown_id_is_a_no_op(self):

        registry = SensorRegistry()
        registry.unregister("does-not-exist")  # must not raise

    def test_all_four_sensor_kinds_are_grouped_independently(self):

        registry = SensorRegistry()
        registry.register(_FixedCCTVSensor("cam-1"))
        registry.register(_FixedSmokeDetectorSensor("smoke-1"))
        registry.register(_FixedHeatDetectorSensor("heat-1"))
        registry.register(_FixedFACPSensor("facp-1"))

        self.assertEqual(len(registry.sensors_of_kind(SensorKind.CCTV)), 1)
        self.assertEqual(len(registry.sensors_of_kind(SensorKind.SMOKE_DETECTOR)), 1)
        self.assertEqual(len(registry.sensors_of_kind(SensorKind.HEAT_DETECTOR)), 1)
        self.assertEqual(len(registry.sensors_of_kind(SensorKind.FIRE_ALARM_CONTROL_PANEL)), 1)
        self.assertEqual(len(registry.all_sensors()), 4)

    def test_read_all_collects_only_sensors_that_reported_this_cycle(self):

        registry = SensorRegistry()
        registry.register(_FixedCCTVSensor(
            "cam-1", CameraFrameObservation(camera_id="cam-1", timestamp=5.0, estimated_occupant_count=3.0),
        ))
        registry.register(_FixedCCTVSensor("cam-2", None))  # offline this cycle
        registry.register(_FixedSmokeDetectorSensor(
            "smoke-1", SmokeDetectorReading(detector_id="smoke-1", timestamp=5.0, alarm_active=True),
        ))

        readings = registry.read_all(5.0)

        self.assertEqual(readings.timestamp, 5.0)
        self.assertEqual(len(readings.camera_observations), 1)
        self.assertEqual(readings.camera_observations[0].camera_id, "cam-1")
        self.assertEqual(len(readings.smoke_detector_readings), 1)
        self.assertEqual(readings.heat_detector_readings, [])
        self.assertEqual(readings.facp_readings, [])

    def test_read_all_with_no_registered_sensors_returns_empty_readings(self):

        registry = SensorRegistry()
        readings = registry.read_all(0.0)

        self.assertEqual(readings.camera_observations, [])
        self.assertEqual(readings.smoke_detector_readings, [])
        self.assertEqual(readings.heat_detector_readings, [])
        self.assertEqual(readings.facp_readings, [])


# =====================================================
# Phase 5 -- State Manager
# =====================================================


class StateManagerTests(unittest.TestCase):

    def test_starts_with_an_empty_default_snapshot(self):

        manager = StateManager()
        snapshot = manager.current()

        self.assertIsNone(snapshot.building_observation)
        self.assertEqual(snapshot.ai_predictions, {})
        self.assertIsNone(snapshot.decision_policy)
        self.assertEqual(snapshot.recommendations, ())

    def test_update_perception_replaces_only_that_field(self):

        manager = StateManager()

        observation_1 = _make_observation(zone_a_count=2.0)
        first = manager.update_perception(observation_1, time=1.0)

        observation_2 = _make_observation(zone_a_count=5.0)
        second = manager.update_perception(observation_2, time=2.0)

        self.assertIs(manager.current(), second)
        self.assertEqual(second.occupancy_for("zone-a").estimated_count, 5.0)
        self.assertEqual(second.timestamp, 2.0)
        self.assertEqual(second.component_timestamps["perception"], 2.0)

        # The snapshot from the first update was never mutated in place.
        self.assertEqual(observation_1.occupancy_observation("zone-a").estimated_count, 2.0)

    def test_updates_to_different_components_do_not_clobber_each_other(self):

        manager = StateManager()

        manager.update_perception(_make_observation(zone_a_count=1.0), time=1.0)
        manager.update_ai_predictions({"evacuation_time": "irrelevant-for-this-test"}, time=2.0)

        snapshot = manager.current()

        self.assertIsNotNone(snapshot.building_observation)
        self.assertIn("evacuation_time", snapshot.ai_predictions)
        self.assertEqual(snapshot.component_timestamps["perception"], 1.0)
        self.assertEqual(snapshot.component_timestamps["ai_predictions"], 2.0)

    def test_update_decision_policy_and_recommendations(self):

        manager = StateManager()

        policy_sentinel = object()
        manager.update_decision_policy(policy_sentinel, time=1.0)
        manager.update_recommendations((object(), object()), time=2.0)

        snapshot = manager.current()

        self.assertIs(snapshot.decision_policy, policy_sentinel)
        self.assertEqual(len(snapshot.recommendations), 2)

    def test_occupancy_and_hazard_accessors_default_when_unobserved(self):

        manager = StateManager()
        snapshot = manager.current()

        self.assertEqual(snapshot.occupancy_for("zone-a"), ObservedOccupancy())
        self.assertEqual(snapshot.hazard_for("zone-a"), ObservedNodeState())

    def test_snapshot_is_frozen(self):

        from dataclasses import FrozenInstanceError

        snapshot = LiveBuildingSnapshot()

        with self.assertRaises(FrozenInstanceError):
            snapshot.timestamp = 99.0


def _make_observation(zone_a_count=None, zone_a_alarm=None):

    from perception.models.building_observation import BuildingObservation

    node_observations = {}
    if zone_a_alarm is not None:
        node_observations["zone-a"] = ObservedNodeState(
            observation_state=ObservationState.OBSERVED, alarm_active=zone_a_alarm,
        )

    occupancy_observations = {}
    if zone_a_count is not None:
        occupancy_observations["zone-a"] = ObservedOccupancy(estimated_count=zone_a_count)

    return BuildingObservation(
        node_observations=node_observations, occupancy_observations=occupancy_observations,
    )


# =====================================================
# Phase 6 -- Incident Manager
# =====================================================


class IncidentManagerTests(unittest.TestCase):

    def test_starts_idle(self):

        manager = IncidentManager()

        self.assertEqual(manager.state, IncidentState.IDLE)
        self.assertFalse(manager.is_active)
        self.assertEqual(manager.history, [])

    def test_legal_transition_updates_state_and_history(self):

        manager = IncidentManager()
        transition = manager.transition_to(IncidentState.ALARM, time=1.0, reason="detector tripped")

        self.assertEqual(manager.state, IncidentState.ALARM)
        self.assertTrue(manager.is_active)
        self.assertEqual(manager.history, [transition])
        self.assertEqual(transition.from_state, IncidentState.IDLE)
        self.assertEqual(transition.to_state, IncidentState.ALARM)
        self.assertEqual(transition.reason, "detector tripped")

    def test_illegal_transition_raises_and_leaves_state_unchanged(self):

        manager = IncidentManager()

        with self.assertRaises(InvalidIncidentTransition):
            manager.transition_to(IncidentState.SUPPRESSION, time=1.0)

        self.assertEqual(manager.state, IncidentState.IDLE)
        self.assertEqual(manager.history, [])

    def test_can_transition_to_reports_legality_without_raising(self):

        manager = IncidentManager()

        self.assertTrue(manager.can_transition_to(IncidentState.ALARM))
        self.assertFalse(manager.can_transition_to(IncidentState.EVACUATION))

    def test_full_lifecycle_idle_to_closed(self):

        manager = IncidentManager()
        path = [
            IncidentState.ALARM, IncidentState.VERIFICATION, IncidentState.EVACUATION,
            IncidentState.SUPPRESSION, IncidentState.RECOVERY, IncidentState.CLOSED,
        ]

        for index, state in enumerate(path):
            manager.transition_to(state, time=float(index))

        self.assertEqual(manager.state, IncidentState.CLOSED)
        self.assertFalse(manager.is_active)
        self.assertEqual(len(manager.history), len(path))

    def test_false_alarm_returns_to_idle_from_verification(self):

        manager = IncidentManager()
        manager.transition_to(IncidentState.ALARM, time=0.0)
        manager.transition_to(IncidentState.VERIFICATION, time=1.0)
        manager.transition_to(IncidentState.IDLE, time=2.0, reason="confirmed false alarm")

        self.assertEqual(manager.state, IncidentState.IDLE)

    def test_reignition_during_recovery_returns_to_alarm(self):

        manager = IncidentManager()
        for state in (
            IncidentState.ALARM, IncidentState.EVACUATION,
            IncidentState.SUPPRESSION, IncidentState.RECOVERY,
        ):
            manager.transition_to(state, time=0.0)

        manager.transition_to(IncidentState.ALARM, time=1.0, reason="re-ignition")

        self.assertEqual(manager.state, IncidentState.ALARM)

    def test_closed_is_terminal(self):

        manager = IncidentManager()
        for state in (
            IncidentState.ALARM, IncidentState.EVACUATION,
            IncidentState.SUPPRESSION, IncidentState.RECOVERY, IncidentState.CLOSED,
        ):
            manager.transition_to(state, time=0.0)

        with self.assertRaises(InvalidIncidentTransition):
            manager.transition_to(IncidentState.IDLE, time=1.0)

    def test_incident_id_defaults_to_a_generated_value_but_is_settable(self):

        auto_id = IncidentManager()
        explicit_id = IncidentManager(incident_id="incident-42")

        self.assertTrue(auto_id.incident_id)
        self.assertEqual(explicit_id.incident_id, "incident-42")


# =====================================================
# Phase 4 -- Update Loop
# =====================================================


class UpdateLoopTests(unittest.TestCase):

    def test_tick_calls_cycle_fn_with_the_given_time_and_returns_its_result(self):

        calls = []

        def cycle_fn(time):
            calls.append(time)
            return f"snapshot-at-{time}"

        loop = UpdateLoop(cycle_fn, interval_seconds=2.0)
        result = loop.tick(5.0)

        self.assertEqual(result, "snapshot-at-5.0")
        self.assertEqual(calls, [5.0])
        self.assertEqual(loop.cycle_count, 1)
        self.assertEqual(loop.last_run_time, 5.0)

    def test_run_for_advances_time_by_interval_seconds_each_cycle(self):

        seen_times = []
        loop = UpdateLoop(lambda time: seen_times.append(time), interval_seconds=0.5)

        loop.run_for(3, start_time=0.0)

        self.assertEqual(seen_times, [0.0, 0.5, 1.0])
        self.assertEqual(loop.cycle_count, 3)

    def test_run_for_continues_from_next_time_when_start_time_is_omitted(self):

        seen_times = []
        loop = UpdateLoop(lambda time: seen_times.append(time), interval_seconds=1.0)

        loop.run_for(2, start_time=10.0)
        loop.run_for(2)

        self.assertEqual(seen_times, [10.0, 11.0, 12.0, 13.0])

    def test_zero_iterations_runs_nothing(self):

        loop = UpdateLoop(lambda time: time, interval_seconds=1.0)
        result = loop.run_for(0, start_time=0.0)

        self.assertEqual(result, [])
        self.assertEqual(loop.cycle_count, 0)

    def test_non_positive_interval_is_rejected_at_construction(self):

        with self.assertRaises(ValueError):
            UpdateLoop(lambda time: time, interval_seconds=0.0)

    def test_interval_seconds_is_configurable_after_construction(self):

        seen_times = []
        loop = UpdateLoop(lambda time: seen_times.append(time), interval_seconds=1.0)
        loop.interval_seconds = 10.0

        loop.run_for(2, start_time=0.0)

        self.assertEqual(seen_times, [0.0, 10.0])

    def test_run_realtime_stops_when_stop_condition_becomes_true(self):

        calls = {"n": 0}

        def stop_condition():
            return calls["n"] >= 3

        loop = UpdateLoop(lambda time: None, interval_seconds=1.0)

        def sleep_fn(seconds):
            calls["n"] += 1

        loop.run_realtime(sleep_fn=sleep_fn, time_fn=lambda: 0.0, stop_condition=stop_condition)

        self.assertEqual(loop.cycle_count, 3)
        self.assertEqual(calls["n"], 3)


# =====================================================
# Phase 7 -- Integration Layer (real composition, no mocks)
# =====================================================


class SensorFusionPerceptionGatewayTests(unittest.TestCase):

    def test_collects_real_readings_into_a_real_building_observation(self):

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={"smoke-1": "zone-a"},
            heat_detector_zone_assignments={},
            camera_zone_assignments={"cam-1": "zone-a"},
        )
        occupancy_estimator = OccupancyEstimator(camera_zone_assignments={"cam-1": "zone-a"})

        gateway = SensorFusionPerceptionGateway(sensor_fusion, occupancy_estimator)

        registry = SensorRegistry()
        registry.register(_FixedCCTVSensor(
            "cam-1", CameraFrameObservation(camera_id="cam-1", timestamp=5.0, estimated_occupant_count=4.0),
        ))
        registry.register(_FixedSmokeDetectorSensor(
            "smoke-1", SmokeDetectorReading(detector_id="smoke-1", timestamp=5.0, alarm_active=True),
        ))

        readings = registry.read_all(5.0)
        observation = gateway.collect(readings, time=5.0)

        self.assertEqual(observation.occupancy_observation("zone-a").estimated_count, 4.0)
        self.assertTrue(observation.node_observation("zone-a").alarm_active)
        self.assertEqual(observation.node_observation("zone-a").estimated_severity, PerceptionSeverity.HIGH)

    def test_no_occupancy_estimator_configured_yields_no_occupancy_observations(self):

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
            camera_zone_assignments={},
        )
        gateway = SensorFusionPerceptionGateway(sensor_fusion, occupancy_estimator=None)

        registry = SensorRegistry()
        observation = gateway.collect(registry.read_all(0.0), time=0.0)

        self.assertEqual(dict(observation.occupancy_observations), {})

    def test_no_human_observation_provider_configured_yields_no_human_observations(self):

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
            camera_zone_assignments={},
        )
        gateway = SensorFusionPerceptionGateway(sensor_fusion, human_observation_provider=None)

        observation = gateway.collect(SensorRegistry().read_all(0.0), time=0.0)

        self.assertEqual(dict(observation.human_observations), {})

    def test_human_observation_provider_is_queried_and_threaded_through(self):

        from perception.models.human_observation import HumanObservation, HumanState

        class _StubHumanObservationProvider:

            def __init__(self):
                self.calls = []

            def observations_at(self, time):
                self.calls.append(time)
                return {"p1": HumanObservation(person_id="p1", zone_id="zone-a", state=HumanState.WALKING)}

        sensor_fusion = SensorFusion(
            smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
            camera_zone_assignments={},
        )
        provider = _StubHumanObservationProvider()
        gateway = SensorFusionPerceptionGateway(sensor_fusion, human_observation_provider=provider)

        observation = gateway.collect(SensorRegistry().read_all(7.0), time=7.0)

        self.assertEqual(provider.calls, [7.0])
        self.assertEqual(observation.human_observation("p1").zone_id, "zone-a")
        self.assertEqual(observation.human_observation("p1").state, HumanState.WALKING)


class _FakeRegressionModel:

    # A minimal, in-memory stand-in for an ai_training.models.base.
    # BaseModel subclass -- avoids needing a real joblib/manifest.json
    # artifact on disk (ai_inference.loader.load_model()'s own job,
    # deliberately not exercised here) just to prove Predictor.predict_all()
    # actually gets invoked with this gateway's feature row.

    is_classifier = False

    def predict(self, X_rows):
        return [len(X_rows[0]) for _ in X_rows]  # deterministic, input-derived


class PredictorAIInferenceGatewayTests(unittest.TestCase):

    def test_predict_invokes_the_real_predictor_with_the_built_feature_row(self):

        loaded_model = LoadedModel(
            model=_FakeRegressionModel(),
            provenance=ModelProvenance(
                experiment_name="exp-1", model_name="evacuation_time", algorithm="dummy",
                model_version="v1", generated_at="2026-07-15T00:00:00",
                train_size=1, test_size=1, metrics={},
            ),
            directory="in-memory",
        )
        predictor = Predictor([loaded_model])

        built_rows = []

        def feature_row_builder(snapshot):
            row = {"zone_a_occupancy": snapshot.occupancy_for("zone-a").estimated_count or 0.0}
            built_rows.append(row)
            return row

        gateway = PredictorAIInferenceGateway(predictor, feature_row_builder)

        snapshot = StateManager().update_perception(_make_observation(zone_a_count=3.0), time=0.0)
        predictions = gateway.predict(snapshot)

        self.assertEqual(built_rows, [{"zone_a_occupancy": 3.0}])
        self.assertIn("evacuation_time", predictions)
        self.assertEqual(predictions["evacuation_time"].value, 1)  # len({"zone_a_occupancy": ...}) == 1


class GeneratePolicyDecisionPolicyGatewayTests(unittest.TestCase):

    def test_evaluate_invokes_the_real_generate_policy(self):

        building = make_building()
        scenario = make_scenario(building)
        ground_truth = GroundTruth(scenario_id="live-incident-1", definition_id="live")

        def decision_inputs_builder(snapshot):
            return DecisionInputs(building=building, scenario=scenario, ground_truth=ground_truth)

        gateway = GeneratePolicyDecisionPolicyGateway(decision_inputs_builder)

        policy = gateway.evaluate(StateManager().current())

        self.assertIsNotNone(policy)
        self.assertEqual(policy.scenario_id, "live-incident-1")

    def test_evaluate_returns_none_when_the_builder_has_insufficient_information(self):

        gateway = GeneratePolicyDecisionPolicyGateway(lambda snapshot: None)

        self.assertIsNone(gateway.evaluate(StateManager().current()))


class DashboardCommandCenterGatewayTests(unittest.TestCase):

    # Exercises the real command_center.dashboard.Dashboard (a PyQt6
    # QWidget) exactly the way tests/test_command_center.py already
    # does -- one shared QApplication instance, per that file's own
    # established convention.

    @classmethod
    def setUpClass(cls):

        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_notify_pushes_a_live_frame_into_the_real_dashboard(self):

        from command_center.dashboard import Dashboard

        building = make_building()
        scenario = make_scenario(building)
        dashboard = Dashboard()

        gateway = DashboardCommandCenterGateway(dashboard, building, scenario)

        observation = _make_observation(zone_a_count=7.0, zone_a_alarm=True)
        snapshot = StateManager().update_perception(observation, time=12.0)

        gateway.notify(snapshot)

        self.assertEqual(dashboard.building_view._building, building)
        self.assertIn("zone-a", dashboard.building_view._frame.current_fire_zone_ids)
        self.assertEqual(dashboard.building_view._frame.zone_occupancy.get("zone-a"), 7.0)

    def test_notify_pushes_human_observations_through_to_the_human_panel(self):

        from command_center.dashboard import Dashboard
        from perception.models.building_observation import BuildingObservation
        from perception.models.human_observation import HumanObservation, HumanState

        building = make_building()
        scenario = make_scenario(building)
        dashboard = Dashboard()

        gateway = DashboardCommandCenterGateway(dashboard, building, scenario)

        person = HumanObservation(
            person_id="person-42", zone_id="zone-a", state=HumanState.WAITING, confidence=0.6,
        )
        observation = BuildingObservation(human_observations={"person-42": person})
        snapshot = StateManager().update_perception(observation, time=3.0)

        gateway.notify(snapshot)

        self.assertIn("person-42", dashboard.human_panel._frame.human_observations)
        self.assertEqual(dashboard.human_panel.people_table.rowCount(), 1)
        self.assertEqual(dashboard.human_panel.people_table.item(0, 0).text(), "person-42")


# =====================================================
# Phase 1 -- Live Orchestrator (end-to-end, mocked gateways)
# =====================================================


class _StubPerceptionGateway:

    def __init__(self, alarm_active=False):
        self.alarm_active = alarm_active
        self.calls = []

    def collect(self, readings, time):
        self.calls.append((readings, time))
        return _make_observation(zone_a_count=1.0, zone_a_alarm=self.alarm_active)


class _StubAIInferenceGateway:

    def __init__(self):
        self.calls = []

    def predict(self, snapshot):
        self.calls.append(snapshot)
        return {"stub_prediction": "value"}


class _StubDecisionPolicyGateway:

    def __init__(self):
        self.calls = []

    def evaluate(self, snapshot):
        self.calls.append(snapshot)
        return "stub-policy"


class _StubCommandCenterGateway:

    def __init__(self):
        self.notified = []

    def notify(self, snapshot):
        self.notified.append(snapshot)


class LiveOrchestratorStartupTests(unittest.TestCase):

    def test_starts_not_running(self):

        orchestrator = LiveOrchestrator()
        self.assertFalse(orchestrator.is_running)

    def test_run_cycle_before_start_raises(self):

        orchestrator = LiveOrchestrator()

        with self.assertRaises(LiveSystemNotRunningError):
            orchestrator.run_cycle(0.0)

    def test_start_then_run_cycle_succeeds(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsInstance(snapshot, LiveBuildingSnapshot)
        self.assertTrue(orchestrator.is_running)

    def test_double_start_raises(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        with self.assertRaises(LiveSystemAlreadyRunningError):
            orchestrator.start()

    def test_stop_then_run_cycle_raises_again(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()
        orchestrator.stop()

        with self.assertRaises(LiveSystemNotRunningError):
            orchestrator.run_cycle(0.0)

    def test_default_collaborators_are_constructed_when_not_injected(self):

        orchestrator = LiveOrchestrator()

        self.assertIsInstance(orchestrator.sensor_registry, SensorRegistry)
        self.assertIsInstance(orchestrator.event_bus, EventBus)
        self.assertIsInstance(orchestrator.state_manager, StateManager)
        self.assertIsInstance(orchestrator.incident_manager, IncidentManager)

    def test_injected_collaborators_are_used_verbatim(self):

        registry = SensorRegistry()
        bus = EventBus()

        orchestrator = LiveOrchestrator(sensor_registry=registry, event_bus=bus)

        self.assertIs(orchestrator.sensor_registry, registry)
        self.assertIs(orchestrator.event_bus, bus)


class LiveOrchestratorSensorRegistrationTests(unittest.TestCase):

    def test_registered_sensors_are_read_every_cycle(self):

        registry = SensorRegistry()
        sensor = _FixedCCTVSensor(
            "cam-1", CameraFrameObservation(camera_id="cam-1", timestamp=0.0, estimated_occupant_count=2.0),
        )
        registry.register(sensor)

        perception_gateway = _StubPerceptionGateway()
        orchestrator = LiveOrchestrator(
            sensor_registry=registry, perception_gateway=perception_gateway,
        )
        orchestrator.start()
        orchestrator.run_cycle(0.0)

        self.assertEqual(len(perception_gateway.calls), 1)
        readings, time = perception_gateway.calls[0]
        self.assertEqual(len(readings.camera_observations), 1)
        self.assertEqual(time, 0.0)


class LiveOrchestratorEventFlowTests(unittest.TestCase):

    def setUp(self):

        self.bus = EventBus()
        self.received = []
        self.bus.subscribe_all(self.received.append)

    def test_a_full_cycle_publishes_sensor_occupancy_and_hazard_events(self):

        orchestrator = LiveOrchestrator(
            event_bus=self.bus, perception_gateway=_StubPerceptionGateway(),
        )
        orchestrator.start()
        orchestrator.run_cycle(0.0)

        published_types = [event.event_type for event in self.received]

        self.assertIn(EventType.SENSOR_UPDATE, published_types)
        self.assertIn(EventType.OCCUPANCY_UPDATED, published_types)
        self.assertIn(EventType.HAZARD_UPDATED, published_types)

    def test_recommendation_updated_is_published_only_when_a_builder_is_configured(self):

        without_builder = LiveOrchestrator(event_bus=self.bus)
        without_builder.start()
        without_builder.run_cycle(0.0)

        self.assertNotIn(
            EventType.RECOMMENDATION_UPDATED, [event.event_type for event in self.received],
        )

        self.received.clear()

        with_builder = LiveOrchestrator(
            event_bus=self.bus, recommendation_builder=lambda snapshot: (object(),),
        )
        with_builder.start()
        with_builder.run_cycle(0.0)

        self.assertIn(
            EventType.RECOMMENDATION_UPDATED, [event.event_type for event in self.received],
        )

    def test_operator_acknowledgement_is_published(self):

        orchestrator = LiveOrchestrator(event_bus=self.bus)
        orchestrator.start()

        orchestrator.acknowledge("operator-1", time=5.0, note="on scene")

        acks = self.bus.history_of(EventType.OPERATOR_ACKNOWLEDGEMENT)
        self.assertEqual(len(acks), 1)
        self.assertEqual(acks[0].payload.operator_id, "operator-1")
        self.assertEqual(acks[0].payload.note, "on scene")

    def test_alarm_activated_is_published_when_perception_reports_an_alarm(self):

        orchestrator = LiveOrchestrator(
            event_bus=self.bus, perception_gateway=_StubPerceptionGateway(alarm_active=True),
        )
        orchestrator.start()
        orchestrator.run_cycle(0.0)

        alarms = self.bus.history_of(EventType.ALARM_ACTIVATED)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(orchestrator.incident_manager.state, IncidentState.ALARM)

    def test_alarm_activated_is_not_republished_on_a_later_still_alarmed_cycle(self):

        orchestrator = LiveOrchestrator(
            event_bus=self.bus, perception_gateway=_StubPerceptionGateway(alarm_active=True),
        )
        orchestrator.start()
        orchestrator.run_cycle(0.0)
        orchestrator.run_cycle(1.0)

        self.assertEqual(len(self.bus.history_of(EventType.ALARM_ACTIVATED)), 1)


class LiveOrchestratorStateUpdatesTests(unittest.TestCase):

    def test_perception_result_is_reflected_in_the_state_manager(self):

        orchestrator = LiveOrchestrator(perception_gateway=_StubPerceptionGateway())
        orchestrator.start()

        snapshot = orchestrator.run_cycle(3.0)

        self.assertIs(orchestrator.state_manager.current(), snapshot)
        self.assertEqual(snapshot.occupancy_for("zone-a").estimated_count, 1.0)
        self.assertEqual(snapshot.timestamp, 3.0)


class LiveOrchestratorAIAndDecisionPolicyInvocationTests(unittest.TestCase):

    def test_ai_inference_gateway_is_invoked_with_the_current_snapshot(self):

        ai_gateway = _StubAIInferenceGateway()
        orchestrator = LiveOrchestrator(
            perception_gateway=_StubPerceptionGateway(), ai_inference_gateway=ai_gateway,
        )
        orchestrator.start()
        snapshot = orchestrator.run_cycle(0.0)

        self.assertEqual(len(ai_gateway.calls), 1)
        self.assertEqual(snapshot.ai_predictions, {"stub_prediction": "value"})

    def test_decision_policy_gateway_is_invoked_after_ai_inference(self):

        call_order = []

        class OrderTrackingAI(_StubAIInferenceGateway):
            def predict(self, snapshot):
                call_order.append("ai")
                return super().predict(snapshot)

        class OrderTrackingPolicy(_StubDecisionPolicyGateway):
            def evaluate(self, snapshot):
                call_order.append("policy")
                return super().evaluate(snapshot)

        orchestrator = LiveOrchestrator(
            ai_inference_gateway=OrderTrackingAI(), decision_policy_gateway=OrderTrackingPolicy(),
        )
        orchestrator.start()
        snapshot = orchestrator.run_cycle(0.0)

        self.assertEqual(call_order, ["ai", "policy"])
        self.assertEqual(snapshot.decision_policy, "stub-policy")

    def test_stages_with_no_gateway_configured_are_simply_skipped(self):

        # No perception/AI/decision-policy/command-center gateway at
        # all -- a fully "bare" orchestrator must still run a cycle
        # cleanly rather than raising, per every gateway being
        # documented optional.

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        snapshot = orchestrator.run_cycle(0.0)

        self.assertIsNone(snapshot.building_observation)
        self.assertEqual(snapshot.ai_predictions, {})
        self.assertIsNone(snapshot.decision_policy)

    def test_command_center_gateway_is_notified_last_with_the_fully_assembled_snapshot(self):

        command_center = _StubCommandCenterGateway()
        orchestrator = LiveOrchestrator(
            perception_gateway=_StubPerceptionGateway(),
            ai_inference_gateway=_StubAIInferenceGateway(),
            decision_policy_gateway=_StubDecisionPolicyGateway(),
            recommendation_builder=lambda snapshot: (object(),),
            command_center_gateway=command_center,
        )
        orchestrator.start()
        snapshot = orchestrator.run_cycle(0.0)

        self.assertEqual(len(command_center.notified), 1)
        notified_snapshot = command_center.notified[0]

        self.assertIs(notified_snapshot, snapshot)
        self.assertIsNotNone(notified_snapshot.building_observation)
        self.assertEqual(notified_snapshot.ai_predictions, {"stub_prediction": "value"})
        self.assertEqual(notified_snapshot.decision_policy, "stub-policy")
        self.assertEqual(len(notified_snapshot.recommendations), 1)


class LiveOrchestratorIncidentLifecycleTests(unittest.TestCase):

    def test_manual_transition_through_the_full_lifecycle(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        orchestrator.transition_incident(IncidentState.ALARM, time=0.0)
        orchestrator.transition_incident(IncidentState.EVACUATION, time=1.0)
        orchestrator.transition_incident(IncidentState.SUPPRESSION, time=2.0)
        orchestrator.transition_incident(IncidentState.RECOVERY, time=3.0)
        orchestrator.transition_incident(IncidentState.CLOSED, time=4.0)

        self.assertEqual(orchestrator.incident_manager.state, IncidentState.CLOSED)

    def test_illegal_manual_transition_raises_via_the_orchestrator_too(self):

        orchestrator = LiveOrchestrator()
        orchestrator.start()

        with self.assertRaises(InvalidIncidentTransition):
            orchestrator.transition_incident(IncidentState.SUPPRESSION, time=0.0)


class LiveOrchestratorContinuousUpdateLoopTests(unittest.TestCase):

    def test_update_loop_drives_repeated_deterministic_cycles(self):

        perception_gateway = _StubPerceptionGateway()
        orchestrator = LiveOrchestrator(
            perception_gateway=perception_gateway, interval_seconds=2.0,
        )
        orchestrator.start()

        snapshots = orchestrator.update_loop.run_for(4, start_time=0.0)

        self.assertEqual([snapshot.timestamp for snapshot in snapshots], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(len(perception_gateway.calls), 4)
        self.assertEqual(orchestrator.update_loop.cycle_count, 4)

    def test_update_loop_refuses_to_run_before_start(self):

        orchestrator = LiveOrchestrator()

        with self.assertRaises(LiveSystemNotRunningError):
            orchestrator.update_loop.run_for(1, start_time=0.0)

    def test_repeated_runs_over_identical_mocked_input_are_deterministic(self):

        def build_and_run():

            registry = SensorRegistry()
            registry.register(_FixedCCTVSensor(
                "cam-1",
                CameraFrameObservation(camera_id="cam-1", timestamp=0.0, estimated_occupant_count=6.0),
            ))

            sensor_fusion = SensorFusion(
                smoke_detector_zone_assignments={}, heat_detector_zone_assignments={},
                camera_zone_assignments={"cam-1": "zone-a"},
            )
            occupancy_estimator = OccupancyEstimator(camera_zone_assignments={"cam-1": "zone-a"})

            orchestrator = LiveOrchestrator(
                sensor_registry=registry,
                perception_gateway=SensorFusionPerceptionGateway(sensor_fusion, occupancy_estimator),
                interval_seconds=1.0,
            )
            orchestrator.start()

            return orchestrator.update_loop.run_for(3, start_time=0.0)

        first_run = build_and_run()
        second_run = build_and_run()

        self.assertEqual(
            [s.occupancy_for("zone-a").estimated_count for s in first_run],
            [s.occupancy_for("zone-a").estimated_count for s in second_run],
        )
        self.assertEqual(
            [s.timestamp for s in first_run], [s.timestamp for s in second_run],
        )


# =====================================================
# Canonical Live BuildingState Runtime Assembly milestone -- Phase 10
# architecture guards.
# =====================================================


class LiveSystemPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase enforces (see e.g.
    # tests.test_sensor_manager.SensorManagerPackageDependencyDirectionTests,
    # tests.test_no_cv_dependencies). live_system MAY depend on
    # building_state (that dependency is this milestone's own point --
    # see live_system/building_state_gateway.py) -- what stays forbidden
    # is live_system reaching past building_state's already-computed
    # BuildingState/CameraStatus/SensorStatus/FusionResult/FACPSnapshot/
    # ControlStateSnapshot value types into the manager/engine/controller
    # classes that OWN or CONTROL a real or replayed device
    # (CameraManager, SensorManager, MultiCameraFusionEngine,
    # SimulatedFACP, FACPEventProvider, BuildingControlController, any
    # BuildingControlProvider), or into any concrete frame-source
    # implementation (ReplayFrameSource today; a future RTSPFrameSource)
    # or computer-vision/streaming library -- Phase 7's own explicit
    # "the orchestrator coordinates already-produced data; it must not
    # directly construct or own CameraManager/SensorManager/
    # MultiCameraFusionEngine/FACP internals" requirement, made
    # mechanical.

    def test_never_imports_device_managers_control_internals_or_frame_sources(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_system"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(camera_manager\.manager|sensor_manager\.manager|"
            r"multi_camera_fusion\.engine|facp\.engine|facp\.provider|"
            r"building_control\.controller|building_control\.providers|"
            r"live_camera_pipeline\.replay_frame_source|live_camera_pipeline\.frame_source|"
            r"cv2|torch|ultralytics|onvif)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"live_system/{path.name} imports a device-manager, panel/control-engine, "
                f"frame-source, or computer-vision/streaming module directly -- "
                f"live_system must only coordinate already-produced values via an injected "
                f"Gateway/provider, never construct or own a real device manager itself",
            )

    # =====================================================
    # Live AI Inference Runtime Integration milestone -- the AI boundary
    # must stay BuildingState -> AI, never CameraFrame -> AI and never
    # Scenario -> AI (Phase 11's own explicit requirement), and AI must
    # never be able to reach control/broadcast authority (Phase 26/27).

    def test_never_imports_scenario_ground_truth_control_or_voice_modules(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_system"

        # building_control.snapshot (the read-only ControlStateSnapshot
        # value type) is a pre-existing, already-narrowly-guarded
        # exception -- see test_never_imports_device_managers_control_
        # internals_or_frame_sources above, which forbids specifically
        # building_control.controller/.providers (the control-CAPABLE
        # modules). This guard does not re-forbid building_control
        # itself to avoid conflicting with that already-correct rule.
        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_definition|scenario_generator|scenario_runner|ground_truth|"
            r"voice_evacuation|speaker_manager)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"live_system/{path.name} imports {forbidden!r} -- the AI boundary must stay "
                f"BuildingState -> AI only (never Scenario/GroundTruth -> AI), and AI must never "
                f"reach voice-broadcast authority directly.",
            )


if __name__ == "__main__":
    unittest.main()

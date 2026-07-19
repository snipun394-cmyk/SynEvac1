import unittest

from dataclasses import FrozenInstanceError

from perception import (
    BuildingObservation,
    CameraFrameObservation,
    CameraProvider,
    HeatDetectorProvider,
    HeatDetectorReading,
    ObservationState,
    ObservedEdgeState,
    ObservedNodeState,
    ObservedOccupancy,
    PerceptionProvider,
    PerceptionSeverity,
    PerceptionSystemStatus,
    SmokeDetectorProvider,
    SmokeDetectorReading,
)


class ObservationModelConstructionTests(unittest.TestCase):

    def test_camera_frame_observation_defaults_to_no_reading(self):

        observation = CameraFrameObservation(camera_id="cam1", timestamp=0.0)

        self.assertIsNone(observation.estimated_occupant_count)
        self.assertIsNone(observation.visibility_estimate)
        self.assertIsNone(observation.confidence)

    def test_smoke_detector_reading_requires_alarm_state(self):

        reading = SmokeDetectorReading(detector_id="s1", timestamp=0.0, alarm_active=True)

        self.assertTrue(reading.alarm_active)
        self.assertIsNone(reading.confidence)

    def test_heat_detector_reading_requires_alarm_state(self):

        reading = HeatDetectorReading(detector_id="h1", timestamp=0.0, alarm_active=False)

        self.assertFalse(reading.alarm_active)
        self.assertIsNone(reading.confidence)

    def test_heat_detector_reading_has_no_rate_of_rise_field(self):

        # SynEvac V1 supports Fixed Temperature heat detectors only --
        # Rate-of-Rise is out of scope, and this type must not carry a
        # dormant placeholder field for it (see
        # perception/models/heat_detector_observation.py).
        from dataclasses import fields

        field_names = {field.name for field in fields(HeatDetectorReading)}
        self.assertNotIn("rate_of_rise_triggered", field_names)

    def test_observation_models_are_frozen(self):

        observation = CameraFrameObservation(camera_id="cam1", timestamp=0.0)

        with self.assertRaises(FrozenInstanceError):
            observation.camera_id = "cam2"


class ObservedNodeStateTests(unittest.TestCase):

    def test_defaults_to_unobserved(self):

        state = ObservedNodeState()

        self.assertEqual(state.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(state.alarm_active)
        self.assertEqual(state.alarm_source_types, [])
        self.assertIsNone(state.estimated_severity)

    def test_is_frozen(self):

        state = ObservedNodeState()

        with self.assertRaises(FrozenInstanceError):
            state.observation_state = ObservationState.OBSERVED


class BuildingObservationTests(unittest.TestCase):

    def test_default_observation_has_no_entries(self):

        observation = BuildingObservation()

        self.assertEqual(len(observation.node_observations), 0)
        self.assertEqual(len(observation.occupancy_observations), 0)
        self.assertEqual(len(observation.edge_observations), 0)

    def test_missing_node_is_unobserved_not_clear(self):

        observation = BuildingObservation(
            node_observations={
                "z1": ObservedNodeState(
                    observation_state=ObservationState.OBSERVED, alarm_active=False,
                ),
            },
        )

        self.assertEqual(observation.node_observation("z1").observation_state, ObservationState.OBSERVED)

        unobserved = observation.node_observation("z2")
        self.assertEqual(unobserved.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(unobserved.alarm_active)

    def test_missing_occupancy_returns_no_reading_not_zero(self):

        observation = BuildingObservation()

        occupancy = observation.occupancy_observation("z1")
        self.assertIsNone(occupancy.estimated_count)

    def test_missing_edge_returns_no_opinion(self):

        observation = BuildingObservation()

        self.assertIsNone(observation.edge_observation("e1").blocked_estimate)

    def test_is_frozen(self):

        observation = BuildingObservation()

        with self.assertRaises(FrozenInstanceError):
            observation.timestamp = 5.0

    def test_mappings_are_read_only(self):

        observation = BuildingObservation(node_observations={"z1": ObservedNodeState()})

        with self.assertRaises(TypeError):
            observation.node_observations["z2"] = ObservedNodeState()

    def test_observation_ids_are_unique_by_default(self):

        first = BuildingObservation()
        second = BuildingObservation()

        self.assertNotEqual(first.observation_id, second.observation_id)

    def test_default_system_status_reports_no_failure(self):

        observation = BuildingObservation()

        self.assertTrue(observation.system_status.panel_communication_ok)


class ProviderInterfaceTests(unittest.TestCase):

    def test_perception_provider_base_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            PerceptionProvider().observation_at(0.0)

    def test_camera_provider_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            CameraProvider().frame_observation_at("cam1", 0.0)

    def test_smoke_detector_provider_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            SmokeDetectorProvider().alarm_states_at(0.0)

    def test_heat_detector_provider_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            HeatDetectorProvider().alarm_states_at(0.0)

    def test_camera_and_detector_providers_specialize_the_raw_sensor_contracts(self):

        from sensors.provider import CameraProvider as RawCameraProvider
        from sensors.provider import DetectorProvider as RawDetectorProvider

        self.assertTrue(issubclass(CameraProvider, RawCameraProvider))
        self.assertTrue(issubclass(SmokeDetectorProvider, RawDetectorProvider))
        self.assertTrue(issubclass(HeatDetectorProvider, RawDetectorProvider))


class PerceptionPackageDependencyDirectionTests(unittest.TestCase):

    def test_models_never_import_ground_truth_or_ai_frameworks(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "perception" / "models"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(hazard|hazard_evolution|occupancy|navigation|pathfinding|simulator|behavior|"
            r"models|designer|sensors|gymnasium|gym|numpy|torch)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"perception/models/{path.name} imports a Ground Truth, downstream, or "
                f"AI-framework module -- BuildingObservation and its component types must "
                f"stay independent of all of them",
            )


if __name__ == "__main__":
    unittest.main()

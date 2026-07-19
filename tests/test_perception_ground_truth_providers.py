import unittest

from dataclasses import fields

from hazard.node_state import HazardNodeState
from hazard.provider import ManualHazardProvider
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.provider import ManualOccupancyProvider
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.detector import Detector
from models.zone import Zone

from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading

from perception.providers.ground_truth_camera_provider import GroundTruthCameraProvider
from perception.providers.ground_truth_heat_detector_provider import GroundTruthHeatDetectorProvider
from perception.providers.ground_truth_smoke_detector_provider import GroundTruthSmokeDetectorProvider


GROUND_TRUTH_TYPES = (
    HazardSnapshot,
    HazardNodeState,
    OccupancySnapshot,
    OccupancyObservation,
)


class GroundTruthCameraProviderTests(unittest.TestCase):

    def setUp(self):

        # zone_near sits directly in front of the camera, well inside
        # its coverage fan; zone_far sits behind and far away from it,
        # outside both the field of view and the max range.
        self.zone_near = Zone(id="zone-near", floor_id="f1", x=5.0, y=-1.0, width=2.0, height=2.0)
        self.zone_far = Zone(id="zone-far", floor_id="f1", x=-30.0, y=-30.0, width=2.0, height=2.0)

        self.camera = Camera(
            id="cam1", floor_id="f1", position=(0.0, 0.0), rotation=0.0,
            horizontal_fov=90.0, max_range=20.0,
        )

        self.occupancy_provider = ManualOccupancyProvider(
            OccupancySnapshot(
                observations={
                    "zone-near": OccupancyObservation(occupant_count=3.0),
                    "zone-far": OccupancyObservation(occupant_count=7.0),
                },
            ),
        )

        self.provider = GroundTruthCameraProvider(
            cameras=[self.camera],
            zones=[self.zone_near, self.zone_far],
            occupancy_provider=self.occupancy_provider,
        )

    def test_only_the_covered_zones_occupancy_is_reported(self):

        observation = self.provider.frame_observation_at("cam1", time=0.0)

        self.assertEqual(observation.estimated_occupant_count, 3.0)

    def test_inactive_camera_reports_no_reading(self):

        inactive_camera = Camera(
            id="cam2", floor_id="f1", position=(0.0, 0.0), rotation=0.0,
            horizontal_fov=90.0, max_range=20.0, active=False,
        )
        provider = GroundTruthCameraProvider(
            cameras=[inactive_camera], zones=[self.zone_near], occupancy_provider=self.occupancy_provider,
        )

        observation = provider.frame_observation_at("cam2", time=0.0)

        self.assertIsNone(observation.estimated_occupant_count)

    def test_camera_covering_nothing_reports_no_reading(self):

        provider = GroundTruthCameraProvider(
            cameras=[self.camera], zones=[self.zone_far], occupancy_provider=self.occupancy_provider,
        )

        observation = provider.frame_observation_at("cam1", time=0.0)

        self.assertIsNone(observation.estimated_occupant_count)

    def test_returns_typed_camera_frame_observation(self):

        observation = self.provider.frame_observation_at("cam1", time=0.0)

        self.assertIsInstance(observation, CameraFrameObservation)


class GroundTruthSmokeDetectorProviderTests(unittest.TestCase):

    def setUp(self):

        self.zone_smoky = Zone(id="zone-smoky", floor_id="f1", x=0.0, y=0.0, width=4.0, height=4.0)
        self.zone_clear = Zone(id="zone-clear", floor_id="f1", x=10.0, y=10.0, width=4.0, height=4.0)

        self.detector_in_smoky_zone = Detector(
            id="smoke1", floor_id="f1", position=(1.0, 1.0), detector_type="Smoke",
        )
        self.detector_in_clear_zone = Detector(
            id="smoke2", floor_id="f1", position=(11.0, 11.0), detector_type="Smoke",
        )

        self.hazard_provider = ManualHazardProvider(
            HazardSnapshot(
                node_states={
                    "zone-smoky": HazardNodeState(smoke_level=0.5),
                    "zone-clear": HazardNodeState(smoke_level=0.05),
                },
            ),
        )

        self.provider = GroundTruthSmokeDetectorProvider(
            detectors=[self.detector_in_smoky_zone, self.detector_in_clear_zone],
            zones=[self.zone_smoky, self.zone_clear],
            hazard_provider=self.hazard_provider,
        )

    def test_each_detector_reports_only_its_own_zones_smoke(self):

        readings_by_id = {
            reading.detector_id: reading
            for reading in self.provider.alarm_states_at(time=0.0)
        }

        self.assertTrue(readings_by_id["smoke1"].alarm_active)
        self.assertFalse(readings_by_id["smoke2"].alarm_active)

    def test_detector_outside_every_zone_produces_no_reading(self):

        stray_detector = Detector(id="smoke3", floor_id="f1", position=(999.0, 999.0), detector_type="Smoke")
        provider = GroundTruthSmokeDetectorProvider(
            detectors=[stray_detector], zones=[self.zone_smoky, self.zone_clear],
            hazard_provider=self.hazard_provider,
        )

        readings = provider.alarm_states_at(time=0.0)

        self.assertEqual(readings, [])

    def test_zone_with_no_smoke_level_ground_truth_produces_no_reading(self):

        provider = GroundTruthSmokeDetectorProvider(
            detectors=[self.detector_in_smoky_zone],
            zones=[self.zone_smoky],
            hazard_provider=ManualHazardProvider(HazardSnapshot()),
        )

        readings = provider.alarm_states_at(time=0.0)

        self.assertEqual(readings, [])

    def test_heat_type_detectors_are_ignored(self):

        heat_detector = Detector(id="heat1", floor_id="f1", position=(1.0, 1.0), detector_type="Heat")
        provider = GroundTruthSmokeDetectorProvider(
            detectors=[heat_detector], zones=[self.zone_smoky], hazard_provider=self.hazard_provider,
        )

        self.assertEqual(provider.alarm_states_at(time=0.0), [])

    def test_returns_typed_smoke_detector_readings(self):

        for reading in self.provider.alarm_states_at(time=0.0):
            self.assertIsInstance(reading, SmokeDetectorReading)


class GroundTruthHeatDetectorProviderTests(unittest.TestCase):

    def setUp(self):

        self.zone_hot = Zone(id="zone-hot", floor_id="f1", x=0.0, y=0.0, width=4.0, height=4.0)
        self.zone_cool = Zone(id="zone-cool", floor_id="f1", x=10.0, y=10.0, width=4.0, height=4.0)

        self.detector_in_hot_zone = Detector(
            id="heat1", floor_id="f1", position=(1.0, 1.0), detector_type="Heat",
        )
        self.detector_in_cool_zone = Detector(
            id="heat2", floor_id="f1", position=(11.0, 11.0), detector_type="Heat",
        )

        self.hazard_provider = ManualHazardProvider(
            HazardSnapshot(
                node_states={
                    "zone-hot": HazardNodeState(temperature=80.0),
                    "zone-cool": HazardNodeState(temperature=22.0),
                },
            ),
        )

        self.provider = GroundTruthHeatDetectorProvider(
            detectors=[self.detector_in_hot_zone, self.detector_in_cool_zone],
            zones=[self.zone_hot, self.zone_cool],
            hazard_provider=self.hazard_provider,
        )

    def test_each_detector_reports_only_its_own_zones_temperature(self):

        readings_by_id = {
            reading.detector_id: reading
            for reading in self.provider.alarm_states_at(time=0.0)
        }

        self.assertTrue(readings_by_id["heat1"].alarm_active)
        self.assertFalse(readings_by_id["heat2"].alarm_active)

    def test_zone_with_no_temperature_ground_truth_produces_no_reading(self):

        provider = GroundTruthHeatDetectorProvider(
            detectors=[self.detector_in_hot_zone],
            zones=[self.zone_hot],
            hazard_provider=ManualHazardProvider(HazardSnapshot()),
        )

        self.assertEqual(provider.alarm_states_at(time=0.0), [])

    def test_returns_typed_heat_detector_readings(self):

        for reading in self.provider.alarm_states_at(time=0.0):
            self.assertIsInstance(reading, HeatDetectorReading)


class NoGroundTruthEscapesPublicApiTests(unittest.TestCase):

    # Every field on every observation model these adapters return
    # must be a plain Python value -- never a Ground Truth object
    # (HazardSnapshot/HazardNodeState/OccupancySnapshot/
    # OccupancyObservation) passed through unwrapped. This is the
    # literal type-level check for "no Ground Truth object escapes
    # through any public API".

    def test_camera_frame_observation_never_carries_a_ground_truth_object(self):

        zone = Zone(id="z1", floor_id="f1", x=-5.0, y=-5.0, width=10.0, height=10.0)
        camera = Camera(id="cam1", floor_id="f1", position=(0.0, 0.0), horizontal_fov=360.0, max_range=20.0)
        occupancy_provider = ManualOccupancyProvider(
            OccupancySnapshot(observations={"z1": OccupancyObservation(occupant_count=2.0)}),
        )
        provider = GroundTruthCameraProvider(
            cameras=[camera], zones=[zone], occupancy_provider=occupancy_provider,
        )

        observation = provider.frame_observation_at("cam1", time=0.0)

        self._assert_no_ground_truth_field(observation)

    def test_smoke_detector_reading_never_carries_a_ground_truth_object(self):

        zone = Zone(id="z1", floor_id="f1", x=0.0, y=0.0, width=4.0, height=4.0)
        detector = Detector(id="s1", floor_id="f1", position=(1.0, 1.0), detector_type="Smoke")
        hazard_provider = ManualHazardProvider(
            HazardSnapshot(node_states={"z1": HazardNodeState(smoke_level=0.5)}),
        )
        provider = GroundTruthSmokeDetectorProvider(
            detectors=[detector], zones=[zone], hazard_provider=hazard_provider,
        )

        for reading in provider.alarm_states_at(time=0.0):
            self._assert_no_ground_truth_field(reading)

    def test_heat_detector_reading_never_carries_a_ground_truth_object(self):

        zone = Zone(id="z1", floor_id="f1", x=0.0, y=0.0, width=4.0, height=4.0)
        detector = Detector(id="h1", floor_id="f1", position=(1.0, 1.0), detector_type="Heat")
        hazard_provider = ManualHazardProvider(
            HazardSnapshot(node_states={"z1": HazardNodeState(temperature=80.0)}),
        )
        provider = GroundTruthHeatDetectorProvider(
            detectors=[detector], zones=[zone], hazard_provider=hazard_provider,
        )

        for reading in provider.alarm_states_at(time=0.0):
            self._assert_no_ground_truth_field(reading)

    def _assert_no_ground_truth_field(self, observation):

        self.assertNotIsInstance(observation, GROUND_TRUTH_TYPES)

        for observation_field in fields(observation):

            value = getattr(observation, observation_field.name)
            self.assertNotIsInstance(value, GROUND_TRUTH_TYPES)


class GroundTruthAdapterDependencyDirectionTests(unittest.TestCase):

    def test_adapters_never_import_simulator_sandbox_behavior_designer_or_ai_decision(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "perception" / "providers"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|sandbox|behavior|designer|ai_decision|hazard_evolution|"
            r"gymnasium|gym|numpy|torch)\b"
        )

        adapter_files = (
            "ground_truth_camera_provider.py",
            "ground_truth_smoke_detector_provider.py",
            "ground_truth_heat_detector_provider.py",
        )

        for filename in adapter_files:

            text = (package_dir / filename).read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"perception/providers/{filename} imports a simulator/runtime or "
                f"AI-framework module -- a Ground Truth adapter may only read Building "
                f"Model geometry (models) and the Ground Truth Provider interfaces "
                f"(hazard, occupancy), never simulator/sandbox internals",
            )


if __name__ == "__main__":
    unittest.main()

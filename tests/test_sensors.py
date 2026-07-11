import unittest

from dataclasses import FrozenInstanceError

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.engine import HazardEvolutionEngine
from hazard_evolution.source import HazardSource

from occupancy.observation import OccupancyObservation
from occupancy.provider import OccupancyProvider
from occupancy.snapshot import OccupancySnapshot

from sensors.provider import (
    CameraProvider,
    DetectorProvider,
    NullCameraProvider,
    NullDetectorProvider,
    NullSensorProvider,
    SensorProvider,
)
from sensors.reading import SensorReading
from sensors.replay import SimulatedReplaySensorProvider


class SensorReadingTests(unittest.TestCase):

    def test_default_payload_is_empty(self):

        reading = SensorReading(sensor_id="cam1", timestamp=0.0)
        self.assertEqual(dict(reading.payload), {})

    def test_is_frozen(self):

        reading = SensorReading(sensor_id="cam1", timestamp=0.0)

        with self.assertRaises(FrozenInstanceError):
            reading.sensor_id = "cam2"

    def test_payload_mapping_is_read_only(self):

        reading = SensorReading(sensor_id="cam1", timestamp=0.0, payload={"count": 3})

        with self.assertRaises(TypeError):
            reading.payload["count"] = 4


class SensorProviderInterfaceTests(unittest.TestCase):

    def test_base_interface_is_not_implemented(self):

        with self.assertRaises(NotImplementedError):
            SensorProvider().readings_at(0.0)

    def test_camera_and_detector_providers_specialize_sensor_provider(self):

        self.assertTrue(issubclass(CameraProvider, SensorProvider))
        self.assertTrue(issubclass(DetectorProvider, SensorProvider))

    def test_null_providers_report_no_readings(self):

        self.assertEqual(NullSensorProvider().readings_at(123.0), [])
        self.assertEqual(NullCameraProvider().readings_at(123.0), [])
        self.assertEqual(NullDetectorProvider().readings_at(123.0), [])

    def test_null_camera_and_detector_providers_are_still_their_specialized_type(self):

        self.assertIsInstance(NullCameraProvider(), CameraProvider)
        self.assertIsInstance(NullDetectorProvider(), DetectorProvider)


class SimulatedReplaySensorProviderTests(unittest.TestCase):

    def test_is_a_sensor_provider_a_hazard_source_and_an_occupancy_provider(self):

        provider = SimulatedReplaySensorProvider()

        self.assertIsInstance(provider, SensorProvider)
        self.assertIsInstance(provider, HazardSource)
        self.assertIsInstance(provider, OccupancyProvider)

    def test_replays_scheduled_readings_at_the_exact_requested_time(self):

        reading = SensorReading(sensor_id="cam1", timestamp=5.0, payload={"person_count": 2})
        provider = SimulatedReplaySensorProvider(readings_schedule={5.0: [reading]})

        self.assertEqual(provider.readings_at(5.0), [reading])
        self.assertEqual(provider.readings_at(6.0), [])

    def test_replays_scheduled_hazard_contribution_keyed_by_step_end_time(self):

        contribution = HazardContribution(node_states={"z1": HazardNodeState(hazard_score=0.8)})
        provider = SimulatedReplaySensorProvider(hazard_schedule={10.0: contribution})

        result = provider.propose(HazardSnapshot(), time=8.0, dt=2.0)  # step ends at t=10
        self.assertEqual(result, contribution)

    def test_no_opinion_when_nothing_is_scheduled_for_a_step(self):

        provider = SimulatedReplaySensorProvider()
        result = provider.propose(HazardSnapshot(), time=0.0, dt=1.0)

        self.assertEqual(result, HazardContribution())

    def test_replays_scheduled_occupancy_snapshot_at_the_exact_requested_time(self):

        snapshot = OccupancySnapshot(
            timestamp=5.0, observations={"z1": OccupancyObservation(occupant_count=6.0)},
        )
        provider = SimulatedReplaySensorProvider(occupancy_schedule={5.0: snapshot})

        self.assertIs(provider.snapshot_at(5.0), snapshot)

    def test_unscheduled_occupancy_time_returns_an_empty_snapshot_stamped_with_that_time(self):

        provider = SimulatedReplaySensorProvider()
        snapshot = provider.snapshot_at(42.0)

        self.assertEqual(snapshot.timestamp, 42.0)
        self.assertEqual(len(snapshot.observations), 0)

    def test_schedules_are_copied_at_construction(self):

        readings_schedule = {5.0: [SensorReading(sensor_id="cam1", timestamp=5.0)]}
        provider = SimulatedReplaySensorProvider(readings_schedule=readings_schedule)

        readings_schedule[5.0] = []
        readings_schedule[6.0] = [SensorReading(sensor_id="cam2", timestamp=6.0)]

        self.assertEqual(len(provider.readings_at(5.0)), 1)
        self.assertEqual(provider.readings_at(6.0), [])


class SensorsPackageDependencyDirectionTests(unittest.TestCase):

    def test_core_contracts_stay_independent_of_hazard_occupancy_and_downstream_layers(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "sensors"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(navigation|pathfinding|simulator|behavior|models|designer|"
            r"hazard|hazard_evolution|occupancy)\b"
        )

        for filename in ("reading.py", "provider.py"):

            text = (package_dir / filename).read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"sensors/{filename} imports a Hazard/Occupancy/downstream layer -- "
                f"the core sensor contracts must stay independent of what a reading "
                f"eventually feeds",
            )

    def test_replay_bridge_never_imports_navigation_pathfinding_simulation_behavior_or_models(self):

        import pathlib
        import re

        text = (
            pathlib.Path(__file__).resolve().parent.parent / "sensors" / "replay.py"
        ).read_text()

        forbidden = r"^\s*(from|import)\s+(navigation|pathfinding|simulator|behavior|models|designer)\b"

        self.assertIsNone(
            re.search(forbidden, text, re.MULTILINE),
            "sensors/replay.py imports a downstream/engineering layer -- it may bridge "
            "into hazard_evolution/occupancy (that is its documented purpose) but must "
            "never reach into Navigation, Pathfinding, Simulation, Behavior, or the "
            "Building Model",
        )


class CompatibilityWithExistingAIPipelineTests(unittest.TestCase):

    # Proves the long-term goal directly: a live-sensor-shaped source
    # plugs into today's HazardEvolutionEngine exactly like
    # FireGrowthModel/SmokePropagationModel/TenabilityModel do, with
    # zero engine changes.

    def test_replay_provider_participates_in_hazard_evolution_like_any_other_source(self):

        class OtherHazardSource(HazardSource):
            def propose(self, previous_snapshot, time, dt):
                return HazardContribution(node_states={"z1": HazardNodeState(hazard_score=0.2)})

        contribution = HazardContribution(node_states={"z1": HazardNodeState(hazard_score=0.9)})
        replay = SimulatedReplaySensorProvider(hazard_schedule={1.0: contribution})

        engine = HazardEvolutionEngine(sources=[OtherHazardSource(), replay])
        result = engine.evolve(HazardSnapshot(timestamp=0.0), time=0.0, dt=1.0)

        # Worst-case-wins merge (unchanged DefaultHazardMergeStrategy)
        # picks the replay's higher value with no special-casing.
        self.assertEqual(result.node_state("z1").hazard_score, 0.9)

    def test_replay_provider_serves_occupancy_independently_of_the_hazard_result(self):

        occupancy_snapshot = OccupancySnapshot(
            timestamp=1.0, observations={"z1": OccupancyObservation(occupant_count=12.0)},
        )
        replay = SimulatedReplaySensorProvider(occupancy_schedule={1.0: occupancy_snapshot})

        result = replay.snapshot_at(1.0)

        self.assertEqual(result.observation_at("z1").occupant_count, 12.0)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from hazard.node_state import HazardNodeState
from hazard.provider import ManualHazardProvider
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from models.building import Building
from models.detector import Detector
from models.detector_migration import adapt_legacy_detector
from models.heat_detector import HeatDetector
from models.sensor_asset import DetectorState
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from perception.providers.ground_truth_heat_detector_provider import GroundTruthHeatDetectorProvider
from perception.providers.ground_truth_smoke_detector_provider import GroundTruthSmokeDetectorProvider

from sensor_manager.manager import SensorManager

from serialization.serializer import Serializer

from building_state.estimator import BuildingStateEstimator


def make_zone(zone_id, floor_id, x=0.0, y=0.0, width=10.0, height=10.0):

    return Zone(id=zone_id, floor_id=floor_id, x=x, y=y, width=width, height=height)


class AdaptLegacyDetectorTests(unittest.TestCase):

    def setUp(self):

        self.zone = make_zone("zone-a", "floor1", x=0.0, y=0.0, width=10.0, height=10.0)

    def test_adapts_a_legacy_smoke_detector(self):

        legacy = Detector(
            id="SD-001", name="Smoke SD-001", floor_id="floor1",
            position=(1.0, 1.0), detector_type="Smoke", active=True,
        )

        adapted = adapt_legacy_detector(legacy, [self.zone])

        self.assertIsInstance(adapted, SmokeDetector)
        self.assertEqual(adapted.id, "SD-001")
        self.assertEqual(adapted.name, "Smoke SD-001")
        self.assertEqual(adapted.floor_id, "floor1")
        self.assertEqual(adapted.position, (1.0, 1.0))
        self.assertTrue(adapted.active)
        self.assertEqual(adapted.zone_ids, ("zone-a",))

    def test_adapts_a_legacy_heat_detector(self):

        legacy = Detector(
            id="HD-001", floor_id="floor1", position=(2.0, 2.0), detector_type="Heat",
        )

        adapted = adapt_legacy_detector(legacy, [self.zone])

        self.assertIsInstance(adapted, HeatDetector)
        self.assertEqual(adapted.id, "HD-001")
        self.assertEqual(adapted.zone_ids, ("zone-a",))

    def test_flame_and_gas_detectors_have_no_canonical_asset_yet(self):

        flame = Detector(id="FL-001", floor_id="floor1", position=(1.0, 1.0), detector_type="Flame")
        gas = Detector(id="GA-001", floor_id="floor1", position=(1.0, 1.0), detector_type="Gas")

        self.assertIsNone(adapt_legacy_detector(flame, [self.zone]))
        self.assertIsNone(adapt_legacy_detector(gas, [self.zone]))

    def test_detector_outside_every_zone_gets_empty_zone_ids(self):

        legacy = Detector(id="SD-002", floor_id="floor1", position=(999.0, 999.0), detector_type="Smoke")

        adapted = adapt_legacy_detector(legacy, [self.zone])

        self.assertEqual(adapted.zone_ids, ())

    def test_adapting_never_mutates_the_original_legacy_detector(self):

        legacy = Detector(id="SD-003", floor_id="floor1", position=(1.0, 1.0), detector_type="Smoke")

        adapt_legacy_detector(legacy, [self.zone])

        self.assertEqual(legacy.detector_type, "Smoke")
        self.assertFalse(hasattr(legacy, "zone_ids"))

    def test_adapted_asset_defaults_to_the_canonical_alarm_threshold(self):

        legacy_smoke = Detector(id="SD-004", floor_id="floor1", position=(1.0, 1.0), detector_type="Smoke")
        legacy_heat = Detector(id="HD-004", floor_id="floor1", position=(1.0, 1.0), detector_type="Heat")

        adapted_smoke = adapt_legacy_detector(legacy_smoke, [self.zone])
        adapted_heat = adapt_legacy_detector(legacy_heat, [self.zone])

        self.assertEqual(adapted_smoke.activation_threshold, SmokeDetector().activation_threshold)
        self.assertEqual(adapted_heat.activation_threshold, HeatDetector().activation_threshold)


class SensorManagerDiscoversLegacyDetectorsTests(unittest.TestCase):

    def test_discovers_a_legacy_smoke_detector_with_its_own_id(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.zones.append(make_zone("zone-a", floor.id))
        floor.detectors.append(
            Detector(id="SD-001", floor_id=floor.id, position=(1.0, 1.0), detector_type="Smoke"),
        )

        manager = SensorManager()
        manager.discover_sensors(building)

        status = manager.sensor_status("SD-001")

        self.assertEqual(status.sensor_id, "SD-001")
        self.assertEqual(status.sensor_type, "SmokeDetector")
        self.assertEqual(status.zone_ids, ("zone-a",))

    def test_flame_and_gas_legacy_detectors_are_not_discovered_as_sensors(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.detectors.append(
            Detector(id="FL-001", floor_id=floor.id, position=(1.0, 1.0), detector_type="Flame"),
        )

        manager = SensorManager()
        manager.discover_sensors(building)

        self.assertEqual(manager.all_sensors(), ())

    def test_legacy_and_canonical_detectors_coexist_without_id_collisions(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        floor.detectors.append(
            Detector(id="legacy-smoke", floor_id=floor.id, position=(1.0, 1.0), detector_type="Smoke"),
        )
        floor.add_smoke_detector(
            SmokeDetector(id="canonical-smoke", floor_id=floor.id, zone_ids=()),
        )

        manager = SensorManager()
        manager.discover_sensors(building)

        self.assertEqual(
            {sensor.id for sensor in manager.all_sensors()}, {"legacy-smoke", "canonical-smoke"},
        )
        self.assertEqual(len(manager.all_sensors()), 2)

    def test_rediscovery_does_not_duplicate_the_same_legacy_detector(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        floor.detectors.append(
            Detector(id="SD-001", floor_id=floor.id, position=(1.0, 1.0), detector_type="Smoke"),
        )

        manager = SensorManager()
        manager.discover_sensors(building)
        manager.discover_sensors(building)

        self.assertEqual(len(manager.all_sensors()), 1)


class GroundTruthProviderParityTests(unittest.TestCase):

    # A legacy Detector and a canonical SmokeDetector/HeatDetector
    # placed in equivalent positions must produce IDENTICAL alarm
    # behavior through the Ground Truth providers -- there must be no
    # second, diverging implementation of "what counts as an alarm" for
    # the two representations of what should be the same physical
    # device class.

    def test_legacy_and_canonical_smoke_detectors_agree_on_alarm_state(self):

        zone = make_zone("zone-a", "floor1", x=0.0, y=0.0, width=4.0, height=4.0)

        legacy = Detector(id="legacy1", floor_id="floor1", position=(1.0, 1.0), detector_type="Smoke")
        canonical = SmokeDetector(id="canonical1", floor_id="floor1", position=(1.0, 1.0), zone_ids=("zone-a",))

        hazard_provider = ManualHazardProvider(
            HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.5)}),
        )

        provider = GroundTruthSmokeDetectorProvider(
            detectors=[legacy, canonical], zones=[zone], hazard_provider=hazard_provider,
        )

        readings = {r.detector_id: r for r in provider.alarm_states_at(time=0.0)}

        self.assertTrue(readings["legacy1"].alarm_active)
        self.assertTrue(readings["canonical1"].alarm_active)

    def test_legacy_and_canonical_heat_detectors_agree_on_alarm_state(self):

        zone = make_zone("zone-a", "floor1", x=0.0, y=0.0, width=4.0, height=4.0)

        legacy = Detector(id="legacy1", floor_id="floor1", position=(1.0, 1.0), detector_type="Heat")
        canonical = HeatDetector(id="canonical1", floor_id="floor1", position=(1.0, 1.0), zone_ids=("zone-a",))

        hazard_provider = ManualHazardProvider(
            HazardSnapshot(node_states={"zone-a": HazardNodeState(temperature=80.0)}),
        )

        provider = GroundTruthHeatDetectorProvider(
            detectors=[legacy, canonical], zones=[zone], hazard_provider=hazard_provider,
        )

        readings = {r.detector_id: r for r in provider.alarm_states_at(time=0.0)}

        self.assertTrue(readings["legacy1"].alarm_active)
        self.assertTrue(readings["canonical1"].alarm_active)

    def test_faulty_canonical_detector_produces_no_reading(self):

        from models.sensor_asset import HealthStatus

        zone = make_zone("zone-a", "floor1", x=0.0, y=0.0, width=4.0, height=4.0)

        faulty = SmokeDetector(
            id="faulty1", floor_id="floor1", position=(1.0, 1.0), zone_ids=("zone-a",),
            health_status=HealthStatus.FAULT,
        )

        hazard_provider = ManualHazardProvider(
            HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.9)}),
        )

        provider = GroundTruthSmokeDetectorProvider(
            detectors=[faulty], zones=[zone], hazard_provider=hazard_provider,
        )

        self.assertEqual(provider.alarm_states_at(time=0.0), [])


class EndToEndIdentityPipelineTests(unittest.TestCase):

    # Designer Detector -> Save Project -> Reload Project -> SensorManager
    # -> HazardSnapshot -> Perception Detector Observation -> BuildingState,
    # verifying the SAME detector id is used at every stage.

    def _run_pipeline(self, building):

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")

            from models.project import Project
            project = Project(name="Identity Test")
            project.set_building(building)

            Serializer.save(project, path)
            reloaded_project = Serializer.load(path)

        return reloaded_project.building

    def test_legacy_smoke_detector_keeps_the_same_id_through_the_full_pipeline(self):

        detector_id = "SD-001"

        building = Building(name="Identity Building")
        floor = building.create_floor(name="Ground Floor")
        floor.zones.append(make_zone("zone-a", floor.id, x=0.0, y=0.0, width=10.0, height=10.0))
        floor.detectors.append(
            Detector(id=detector_id, name="Lobby Smoke", floor_id=floor.id, position=(1.0, 1.0), detector_type="Smoke"),
        )

        reloaded_building = self._run_pipeline(building)
        reloaded_floor = reloaded_building.floors[0]

        # 1. Reloads with the same id, still in Floor.detectors (backward
        #    compatible -- serialization of legacy Detector is untouched).
        self.assertEqual(reloaded_floor.detectors[0].id, detector_id)

        # 2. Discovered by SensorManager under the same id.
        manager = SensorManager()
        manager.discover_sensors(reloaded_building)
        sensor_status = manager.sensor_status(detector_id)
        self.assertEqual(sensor_status.sensor_id, detector_id)

        # 3. Produces a simulated reading from HazardSnapshot under the
        #    same id.
        hazard_snapshot = HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.9)})
        hazard_provider = ManualHazardProvider(hazard_snapshot)

        smoke_provider = GroundTruthSmokeDetectorProvider(
            detectors=[reloaded_floor.detectors[0]], zones=reloaded_floor.zones,
            hazard_provider=hazard_provider,
        )
        readings = smoke_provider.alarm_states_at(time=0.0)
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0].detector_id, detector_id)
        self.assertTrue(readings[0].alarm_active)

        # 4. Appears in BuildingState under the same id.
        estimator = BuildingStateEstimator()
        state = estimator.estimate(
            0.0,
            hazard_snapshot=hazard_snapshot,
            occupancy_snapshot=OccupancySnapshot(observations={"zone-a": OccupancyObservation(occupant_count=0.0)}),
            smoke_detector_statuses=[sensor_status],
            smoke_detector_readings=readings,
        )

        asset = state.smoke_detector_state(detector_id)
        self.assertIsNotNone(asset)
        self.assertEqual(asset.status.sensor_id, detector_id)
        self.assertEqual(asset.reading.detector_id, detector_id)
        self.assertTrue(asset.reading.alarm_active)

    def test_canonical_smoke_detector_keeps_the_same_id_through_the_full_pipeline(self):

        detector_id = "SD-CANON-001"

        building = Building(name="Identity Building")
        floor = building.create_floor(name="Ground Floor")
        floor.zones.append(make_zone("zone-a", floor.id, x=0.0, y=0.0, width=10.0, height=10.0))
        floor.add_smoke_detector(
            SmokeDetector(id=detector_id, name="Lobby Smoke", floor_id=floor.id, position=(1.0, 1.0), zone_ids=("zone-a",)),
        )

        reloaded_building = self._run_pipeline(building)
        reloaded_floor = reloaded_building.floors[0]

        self.assertEqual(reloaded_floor.smoke_detectors[0].id, detector_id)

        manager = SensorManager()
        manager.discover_sensors(reloaded_building)
        sensor_status = manager.sensor_status(detector_id)
        self.assertEqual(sensor_status.sensor_id, detector_id)

        hazard_snapshot = HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.9)})

        smoke_provider = GroundTruthSmokeDetectorProvider(
            detectors=[reloaded_floor.smoke_detectors[0]], zones=reloaded_floor.zones,
            hazard_provider=ManualHazardProvider(hazard_snapshot),
        )
        readings = smoke_provider.alarm_states_at(time=0.0)
        self.assertEqual(readings[0].detector_id, detector_id)

        estimator = BuildingStateEstimator()
        state = estimator.estimate(
            0.0,
            hazard_snapshot=hazard_snapshot,
            occupancy_snapshot=OccupancySnapshot(),
            smoke_detector_statuses=[sensor_status],
            smoke_detector_readings=readings,
        )

        asset = state.smoke_detector_state(detector_id)
        self.assertIsNotNone(asset)
        self.assertEqual(asset.status.sensor_id, detector_id)
        self.assertEqual(asset.reading.detector_id, detector_id)


class ArchitectureGuardTests(unittest.TestCase):

    # Regression guard -- prevents the platform from silently
    # reintroducing a second, independent "legacy Detector -> canonical
    # asset" conversion somewhere else. SensorManager and both Ground
    # Truth detector providers must all delegate to the ONE shared
    # models.detector_migration.adapt_legacy_detector() function rather
    # than each growing their own copy of this logic.

    def test_sensor_manager_uses_the_shared_adapter(self):

        import pathlib

        text = (pathlib.Path(__file__).resolve().parent.parent / "sensor_manager" / "manager.py").read_text()

        self.assertIn("from models.detector_migration import adapt_legacy_detector", text)
        self.assertIn("adapt_legacy_detector(", text)

    def test_ground_truth_detector_providers_use_the_shared_adapter(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "perception" / "providers"

        for filename in ("ground_truth_smoke_detector_provider.py", "ground_truth_heat_detector_provider.py"):

            text = (package_dir / filename).read_text()

            self.assertIn("from models.detector_migration import adapt_legacy_detector", text)
            self.assertIn("adapt_legacy_detector(", text)

    def test_only_one_detector_migration_module_exists(self):

        import pathlib

        matches = list(pathlib.Path(__file__).resolve().parent.parent.rglob("detector_migration.py"))

        # Excludes anything under a virtual environment / site-packages
        # that might incidentally match, and this repo's own stray
        # worktree checkout, so this guard only ever counts real
        # project source.
        matches = [
            path for path in matches
            if ".venv" not in path.parts and "site-packages" not in path.parts
            and "worktrees" not in path.parts
        ]

        self.assertEqual(len(matches), 1, f"Expected exactly one detector_migration module, found: {matches}")


if __name__ == "__main__":
    unittest.main()

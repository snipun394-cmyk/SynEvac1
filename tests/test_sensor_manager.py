import unittest

from models.building import Building
from models.heat_detector import HeatDetector
from models.sensor_asset import DetectorState, HealthStatus
from models.smoke_detector import SmokeDetector

from sensor_manager.manager import SensorManager
from sensor_manager.status import SensorStatus


def make_smoke(name, floor_id, zone_ids=(), **overrides):

    fields = dict(name=name, floor_id=floor_id, zone_ids=tuple(zone_ids))
    fields.update(overrides)

    return SmokeDetector(**fields)


def make_heat(name, floor_id, zone_ids=(), **overrides):

    fields = dict(name=name, floor_id=floor_id, zone_ids=tuple(zone_ids))
    fields.update(overrides)

    return HeatDetector(**fields)


class DiscoveryTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor_1 = self.building.create_floor(name="Ground Floor")
        self.floor_2 = self.building.create_floor(name="Floor 1", height=3.0)

        self.smoke_1 = make_smoke("Smoke 1", self.floor_1.id)
        self.heat_1 = make_heat("Heat 1", self.floor_2.id)

        self.floor_1.add_smoke_detector(self.smoke_1)
        self.floor_2.add_heat_detector(self.heat_1)

        self.manager = SensorManager()

    def test_discover_sensors_finds_both_types_across_every_floor(self):

        discovered = self.manager.discover_sensors(self.building)

        self.assertEqual(
            {sensor.id for sensor in discovered}, {self.smoke_1.id, self.heat_1.id},
        )
        self.assertEqual(len(self.manager.all_sensors()), 2)

    def test_discover_with_no_building_clears_the_registry(self):

        self.manager.discover_sensors(self.building)
        self.assertEqual(len(self.manager.all_sensors()), 2)

        result = self.manager.discover_sensors(None)

        self.assertEqual(result, ())
        self.assertEqual(self.manager.all_sensors(), ())

    def test_rediscovering_picks_up_a_newly_added_sensor(self):

        self.manager.discover_sensors(self.building)

        smoke_2 = make_smoke("Smoke 2", self.floor_1.id)
        self.floor_1.add_smoke_detector(smoke_2)

        self.manager.discover_sensors(self.building)

        self.assertEqual(len(self.manager.all_sensors()), 3)
        self.assertIs(self.manager.get_sensor(smoke_2.id), smoke_2)

    def test_rediscovering_drops_a_removed_sensor(self):

        self.manager.discover_sensors(self.building)

        self.floor_1.remove_smoke_detector(self.smoke_1)

        self.manager.discover_sensors(self.building)

        self.assertIsNone(self.manager.get_sensor(self.smoke_1.id))
        self.assertEqual(len(self.manager.all_sensors()), 1)


class RegistrationAndLookupTests(unittest.TestCase):

    def setUp(self):

        self.manager = SensorManager()

        self.smoke = make_smoke("Smoke", "floor-1", zone_ids=("zone-1",))
        self.heat = make_heat("Heat", "floor-1", zone_ids=("zone-2",))
        self.smoke_other_floor = make_smoke("Smoke 2", "floor-2", zone_ids=("zone-1",))

        for sensor in (self.smoke, self.heat, self.smoke_other_floor):
            self.manager.register_sensor(sensor)

    def test_get_sensor(self):

        self.assertIs(self.manager.get_sensor(self.smoke.id), self.smoke)
        self.assertIsNone(self.manager.get_sensor("no-such-id"))

    def test_remove_sensor(self):

        self.manager.remove_sensor(self.smoke.id)
        self.assertIsNone(self.manager.get_sensor(self.smoke.id))

    def test_remove_unknown_sensor_is_a_no_op(self):

        self.manager.remove_sensor("does-not-exist")

    def test_sensors_on_floor(self):

        ids_on_floor_1 = {s.id for s in self.manager.sensors_on_floor("floor-1")}
        self.assertEqual(ids_on_floor_1, {self.smoke.id, self.heat.id})
        self.assertEqual(self.manager.sensors_on_floor("floor-2"), (self.smoke_other_floor,))

    def test_sensors_in_zone(self):

        ids_in_zone_1 = {s.id for s in self.manager.sensors_in_zone("zone-1")}
        self.assertEqual(ids_in_zone_1, {self.smoke.id, self.smoke_other_floor.id})
        self.assertEqual(self.manager.sensors_in_zone("zone-2"), (self.heat,))
        self.assertEqual(self.manager.sensors_in_zone("no-such-zone"), ())


class EnableDisableTests(unittest.TestCase):

    def test_disable_then_enable(self):

        manager = SensorManager()
        sensor = make_smoke("Smoke", "floor-1", active=True)
        manager.register_sensor(sensor)

        manager.disable_sensor(sensor.id)
        self.assertFalse(sensor.active)

        manager.enable_sensor(sensor.id)
        self.assertTrue(sensor.active)

    def test_enable_unknown_sensor_raises(self):

        manager = SensorManager()

        with self.assertRaises(KeyError):
            manager.enable_sensor("no-such-sensor")

    def test_disable_unknown_sensor_raises(self):

        manager = SensorManager()

        with self.assertRaises(KeyError):
            manager.disable_sensor("no-such-sensor")


class GroupingTests(unittest.TestCase):

    def test_sensors_by_type_groups_correctly(self):

        manager = SensorManager()

        smoke_1 = make_smoke("Smoke 1", "floor-1")
        smoke_2 = make_smoke("Smoke 2", "floor-1")
        heat_1 = make_heat("Heat 1", "floor-1")

        for sensor in (smoke_1, smoke_2, heat_1):
            manager.register_sensor(sensor)

        grouped = manager.sensors_by_type()

        self.assertEqual(set(grouped.keys()), {"SmokeDetector", "HeatDetector"})
        self.assertEqual(
            {sensor.id for sensor in grouped["SmokeDetector"]}, {smoke_1.id, smoke_2.id},
        )
        self.assertEqual(grouped["HeatDetector"], (heat_1,))


class StatusTests(unittest.TestCase):

    def setUp(self):

        self.manager = SensorManager()
        self.sensor = make_smoke(
            "Smoke", "floor-1", zone_ids=("zone-1",), active=True,
            health_status=HealthStatus.OK,
        )
        self.manager.register_sensor(self.sensor)

    def test_sensor_status_reflects_current_fields(self):

        status = self.manager.sensor_status(self.sensor.id)

        self.assertIsInstance(status, SensorStatus)
        self.assertEqual(status.sensor_id, self.sensor.id)
        self.assertEqual(status.sensor_type, "SmokeDetector")
        self.assertEqual(status.name, "Smoke")
        self.assertEqual(status.floor_id, "floor-1")
        self.assertEqual(status.zone_ids, ("zone-1",))
        self.assertTrue(status.active)
        self.assertEqual(status.health_status, HealthStatus.OK)
        self.assertIsNone(status.current_state)

    def test_sensor_status_carries_a_supplied_current_state(self):

        status = self.manager.sensor_status(self.sensor.id, current_state=DetectorState.ALARM)

        self.assertEqual(status.current_state, DetectorState.ALARM)

    def test_status_for_unknown_sensor_raises(self):

        with self.assertRaises(KeyError):
            self.manager.sensor_status("no-such-sensor")

    def test_all_statuses_covers_every_sensor_and_applies_supplied_states(self):

        sensor_2 = make_smoke("Smoke 2", "floor-1")
        self.manager.register_sensor(sensor_2)

        statuses = self.manager.all_statuses(states={self.sensor.id: DetectorState.ALARM})

        by_id = {status.sensor_id: status for status in statuses}
        self.assertEqual(by_id[self.sensor.id].current_state, DetectorState.ALARM)
        self.assertIsNone(by_id[sensor_2.id].current_state)


class AggregateAlarmStateTests(unittest.TestCase):

    def setUp(self):

        self.manager = SensorManager()

        self.sensor_1 = make_smoke("Smoke 1", "floor-1")
        self.sensor_2 = make_heat("Heat 1", "floor-1")

        self.manager.register_sensor(self.sensor_1)
        self.manager.register_sensor(self.sensor_2)

    def test_all_normal(self):

        states = {self.sensor_1.id: DetectorState.NORMAL, self.sensor_2.id: DetectorState.NORMAL}

        self.assertEqual(self.manager.aggregate_alarm_state(states), DetectorState.NORMAL)

    def test_any_alarm_wins(self):

        states = {self.sensor_1.id: DetectorState.ALARM, self.sensor_2.id: DetectorState.FAULT}

        self.assertEqual(self.manager.aggregate_alarm_state(states), DetectorState.ALARM)

    def test_fault_outranks_normal_when_no_alarm(self):

        states = {self.sensor_1.id: DetectorState.FAULT, self.sensor_2.id: DetectorState.NORMAL}

        self.assertEqual(self.manager.aggregate_alarm_state(states), DetectorState.FAULT)

    def test_empty_states_mapping_is_normal(self):

        self.assertEqual(self.manager.aggregate_alarm_state({}), DetectorState.NORMAL)

    def test_states_for_unregistered_sensors_are_ignored(self):

        states = {"not-a-real-sensor": DetectorState.ALARM}

        self.assertEqual(self.manager.aggregate_alarm_state(states), DetectorState.NORMAL)


class BackwardCompatibilityTests(unittest.TestCase):

    def test_discovers_a_legacy_shaped_smoke_detector(self):

        from models.floor import Floor

        floor_data = {
            "id": "floor-legacy", "name": "Ground Floor", "display_order": 0, "height": 3.0,
            "floor_plan": "", "visible": True, "locked": False,
            "zones": [], "exits": [], "stairs": [], "elevators": [], "cameras": [], "detectors": [],
            "smoke_detectors": [
                {
                    "id": "smoke-legacy", "name": "Legacy Smoke", "object_type": "SmokeDetector",
                    "properties": {}, "created_at": "", "modified_at": "",
                    "floor_id": "floor-legacy", "position": (1.0, 1.0), "active": True,
                }
            ],
            "heat_detectors": [],
            "assembly_points": [], "obstacles": [], "doors": [],
        }

        floor = Floor.from_dict(floor_data)

        building = Building(name="Legacy Building")
        building.floors.append(floor)

        manager = SensorManager()
        manager.discover_sensors(building)

        self.assertEqual(len(manager.all_sensors()), 1)
        status = manager.sensor_status("smoke-legacy")
        self.assertEqual(status.health_status, HealthStatus.OK)
        self.assertTrue(status.active)


class SensorManagerPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase enforces -- SensorManager must mirror
    # CameraManager's design principles WITHOUT coupling to it (Phase
    # 5's own explicit instruction), and must never perform hazard/
    # physics computation itself.

    def test_never_imports_camera_manager_or_simulation_internals(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "sensor_manager"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(camera_manager|virtual_camera|visibility|multi_camera_fusion|camera_validation|"
            r"simulator|ground_truth|behavior|behavior_library|behaviour_profile_resolver|"
            r"simulation_runtime|hazard|hazard_evolution|ai_training|rl_training|"
            r"advisory_system|command_center|designer)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"sensor_manager/{path.name} imports a camera-runtime, simulation, or "
                f"hazard-computation module directly -- it must only manage sensor assets "
                f"and aggregate already-computed DetectorState values",
            )


if __name__ == "__main__":
    unittest.main()

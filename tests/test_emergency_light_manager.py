import unittest

from models.building import Building
from models.emergency_light import EmergencyLight, EmergencyLightAvailability
from models.floor import Floor
from models.sensor_asset import HealthStatus

from emergency_light_manager.manager import EmergencyLightManager


def _building_with(lights):

    floor = Floor(id="f1", name="F1")

    for light in lights:
        floor.add_emergency_light(light)

    return Building(id="b1", name="B", floors=[floor])


class DiscoveryTests(unittest.TestCase):

    def test_discovers_every_light_across_floors(self):

        f1 = Floor(id="f1", name="F1")
        f1.add_emergency_light(EmergencyLight(id="E1", name="E1", floor_id="f1"))
        f2 = Floor(id="f2", name="F2")
        f2.add_emergency_light(EmergencyLight(id="E2", name="E2", floor_id="f2"))
        building = Building(id="b1", name="B", floors=[f1, f2])

        manager = EmergencyLightManager()
        lights = manager.discover_lights(building)

        self.assertEqual({l.id for l in lights}, {"E1", "E2"})

    def test_no_building_returns_empty(self):

        manager = EmergencyLightManager()
        self.assertEqual(manager.discover_lights(None), ())

    def test_rediscovery_drops_removed_light(self):

        building = _building_with([EmergencyLight(id="E1", name="E1", floor_id="f1")])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        building.floors[0].emergency_lights = []
        manager.discover_lights(building)

        self.assertEqual(manager.all_lights(), ())


class LookupTests(unittest.TestCase):

    def test_lights_on_floor(self):

        building = _building_with([EmergencyLight(id="E1", name="E1", floor_id="f1")])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        self.assertEqual({l.id for l in manager.lights_on_floor("f1")}, {"E1"})

    def test_lights_in_zone(self):

        building = _building_with([EmergencyLight(id="E1", name="E1", floor_id="f1", zone_ids=("z1",))])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        self.assertEqual({l.id for l in manager.lights_in_zone("z1")}, {"E1"})
        self.assertEqual(manager.lights_in_zone("z2"), ())

    def test_get_light_unknown_returns_none(self):

        manager = EmergencyLightManager()
        self.assertIsNone(manager.get_light("nope"))


class EnableDisableTests(unittest.TestCase):

    def test_disable_then_enable(self):

        building = _building_with([EmergencyLight(id="E1", name="E1", floor_id="f1")])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        manager.disable_light("E1")
        self.assertFalse(manager.get_light("E1").active)

        manager.enable_light("E1")
        self.assertTrue(manager.get_light("E1").active)

    def test_disable_unknown_raises(self):

        manager = EmergencyLightManager()
        with self.assertRaises(KeyError):
            manager.disable_light("nope")


class StatusAndAvailabilityTests(unittest.TestCase):

    def test_status_reflects_availability(self):

        building = _building_with([EmergencyLight(id="E1", name="E1", floor_id="f1", zone_ids=("z1",))])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        status = manager.light_status("E1")

        self.assertEqual(status.light_id, "E1")
        self.assertEqual(status.zone_ids, ("z1",))
        self.assertEqual(status.availability, EmergencyLightAvailability.AVAILABLE)

    def test_available_lights_excludes_unavailable_and_fault(self):

        building = _building_with([
            EmergencyLight(id="E1", name="E1", floor_id="f1", active=True),
            EmergencyLight(id="E2", name="E2", floor_id="f1", active=False),
            EmergencyLight(id="E3", name="E3", floor_id="f1", health_status=HealthStatus.FAULT),
        ])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        available_ids = {l.id for l in manager.available_lights()}
        self.assertEqual(available_ids, {"E1"})

    def test_all_statuses_covers_every_light(self):

        building = _building_with([
            EmergencyLight(id="E1", name="E1", floor_id="f1"),
            EmergencyLight(id="E2", name="E2", floor_id="f1"),
        ])
        manager = EmergencyLightManager()
        manager.discover_lights(building)

        self.assertEqual(len(manager.all_statuses()), 2)

    def test_status_of_unknown_light_raises(self):

        manager = EmergencyLightManager()
        with self.assertRaises(KeyError):
            manager.light_status("nope")


if __name__ == "__main__":
    unittest.main()

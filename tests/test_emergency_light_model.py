import unittest

from models.emergency_light import EmergencyLight, EmergencyLightAvailability
from models.floor import Floor
from models.sensor_asset import HealthStatus


class EmergencyLightModelTests(unittest.TestCase):

    def test_default_is_available(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1")

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.AVAILABLE)

    def test_inactive_is_unavailable(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1", active=False)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.UNAVAILABLE)

    def test_offline_health_is_unavailable(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1", health_status=HealthStatus.OFFLINE)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.UNAVAILABLE)

    def test_fault_health_is_fault(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1", health_status=HealthStatus.FAULT)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.FAULT)

    def test_fault_outranks_inactive(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1", active=False, health_status=HealthStatus.FAULT)

        self.assertEqual(light.compute_availability(), EmergencyLightAvailability.FAULT)

    def test_object_type_is_emergency_light(self):

        light = EmergencyLight(id="E1", name="E1", floor_id="f1")

        self.assertEqual(light.object_type, "EmergencyLight")

    def test_not_a_sensor_asset(self):

        from models.sensor_asset import SensorAsset

        self.assertFalse(issubclass(EmergencyLight, SensorAsset))


class EmergencyLightSerializationTests(unittest.TestCase):

    def test_round_trip_preserves_light_type_and_zone(self):

        light = EmergencyLight(
            id="E1", name="E1", floor_id="f1", zone_ids=("z1",), position=(2.0, 3.0), light_type="Ceiling Mounted",
        )

        restored = EmergencyLight.from_dict(light.to_dict())

        self.assertEqual(restored.id, "E1")
        self.assertEqual(restored.zone_ids, ("z1",))
        self.assertEqual(restored.light_type, "Ceiling Mounted")

    def test_missing_light_type_key_defaults(self):

        data = EmergencyLight(id="E1", name="E1", floor_id="f1", light_type="Recessed").to_dict()
        del data["light_type"]

        restored = EmergencyLight.from_dict(data)

        self.assertEqual(restored.light_type, "Wall Mounted")

    def test_floor_round_trip(self):

        floor = Floor(id="f1", name="F1")
        floor.add_emergency_light(EmergencyLight(id="E1", name="E1", floor_id="f1"))

        restored = Floor.from_dict(floor.to_dict())

        self.assertEqual(restored.emergency_light_count, 1)
        self.assertEqual(restored.emergency_lights[0].id, "E1")

    def test_legacy_project_without_emergency_lights_key_loads(self):

        floor = Floor(id="f1", name="F1")
        data = floor.to_dict()
        del data["emergency_lights"]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.emergency_lights, [])
        self.assertEqual(restored.emergency_light_count, 0)


if __name__ == "__main__":
    unittest.main()

import unittest

from models.floor import Floor
from models.manual_call_point import ManualCallPoint
from models.sensor_asset import DetectorState, HealthStatus


class ManualCallPointModelTests(unittest.TestCase):

    def test_default_state_is_normal(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1")

        self.assertEqual(mcp.compute_state(), DetectorState.NORMAL)
        self.assertFalse(mcp.activated)

    def test_activate_produces_alarm(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1")

        mcp.activate()

        self.assertTrue(mcp.activated)
        self.assertEqual(mcp.compute_state(), DetectorState.ALARM)

    def test_activate_records_last_activation_time(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1")

        mcp.activate()
        mcp.compute_state(time=42.0)

        self.assertEqual(mcp.last_activation_time, 42.0)

    def test_restore_clears_activation(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1")

        mcp.activate()
        self.assertEqual(mcp.compute_state(), DetectorState.ALARM)

        mcp.restore()

        self.assertFalse(mcp.activated)
        self.assertEqual(mcp.compute_state(), DetectorState.NORMAL)

    def test_fault_health_outranks_activation(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", health_status=HealthStatus.FAULT)

        mcp.activate()

        self.assertEqual(mcp.compute_state(), DetectorState.FAULT)

    def test_inactive_device_never_alarms(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", active=False)

        mcp.activate()

        self.assertEqual(mcp.compute_state(), DetectorState.NORMAL)

    def test_object_type_is_manual_call_point(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1")

        self.assertEqual(mcp.object_type, "ManualCallPoint")


class ManualCallPointSerializationTests(unittest.TestCase):

    def test_round_trip_preserves_activated_and_zone(self):

        mcp = ManualCallPoint(
            id="M1", name="M1", floor_id="f1", zone_ids=("z1",), position=(3.0, 4.0), activated=True,
        )

        restored = ManualCallPoint.from_dict(mcp.to_dict())

        self.assertEqual(restored.id, "M1")
        self.assertEqual(restored.zone_ids, ("z1",))
        self.assertEqual(restored.position, (3.0, 4.0))
        self.assertTrue(restored.activated)

    def test_missing_activated_key_defaults_to_false(self):

        data = ManualCallPoint(id="M1", name="M1", floor_id="f1", activated=True).to_dict()
        del data["activated"]

        restored = ManualCallPoint.from_dict(data)

        self.assertFalse(restored.activated)

    def test_floor_round_trip(self):

        floor = Floor(id="f1", name="F1")
        floor.add_manual_call_point(ManualCallPoint(id="M1", name="M1", floor_id="f1"))

        restored = Floor.from_dict(floor.to_dict())

        self.assertEqual(restored.manual_call_point_count, 1)
        self.assertEqual(restored.manual_call_points[0].id, "M1")

    def test_legacy_project_without_manual_call_points_key_loads(self):

        floor = Floor(id="f1", name="F1")
        data = floor.to_dict()
        del data["manual_call_points"]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.manual_call_points, [])
        self.assertEqual(restored.manual_call_point_count, 0)


if __name__ == "__main__":
    unittest.main()

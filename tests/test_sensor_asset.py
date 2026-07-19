import unittest

from models.engineering_asset import DeviceMode
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.sensor_asset import DetectorState, HealthStatus, SensorAsset
from models.smoke_detector import SmokeDetector


class SensorAssetTests(unittest.TestCase):

    def test_defaults(self):

        sensor = SensorAsset()

        self.assertEqual(sensor.health_status, HealthStatus.OK)
        self.assertEqual(sensor.installation_date, "")
        self.assertIsNone(sensor.last_activation_time)

    def test_reuses_engineering_asset_fields_without_redeclaring_them(self):

        sensor = SensorAsset(
            floor_id="floor-1", zone_ids=("zone-1",), position=(2.0, 3.0),
            active=False, mode=DeviceMode.LIVE,
        )

        self.assertEqual(sensor.floor_id, "floor-1")
        self.assertEqual(sensor.zone_ids, ("zone-1",))
        self.assertEqual(sensor.position, (2.0, 3.0))
        self.assertFalse(sensor.active)
        self.assertEqual(sensor.mode, DeviceMode.LIVE)

    def test_sensor_dict_round_trip_via_kwargs(self):

        sensor = SensorAsset(
            name="Sensor", floor_id="floor-1", health_status=HealthStatus.FAULT,
            installation_date="2026-01-01", last_activation_time=42.0,
        )

        data = sensor._sensor_dict()
        kwargs = SensorAsset._sensor_kwargs(data)
        restored = SensorAsset(**kwargs)

        self.assertEqual(restored.health_status, HealthStatus.FAULT)
        self.assertEqual(restored.installation_date, "2026-01-01")
        self.assertEqual(restored.last_activation_time, 42.0)


class SmokeDetectorTests(unittest.TestCase):

    def test_defaults(self):

        detector = SmokeDetector()

        self.assertEqual(detector.object_type, "SmokeDetector")
        self.assertEqual(detector.activation_threshold, 0.2)

    def test_normal_state_below_threshold(self):

        detector = SmokeDetector(activation_threshold=0.2)

        self.assertEqual(detector.compute_state(0.1), DetectorState.NORMAL)

    def test_alarm_state_at_or_above_threshold(self):

        detector = SmokeDetector(activation_threshold=0.2)

        self.assertEqual(detector.compute_state(0.2), DetectorState.ALARM)
        self.assertEqual(detector.compute_state(0.9), DetectorState.ALARM)

    def test_alarm_records_last_activation_time(self):

        detector = SmokeDetector(activation_threshold=0.2)

        self.assertIsNone(detector.last_activation_time)

        detector.compute_state(0.5, time=12.5)

        self.assertEqual(detector.last_activation_time, 12.5)

    def test_alarm_without_a_time_argument_does_not_record_activation(self):

        detector = SmokeDetector(activation_threshold=0.2)

        detector.compute_state(0.5)

        self.assertIsNone(detector.last_activation_time)

    def test_no_reading_is_normal_not_fault(self):

        detector = SmokeDetector()

        self.assertEqual(detector.compute_state(None), DetectorState.NORMAL)

    def test_inactive_detector_is_normal_regardless_of_reading(self):

        detector = SmokeDetector(active=False, activation_threshold=0.1)

        self.assertEqual(detector.compute_state(0.9), DetectorState.NORMAL)

    def test_faulty_health_status_always_reports_fault(self):

        detector = SmokeDetector(health_status=HealthStatus.FAULT, activation_threshold=0.1)

        self.assertEqual(detector.compute_state(0.9), DetectorState.FAULT)
        self.assertEqual(detector.compute_state(0.0), DetectorState.FAULT)

    def test_offline_health_status_always_reports_fault(self):

        detector = SmokeDetector(health_status=HealthStatus.OFFLINE)

        self.assertEqual(detector.compute_state(0.9), DetectorState.FAULT)

    def test_to_dict_from_dict_round_trip(self):

        detector = SmokeDetector(
            name="S1", floor_id="floor-1", zone_ids=("zone-1",), position=(1.0, 2.0),
            activation_threshold=0.35, health_status=HealthStatus.FAULT,
            installation_date="2026-02-02", last_activation_time=5.0,
        )

        restored = SmokeDetector.from_dict(detector.to_dict())

        self.assertEqual(restored.name, "S1")
        self.assertEqual(restored.floor_id, "floor-1")
        self.assertEqual(restored.zone_ids, ("zone-1",))
        self.assertEqual(restored.position, (1.0, 2.0))
        self.assertEqual(restored.activation_threshold, 0.35)
        self.assertEqual(restored.health_status, HealthStatus.FAULT)
        self.assertEqual(restored.installation_date, "2026-02-02")
        self.assertEqual(restored.last_activation_time, 5.0)

    def test_from_dict_defaults_missing_fields(self):

        detector = SmokeDetector.from_dict({"id": "s-1"})

        self.assertEqual(detector.activation_threshold, 0.2)
        self.assertEqual(detector.health_status, HealthStatus.OK)
        self.assertIsNone(detector.last_activation_time)


class HeatDetectorTests(unittest.TestCase):

    def test_defaults(self):

        detector = HeatDetector()

        self.assertEqual(detector.object_type, "HeatDetector")
        self.assertEqual(detector.activation_threshold, 57.0)

    def test_normal_below_threshold_alarm_at_or_above(self):

        detector = HeatDetector(activation_threshold=57.0)

        self.assertEqual(detector.compute_state(20.0), DetectorState.NORMAL)
        self.assertEqual(detector.compute_state(57.0), DetectorState.ALARM)
        self.assertEqual(detector.compute_state(100.0), DetectorState.ALARM)

    def test_faulty_health_status_always_reports_fault(self):

        detector = HeatDetector(health_status=HealthStatus.FAULT)

        self.assertEqual(detector.compute_state(20.0), DetectorState.FAULT)

    def test_to_dict_from_dict_round_trip(self):

        detector = HeatDetector(name="H1", floor_id="floor-1", activation_threshold=65.0)

        restored = HeatDetector.from_dict(detector.to_dict())

        self.assertEqual(restored.name, "H1")
        self.assertEqual(restored.activation_threshold, 65.0)


class FloorSensorIntegrationTests(unittest.TestCase):

    def test_floor_stores_and_round_trips_both_sensor_types(self):

        floor = Floor(name="Ground Floor")

        smoke = SmokeDetector(name="Smoke 1", floor_id=floor.id, activation_threshold=0.3)
        heat = HeatDetector(name="Heat 1", floor_id=floor.id, activation_threshold=60.0)

        floor.add_smoke_detector(smoke)
        floor.add_heat_detector(heat)

        self.assertEqual(floor.smoke_detector_count, 1)
        self.assertEqual(floor.heat_detector_count, 1)

        restored_floor = Floor.from_dict(floor.to_dict())

        self.assertEqual(len(restored_floor.smoke_detectors), 1)
        self.assertEqual(len(restored_floor.heat_detectors), 1)
        self.assertEqual(restored_floor.smoke_detectors[0].activation_threshold, 0.3)
        self.assertEqual(restored_floor.heat_detectors[0].activation_threshold, 60.0)

    def test_remove_sensors(self):

        floor = Floor(name="Ground Floor")

        smoke = SmokeDetector(name="Smoke 1", floor_id=floor.id)
        floor.add_smoke_detector(smoke)

        floor.remove_smoke_detector(smoke)

        self.assertEqual(floor.smoke_detector_count, 0)

    def test_existing_generic_detector_list_is_unaffected(self):

        # The pre-existing generic Detector class/list must keep working
        # exactly as before -- this milestone is additive, not a
        # replacement.
        from models.detector import Detector

        floor = Floor(name="Ground Floor")
        floor.add_detector(Detector(name="Legacy Smoke", detector_type="Smoke"))

        self.assertEqual(floor.detector_count, 1)
        self.assertEqual(floor.smoke_detector_count, 0)

    def test_legacy_floor_dict_without_sensor_keys_still_loads(self):

        floor_data = {
            "id": "floor-legacy",
            "name": "Ground Floor",
            "display_order": 0,
            "height": 3.0,
            "floor_plan": "",
            "visible": True,
            "locked": False,
            "zones": [], "exits": [], "stairs": [], "elevators": [],
            "cameras": [], "detectors": [],
            "assembly_points": [], "obstacles": [], "doors": [],
        }

        floor = Floor.from_dict(floor_data)

        self.assertEqual(floor.smoke_detectors, [])
        self.assertEqual(floor.heat_detectors, [])


if __name__ == "__main__":
    unittest.main()

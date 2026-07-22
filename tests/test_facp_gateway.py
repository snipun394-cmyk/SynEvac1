import unittest

from facp.engine import SimulatedFACP
from facp.models import PanelState

from sensor_manager.manager import SensorManager
from sensor_manager.status import SensorStatus

from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.models.heat_detector_observation import HeatDetectorReading

from live_system.facp_gateway import EngineFACPGateway


class _FakeSensorManager:

    def __init__(self, statuses):
        self._statuses = statuses

    def all_statuses(self):
        return tuple(self._statuses)


def _status(sensor_id, sensor_type, zone_ids=("z1",), floor_id="f1", health="OK"):

    return SensorStatus(
        sensor_id=sensor_id, sensor_type=sensor_type, name=sensor_id, floor_id=floor_id,
        zone_ids=zone_ids, active=True, mode="Simulation", health_status=health,
    )


class FACPGatewayBasicTests(unittest.TestCase):

    def test_evaluate_reaches_alarm_from_a_smoke_reading(self):

        facp = SimulatedFACP()
        sensor_manager = _FakeSensorManager([_status("SD-1", "SmokeDetector")])

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=True)]

        gateway = EngineFACPGateway(facp, sensor_manager, smoke_detector_reading_provider=smoke_readings)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIn("SD-1", facp.active_alarm_source_ids)

    def test_evaluate_reaches_alarm_from_a_heat_reading(self):

        facp = SimulatedFACP()
        sensor_manager = _FakeSensorManager([_status("HD-1", "HeatDetector")])

        def heat_readings(time):
            return [HeatDetectorReading(detector_id="HD-1", timestamp=time, alarm_active=True)]

        gateway = EngineFACPGateway(facp, sensor_manager, heat_detector_reading_provider=heat_readings)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIn("HD-1", facp.active_alarm_source_ids)

    def test_fault_health_status_reaches_fault(self):

        facp = SimulatedFACP()
        sensor_manager = _FakeSensorManager([_status("SD-1", "SmokeDetector", health="Fault")])

        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.FAULT)

    def test_no_reading_provider_stays_normal(self):

        facp = SimulatedFACP()
        sensor_manager = _FakeSensorManager([_status("SD-1", "SmokeDetector")])

        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)

    def test_no_sensors_at_all_stays_normal(self):

        facp = SimulatedFACP()
        sensor_manager = _FakeSensorManager([])

        gateway = EngineFACPGateway(facp, sensor_manager)
        result = gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)
        self.assertEqual(result, ())

    def test_evaluate_never_raises_on_a_broken_sensor_manager(self):

        class BrokenSensorManager:
            def all_statuses(self):
                raise RuntimeError("boom")

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, BrokenSensorManager())

        result = gateway.evaluate(1.0)

        self.assertIsNone(result)
        # The panel itself must be left completely untouched by a
        # failed evaluation attempt -- never a partially-applied state.
        self.assertEqual(facp.panel_state, PanelState.NORMAL)

    def test_never_calls_acknowledge_silence_or_reset(self):

        # Structural guard: EngineFACPGateway must never call anything
        # but evaluate() on the panel.
        import inspect

        source = inspect.getsource(EngineFACPGateway)

        self.assertNotIn(".acknowledge(", source)
        self.assertNotIn(".silence(", source)
        self.assertNotIn(".reset(", source)

    def test_uses_a_real_sensor_manager_instance_too(self):

        # Not just a fake -- confirm the gateway works against the
        # genuine SensorManager class as well.
        sensor_manager = SensorManager()
        facp = SimulatedFACP()

        gateway = EngineFACPGateway(facp, sensor_manager)
        result = gateway.evaluate(1.0)

        self.assertEqual(result, ())
        self.assertEqual(facp.panel_state, PanelState.NORMAL)


if __name__ == "__main__":
    unittest.main()

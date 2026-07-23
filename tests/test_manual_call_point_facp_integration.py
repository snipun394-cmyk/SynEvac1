import unittest

from models.building import Building
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.manual_call_point import ManualCallPoint
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from sensor_manager.manager import SensorManager

from facp.engine import SimulatedFACP
from facp.models import PanelState

from live_system.facp_gateway import EngineFACPGateway


# =====================================================
# Manual Call Points & Emergency Lighting milestone, Phase 3/13 -- MCP
# -> FACP integration and mixed detector/MCP alarm source behavior.
# =====================================================


def _building_with(mcps=(), smoke=(), heat=()):

    floor = Floor(id="f1", name="F1")
    floor.add_zone(Zone(id="z1", name="Z1", floor_id="f1", width=5.0, height=5.0))

    for mcp in mcps:
        floor.add_manual_call_point(mcp)

    for sd in smoke:
        floor.add_smoke_detector(sd)

    for hd in heat:
        floor.add_heat_detector(hd)

    return Building(id="b1", name="B", floors=[floor])


class MCPBasicFACPIntegrationTests(unittest.TestCase):

    def test_normal_mcp_produces_no_alarm_source(self):

        building = _building_with(mcps=[ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)
        self.assertEqual(facp.active_alarm_source_ids, ())

    def test_activated_mcp_produces_facp_alarm(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()

        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIn("M1", facp.active_alarm_source_ids)

    def test_acknowledge_after_mcp_alarm(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        facp.acknowledge(2.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM_ACKNOWLEDGED)

    def test_silence_after_mcp_alarm(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        facp.silence(2.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM_SILENCED)

    def test_mcp_remains_activated_reset_cannot_falsely_clear(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        facp.reset(2.0)

        # The MCP condition is still genuinely active -- reset() must
        # not force NORMAL while it remains so.
        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_mcp_restored_then_legal_reset_reaches_normal(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        mcp.restore()
        gateway.evaluate(2.0)

        # Restoration alone never auto-clears (latched, per existing
        # FACP semantics) -- still ALARM until an explicit reset().
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        facp.reset(3.0)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)

    def test_new_mcp_activation_while_silenced_re_alerts(self):

        mcp1 = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp2 = ManualCallPoint(id="M2", name="M2", floor_id="f1", zone_ids=("z1",))
        mcp1.activate()

        building = _building_with(mcps=[mcp1, mcp2])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        facp.silence(2.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM_SILENCED)

        mcp2.activate()
        gateway.evaluate(3.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_mcp_fault_without_activation(self):

        from models.sensor_asset import HealthStatus

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",), health_status=HealthStatus.FAULT)
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.FAULT)
        self.assertIn("M1", facp.active_fault_source_ids)

    def test_inactive_mcp_never_alarms(self):

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",), active=False)
        mcp.activate()
        building = _building_with(mcps=[mcp])

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)


class MixedAlarmSourceTests(unittest.TestCase):

    def test_two_mcps_activated_simultaneously(self):

        mcp1 = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp2 = ManualCallPoint(id="M2", name="M2", floor_id="f1", zone_ids=("z1",))
        mcp1.activate()
        mcp2.activate()

        building = _building_with(mcps=[mcp1, mcp2])
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(1.0)

        self.assertEqual(set(facp.active_alarm_source_ids), {"M1", "M2"})

    def test_mcp_and_heat_detector_both_alarm(self):

        from perception.models.heat_detector_observation import HeatDetectorReading

        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))
        mcp.activate()
        heat = HeatDetector(id="H1", name="H1", floor_id="f1", zone_ids=("z1",))

        building = _building_with(mcps=[mcp], heat=[heat])
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        def heat_readings(time):
            return [HeatDetectorReading(detector_id="H1", timestamp=time, alarm_active=True)]

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager, heat_detector_reading_provider=heat_readings)
        gateway.evaluate(1.0)

        self.assertEqual(set(facp.active_alarm_source_ids), {"M1", "H1"})

    def test_smoke_then_mcp_then_smoke_clears_mcp_remains(self):

        from perception.models.smoke_detector_observation import SmokeDetectorReading

        smoke = SmokeDetector(id="S1", name="S1", floor_id="f1", zone_ids=("z1",))
        mcp = ManualCallPoint(id="M1", name="M1", floor_id="f1", zone_ids=("z1",))

        building = _building_with(mcps=[mcp], smoke=[smoke])
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        smoke_active = {"value": True}

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="S1", timestamp=time, alarm_active=smoke_active["value"])]

        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager, smoke_detector_reading_provider=smoke_readings)

        # Cycle 1: SD-1 alarms.
        gateway.evaluate(1.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertEqual(set(facp.active_alarm_source_ids), {"S1"})

        # Cycle 2: MCP-1 activates too -- both source identities present.
        mcp.activate()
        gateway.evaluate(2.0)
        self.assertEqual(set(facp.active_alarm_source_ids), {"S1", "M1"})

        # Cycle 3: smoke clears -- MCP condition remains represented.
        smoke_active["value"] = False
        gateway.evaluate(3.0)
        self.assertEqual(facp.active_alarm_source_ids, ("M1",))
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        # Cycle 4: MCP restores -- panel remains latched (no auto-clear).
        mcp.restore()
        gateway.evaluate(4.0)
        self.assertEqual(facp.active_alarm_source_ids, ())
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        # Legal reset -> NORMAL.
        facp.reset(5.0)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)


if __name__ == "__main__":
    unittest.main()

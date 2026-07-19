import unittest

from hazard.node_state import HazardNodeState
from hazard.provider import ManualHazardProvider
from hazard.snapshot import HazardSnapshot

from models.building import Building
from models.detector import Detector
from models.sensor_asset import DetectorState, HealthStatus
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from perception.providers.ground_truth_heat_detector_provider import GroundTruthHeatDetectorProvider
from perception.providers.ground_truth_smoke_detector_provider import GroundTruthSmokeDetectorProvider

from sensor_manager.manager import SensorManager

from building_state.estimator import BuildingStateEstimator

from facp.engine import SimulatedFACP
from facp.event_log import PanelEventLog
from facp.models import (
    DetectorConditionReport,
    FACPSnapshot,
    InvalidPanelOperation,
    PanelEvent,
    PanelEventType,
    PanelState,
)
from facp.provider import FACPEventProvider


def report(asset_id, state, asset_type="SmokeDetector", floor_id="floor1", zone_ids=("zone-a",)):

    return DetectorConditionReport(
        asset_id=asset_id, asset_type=asset_type, state=state, floor_id=floor_id, zone_ids=zone_ids,
    )


class SingleAndMultipleAlarmTests(unittest.TestCase):

    def test_one_detector_alarm(self):

        facp = SimulatedFACP()
        events = facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, PanelEventType.DETECTOR_ALARM)
        self.assertEqual(events[0].source_asset_id, "SD-001")
        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertEqual(facp.active_alarm_source_ids, ("SD-001",))

    def test_multiple_simultaneous_detector_alarms(self):

        facp = SimulatedFACP()
        events = facp.evaluate(
            {
                "SD-001": report("SD-001", DetectorState.ALARM),
                "SD-002": report("SD-002", DetectorState.ALARM),
                "SD-003": report("SD-003", DetectorState.ALARM),
            },
            time=1.0,
        )

        self.assertEqual(len(events), 3)
        self.assertEqual(facp.active_alarm_source_ids, ("SD-001", "SD-002", "SD-003"))
        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_smoke_and_heat_detector_alarms_together(self):

        facp = SimulatedFACP()
        events = facp.evaluate(
            {
                "SD-001": report("SD-001", DetectorState.ALARM, asset_type="SmokeDetector"),
                "HD-001": report("HD-001", DetectorState.ALARM, asset_type="HeatDetector"),
            },
            time=1.0,
        )

        event_types_by_asset = {event.source_asset_type: event.event_type for event in events}
        self.assertEqual(event_types_by_asset["SmokeDetector"], PanelEventType.DETECTOR_ALARM)
        self.assertEqual(event_types_by_asset["HeatDetector"], PanelEventType.DETECTOR_ALARM)
        self.assertEqual(set(facp.active_alarm_source_ids), {"SD-001", "HD-001"})


class FaultAndRestorationTests(unittest.TestCase):

    def test_detector_fault(self):

        facp = SimulatedFACP()
        events = facp.evaluate({"SD-001": report("SD-001", DetectorState.FAULT)}, time=1.0)

        self.assertEqual(events[0].event_type, PanelEventType.DETECTOR_FAULT)
        self.assertEqual(facp.panel_state, PanelState.FAULT)
        self.assertEqual(facp.active_fault_source_ids, ("SD-001",))

    def test_detector_restoration_emits_restore_event(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)

        events = facp.evaluate({"SD-001": report("SD-001", DetectorState.NORMAL)}, time=2.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, PanelEventType.DETECTOR_RESTORE)
        self.assertEqual(facp.active_alarm_source_ids, ())

    def test_restoration_alone_does_not_clear_the_panel(self):

        # A real panel stays latched in ALARM until an operator
        # explicitly presses Reset -- restoration only clears the
        # underlying condition, never the displayed panel state.
        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)
        facp.evaluate({"SD-001": report("SD-001", DetectorState.NORMAL)}, time=2.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_no_duplicate_events_when_condition_is_unchanged(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)
        events = facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=2.0)

        self.assertEqual(events, ())


class AcknowledgeSilenceResetTests(unittest.TestCase):

    def test_alarm_acknowledgement(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)

        event = facp.acknowledge(2.0, operator_id="op1")

        self.assertEqual(event.event_type, PanelEventType.PANEL_ACKNOWLEDGED)
        self.assertEqual(facp.panel_state, PanelState.ALARM_ACKNOWLEDGED)
        # The underlying fire condition must still be visible.
        self.assertEqual(facp.active_alarm_source_ids, ("SD-001",))

    def test_alarm_silence(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)

        event = facp.silence(2.0)

        self.assertEqual(event.event_type, PanelEventType.ALARM_SILENCED)
        self.assertEqual(facp.panel_state, PanelState.ALARM_SILENCED)
        self.assertEqual(facp.active_alarm_source_ids, ("SD-001",))

    def test_acknowledge_outside_alarm_raises(self):

        facp = SimulatedFACP()

        with self.assertRaises(InvalidPanelOperation):
            facp.acknowledge(1.0)

    def test_silence_outside_alarm_or_acknowledged_raises(self):

        facp = SimulatedFACP()

        with self.assertRaises(InvalidPanelOperation):
            facp.silence(1.0)

    def test_new_alarm_source_re_alerts_a_silenced_panel(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)
        facp.acknowledge(2.0)
        facp.silence(3.0)

        facp.evaluate(
            {"SD-001": report("SD-001", DetectorState.ALARM), "HD-001": report("HD-001", DetectorState.ALARM, asset_type="HeatDetector")},
            time=4.0,
        )

        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_reset_with_no_active_alarm(self):

        facp = SimulatedFACP()
        event = facp.reset(1.0)

        self.assertEqual(event.event_type, PanelEventType.SYSTEM_RESET)
        self.assertEqual(event.panel_state_after, PanelState.NORMAL)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)

    def test_reset_while_an_alarm_condition_remains_active(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)

        event = facp.reset(2.0)

        # Must NOT be forced to NORMAL -- the source is still alarming.
        self.assertEqual(event.panel_state_after, PanelState.ALARM)
        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertEqual(facp.active_alarm_source_ids, ("SD-001",))

    def test_reset_clears_ack_and_silence_flags(self):

        facp = SimulatedFACP()
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)
        facp.evaluate({"SD-001": report("SD-001", DetectorState.NORMAL)}, time=2.0)

        facp.reset(3.0)

        self.assertEqual(facp.panel_state, PanelState.NORMAL)
        # A fresh alarm after reset must behave like a brand new alarm.
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=4.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

    def test_manual_alarm_and_reset_clears_the_manual_latch(self):

        facp = SimulatedFACP()
        event = facp.manual_alarm("MCP-001", 1.0, floor_id="floor1", zone_ids=("zone-a",))

        self.assertEqual(event.event_type, PanelEventType.MANUAL_ALARM)
        self.assertEqual(event.source_asset_type, "ManualCallPoint")
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        facp.reset(2.0)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)


class DeterministicOrderingTests(unittest.TestCase):

    def test_events_fire_in_sorted_asset_id_order_regardless_of_input_order(self):

        facp = SimulatedFACP()

        conditions = {
            "SD-003": report("SD-003", DetectorState.ALARM),
            "SD-001": report("SD-001", DetectorState.ALARM),
            "SD-002": report("SD-002", DetectorState.ALARM),
        }

        events = facp.evaluate(conditions, time=1.0)

        self.assertEqual([event.source_asset_id for event in events], ["SD-001", "SD-002", "SD-003"])

    def test_repeated_runs_with_the_same_input_produce_the_same_event_sequence(self):

        conditions = {
            "SD-002": report("SD-002", DetectorState.ALARM),
            "SD-001": report("SD-001", DetectorState.ALARM),
        }

        facp_a = SimulatedFACP()
        facp_b = SimulatedFACP()

        events_a = facp_a.evaluate(conditions, time=1.0)
        events_b = facp_b.evaluate(conditions, time=1.0)

        self.assertEqual(
            [(e.source_asset_id, e.event_type) for e in events_a],
            [(e.source_asset_id, e.event_type) for e in events_b],
        )


class PanelEventLogQueryTests(unittest.TestCase):

    def setUp(self):

        self.facp = SimulatedFACP()
        self.facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM, floor_id="floor1", zone_ids=("zone-a",))}, time=1.0)
        self.facp.evaluate({"HD-001": report("HD-001", DetectorState.FAULT, asset_type="HeatDetector", floor_id="floor2", zone_ids=("zone-b",))}, time=2.0)
        self.facp.acknowledge(3.0)

    def test_query_by_type(self):

        alarms = self.facp.event_log.events_of_type(PanelEventType.DETECTOR_ALARM)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0].source_asset_id, "SD-001")

    def test_query_by_source_asset(self):

        events = self.facp.event_log.events_for_asset("HD-001")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, PanelEventType.DETECTOR_FAULT)

    def test_query_by_floor(self):

        events = self.facp.event_log.events_for_floor("floor2")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_asset_id, "HD-001")

    def test_query_by_zone(self):

        events = self.facp.event_log.events_for_zone("zone-a")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_asset_id, "SD-001")

    def test_query_by_time_range(self):

        events = self.facp.event_log.events_between(1.5, 2.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_asset_id, "HD-001")

    def test_all_events_includes_panel_level_events(self):

        self.assertEqual(len(self.facp.event_log.all_events()), 3)


class FACPSnapshotTests(unittest.TestCase):

    def test_current_snapshot_reflects_state(self):

        facp = SimulatedFACP(panel_id="FACP-X")
        facp.evaluate({"SD-001": report("SD-001", DetectorState.ALARM)}, time=1.0)
        facp.acknowledge(2.0)

        snapshot = facp.current_snapshot(2.0)

        self.assertIsInstance(snapshot, FACPSnapshot)
        self.assertEqual(snapshot.panel_id, "FACP-X")
        self.assertEqual(snapshot.panel_state, PanelState.ALARM_ACKNOWLEDGED)
        self.assertTrue(snapshot.acknowledged)
        self.assertFalse(snapshot.silenced)
        self.assertEqual(snapshot.active_alarm_source_ids, ("SD-001",))
        self.assertGreaterEqual(len(snapshot.recent_events), 2)


class DetectorConditionReportFromStatusTests(unittest.TestCase):

    def _status(self, sensor_id, sensor_type="SmokeDetector", health_status=HealthStatus.OK, active=True):

        from sensor_manager.status import SensorStatus

        return SensorStatus(
            sensor_id=sensor_id, sensor_type=sensor_type, name=sensor_id, floor_id="floor1",
            zone_ids=("zone-a",), active=active, mode="Simulation", health_status=health_status,
        )

    def test_faulty_status_outranks_a_clear_reading(self):

        from perception.models.smoke_detector_observation import SmokeDetectorReading

        status = self._status("SD-001", health_status=HealthStatus.FAULT)
        reading = SmokeDetectorReading(detector_id="SD-001", timestamp=1.0, alarm_active=False)

        result = DetectorConditionReport.from_status_and_reading(status, reading)

        self.assertEqual(result.state, DetectorState.FAULT)
        self.assertEqual(result.asset_id, "SD-001")

    def test_alarm_reading_with_healthy_status(self):

        from perception.models.smoke_detector_observation import SmokeDetectorReading

        status = self._status("SD-001")
        reading = SmokeDetectorReading(detector_id="SD-001", timestamp=1.0, alarm_active=True)

        result = DetectorConditionReport.from_status_and_reading(status, reading)

        self.assertEqual(result.state, DetectorState.ALARM)

    def test_no_reading_and_healthy_status_is_normal(self):

        status = self._status("SD-001")

        result = DetectorConditionReport.from_status_and_reading(status, None)

        self.assertEqual(result.state, DetectorState.NORMAL)


class EndToEndFACPIdentityPipelineTests(unittest.TestCase):

    # HazardSnapshot -> Canonical Smoke/Heat Detector -> Detector
    # Reading/State -> SimulatedFACP -> Canonical Panel Event ->
    # BuildingState, verifying the SAME detector id survives throughout.

    def test_canonical_smoke_detector_id_survives_the_full_facp_pipeline(self):

        detector_id = "SD-EE-001"

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        zone = Zone(id="zone-a", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        floor.zones.append(zone)
        floor.add_smoke_detector(
            SmokeDetector(id=detector_id, name="Lobby Smoke", floor_id=floor.id, position=(1.0, 1.0), zone_ids=("zone-a",)),
        )

        manager = SensorManager()
        manager.discover_sensors(building)
        sensor_status = manager.sensor_status(detector_id)

        hazard_snapshot = HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.9)})

        smoke_provider = GroundTruthSmokeDetectorProvider(
            detectors=[floor.smoke_detectors[0]], zones=[zone],
            hazard_provider=ManualHazardProvider(hazard_snapshot),
        )
        readings = smoke_provider.alarm_states_at(time=0.0)
        self.assertEqual(readings[0].detector_id, detector_id)

        detector_condition = DetectorConditionReport.from_status_and_reading(sensor_status, readings[0])
        self.assertEqual(detector_condition.asset_id, detector_id)
        self.assertEqual(detector_condition.state, DetectorState.ALARM)

        facp = SimulatedFACP()
        fired = facp.evaluate({detector_id: detector_condition}, time=0.0)

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].source_asset_id, detector_id)
        self.assertEqual(fired[0].event_type, PanelEventType.DETECTOR_ALARM)

        facp_snapshot = facp.current_snapshot(0.0)
        self.assertIn(detector_id, facp_snapshot.active_alarm_source_ids)

        estimator = BuildingStateEstimator()
        state = estimator.estimate(
            0.0,
            hazard_snapshot=hazard_snapshot,
            occupancy_snapshot=OccupancySnapshot(observations={"zone-a": OccupancyObservation(occupant_count=0.0)}),
            smoke_detector_statuses=[sensor_status],
            smoke_detector_readings=readings,
            facp_snapshot=facp_snapshot,
        )

        self.assertIn(detector_id, state.facp_status.active_alarm_source_ids)
        self.assertEqual(state.smoke_detector_state(detector_id).status.sensor_id, detector_id)
        self.assertEqual(state.facp_status.recent_events[0].source_asset_id, detector_id)

    def test_legacy_detector_adapter_id_survives_the_full_facp_pipeline(self):

        detector_id = "SD-LEGACY-001"

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")
        zone = Zone(id="zone-a", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
        floor.zones.append(zone)
        floor.detectors.append(
            Detector(id=detector_id, name="Legacy Smoke", floor_id=floor.id, position=(1.0, 1.0), detector_type="Smoke"),
        )

        manager = SensorManager()
        manager.discover_sensors(building)
        sensor_status = manager.sensor_status(detector_id)

        hazard_snapshot = HazardSnapshot(node_states={"zone-a": HazardNodeState(smoke_level=0.9)})

        smoke_provider = GroundTruthSmokeDetectorProvider(
            detectors=[floor.detectors[0]], zones=[zone],
            hazard_provider=ManualHazardProvider(hazard_snapshot),
        )
        readings = smoke_provider.alarm_states_at(time=0.0)
        self.assertEqual(readings[0].detector_id, detector_id)

        detector_condition = DetectorConditionReport.from_status_and_reading(sensor_status, readings[0])

        facp = SimulatedFACP()
        fired = facp.evaluate({detector_id: detector_condition}, time=0.0)

        self.assertEqual(fired[0].source_asset_id, detector_id)
        self.assertIn(detector_id, facp.active_alarm_source_ids)


class BuildingStateBackwardCompatibilityTests(unittest.TestCase):

    def test_building_state_still_constructs_without_a_facp_status(self):

        from building_state.models import BuildingState

        state = BuildingState(timestamp=1.0)

        self.assertIsNone(state.facp_status)

    def test_building_state_retains_individual_detector_states_alongside_facp_status(self):

        from sensor_manager.status import SensorStatus

        from perception.models.smoke_detector_observation import SmokeDetectorReading

        status = SensorStatus(
            sensor_id="SD-001", sensor_type="SmokeDetector", name="SD-001", floor_id="floor1",
            zone_ids=("zone-a",), active=True, mode="Simulation", health_status=HealthStatus.OK,
        )
        reading = SmokeDetectorReading(detector_id="SD-001", timestamp=1.0, alarm_active=True)

        facp = SimulatedFACP()
        facp.evaluate(
            {"SD-001": DetectorConditionReport.from_status_and_reading(status, reading)}, time=1.0,
        )

        estimator = BuildingStateEstimator()
        state = estimator.estimate(
            1.0,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            smoke_detector_statuses=[status],
            smoke_detector_readings=[reading],
            facp_snapshot=facp.current_snapshot(1.0),
        )

        # facp_status is additive -- individual detector state is
        # unaffected and still directly queryable.
        self.assertIsNotNone(state.facp_status)
        self.assertEqual(state.smoke_detector_state("SD-001").reading.alarm_active, True)


class ArchitectureGuardTests(unittest.TestCase):

    def test_facp_package_never_touches_hazard_physics_or_other_frozen_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "facp"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(hazard|hazard_evolution|fire_growth|smoke_propagation|perception|"
            r"sensor_manager|camera_manager|sensors|simulator|sandbox|designer|"
            r"ai_decision|ai_training|rl_training|advisory_system|command_center|"
            r"gymnasium|gym|numpy|torch)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"facp/{path.name} imports a hazard-physics, perception, sensor-management, "
                f"simulation, or AI/decision module -- facp must only consume already-computed "
                f"DetectorConditionReport values and must never replace SensorManager or "
                f"recompute hazard physics.",
            )

    def test_facp_package_has_no_vendor_hardware_protocol_dependencies(self):

        # Anchored to actual import statements only (same convention as
        # this file's other dependency-direction guard above) -- the
        # module docstrings in this package legitimately *mention*
        # protocol names like Modbus/BACnet/MQTT/OPC-UA in prose to
        # explain what a future adapter would translate, which must not
        # trip this guard; only an actual `import`/`from ... import` of
        # one would.

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "facp"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(pymodbus|modbus|bacpypes|bacnet|paho|mqtt|opcua|freeopcua|pyserial|serial|socket|requests)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE | re.IGNORECASE),
                f"facp/{path.name} imports a vendor hardware protocol library -- "
                f"real hardware communication must not be implemented yet.",
            )

    def test_facp_events_always_preserve_the_source_asset_id_given(self):

        facp = SimulatedFACP()
        events = facp.evaluate({"SD-XYZ": report("SD-XYZ", DetectorState.ALARM)}, time=1.0)

        self.assertEqual(events[0].source_asset_id, "SD-XYZ")

    def test_future_live_adapter_can_satisfy_the_provider_interface(self):

        # A minimal stand-in for a future vendor adapter -- proves the
        # seam is usable without this test needing to implement any
        # actual protocol.

        class _FixedFACPEventProvider(FACPEventProvider):

            def events_at(self, time):
                return (
                    PanelEvent(
                        timestamp=time, event_type=PanelEventType.DETECTOR_ALARM,
                        source_asset_id="SD-LIVE-001", source_asset_type="SmokeDetector",
                    ),
                )

        provider = _FixedFACPEventProvider()
        events = provider.events_at(5.0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_asset_id, "SD-LIVE-001")

    def test_unimplemented_provider_raises_not_implemented(self):

        provider = FACPEventProvider()

        with self.assertRaises(NotImplementedError):
            provider.events_at(0.0)

    def test_panel_event_log_is_independently_usable_without_simulated_facp(self):

        # Confirms PanelEventLog (the Phase 4 query surface) is its own
        # reusable class, not something only SimulatedFACP can build --
        # a future Live/Replay adapter could populate one directly from
        # FACPEventProvider.events_at() without ever constructing a
        # SimulatedFACP.

        log = PanelEventLog()
        log.append(PanelEvent(timestamp=1.0, event_type=PanelEventType.DETECTOR_ALARM, source_asset_id="SD-001"))

        self.assertEqual(len(log), 1)
        self.assertEqual(len(log.events_for_asset("SD-001")), 1)


if __name__ == "__main__":
    unittest.main()

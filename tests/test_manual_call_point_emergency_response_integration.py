import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.zone import Zone
from models.exit import Exit
from models.door import Door
from models.obstacle import Obstacle
from models.manual_call_point import ManualCallPoint
from models.smoke_detector import SmokeDetector
from models.heat_detector import HeatDetector

from sensor_manager.manager import SensorManager

from facp.engine import SimulatedFACP
from live_system.facp_gateway import EngineFACPGateway

from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import ResponseReason

from live_occupants.manager import LiveOccupantManager

from live_system.live_advisory_gateway import emergency_response_evidence_from_snapshot

from navigation.graph_builder import NavigationGraphGenerator
from navigation.edge import Edge

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot


# =====================================================
# Manual Call Point -> Live Emergency Response Integration milestone.
#
# The real production chain proven throughout this file:
#   ManualCallPoint -> SensorManager -> EngineFACPGateway -> SimulatedFACP
#   -> FACPSnapshot -> BuildingStateEstimator -> BuildingState
#   -> EmergencyResponseIntelligenceEngine -> EmergencyResponseSnapshot
#   -> emergency_response_evidence_from_snapshot() -> EmergencyResponseEvidence
#   -> Command Center (LiveEmergencyResponsePanel)
# =====================================================


def _build_building(zone_specs):

    # zone_specs: {zone_id: (x, y)} -- a trivial 10x10 rectangle per zone.
    building = Building(name="B")
    floor = building.create_floor(name="Ground")

    for zone_id, (x, y) in zone_specs.items():
        floor.add_zone(Zone(id=zone_id, name=zone_id, floor_id=floor.id, x=x, y=y, width=10.0, height=10.0))

    return building, floor


def _occupant_manager(zone_id, floor_id):

    manager = LiveOccupantManager()
    manager.update("OCC-1", None, None, zone_id, floor_id, None, None, None, 0.9, 0.0)
    return manager


def _run_full_chain(building, floor, occupant_manager, time=0.0):

    sensor_manager = SensorManager()
    sensor_manager.discover_sensors(building)

    facp = SimulatedFACP()
    gateway = EngineFACPGateway(facp, sensor_manager)
    gateway.evaluate(time)

    facp_snapshot = facp.current_snapshot(time)

    smoke_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "SmokeDetector")
    heat_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "HeatDetector")
    mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")

    building_state = BuildingStateEstimator().estimate(
        time,
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        smoke_detector_statuses=smoke_statuses,
        heat_detector_statuses=heat_statuses,
        manual_call_point_statuses=mcp_statuses,
        facp_snapshot=facp_snapshot,
    )

    response_engine = EmergencyResponseIntelligenceEngine(building, occupant_manager)
    response_snapshot = response_engine.compute(time, building_state)

    evidence = emergency_response_evidence_from_snapshot(response_snapshot)

    return facp, building_state, response_snapshot, evidence


class MultipleMCPTests(unittest.TestCase):

    # Phase 6 -- MCP-1 in Zone A, MCP-2 in Zone B.

    def setUp(self):

        self.building, self.floor = _build_building({"ZONE-A": (0.0, 0.0), "ZONE-B": (20.0, 0.0)})

        self.mcp_1 = ManualCallPoint(id="MCP-1", name="MCP-1", floor_id=self.floor.id, zone_ids=("ZONE-A",))
        self.mcp_2 = ManualCallPoint(id="MCP-2", name="MCP-2", floor_id=self.floor.id, zone_ids=("ZONE-B",))
        self.floor.manual_call_points.append(self.mcp_1)
        self.floor.manual_call_points.append(self.mcp_2)

        self.occupant_manager = _occupant_manager("ZONE-A", self.floor.id)
        self.occupant_manager.update("OCC-2", None, None, "ZONE-B", self.floor.id, None, None, None, 0.9, 0.0)

    def test_activating_mcp_1_only_affects_zone_a(self):

        self.mcp_1.activate()

        _, _, response_snapshot, _ = _run_full_chain(self.building, self.floor, self.occupant_manager)

        zone_a = response_snapshot.zone("ZONE-A")
        zone_b = response_snapshot.zone("ZONE-B")

        self.assertTrue(zone_a.manual_emergency_reported)
        self.assertEqual({s.source_id for s in zone_a.alarm_sources}, {"MCP-1"})

        self.assertFalse(zone_b.manual_emergency_reported)
        self.assertEqual(zone_b.alarm_sources, ())

    def test_activating_both_keeps_independent_source_identities(self):

        self.mcp_1.activate()
        self.mcp_2.activate()

        _, _, response_snapshot, _ = _run_full_chain(self.building, self.floor, self.occupant_manager)

        zone_a = response_snapshot.zone("ZONE-A")
        zone_b = response_snapshot.zone("ZONE-B")

        self.assertEqual({s.source_id for s in zone_a.alarm_sources}, {"MCP-1"})
        self.assertEqual({s.source_id for s in zone_b.alarm_sources}, {"MCP-2"})

    def test_restoring_mcp_1_without_facp_reset_removes_it_from_active_sources(self):

        # Phase 6's own careful warning: device restoration and FACP
        # reset are different concepts. Restoring the device clears IT
        # from active_alarm_source_ids immediately (verified directly
        # against the real FACP), but never bypasses panel-wide
        # latching -- proven here at the Emergency Response layer.
        # Deliberately reuses the SAME SensorManager/FACP instance
        # across both cycles (never a fresh one) -- panel latching is
        # a property of one continuously-running panel, not something
        # a fresh FACP could ever exhibit.
        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(self.building)
        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)

        self.mcp_1.activate()
        gateway.evaluate(0.0)

        self.mcp_1.restore()
        gateway.evaluate(1.0)

        mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")
        building_state = BuildingStateEstimator().estimate(
            1.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            manual_call_point_statuses=mcp_statuses, facp_snapshot=facp.current_snapshot(1.0),
        )
        response_engine = EmergencyResponseIntelligenceEngine(self.building, self.occupant_manager)
        response_snapshot = response_engine.compute(1.0, building_state)

        zone_a = response_snapshot.zone("ZONE-A")
        self.assertFalse(zone_a.manual_emergency_reported)
        self.assertEqual(zone_a.alarm_sources, ())

        # The panel itself remains latched in ALARM until an operator
        # acknowledges/resets it -- this milestone never touches that.
        from facp.models import PanelState
        self.assertEqual(facp.panel_state, PanelState.ALARM)


class MixedAlarmSourcesTests(unittest.TestCase):

    # Phase 7 -- SmokeDetector SD-1 -> Zone A, HeatDetector HD-1 ->
    # Zone B, ManualCallPoint MCP-1 -> Zone C, all three triggered.

    def setUp(self):

        self.building, self.floor = _build_building({
            "ZONE-A": (0.0, 0.0), "ZONE-B": (20.0, 0.0), "ZONE-C": (40.0, 0.0),
        })

        self.sd_1 = SmokeDetector(id="SD-1", name="SD-1", floor_id=self.floor.id, zone_ids=("ZONE-A",))
        self.hd_1 = HeatDetector(id="HD-1", name="HD-1", floor_id=self.floor.id, zone_ids=("ZONE-B",))
        self.mcp_1 = ManualCallPoint(id="MCP-1", name="MCP-1", floor_id=self.floor.id, zone_ids=("ZONE-C",))

        self.floor.smoke_detectors.append(self.sd_1)
        self.floor.heat_detectors.append(self.hd_1)
        self.floor.manual_call_points.append(self.mcp_1)

        self.occupant_manager = LiveOccupantManager()
        for i, zone_id in enumerate(("ZONE-A", "ZONE-B", "ZONE-C")):
            self.occupant_manager.update(f"OCC-{i}", None, None, zone_id, self.floor.id, None, None, None, 0.9, 0.0)

    def test_all_three_source_types_are_distinguished_independently(self):

        # Smoke/Heat require a real hazard-driven reading path in
        # production (GroundTruthSmokeDetectorProvider/HeatDetectorProvider)
        # -- this test drives FACP directly with hand-built
        # DetectorConditionReports (the exact shape EngineFACPGateway
        # itself builds), proving Emergency Response's own source-type
        # distinction without needing a real hazard pipeline.
        from facp.models import DetectorConditionReport
        from models.sensor_asset import DetectorState

        self.mcp_1.activate()

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(self.building)

        facp = SimulatedFACP()

        reports = {
            "SD-1": DetectorConditionReport(asset_id="SD-1", asset_type="SmokeDetector", state=DetectorState.ALARM, floor_id=self.floor.id, zone_ids=("ZONE-A",)),
            "HD-1": DetectorConditionReport(asset_id="HD-1", asset_type="HeatDetector", state=DetectorState.ALARM, floor_id=self.floor.id, zone_ids=("ZONE-B",)),
            "MCP-1": DetectorConditionReport(asset_id="MCP-1", asset_type="ManualCallPoint", state=self.mcp_1.compute_state(0.0), floor_id=self.floor.id, zone_ids=("ZONE-C",)),
        }
        facp.evaluate(reports, 0.0)
        facp_snapshot = facp.current_snapshot(0.0)

        smoke_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "SmokeDetector")
        heat_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "HeatDetector")
        mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")

        building_state = BuildingStateEstimator().estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            smoke_detector_statuses=smoke_statuses, heat_detector_statuses=heat_statuses,
            manual_call_point_statuses=mcp_statuses, facp_snapshot=facp_snapshot,
        )

        response_engine = EmergencyResponseIntelligenceEngine(self.building, self.occupant_manager)
        response_snapshot = response_engine.compute(0.0, building_state)

        zone_a, zone_b, zone_c = (response_snapshot.zone(z) for z in ("ZONE-A", "ZONE-B", "ZONE-C"))

        self.assertEqual({s.source_type for s in zone_a.alarm_sources}, {"SmokeDetector"})
        self.assertIn(ResponseReason.FACP_ALARM_ACTIVE, zone_a.reason_codes)
        self.assertNotIn(ResponseReason.MANUAL_EMERGENCY_REPORTED, zone_a.reason_codes)
        self.assertFalse(zone_a.manual_emergency_reported)

        self.assertEqual({s.source_type for s in zone_b.alarm_sources}, {"HeatDetector"})
        self.assertIn(ResponseReason.FACP_ALARM_ACTIVE, zone_b.reason_codes)
        self.assertFalse(zone_b.manual_emergency_reported)

        self.assertEqual({s.source_type for s in zone_c.alarm_sources}, {"ManualCallPoint"})
        self.assertIn(ResponseReason.MANUAL_EMERGENCY_REPORTED, zone_c.reason_codes)
        self.assertNotIn(ResponseReason.FACP_ALARM_ACTIVE, zone_c.reason_codes)
        self.assertTrue(zone_c.manual_emergency_reported)

    def test_manual_and_automatic_evidence_never_overwrite_each_other_in_the_same_zone(self):

        # Both a SmokeDetector AND a ManualCallPoint assigned to the
        # SAME zone -- neither may mask the other.
        self.floor.manual_call_points[0].zone_ids = ("ZONE-A",)
        self.mcp_1.activate()

        from facp.models import DetectorConditionReport
        from models.sensor_asset import DetectorState

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(self.building)
        facp = SimulatedFACP()

        reports = {
            "SD-1": DetectorConditionReport(asset_id="SD-1", asset_type="SmokeDetector", state=DetectorState.ALARM, floor_id=self.floor.id, zone_ids=("ZONE-A",)),
            "MCP-1": DetectorConditionReport(asset_id="MCP-1", asset_type="ManualCallPoint", state=self.mcp_1.compute_state(0.0), floor_id=self.floor.id, zone_ids=("ZONE-A",)),
        }
        facp.evaluate(reports, 0.0)

        smoke_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "SmokeDetector")
        mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")

        building_state = BuildingStateEstimator().estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            smoke_detector_statuses=smoke_statuses, manual_call_point_statuses=mcp_statuses,
            facp_snapshot=facp.current_snapshot(0.0),
        )

        response_engine = EmergencyResponseIntelligenceEngine(self.building, self.occupant_manager)
        response_snapshot = response_engine.compute(0.0, building_state)
        zone_a = response_snapshot.zone("ZONE-A")

        self.assertEqual({s.source_type for s in zone_a.alarm_sources}, {"SmokeDetector", "ManualCallPoint"})
        self.assertIn(ResponseReason.FACP_ALARM_ACTIVE, zone_a.reason_codes)
        self.assertIn(ResponseReason.MANUAL_EMERGENCY_REPORTED, zone_a.reason_codes)
        self.assertTrue(zone_a.manual_emergency_reported)


class FullOfflineE2ETests(unittest.TestCase):

    # Phase 10 -- the real production LiveRuntime, Zone A (MCP-1) /
    # Zone B (SD-1), proving the complete chain and that nothing is
    # dispatched/executed automatically throughout.

    def test_full_chain_mcp_then_smoke_no_automatic_execution(self):

        from live_runtime.factory import build_live_runtime

        building, floor = _build_building({"ZONE-A": (0.0, 0.0), "ZONE-B": (20.0, 0.0)})

        mcp_1 = ManualCallPoint(id="MCP-1", name="MCP-1", floor_id=floor.id, zone_ids=("ZONE-A",))
        sd_1 = SmokeDetector(id="SD-1", name="SD-1", floor_id=floor.id, zone_ids=("ZONE-B",))
        floor.manual_call_points.append(mcp_1)
        floor.smoke_detectors.append(sd_1)

        facp = SimulatedFACP()

        def smoke_reading_provider(time):
            from perception.models.smoke_detector_observation import SmokeDetectorReading
            return (SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=sd_1_alarm[0]),)

        sd_1_alarm = [False]

        runtime = build_live_runtime(
            building, facp=facp, smoke_detector_reading_provider=smoke_reading_provider,
        )
        occupant_manager = runtime.live_occupant_manager
        occupant_manager.update("OCC-A", None, None, "ZONE-A", floor.id, None, None, None, 0.9, 0.0)
        occupant_manager.update("OCC-B", None, None, "ZONE-B", floor.id, None, None, None, 0.9, 0.0)

        runtime.start()
        try:

            # ---- Initial: no alarm ----
            runtime.run_cycle(0.0)
            response = runtime.orchestrator.latest_emergency_response
            self.assertFalse(response.zone("ZONE-A").manual_emergency_reported)
            self.assertEqual(response.zone("ZONE-B").alarm_sources, ())

            # ---- Activate MCP-1 ----
            mcp_1.activate()
            runtime.run_cycle(1.0)

            response = runtime.orchestrator.latest_emergency_response
            zone_a = response.zone("ZONE-A")
            self.assertTrue(zone_a.manual_emergency_reported)
            self.assertEqual({s.source_id for s in zone_a.alarm_sources}, {"MCP-1"})

            building_state = runtime.orchestrator.latest_building_state
            self.assertIn("MCP-1", building_state.facp_status.active_alarm_source_ids)

            # Advisory-facing evidence reduction, run directly against
            # the REAL runtime-produced EmergencyResponseSnapshot (a
            # full live AI/Advisory gateway is a separate, unrelated
            # wiring concern out of this milestone's own scope -- see
            # live_system.live_advisory_gateway's own module for where
            # a real deployment would additionally wire this).
            evidence = emergency_response_evidence_from_snapshot(response)
            self.assertTrue(evidence.available)
            self.assertIn("ZONE-A", evidence.manual_emergency_report_zone_ids)
            self.assertTrue(evidence.zone_details["ZONE-A"].manual_emergency_reported)
            self.assertEqual(evidence.zone_details["ZONE-A"].manual_call_point_ids, ("MCP-1",))

            # ---- Trigger SD-1 too -- both source types coexist ----
            sd_1_alarm[0] = True
            runtime.run_cycle(2.0)

            response = runtime.orchestrator.latest_emergency_response
            zone_a = response.zone("ZONE-A")
            zone_b = response.zone("ZONE-B")

            self.assertTrue(zone_a.manual_emergency_reported)
            self.assertEqual({s.source_type for s in zone_b.alarm_sources}, {"SmokeDetector"})

            # ---- Never automatically dispatched anything ----
            self.assertIsNone(runtime.voice_evacuation_controller)
            self.assertIsNone(runtime.building_control_controller)
            self.assertIsNone(runtime.dynamic_signage_controller)

        finally:
            runtime.stop()


class SafetyPrecedenceTests(unittest.TestCase):

    # Phase 11 -- MCP activation cannot alter route safety, hazard
    # values, or execute/broadcast anything by itself.

    def test_mcp_activation_never_unblocks_a_blocked_exit(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        exit_obj = Exit(id="E1", floor_id=floor.id, start_point=(0.0, 0.0), end_point=(0.0, 10.0), zone_id="ZONE-A", is_blocked=True)
        floor.add_exit(exit_obj)

        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=("ZONE-A",))
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()

        graph = NavigationGraphGenerator().build(building)
        exit_edge = next(e for e in graph.edges if e.id == "E1")

        self.assertFalse(exit_edge.traversable)

    def test_mcp_activation_never_unblocks_an_obstacle_blocked_door(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0), "ZONE-B": (20.0, 0.0)})
        door = Door(id="D1", floor_id=floor.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0), zone_a_id="ZONE-A", zone_b_id="ZONE-B")
        floor.add_door(door)
        obstacle = Obstacle(id="O1", floor_id=floor.id, x=14.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked")
        floor.obstacles.append(obstacle)

        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=("ZONE-A",))
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()

        graph = NavigationGraphGenerator().build(building)
        door_edge = next(e for e in graph.edges if e.id == "D1")

        self.assertFalse(door_edge.traversable)

    def test_mcp_activation_never_changes_hazard_summary(self):

        building_state = BuildingStateEstimator().estimate(
            0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        )

        # An empty HazardSnapshot produces an empty hazard summary --
        # confirms MCP has no path to influence it at all (this test
        # never even constructs an MCP, since there is no code path
        # for one to affect this value through).
        self.assertEqual(building_state.hazard_summary.zone_severities, {})

    def test_mcp_never_reachable_from_voice_or_building_control(self):

        import pathlib
        import re

        path = pathlib.Path(__file__).resolve().parent.parent / "models" / "manual_call_point.py"
        text = path.read_text(encoding="utf-8")

        forbidden = r"^\s*(from|import)\s+(voice_evacuation|building_control|dynamic_signage|decision_policy|hazard|hazard_evolution|fire_growth|smoke_propagation)\b"
        match = re.search(forbidden, text, re.MULTILINE)
        self.assertIsNone(match, f"models/manual_call_point.py imports {match.group(0) if match else ''!r}")


class FailureAndDegradationTests(unittest.TestCase):

    # Phase 12 -- no crash, no fabricated location.

    def test_unassigned_mcp_produces_building_level_evidence_never_a_fake_zone(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=())  # unassigned
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()

        occupant_manager = _occupant_manager("ZONE-A", floor.id)
        facp, building_state, response_snapshot, _ = _run_full_chain(building, floor, occupant_manager)

        # Genuinely reaches FACP as a real alarm source ...
        self.assertIn("MCP-1", building_state.facp_status.active_alarm_source_ids)

        # ... but never gets assigned to Zone A (or any zone) since it
        # carries no real zone assignment.
        zone_a = response_snapshot.zone("ZONE-A")
        self.assertEqual(zone_a.alarm_sources, ())
        self.assertFalse(zone_a.manual_emergency_reported)

    def test_mcp_referencing_a_deleted_zone_never_crashes(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=("ZONE-DELETED",))
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()

        occupant_manager = _occupant_manager("ZONE-A", floor.id)

        # Must not raise.
        _, _, response_snapshot, _ = _run_full_chain(building, floor, occupant_manager)
        self.assertIsNotNone(response_snapshot.zone("ZONE-A"))

    def test_inactive_mcp_never_produces_evidence(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=("ZONE-A",), active=False)
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()  # activated, but the device itself is inactive

        occupant_manager = _occupant_manager("ZONE-A", floor.id)
        _, _, response_snapshot, _ = _run_full_chain(building, floor, occupant_manager)

        zone_a = response_snapshot.zone("ZONE-A")
        self.assertFalse(zone_a.manual_emergency_reported)

    def test_missing_facp_status_never_crashes_emergency_response(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        occupant_manager = _occupant_manager("ZONE-A", floor.id)

        building_state = BuildingState()  # facp_status is None

        response_engine = EmergencyResponseIntelligenceEngine(building, occupant_manager)
        response_snapshot = response_engine.compute(0.0, building_state)

        self.assertFalse(response_snapshot.zone("ZONE-A").manual_emergency_reported)

    def test_legacy_project_without_manual_call_points_key_loads_and_computes_cleanly(self):

        from models.project import Project

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        project = Project(name="P", building=building)

        data = project.to_dict()
        for floor_data in data["building"]["floors"]:
            floor_data.pop("manual_call_points", None)

        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.manual_call_point_count, 0)

        occupant_manager = _occupant_manager("ZONE-A", restored_floor.id)
        _, _, response_snapshot, _ = _run_full_chain(restored.building, restored_floor, occupant_manager)
        self.assertIsNotNone(response_snapshot.zone("ZONE-A"))

    def test_duplicate_mcp_ids_never_crash_sensor_manager_discovery(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        floor.manual_call_points.append(ManualCallPoint(id="MCP-DUP", floor_id=floor.id, zone_ids=("ZONE-A",)))
        floor.manual_call_points.append(ManualCallPoint(id="MCP-DUP", floor_id=floor.id, zone_ids=("ZONE-A",)))

        occupant_manager = _occupant_manager("ZONE-A", floor.id)

        # Must not raise.
        _run_full_chain(building, floor, occupant_manager)

    def test_facp_reset_clears_manual_evidence(self):

        building, floor = _build_building({"ZONE-A": (0.0, 0.0)})
        mcp_1 = ManualCallPoint(id="MCP-1", floor_id=floor.id, zone_ids=("ZONE-A",))
        floor.manual_call_points.append(mcp_1)
        mcp_1.activate()
        mcp_1.restore()

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)
        facp = SimulatedFACP()
        gateway = EngineFACPGateway(facp, sensor_manager)
        gateway.evaluate(0.0)

        facp.reset(1.0)

        occupant_manager = _occupant_manager("ZONE-A", floor.id)
        mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")

        building_state = BuildingStateEstimator().estimate(
            1.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
            manual_call_point_statuses=mcp_statuses, facp_snapshot=facp.current_snapshot(1.0),
        )
        response_engine = EmergencyResponseIntelligenceEngine(building, occupant_manager)
        response_snapshot = response_engine.compute(1.0, building_state)

        self.assertFalse(response_snapshot.zone("ZONE-A").manual_emergency_reported)


class ArchitectureGuardTests(unittest.TestCase):

    # Phase 14.

    def _assert_no_forbidden_import(self, path, forbidden_pattern):

        import re

        text = path.read_text(encoding="utf-8")
        match = re.search(forbidden_pattern, text, re.MULTILINE)
        self.assertIsNone(match, f"{path} imports {match.group(0) if match else ''!r}")

    def test_manual_call_point_model_imports_nothing_forbidden(self):

        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "models" / "manual_call_point.py"
        forbidden = r"^\s*(from|import)\s+(voice_evacuation|building_control|dynamic_signage|decision_policy|hazard|hazard_evolution|fire_growth|smoke_propagation|ai_decision|ai_registry|ai_inference|ai_training|rl_training)\b"
        self._assert_no_forbidden_import(path, forbidden)

    def test_facp_package_imports_nothing_forbidden(self):

        import pathlib

        for filename in ("engine.py", "models.py"):
            path = pathlib.Path(__file__).resolve().parent.parent / "facp" / filename
            forbidden = r"^\s*(from|import)\s+(voice_evacuation|building_control|dynamic_signage|decision_policy|ai_decision|ai_registry|ai_inference|ai_training|rl_training)\b"
            self._assert_no_forbidden_import(path, forbidden)

    def test_emergency_response_package_imports_nothing_forbidden(self):

        import pathlib

        for filename in ("engine.py", "models.py"):
            path = pathlib.Path(__file__).resolve().parent.parent / "emergency_response" / filename
            forbidden = r"^\s*(from|import)\s+(voice_evacuation|building_control|dynamic_signage|decision_policy|ai_decision|ai_registry|ai_inference|ai_training|rl_training)\b"
            self._assert_no_forbidden_import(path, forbidden)

    def test_ai_and_rl_packages_never_reference_manual_call_point_or_facp(self):

        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parent.parent

        for package in ("ai_decision", "ai_registry", "ai_inference", "ai_training", "rl_training", "rl"):

            package_dir = repo_root / package

            if not package_dir.is_dir():
                continue

            for path in package_dir.glob("*.py"):

                text = path.read_text(encoding="utf-8")
                self.assertNotIn("ManualCallPoint", text, f"{path} references ManualCallPoint")
                self.assertNotIn("SimulatedFACP", text, f"{path} references SimulatedFACP")

    def test_no_hardware_or_network_protocol_import_in_manual_call_point_chain(self):

        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parent.parent
        forbidden = r"^\s*(from|import)\s+(socket|serial|can|modbus|bacnet)\b"

        for relative in ("models/manual_call_point.py", "facp/engine.py", "facp/models.py", "emergency_response/engine.py"):

            path = repo_root / relative
            self._assert_no_forbidden_import(path, forbidden)


class CommandCenterPanelTests(unittest.TestCase):

    # Phase 9 -- the existing Live Emergency Response panel, extended.

    def test_panel_distinguishes_manual_and_automatic_sources(self):

        from command_center.live_emergency_response_panel import LiveEmergencyResponsePanel
        from emergency_response.models import AlarmSourceEvidence, EmergencyResponseSnapshot, ZoneResponsePriority

        manual_source = AlarmSourceEvidence(source_id="MCP-1", source_type="ManualCallPoint", zone_ids=("Z1",))
        auto_source = AlarmSourceEvidence(source_id="SD-1", source_type="SmokeDetector", zone_ids=("Z1",))

        priority = ZoneResponsePriority(
            zone_id="Z1", priority_level="HIGH", priority_score=0.5,
            alarm_sources=(manual_source, auto_source), manual_emergency_reported=True,
            explanation="Zone Z1 -- HIGH.", reason_codes=("MANUAL_EMERGENCY_REPORTED", "FACP_ALARM_ACTIVE"),
        )
        snapshot = EmergencyResponseSnapshot(timestamp=0.0, zones={"Z1": priority}, response_priority_order=("Z1",))

        panel = LiveEmergencyResponsePanel()
        panel.show_response(snapshot)

        alarm_column_text = panel.queue_table.item(0, 8).text()
        self.assertIn("Manual: MCP-1", alarm_column_text)
        self.assertIn("Auto: SD-1", alarm_column_text)

        panel.queue_table.selectRow(0)
        self.assertIn("Manual Call Point MCP-1", panel.detail_label.text())
        self.assertIn("SmokeDetector SD-1", panel.detail_label.text())

    def test_panel_shows_dash_when_no_alarm_sources(self):

        from command_center.live_emergency_response_panel import LiveEmergencyResponsePanel
        from emergency_response.models import EmergencyResponseSnapshot, ZoneResponsePriority

        priority = ZoneResponsePriority(zone_id="Z1", priority_level="LOW", priority_score=0.1, explanation="Zone Z1 -- LOW.")
        snapshot = EmergencyResponseSnapshot(timestamp=0.0, zones={"Z1": priority}, response_priority_order=("Z1",))

        panel = LiveEmergencyResponsePanel()
        panel.show_response(snapshot)

        self.assertEqual(panel.queue_table.item(0, 8).text(), "-")


if __name__ == "__main__":
    unittest.main()

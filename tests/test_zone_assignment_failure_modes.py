import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.items.smoke_detector_item import SmokeDetectorItem

from models.building import Building
from models.detector import Detector
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.zone import Zone

from sensor_manager.manager import SensorManager
from speaker_manager.manager import SpeakerManager

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport, PanelState


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 14 -- failure/degradation coverage not already exercised by
# tests/test_property_panel_zone_assignment.py, tests/
# test_designer_zone_autoassignment.py, tests/test_facp_gateway.py, or
# tests/test_live_runtime_facp_production.py.
# =====================================================


class AssetMovedZoneStaysExplicitTests(unittest.TestCase):

    def test_moving_a_detector_does_not_silently_change_its_zone_assignment(self):

        # Zone re-derivation happens only at PLACEMENT time (Phase 4's
        # own explicit scope) -- moving an already-placed detector to a
        # different physical zone does NOT silently follow it. This is
        # a deliberate, documented limitation, never a crash: the old
        # assignment is honestly retained until a human explicitly
        # reassigns it (Property Panel), never guessed again on every
        # drag.
        zone_a = Zone(id="Z-A", name="A", floor_id="f1", x=0.0, y=0.0, width=10.0, height=10.0)
        zone_b = Zone(id="Z-B", name="B", floor_id="f1", x=20.0, y=0.0, width=10.0, height=10.0)

        model = SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", position=(5.0, 5.0), zone_ids=("Z-A",))
        item = SmokeDetectorItem(250, 250, model=model)

        item.setPos(1250, 250)  # (25, 5)m -- now geometrically inside Zone B
        item.sync_to_model()

        self.assertEqual(model.position, (25.0, 5.0))
        # zone_ids is untouched by the move itself.
        self.assertEqual(model.zone_ids, ("Z-A",))


class InactiveAndFaultDeviceTests(unittest.TestCase):

    def test_inactive_speaker_not_selected_for_broadcast(self):

        floor = Floor(id="f1", name="F1")
        floor.add_speaker(Speaker(id="SP-1", name="SP-1", floor_id="f1", zone_ids=("z1",), active=False))
        building = Building(id="b1", name="B", floors=[floor])

        manager = SpeakerManager()
        manager.discover_speakers(building)

        self.assertEqual(manager.active_speakers_in_zone("z1"), ())
        self.assertEqual(len(manager.speakers_in_zone("z1")), 1)

    def test_inactive_detector_computes_normal_not_alarm(self):

        detector = SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", active=False)

        self.assertEqual(detector.compute_state(0.9).name, "NORMAL")

    def test_fault_detector_reported_honestly_to_facp(self):

        from models.sensor_asset import HealthStatus
        from sensor_manager.status import SensorStatus

        status = SensorStatus(
            sensor_id="SD-1", sensor_type="SmokeDetector", name="SD-1", floor_id="f1",
            zone_ids=("z1",), active=True, mode="Simulation", health_status=HealthStatus.FAULT,
        )

        report = DetectorConditionReport.from_status_and_reading(status, None)

        facp = SimulatedFACP()
        facp.evaluate({"SD-1": report}, 1.0)

        self.assertEqual(facp.panel_state, PanelState.FAULT)


class LegacyAndDuplicateDetectorTests(unittest.TestCase):

    def test_legacy_detector_round_trips_and_participates_in_facp(self):

        floor = Floor(id="f1", name="F1")
        floor.add_zone(Zone(id="z1", name="Z1", floor_id="f1", x=0.0, y=0.0, width=10.0, height=10.0))
        floor.add_detector(Detector(id="LEGACY-1", name="Legacy Smoke", floor_id="f1", position=(5.0, 5.0), detector_type="Smoke"))
        building = Building(id="b1", name="B", floors=[floor])

        manager = SensorManager()
        manager.discover_sensors(building)

        statuses = manager.all_statuses()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].sensor_id, "LEGACY-1")
        self.assertEqual(statuses[0].zone_ids, ("z1",))  # geometrically derived

        # Preserved on Floor.detectors -- never migrated into a second,
        # independently-persisted SmokeDetector.
        self.assertEqual(floor.smoke_detector_count, 0)
        self.assertEqual(floor.detector_count, 1)

        report = DetectorConditionReport.from_status_and_reading(
            statuses[0], None,
        )
        facp = SimulatedFACP()
        facp.evaluate({"LEGACY-1": report}, 1.0)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)

    def test_duplicate_detector_id_does_not_double_register(self):

        # A pathological (should-never-happen) case: two distinct
        # asset objects sharing the same id -- SensorManager's own
        # dict-keyed-by-id storage means the second registration simply
        # replaces the first; it never silently double-counts one
        # physical identity as two entries, and FACP receives exactly
        # one condition report per id, never two.
        floor = Floor(id="f1", name="F1")
        floor.add_smoke_detector(SmokeDetector(id="SAME-ID", name="First", floor_id="f1", zone_ids=("z1",)))
        floor.add_heat_detector(HeatDetector(id="SAME-ID", name="Second", floor_id="f1", zone_ids=("z2",)))
        building = Building(id="b1", name="B", floors=[floor])

        manager = SensorManager()
        manager.discover_sensors(building)

        self.assertEqual(len(manager.all_sensors()), 1)


class EmptyBuildingTests(unittest.TestCase):

    def test_no_detectors_no_speakers_no_crash(self):

        floor = Floor(id="f1", name="F1")
        building = Building(id="b1", name="B", floors=[floor])

        sensor_manager = SensorManager()
        speaker_manager = SpeakerManager()

        self.assertEqual(sensor_manager.discover_sensors(building), ())
        self.assertEqual(speaker_manager.discover_speakers(building), ())

        facp = SimulatedFACP()
        result = facp.evaluate({}, 1.0)

        self.assertEqual(result, ())
        self.assertEqual(facp.panel_state, PanelState.NORMAL)


class OldProjectWithoutZoneIdsTests(unittest.TestCase):

    def test_asset_dict_missing_zone_ids_key_loads_as_empty(self):

        data = SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", zone_ids=("z1",)).to_dict()
        del data["zone_ids"]

        restored = SmokeDetector.from_dict(data)

        self.assertEqual(restored.zone_ids, ())

    def test_floor_missing_speakers_key_entirely_loads_safely(self):

        floor = Floor(id="f1", name="F1")
        data = floor.to_dict()
        del data["speakers"]
        del data["smoke_detectors"]
        del data["heat_detectors"]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.speakers, [])
        self.assertEqual(restored.smoke_detectors, [])
        self.assertEqual(restored.heat_detectors, [])


if __name__ == "__main__":
    unittest.main()

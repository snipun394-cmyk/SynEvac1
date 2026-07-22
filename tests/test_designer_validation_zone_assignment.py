import unittest

from designer.validation import validate_building_authoring
from navigation.validation import ValidationReport

from models.building import Building
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.speaker import Speaker
from models.zone import Zone


def _make_building_with_floor():

    floor = Floor(id="f1", name="Floor 1")
    building = Building(id="b1", name="B", floors=[floor])

    return building, floor


class ZoneAssignmentWarningTests(unittest.TestCase):

    def test_unassigned_speaker_produces_warning(self):

        building, floor = _make_building_with_floor()
        floor.add_speaker(Speaker(id="SP-1", name="SP-1", floor_id="f1"))

        report = validate_building_authoring(building)

        codes = {issue.code for issue in report.warnings}
        self.assertIn("speaker_missing_zone", codes)

    def test_unassigned_smoke_detector_produces_warning(self):

        building, floor = _make_building_with_floor()
        floor.add_smoke_detector(SmokeDetector(id="SD-1", name="SD-1", floor_id="f1"))

        report = validate_building_authoring(building)

        codes = {issue.code for issue in report.warnings}
        self.assertIn("smoke_detector_missing_zone", codes)

    def test_unassigned_heat_detector_produces_warning(self):

        building, floor = _make_building_with_floor()
        floor.add_heat_detector(HeatDetector(id="HD-1", name="HD-1", floor_id="f1"))

        report = validate_building_authoring(building)

        codes = {issue.code for issue in report.warnings}
        self.assertIn("heat_detector_missing_zone", codes)

    def test_assigned_assets_produce_no_warning(self):

        building, floor = _make_building_with_floor()
        floor.add_zone(Zone(id="Z1", name="Z1", floor_id="f1"))
        floor.add_speaker(Speaker(id="SP-1", name="SP-1", floor_id="f1", zone_ids=("Z1",)))
        floor.add_smoke_detector(SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", zone_ids=("Z1",)))
        floor.add_heat_detector(HeatDetector(id="HD-1", name="HD-1", floor_id="f1", zone_ids=("Z1",)))

        report = validate_building_authoring(building)

        codes = {issue.code for issue in report.warnings}
        self.assertNotIn("speaker_missing_zone", codes)
        self.assertNotIn("smoke_detector_missing_zone", codes)
        self.assertNotIn("heat_detector_missing_zone", codes)

    def test_severity_is_warning_not_error(self):

        # Deliberately softer than Door/Exit/Stair's own ERROR severity
        # -- an unassigned device is still a real, functioning asset,
        # never a structurally broken Navigation Graph edge.
        building, floor = _make_building_with_floor()
        floor.add_speaker(Speaker(id="SP-1", name="SP-1", floor_id="f1"))

        report = validate_building_authoring(building)

        matching = [issue for issue in report.issues if issue.code == "speaker_missing_zone"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].severity, ValidationReport.WARNING)

    def test_door_exit_stair_errors_unaffected(self):

        # Regression: this milestone must not change Door/Exit/Stair's
        # own pre-existing ERROR-severity checks.
        from models.door import Door

        building, floor = _make_building_with_floor()
        floor.add_door(Door(id="D1", name="D1", floor_id="f1"))

        report = validate_building_authoring(building)

        codes = {issue.code for issue in report.errors}
        self.assertIn("door_missing_zone_a", codes)
        self.assertIn("door_missing_zone_b", codes)


if __name__ == "__main__":
    unittest.main()

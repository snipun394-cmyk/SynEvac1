import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from navigation.validation import ValidationReport

from designer.validation import validate_building_authoring


def make_zone(name, **kwargs):

    fields = dict(x=0.0, y=0.0, width=2.0, height=2.0)
    fields.update(kwargs)

    return Zone(name=name, **fields)


def build_two_floor_building():

    building = Building(name="B")
    ground = building.create_floor(name="Ground Floor")
    floor1 = building.create_floor(name="Floor 1")

    room = make_zone("Room")
    lobby = make_zone("Lobby", x=10.0)
    ground.add_zone(room)
    ground.add_zone(lobby)

    upstairs = make_zone("Upstairs")
    floor1.add_zone(upstairs)

    return building, ground, floor1, room, lobby, upstairs


class FullyWiredBuildingTests(unittest.TestCase):

    def test_a_fully_connected_building_reports_no_issues(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        door = Door(name="D1", zone_a_id=room.id, zone_b_id=lobby.id, floor_id=ground.id)
        ground.add_door(door)

        exit_obj = Exit(name="Ex", zone_id=lobby.id, floor_id=ground.id)
        ground.add_exit(exit_obj)

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_none_building_returns_an_empty_report(self):

        report = validate_building_authoring(None)

        self.assertEqual(list(report.issues), [])


class DoorValidationTests(unittest.TestCase):

    def test_missing_zone_a_is_an_error(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        door = Door(name="D1", zone_a_id="", zone_b_id=lobby.id, floor_id=ground.id)
        ground.add_door(door)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("door_missing_zone_a", codes)
        self.assertNotIn("door_missing_zone_b", codes)

    def test_missing_zone_b_is_an_error(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        door = Door(name="D1", zone_a_id=room.id, zone_b_id="", floor_id=ground.id)
        ground.add_door(door)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("door_missing_zone_b", codes)


class ExitValidationTests(unittest.TestCase):

    def test_missing_zone_is_an_error(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        exit_obj = Exit(name="Ex", zone_id="", floor_id=ground.id)
        ground.add_exit(exit_obj)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("exit_missing_zone", codes)
        self.assertEqual(report.errors[0].severity, ValidationReport.ERROR)


class StairValidationTests(unittest.TestCase):

    def test_missing_from_zone_is_an_error(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id="", to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("stair_missing_from_zone", codes)
        self.assertNotIn("stair_missing_to_zone", codes)

    def test_missing_to_zone_is_an_error(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id="",
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("stair_missing_to_zone", codes)

    def test_missing_both_zones_reports_both_errors(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id="", to_zone_id="",
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        codes = [issue.code for issue in report.errors]
        self.assertIn("stair_missing_from_zone", codes)
        self.assertIn("stair_missing_to_zone", codes)

    def test_fully_wired_stair_is_not_flagged(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        self.assertEqual(report.errors, [])


class DuplicateStairDetectionTests(unittest.TestCase):

    def test_two_stairs_connecting_the_same_zone_pair_are_flagged_as_duplicates(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair_a = Staircase(
            id="stair-a", name="Stair A", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair_a)

        # The reported bug: a second, independent Staircase object
        # created from the other floor, representing the same
        # physical connection.
        stair_b = Staircase(
            id="stair-b", name="Stair B", from_floor_id=floor1.id, to_floor_id=ground.id,
            from_zone_id=upstairs.id, to_zone_id=room.id,
        )
        floor1.add_stair(stair_b)

        report = validate_building_authoring(building)

        duplicate_issues = report.by_code("potential_duplicate_stair")
        self.assertEqual(len(duplicate_issues), 1)
        self.assertEqual(duplicate_issues[0].severity, ValidationReport.WARNING)

    def test_two_stairs_between_the_same_floors_but_different_zones_are_not_flagged(self):

        # A legitimate building with two distinct physical staircases
        # between the same two floors (different zone pairs) must
        # never be reported as a duplicate.
        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        second_upstairs = make_zone("Second Upstairs", x=20.0)
        floor1.add_zone(second_upstairs)

        stair_a = Staircase(
            name="North Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair_a)

        stair_b = Staircase(
            name="South Stair", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=lobby.id, to_zone_id=second_upstairs.id,
        )
        ground.add_stair(stair_b)

        report = validate_building_authoring(building)

        self.assertEqual(report.by_code("potential_duplicate_stair"), [])

    def test_a_single_stair_is_never_flagged_as_a_duplicate_of_itself(self):

        building, ground, floor1, room, lobby, upstairs = build_two_floor_building()

        stair = Staircase(
            name="S1", from_floor_id=ground.id, to_floor_id=floor1.id,
            from_zone_id=room.id, to_zone_id=upstairs.id,
        )
        ground.add_stair(stair)

        report = validate_building_authoring(building)

        self.assertEqual(report.by_code("potential_duplicate_stair"), [])


if __name__ == "__main__":
    unittest.main()

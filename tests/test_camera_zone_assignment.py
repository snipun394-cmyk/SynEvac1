import unittest

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.zone import Zone


# =====================================================
# Camera -> Zone Assignment milestone.
#
# Camera.zone_ids (models/engineering_asset.py) already existed as a
# plain Tuple[str, ...] before this milestone -- these tests cover what
# this milestone actually adds: Floor.get_zone() (the lookup a runtime
# caller needs to turn zone_ids into real Zone objects) and a full
# Building-level round trip, the level that actually gets written by
# serialization.Serializer.save()/load(). Camera.to_dict()/from_dict()'s
# own zone_ids round trip and backward compatibility are already covered
# by tests/test_camera.py -- not repeated here.
# =====================================================


def _building_with_camera_and_zones(zone_ids_on_camera=()):

    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    zone_a = Zone(id="ZONE-A", name="Lobby", floor_id=floor.id, zone_type="Lobby")
    zone_b = Zone(id="ZONE-B", name="Corridor", floor_id=floor.id, zone_type="Corridor")
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)

    camera = Camera(
        id="CAM-1", name="Camera 1", floor_id=floor.id, zone_ids=zone_ids_on_camera,
    )
    floor.add_camera(camera)

    return building, floor, camera


class FloorGetZoneTests(unittest.TestCase):

    def test_returns_the_matching_zone(self):

        building, floor, camera = _building_with_camera_and_zones()

        zone = floor.get_zone("ZONE-A")

        self.assertIsNotNone(zone)
        self.assertEqual(zone.name, "Lobby")

    def test_returns_none_for_an_unknown_id(self):

        building, floor, camera = _building_with_camera_and_zones()

        self.assertIsNone(floor.get_zone("does-not-exist"))

    def test_returns_none_on_an_empty_floor(self):

        floor = Floor(name="Empty Floor")

        self.assertIsNone(floor.get_zone("anything"))


class CameraObservedZonesQueryTests(unittest.TestCase):

    # The exact query this milestone exists to answer: "which zones does
    # this Camera observe" -- Camera.zone_ids is the storage (a tuple of
    # ids, zero hard-coded anywhere), Floor.get_zone() is how a runtime
    # caller resolves those ids into the real Zone objects/names.

    def test_camera_with_no_assignment_observes_nothing(self):

        building, floor, camera = _building_with_camera_and_zones(zone_ids_on_camera=())

        observed = [floor.get_zone(zone_id) for zone_id in camera.zone_ids]

        self.assertEqual(observed, [])

    def test_camera_assigned_to_one_zone(self):

        building, floor, camera = _building_with_camera_and_zones(zone_ids_on_camera=("ZONE-A",))

        observed = [floor.get_zone(zone_id) for zone_id in camera.zone_ids]

        self.assertEqual([zone.name for zone in observed], ["Lobby"])

    def test_camera_assigned_to_multiple_zones(self):

        # One camera may observe multiple zones -- the milestone's own
        # first requirement. No hard-coded zone id anywhere in this
        # path: both ids come from the Zones actually added to the
        # Floor above.
        building, floor, camera = _building_with_camera_and_zones(
            zone_ids_on_camera=("ZONE-A", "ZONE-B"),
        )

        observed = {floor.get_zone(zone_id).name for zone_id in camera.zone_ids}

        self.assertEqual(observed, {"Lobby", "Corridor"})

    def test_a_stale_zone_id_resolves_to_none_rather_than_raising(self):

        # A zone referenced by zone_ids may later be deleted from the
        # Floor -- the query must degrade honestly (None for that one
        # id), never raise and never fabricate a Zone.
        building, floor, camera = _building_with_camera_and_zones(
            zone_ids_on_camera=("ZONE-A", "ZONE-GONE"),
        )

        observed = [floor.get_zone(zone_id) for zone_id in camera.zone_ids]

        self.assertEqual(observed[0].name, "Lobby")
        self.assertIsNone(observed[1])


class BuildingLevelRoundTripTests(unittest.TestCase):

    # The level that actually gets written by serialization.Serializer
    # (Project.to_dict() -> Building.to_dict() -> Floor.to_dict() ->
    # Camera.to_dict()) -- proves the assignment survives a full
    # Building round trip, not just Camera's or Floor's own in
    # isolation (already covered by tests/test_camera.py).

    def test_camera_zone_assignment_survives_a_full_building_round_trip(self):

        building, floor, camera = _building_with_camera_and_zones(
            zone_ids_on_camera=("ZONE-A", "ZONE-B"),
        )

        restored = Building.from_dict(building.to_dict())

        restored_floor = restored.get_floor(floor.id)
        restored_camera = restored_floor.cameras[0]

        self.assertEqual(set(restored_camera.zone_ids), {"ZONE-A", "ZONE-B"})

        observed = {restored_floor.get_zone(zone_id).name for zone_id in restored_camera.zone_ids}
        self.assertEqual(observed, {"Lobby", "Corridor"})

    def test_a_project_with_no_zone_ids_key_at_all_still_loads(self):

        # An existing .syn project saved before Camera.zone_ids existed
        # -- the whole point of the backward-compatibility requirement:
        # it must continue to load normally, with the camera simply
        # unassigned (never a crash, never a fabricated assignment).
        legacy_building_dict = {
            "id": "b1",
            "name": "Legacy Building",
            "floors": [
                {
                    "id": "f1",
                    "name": "Ground Floor",
                    "display_order": 0,
                    "height": 3.0,
                    "floor_plan": "",
                    "visible": True,
                    "locked": False,
                    "zones": [],
                    "exits": [],
                    "stairs": [],
                    "elevators": [],
                    "cameras": [
                        {
                            "id": "cam-legacy",
                            "name": "Legacy Camera",
                            "object_type": "Camera",
                            "properties": {},
                            "created_at": "",
                            "modified_at": "",
                            "position": (0.0, 0.0),
                            "floor_id": "f1",
                            "rotation": 0.0,
                            "horizontal_fov": 90.0,
                            "max_range": 25.0,
                            "mount_height": 3.0,
                            "active": True,
                            # No "zone_ids" key at all.
                        }
                    ],
                    "detectors": [],
                    "assembly_points": [],
                    "obstacles": [],
                    "doors": [],
                }
            ],
        }

        restored = Building.from_dict(legacy_building_dict)
        restored_camera = restored.get_floor("f1").cameras[0]

        self.assertEqual(restored_camera.zone_ids, ())


if __name__ == "__main__":
    unittest.main()

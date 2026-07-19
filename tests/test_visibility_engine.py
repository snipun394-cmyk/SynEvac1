import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.obstacle import Obstacle
from models.zone import Zone

from visibility.engine import VisibilityEngine


def make_zone(name, x, y, width, height, floor_id=""):

    return Zone(name=name, x=x, y=y, width=width, height=height, floor_id=floor_id)


class SingleZoneVisibilityTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.engine = VisibilityEngine()

    def test_camera_facing_into_its_own_zone_sees_it(self):

        # A wide-but-not-omnidirectional FOV from a point near the
        # zone's own wall still leaves the two corners closest to the
        # camera's own position (almost directly "sideways" from it)
        # outside the wedge -- this asserts "sees most of it", not
        # "sees literally every square inch", which is the geometric
        # reality of a fixed-position, fixed-FOV camera.
        camera = Camera(
            position=(1.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=150.0, max_range=20.0,
        )

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertNotIn(self.zone.id, visibility.hidden_zone_ids)
        self.assertGreater(visibility.zone_coverage[self.zone.id], 0.7)

    def test_camera_facing_away_from_its_own_zone_sees_nothing(self):

        camera = Camera(
            position=(1.0, 5.0), floor_id=self.floor.id,
            # Facing -x, straight out of the zone's own left wall.
            rotation=180.0, horizontal_fov=60.0, max_range=20.0,
        )

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertIn(self.zone.id, visibility.hidden_zone_ids)
        self.assertEqual(visibility.zone_coverage[self.zone.id], 0.0)

    def test_short_range_only_partially_covers_a_large_zone(self):

        camera = Camera(
            position=(0.5, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=3.0,
        )

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertIn(self.zone.id, visibility.partially_visible_zone_ids)
        self.assertGreater(visibility.zone_coverage[self.zone.id], 0.0)
        self.assertLess(visibility.zone_coverage[self.zone.id], 1.0)

    def test_narrower_fov_covers_less_of_the_zone_than_a_wider_one(self):

        narrow = Camera(
            position=(0.5, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=20.0, max_range=20.0,
        )
        wide = Camera(
            position=(0.5, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=170.0, max_range=20.0,
        )

        narrow_visibility = self.engine.compute_camera_visibility(narrow, self.building)
        wide_visibility = self.engine.compute_camera_visibility(wide, self.building)

        self.assertLess(
            narrow_visibility.zone_coverage[self.zone.id],
            wide_visibility.zone_coverage[self.zone.id],
        )

    def test_max_visible_distance_never_exceeds_the_cameras_own_range(self):

        camera = Camera(
            position=(1.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=6.0,
        )

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertLessEqual(visibility.max_visible_distance, 6.0 + 1e-6)
        self.assertGreater(visibility.max_visible_distance, 0.0)

    def test_inactive_camera_sees_nothing(self):

        camera = Camera(
            position=(1.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=20.0, active=False,
        )

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertEqual(visibility.zone_coverage[self.zone.id], 0.0)
        self.assertIn(self.zone.id, visibility.hidden_zone_ids)
        self.assertEqual(visibility.visibility_polygon, ())

    def test_camera_on_a_floor_with_no_matching_floor_id_is_handled_gracefully(self):

        camera = Camera(position=(1.0, 5.0), floor_id="nonexistent-floor")

        visibility = self.engine.compute_camera_visibility(camera, self.building)

        self.assertEqual(visibility.zone_coverage, {})
        self.assertEqual(visibility.visibility_polygon, ())


class WallOcclusionTests(unittest.TestCase):

    # Two adjacent zones sharing a wall at x=5 -- Zone B is only
    # visible from Zone A through an explicit Door opening, never
    # straight through the solid wall.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("Zone A", 0.0, 0.0, 5.0, 5.0, floor_id=self.floor.id)
        self.zone_b = make_zone("Zone B", 5.0, 0.0, 5.0, 5.0, floor_id=self.floor.id)

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

    def test_zone_b_is_hidden_when_there_is_no_door_between_them(self):

        camera = Camera(
            position=(1.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=20.0,
        )

        visibility = VisibilityEngine().compute_camera_visibility(camera, self.building)

        self.assertEqual(visibility.zone_coverage[self.zone_b.id], 0.0)
        self.assertIn(self.zone_b.id, visibility.hidden_zone_ids)

    def test_zone_b_becomes_partially_visible_through_an_explicit_door(self):

        door = Door(
            name="D", start_point=(5.0, 2.0), end_point=(5.0, 3.0),
            floor_id=self.floor.id, width=1.0,
            zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
        )
        self.floor.add_door(door)

        camera = Camera(
            position=(1.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=20.0,
        )

        visibility = VisibilityEngine().compute_camera_visibility(camera, self.building)

        self.assertGreater(visibility.zone_coverage[self.zone_b.id], 0.0)
        self.assertIn(door.id, visibility.visible_door_ids)

    def test_door_belonging_to_a_different_zone_pair_does_not_open_this_wall(self):

        # A Door that merely happens to be geometrically drawn on the
        # A/B wall, but whose connectivity references neither zone,
        # must not carve a gap -- connectivity is explicit, never
        # inferred from geometry (see visibility/segments.py).
        door = Door(
            name="Unrelated Door", start_point=(5.0, 2.0), end_point=(5.0, 3.0),
            floor_id=self.floor.id, width=1.0,
            zone_a_id="some-other-zone", zone_b_id="yet-another-zone",
        )
        self.floor.add_door(door)

        camera = Camera(
            position=(1.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=20.0,
        )

        visibility = VisibilityEngine().compute_camera_visibility(camera, self.building)

        self.assertEqual(visibility.zone_coverage[self.zone_b.id], 0.0)


class ObstacleOcclusionTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", 0.0, 0.0, 20.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

    def test_blocked_obstacle_shortens_visible_distance(self):

        obstacle = Obstacle(
            name="Wall Segment", x=5.0, y=4.0, length=0.2, width=2.0,
            floor_id=self.floor.id, traversability="Blocked",
        )
        self.floor.add_obstacle(obstacle)

        camera = Camera(
            position=(0.5, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=10.0, max_range=20.0,
        )

        visibility = VisibilityEngine().compute_camera_visibility(camera, self.building)

        self.assertLess(visibility.max_visible_distance, 10.0)

    def test_passable_obstacle_does_not_block_sight(self):

        obstacle = Obstacle(
            name="Low Furniture", x=5.0, y=4.0, length=0.2, width=2.0,
            floor_id=self.floor.id, traversability="Passable",
        )
        self.floor.add_obstacle(obstacle)

        camera = Camera(
            position=(0.5, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=10.0, max_range=20.0,
        )

        visibility = VisibilityEngine().compute_camera_visibility(camera, self.building)

        self.assertGreater(visibility.max_visible_distance, 15.0)


class PhaseSixSeamTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.camera = Camera(
            position=(1.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=90.0, max_range=20.0,
        )

        self.engine = VisibilityEngine()

    def test_point_is_visible_true_for_a_point_in_front_of_the_camera(self):

        self.assertTrue(
            self.engine.point_is_visible(self.camera, self.building, (5.0, 5.0))
        )

    def test_point_is_visible_false_for_a_point_behind_the_camera(self):

        self.assertFalse(
            self.engine.point_is_visible(self.camera, self.building, (-5.0, 5.0))
        )

    def test_is_within_expected_area_matches_point_is_visible(self):

        for point in ((5.0, 5.0), (-5.0, 5.0), (9.0, 1.0)):

            self.assertEqual(
                self.engine.is_within_expected_area(self.camera, self.building, point),
                self.engine.point_is_visible(self.camera, self.building, point),
            )


if __name__ == "__main__":
    unittest.main()

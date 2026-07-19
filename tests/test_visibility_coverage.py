import unittest

from models.building import Building
from models.camera import Camera
from models.zone import Zone

from visibility.coverage import compute_floor_coverage


def make_zone(name, x, y, width, height, floor_id=""):

    return Zone(name=name, x=x, y=y, width=width, height=height, floor_id=floor_id)


class FloorCoverageTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        # Zone A: covered by camera 1 only.
        # Zone B: covered by both cameras (overlap).
        # Zone C: covered by neither (blind).
        self.zone_a = make_zone("Zone A", 0.0, 0.0, 5.0, 5.0, floor_id=self.floor.id)
        self.zone_b = make_zone("Zone B", 5.0, 0.0, 5.0, 5.0, floor_id=self.floor.id)
        self.zone_c = make_zone("Zone C", 10.0, 0.0, 5.0, 5.0, floor_id=self.floor.id)

        for zone in (self.zone_a, self.zone_b, self.zone_c):
            self.floor.add_zone(zone)

        # Each zone's own walls are opaque on every side with no
        # doors cut into them here, so a camera only ever sees the
        # zone it physically stands in -- exactly the wall-occlusion
        # behavior WallOcclusionTests (tests/test_visibility_engine.py)
        # already proves. Camera 1 alone in Zone A; Cameras 2 and 3
        # both in Zone B (the overlap case); Zone C gets none (the
        # blind case).
        self.camera_1 = Camera(
            position=(2.5, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=15.0,
        )

        self.camera_2 = Camera(
            position=(6.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=15.0,
        )

        self.camera_3 = Camera(
            position=(9.0, 2.5), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=15.0,
        )

        self.cameras = [self.camera_1, self.camera_2, self.camera_3]

    def test_zone_seen_by_only_one_camera_is_not_overlapping(self):

        result = compute_floor_coverage(self.cameras, self.building, self.floor)

        self.assertGreater(result.zone_combined_coverage[self.zone_a.id], 0.0)
        self.assertNotIn(self.zone_a.id, result.overlapping_zone_ids)

    def test_zone_seen_by_both_cameras_is_overlapping(self):

        result = compute_floor_coverage(self.cameras, self.building, self.floor)

        self.assertIn(self.zone_b.id, result.overlapping_zone_ids)
        self.assertGreater(result.zone_overlap_coverage[self.zone_b.id], 0.0)

    def test_zone_seen_by_neither_camera_is_uncovered(self):

        result = compute_floor_coverage(self.cameras, self.building, self.floor)

        self.assertIn(self.zone_c.id, result.uncovered_zone_ids)
        self.assertEqual(result.zone_combined_coverage[self.zone_c.id], 0.0)

    def test_total_floor_coverage_is_area_weighted(self):

        # All three zones are equal-area (5x5=25 each); Zone C is
        # entirely uncovered, so the floor-wide fraction should sit
        # meaningfully below 100% but above 0%, roughly reflecting
        # "2 covered zones out of 3 equal-area zones".
        result = compute_floor_coverage(self.cameras, self.building, self.floor)

        self.assertGreater(result.total_floor_coverage_fraction, 0.4)
        self.assertLess(result.total_floor_coverage_fraction, 0.8)

    def test_per_camera_results_are_included_and_keyed_by_camera_id(self):

        result = compute_floor_coverage(self.cameras, self.building, self.floor)

        self.assertEqual(
            set(result.per_camera.keys()),
            {self.camera_1.id, self.camera_2.id, self.camera_3.id},
        )

    def test_camera_on_a_different_floor_is_excluded(self):

        other_floor = self.building.create_floor(name="Floor 2", height=3.0)

        elsewhere_camera = Camera(
            position=(2.5, 2.5), floor_id=other_floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=15.0,
        )

        result = compute_floor_coverage(
            [self.camera_1, elsewhere_camera], self.building, self.floor,
        )

        self.assertEqual(set(result.per_camera.keys()), {self.camera_1.id})

    def test_no_cameras_leaves_every_zone_uncovered(self):

        result = compute_floor_coverage([], self.building, self.floor)

        self.assertEqual(result.total_floor_coverage_fraction, 0.0)
        self.assertEqual(
            set(result.uncovered_zone_ids),
            {self.zone_a.id, self.zone_b.id, self.zone_c.id},
        )


if __name__ == "__main__":
    unittest.main()

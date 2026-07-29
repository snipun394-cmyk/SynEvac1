import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase, StairObservableRegion

from camera_calibration.stair_lookup import build_stairs_by_floor, covered_stair_ids, locate_stair


# =====================================================
# Observable Stair Perception milestone -- deterministic, offline unit
# tests for the observable-region model (Phase 2/3) and the spatial
# lookup (Phase 5). No randomness anywhere in this file.
# =====================================================


def make_building_with_stair(with_regions=True):

    building = Building(name="Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0)
    floor_2 = Floor(name="Floor 2", display_order=1)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stair = Staircase(
        name="Stair S1",
        from_floor_id=floor_1.id, to_floor_id=floor_2.id,
        from_position=(10.0, 10.0), to_position=(50.0, 50.0),
        width=1.5,
    )

    if with_regions:
        stair.from_observable_region = StairObservableRegion(center_x=10.0, center_y=10.0, width=4.0, depth=4.0)
        stair.to_observable_region = StairObservableRegion(center_x=50.0, center_y=50.0, width=4.0, depth=4.0)

    floor_1.add_stair(stair)

    return building, floor_1, floor_2, stair


class StairObservableRegionGeometryTests(unittest.TestCase):

    def test_1_contains_inside_and_outside_points(self):

        region = StairObservableRegion(center_x=10.0, center_y=10.0, width=4.0, depth=2.0)

        self.assertTrue(region.contains(10.0, 10.0))
        self.assertTrue(region.contains(11.9, 10.9))
        self.assertFalse(region.contains(12.1, 10.0))
        self.assertFalse(region.contains(10.0, 11.1))

    def test_2_boundary_points_are_inclusive(self):

        region = StairObservableRegion(center_x=0.0, center_y=0.0, width=4.0, depth=2.0)

        self.assertTrue(region.contains(2.0, 1.0))    # exact top-right corner
        self.assertTrue(region.contains(-2.0, -1.0))   # exact bottom-left corner

    def test_3_serialization_round_trip(self):

        region = StairObservableRegion(center_x=1.5, center_y=2.5, width=3.0, depth=1.0)
        restored = StairObservableRegion.from_dict(region.to_dict())

        self.assertEqual(restored, region)


class StaircaseObservableRegionLookupTests(unittest.TestCase):

    def test_4_contains_world_point_on_from_floor(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        self.assertTrue(stair.contains_world_point(floor_1.id, (10.0, 10.0)))
        self.assertFalse(stair.contains_world_point(floor_1.id, (50.0, 50.0)))

    def test_5_contains_world_point_on_to_floor_uses_different_region(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        self.assertTrue(stair.contains_world_point(floor_2.id, (50.0, 50.0)))
        # The from-floor region's coordinates must NOT leak into a
        # to-floor lookup -- (10.0, 10.0) is inside from_observable_region
        # but outside to_observable_region (centered at 50,50).
        self.assertFalse(stair.contains_world_point(floor_2.id, (10.0, 10.0)))

    def test_6_wrong_floor_never_matches(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        other_floor = Floor(name="Floor 3", display_order=2)
        building.add_floor(other_floor)

        self.assertFalse(stair.contains_world_point(other_floor.id, (10.0, 10.0)))

    def test_7_missing_region_means_unavailable_not_fabricated(self):

        building, floor_1, floor_2, stair = make_building_with_stair(with_regions=False)

        self.assertIsNone(stair.observable_region_for_floor(floor_1.id))
        self.assertFalse(stair.contains_world_point(floor_1.id, (10.0, 10.0)))
        self.assertFalse(stair.contains_world_point(floor_2.id, (50.0, 50.0)))

    def test_8_legacy_from_dict_without_region_produces_none(self):

        legacy_data = {
            "id": "STAIR-LEGACY", "name": "Old Stair",
            "from_floor_id": "f1", "to_floor_id": "f2",
            "from_position": (0.0, 0.0), "to_position": (1.0, 1.0),
            "from_zone_id": "", "to_zone_id": "", "width": 1.5,
            # no from_observable_region/to_observable_region key at all --
            # exactly what a pre-milestone .syn file would contain.
        }

        stair = Staircase.from_dict(legacy_data)

        self.assertIsNone(stair.from_observable_region)
        self.assertIsNone(stair.to_observable_region)

    def test_9_serialization_round_trip_with_regions(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        restored = Staircase.from_dict(stair.to_dict())

        self.assertEqual(restored.from_observable_region, stair.from_observable_region)
        self.assertEqual(restored.to_observable_region, stair.to_observable_region)


class LocateStairTests(unittest.TestCase):

    def test_10_no_match_returns_none_not_ambiguous(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        result = locate_stair([stair], floor_1.id, (500.0, 500.0))

        self.assertIsNone(result.stair_id)
        self.assertFalse(result.ambiguous)

    def test_11_exactly_one_match(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        result = locate_stair([stair], floor_1.id, (10.0, 10.0))

        self.assertEqual(result.stair_id, stair.id)
        self.assertFalse(result.ambiguous)

    def test_12_ambiguous_overlap_never_arbitrarily_resolved(self):

        building, floor_1, floor_2, stair_a = make_building_with_stair()

        stair_b = Staircase(
            name="Stair S2", from_floor_id=floor_1.id, to_floor_id=floor_2.id,
            from_position=(10.5, 10.5), to_position=(20.0, 20.0), width=1.5,
        )
        stair_b.from_observable_region = StairObservableRegion(center_x=10.5, center_y=10.5, width=4.0, depth=4.0)
        floor_1.add_stair(stair_b)

        # (10.2, 10.2) genuinely falls inside BOTH stairs' overlapping
        # from_observable_regions.
        result = locate_stair([stair_a, stair_b], floor_1.id, (10.2, 10.2))

        self.assertIsNone(result.stair_id)
        self.assertTrue(result.ambiguous)

    def test_13_deleted_stair_reference_cannot_leak_a_stale_match(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        # Simulates deletion: the caller's CURRENT stairs list no longer
        # contains `stair` at all.
        result = locate_stair([], floor_1.id, (10.0, 10.0))

        self.assertIsNone(result.stair_id)
        self.assertFalse(result.ambiguous)

    def test_14_boundary_point_matches(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        # Exact edge of the 4x4 region centered at (10, 10).
        result = locate_stair([stair], floor_1.id, (12.0, 10.0))

        self.assertEqual(result.stair_id, stair.id)


class BuildStairsByFloorTests(unittest.TestCase):

    def test_15_stair_appears_under_both_floor_ids(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        by_floor = build_stairs_by_floor(building)

        self.assertIn(stair, by_floor[floor_1.id])
        self.assertIn(stair, by_floor[floor_2.id])

    def test_16_lookup_via_prebuilt_mapping_resolves_correct_side(self):

        building, floor_1, floor_2, stair = make_building_with_stair()

        by_floor = build_stairs_by_floor(building)

        from_match = locate_stair(by_floor[floor_1.id], floor_1.id, (10.0, 10.0))
        to_match = locate_stair(by_floor[floor_2.id], floor_2.id, (50.0, 50.0))
        cross_match = locate_stair(by_floor[floor_1.id], floor_1.id, (50.0, 50.0))  # to-side coords on from-floor

        self.assertEqual(from_match.stair_id, stair.id)
        self.assertEqual(to_match.stair_id, stair.id)
        self.assertIsNone(cross_match.stair_id)


class CoveredStairIdsTests(unittest.TestCase):

    def test_17_stair_with_region_and_calibrated_floor_is_covered(self):

        building, floor_1, floor_2, stair = make_building_with_stair()
        by_floor = build_stairs_by_floor(building)

        covered = covered_stair_ids(by_floor, frozenset({floor_1.id}))

        self.assertIn(stair.id, covered)

    def test_18_stair_with_region_but_no_calibrated_floor_is_not_covered(self):

        building, floor_1, floor_2, stair = make_building_with_stair()
        by_floor = build_stairs_by_floor(building)

        covered = covered_stair_ids(by_floor, frozenset())  # no calibrated cameras at all

        self.assertNotIn(stair.id, covered)

    def test_19_stair_without_any_region_is_never_covered_even_if_calibrated(self):

        building, floor_1, floor_2, stair = make_building_with_stair(with_regions=False)
        by_floor = build_stairs_by_floor(building)

        covered = covered_stair_ids(by_floor, frozenset({floor_1.id, floor_2.id}))

        self.assertNotIn(stair.id, covered)


if __name__ == "__main__":
    unittest.main()

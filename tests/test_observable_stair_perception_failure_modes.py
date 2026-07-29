import os
import tempfile
import unittest

from models.building import Building
from models.floor import Floor
from models.project import Project
from models.staircase import Staircase, StairObservableRegion

from camera_calibration.asset_lookup import build_assets_by_floor, covered_asset_ids, locate_asset
from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector
from camera_calibration.stair_lookup import DEFAULT_OBSERVABLE_ASSET_KINDS, build_stairs_by_floor

from live_occupants.manager import LiveOccupantManager

from serialization.serializer import Serializer

from observable_assets.facts import compute_asset_occupancy_snapshot
from observable_assets.models import ObservationStatus


# =====================================================
# Observable Stair Perception milestone, Phase 25 -- degradation/
# failure-mode coverage not already exercised by tests/test_stair_
# observable_geometry.py, tests/test_stair_perception_pipeline.py, or
# tests/test_observable_stair_perception_e2e.py. No crashes, no
# fabricated occupancy anywhere in this file.
# =====================================================


def make_calibration(camera_id="CAM-1", floor_id="floor-1"):

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0)

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


class MissingOrInvalidCalibrationTests(unittest.TestCase):

    def test_1_camera_unavailable_never_crashes_stair_lookup(self):

        stair = Staircase(id="s1", from_floor_id="floor-1", to_floor_id="floor-2")
        stair.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=2.0, depth=2.0)

        # No calibration at all for "CAM-UNKNOWN" -- WorldProjector's
        # own existing "no honest basis to project at all" branch.
        projector = WorldProjector(calibrations={}, zones_by_floor={}, stairs_by_floor={"floor-1": (stair,)})

        result = projector.project("CAM-UNKNOWN", (315.0, 200.0, 325.0, 240.0), 0.9)

        self.assertIsNone(result.world_position)
        self.assertIsNone(result.stair_id)
        self.assertFalse(result.stair_localization_ambiguous)

    def test_2_no_bounding_box_never_crashes_stair_lookup(self):

        stair = Staircase(id="s1", from_floor_id="floor-1", to_floor_id="floor-2")
        stair.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=2.0, depth=2.0)

        projector = WorldProjector(
            calibrations={"CAM-1": make_calibration()}, zones_by_floor={}, stairs_by_floor={"floor-1": (stair,)},
        )

        result = projector.project("CAM-1", None, 0.9)

        self.assertIsNone(result.world_position)
        self.assertIsNone(result.stair_id)

    def test_3_uncovered_floor_reports_unknown_not_zero(self):

        stair = Staircase(id="s1", from_floor_id="floor-1", to_floor_id="floor-2")
        stair.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=2.0, depth=2.0)

        assets_by_floor = {"floor-1": (("Stair", stair),)}

        # No calibrated camera on floor-1 at all this cycle.
        covered = covered_asset_ids(assets_by_floor, frozenset())

        snapshot = compute_asset_occupancy_snapshot(
            asset_ids_by_type={"Stair": ["s1"]}, occupant_ids_by_asset={}, covered_asset_ids=covered, timestamp=0.0,
        )

        self.assertEqual(snapshot.observation_for("s1").status, ObservationStatus.UNKNOWN)


class EmptyBuildingTests(unittest.TestCase):

    def test_4_building_with_no_stairs_never_crashes(self):

        building = Building(name="Empty Building")
        floor = Floor(name="Floor 1", display_order=0)
        building.add_floor(floor)

        by_floor = build_stairs_by_floor(building)
        self.assertEqual(by_floor, {})

        assets_by_floor = build_assets_by_floor(building, DEFAULT_OBSERVABLE_ASSET_KINDS)
        self.assertEqual(assets_by_floor, {})

        covered = covered_asset_ids(assets_by_floor, frozenset({floor.id}))
        self.assertEqual(covered, frozenset())

        snapshot = compute_asset_occupancy_snapshot(
            asset_ids_by_type={}, occupant_ids_by_asset={}, covered_asset_ids=covered, timestamp=0.0,
        )
        self.assertEqual(snapshot.observations, {})

    def test_5_building_with_no_floors_never_crashes(self):

        building = Building(name="Totally Empty Building")

        by_floor = build_stairs_by_floor(building)
        self.assertEqual(by_floor, {})


class DeletedAndOverlappingStairTests(unittest.TestCase):

    def test_6_stair_removed_from_floor_between_cycles_never_matches_again(self):

        floor = Floor(name="Floor 1", display_order=0)
        building = Building(name="B")
        building.add_floor(floor)

        stair = Staircase(name="S1", from_floor_id=floor.id, to_floor_id=floor.id)
        stair.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=2.0, depth=2.0)
        floor.add_stair(stair)

        assets_by_floor_before = build_assets_by_floor(building, DEFAULT_OBSERVABLE_ASSET_KINDS)
        self.assertEqual(locate_asset(assets_by_floor_before[floor.id], floor.id, (3.0, 0.0)).asset_id, stair.id)

        floor.remove_stair(stair)

        assets_by_floor_after = build_assets_by_floor(building, DEFAULT_OBSERVABLE_ASSET_KINDS)
        result = locate_asset(assets_by_floor_after.get(floor.id, ()), floor.id, (3.0, 0.0))

        self.assertIsNone(result.asset_id)
        self.assertFalse(result.ambiguous)

    def test_7_overlapping_regions_across_two_stairs_stay_unresolved(self):

        floor = Floor(name="Floor 1", display_order=0)
        building = Building(name="B")
        building.add_floor(floor)

        stair_a = Staircase(name="S1", from_floor_id=floor.id, to_floor_id=floor.id)
        stair_a.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=4.0, depth=4.0)
        stair_b = Staircase(name="S2", from_floor_id=floor.id, to_floor_id=floor.id)
        stair_b.from_observable_region = StairObservableRegion(center_x=3.0, center_y=0.0, width=4.0, depth=4.0)

        floor.add_stair(stair_a)
        floor.add_stair(stair_b)

        assets_by_floor = build_assets_by_floor(building, DEFAULT_OBSERVABLE_ASSET_KINDS)
        result = locate_asset(assets_by_floor[floor.id], floor.id, (3.0, 0.0))

        self.assertIsNone(result.asset_id)
        self.assertTrue(result.ambiguous)


class OccupantLifecycleFailureModeTests(unittest.TestCase):

    def test_8_occupant_leaving_stair_without_ever_entering_a_zone_stays_honest(self):

        manager = LiveOccupantManager()

        occupant = manager.update(
            "OCC-1", "CAM-1", "T1", None, "floor-1", (3.0, 0.0), None, None, 0.9, 0.0, stair_id="STAIR-1",
        )
        self.assertEqual(occupant.current_stair_id, "STAIR-1")
        self.assertIsNone(occupant.current_zone_id)  # never fabricated

        left = manager.update(
            "OCC-1", "CAM-1", "T1", None, "floor-1", (30.0, 30.0), None, None, 0.9, 1.0, stair_id=None,
        )
        self.assertIsNone(left.current_stair_id)
        self.assertIsNone(left.current_zone_id)  # still honestly unresolved, not silently a stale value


class SaveReloadTests(unittest.TestCase):

    def test_9_project_with_observable_stair_region_round_trips(self):

        floor_1 = Floor(name="Floor 1", display_order=0)
        floor_2 = Floor(name="Floor 2", display_order=1)

        building = Building(name="Round Trip Building")
        building.add_floor(floor_1)
        building.add_floor(floor_2)

        stair = Staircase(
            name="Stair S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id,
            from_position=(1.0, 2.0), to_position=(3.0, 4.0), width=1.5,
        )
        stair.from_observable_region = StairObservableRegion(center_x=1.0, center_y=2.0, width=2.0, depth=2.0)
        stair.to_observable_region = StairObservableRegion(center_x=3.0, center_y=4.0, width=2.0, depth=2.0)
        floor_1.add_stair(stair)

        project = Project(name="Round Trip Project")
        project.set_building(building)

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "roundtrip.syn")
            Serializer.save(project, path)
            loaded = Serializer.load(path)

        loaded_floor_1 = loaded.building.get_floor(floor_1.id)
        loaded_stair = loaded_floor_1.stairs[0]

        self.assertEqual(loaded_stair.from_observable_region, stair.from_observable_region)
        self.assertEqual(loaded_stair.to_observable_region, stair.to_observable_region)

    def test_10_legacy_project_without_observable_region_key_loads_cleanly(self):

        floor_1 = Floor(name="Floor 1", display_order=0)
        floor_2 = Floor(name="Floor 2", display_order=1)

        building = Building(name="Legacy Building")
        building.add_floor(floor_1)
        building.add_floor(floor_2)

        stair = Staircase(
            name="Stair S1", from_floor_id=floor_1.id, to_floor_id=floor_2.id,
            from_position=(1.0, 2.0), to_position=(3.0, 4.0), width=1.5,
        )
        floor_1.add_stair(stair)

        project = Project(name="Legacy Project")
        project.set_building(building)

        data = project.to_dict()

        # Simulate a genuinely pre-milestone .syn file: strip the new
        # keys entirely (to_dict() always emits them as None today, but
        # an OLDER version of this codebase never wrote them at all).
        stair_data = data["building"]["floors"][0]["stairs"][0]
        del stair_data["from_observable_region"]
        del stair_data["to_observable_region"]

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = os.path.join(tmp_dir, "legacy.syn")

            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            loaded = Serializer.load(path)

        loaded_stair = loaded.building.get_floor(floor_1.id).stairs[0]

        self.assertIsNone(loaded_stair.from_observable_region)
        self.assertIsNone(loaded_stair.to_observable_region)


if __name__ == "__main__":
    unittest.main()

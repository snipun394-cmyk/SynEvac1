import unittest

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.camera import Camera
from models.staircase import Staircase, StairObservableRegion

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics

from camera_coverage.discovery import (
    NO_CALIBRATION, NO_OBSERVABLE_REGION, NO_USABLE_SECTOR,
    compute_camera_coverage, compute_camera_coverage_snapshot,
)
from camera_coverage.models import AssetCoverage, CameraCoverage, CameraCoverageSnapshot, CoverageState

from building_state.estimator import BuildingStateEstimator


# =====================================================
# Camera Coverage Intelligence & Observable Asset Mapping milestone --
# deterministic, offline unit tests. No randomness anywhere in this
# file: every camera pose/asset geometry/expected coverage state is
# hand-chosen and independently verifiable.
#
# A calibration built with image_width=1000, focal_length_x=500 yields
# an EXACT 90-degree derived horizontal FOV (fov = 2*atan((1000/2)/500)
# = 2*atan(1) = 90 degrees) -- used throughout so expected sector
# geometry is trivial to reason about by hand: yaw=0 means the sector is
# the 90-degree wedge from -45 to +45 degrees around +x.
# =====================================================


def make_calibration(camera_id="CAM-1", floor_id="floor-1", position=(0.0, 0.0), yaw_degrees=0.0):

    intrinsics = CameraIntrinsics(image_width=1000, image_height=1000, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=position, mount_height=3.0, yaw_degrees=yaw_degrees)

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


def make_camera(camera_id="CAM-1", floor_id="floor-1", max_range=25.0):

    camera = Camera(floor_id=floor_id, max_range=max_range)
    camera.id = camera_id
    return camera


def make_stair(stair_id, from_floor_id="floor-1", to_floor_id="floor-2", region=None):

    stair = Staircase(from_floor_id=from_floor_id, to_floor_id=to_floor_id, from_observable_region=region)
    stair.id = stair_id
    return stair


class SingleCameraSingleAssetTests(unittest.TestCase):

    def test_1_fully_visible_when_region_entirely_inside_sector(self):

        camera = make_camera()
        calibration = make_calibration()

        region = StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)

        coverage = compute_camera_coverage(camera, calibration, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.FULLY_VISIBLE)
        self.assertEqual(entry.asset_type, "Stair")
        self.assertIsNotNone(entry.region_polygon)
        self.assertEqual(len(entry.region_polygon), 4)
        self.assertIn("S1", coverage.covered_asset_ids())

    def test_2_not_visible_when_region_behind_camera(self):

        camera = make_camera()
        calibration = make_calibration()

        region = StairObservableRegion(center_x=-10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)

        coverage = compute_camera_coverage(camera, calibration, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.NOT_VISIBLE)
        self.assertIsNone(entry.region_polygon)
        self.assertNotIn("S1", coverage.covered_asset_ids())

    def test_3_partially_visible_when_region_straddles_max_range(self):

        camera = make_camera(max_range=25.0)
        calibration = make_calibration()

        # Straight ahead (angle 0, always inside the 90deg cone) but
        # straddling the range boundary at x=25: corner at x=22 is in
        # range, corner at x=28 is not.
        region = StairObservableRegion(center_x=25.0, center_y=0.0, width=6.0, depth=2.0)
        stair = make_stair("S1", region=region)

        coverage = compute_camera_coverage(camera, calibration, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.PARTIALLY_VISIBLE)
        self.assertIsNotNone(entry.region_polygon)

    def test_4_unknown_when_camera_has_no_calibration(self):

        camera = make_camera()

        region = StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)

        coverage = compute_camera_coverage(camera, None, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.UNKNOWN)
        self.assertEqual(entry.provenance, NO_CALIBRATION)
        self.assertIsNone(coverage.sector_polygon)

    def test_5_unknown_when_asset_has_no_observable_region_legacy_project(self):

        # Mirrors a .syn project saved before Observable Stair Perception
        # existed, or one where an operator has simply never authored a
        # region yet -- from_observable_region stays None, never a
        # fabricated footprint.
        camera = make_camera()
        calibration = make_calibration()

        stair = make_stair("S1", region=None)

        coverage = compute_camera_coverage(camera, calibration, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.UNKNOWN)
        self.assertEqual(entry.provenance, NO_OBSERVABLE_REGION)

    def test_6_unknown_when_camera_has_no_detection_range(self):

        camera = make_camera(max_range=0.0)
        calibration = make_calibration()

        region = StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)

        coverage = compute_camera_coverage(camera, calibration, candidates=[("Stair", stair)])

        entry = coverage.coverage_for("S1")
        self.assertEqual(entry.state, CoverageState.UNKNOWN)
        self.assertEqual(entry.provenance, NO_USABLE_SECTOR)

    def test_7_absent_asset_id_defaults_to_unknown(self):

        camera = make_camera()
        coverage = compute_camera_coverage(camera, make_calibration(), candidates=[])

        entry = coverage.coverage_for("never-considered")
        self.assertEqual(entry.state, CoverageState.UNKNOWN)


class SingleCameraMultipleAssetTests(unittest.TestCase):

    def test_8_one_camera_covers_multiple_assets(self):

        camera = make_camera()
        calibration = make_calibration()

        stair_a = make_stair("S1", region=StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0))
        stair_b = make_stair("S2", region=StairObservableRegion(center_x=5.0, center_y=3.0, width=2.0, depth=2.0))
        stair_c = make_stair("S3", region=StairObservableRegion(center_x=-10.0, center_y=0.0, width=2.0, depth=2.0))

        coverage = compute_camera_coverage(
            camera, calibration, candidates=[("Stair", stair_a), ("Stair", stair_b), ("Stair", stair_c)],
        )

        self.assertEqual(coverage.coverage_for("S1").state, CoverageState.FULLY_VISIBLE)
        self.assertEqual(coverage.coverage_for("S2").state, CoverageState.FULLY_VISIBLE)
        self.assertEqual(coverage.coverage_for("S3").state, CoverageState.NOT_VISIBLE)
        self.assertEqual(coverage.covered_asset_ids(), frozenset({"S1", "S2"}))


class MultiCameraCoverageSnapshotTests(unittest.TestCase):

    def test_9_multiple_cameras_can_observe_one_asset(self):

        region = StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)

        camera_a = make_camera(camera_id="CAM-A")
        camera_b = make_camera(camera_id="CAM-B")

        calibration_a = make_calibration(camera_id="CAM-A", position=(0.0, 0.0), yaw_degrees=0.0)
        # Looking back at the same region from the opposite side.
        calibration_b = make_calibration(camera_id="CAM-B", position=(20.0, 0.0), yaw_degrees=180.0)

        snapshot = compute_camera_coverage_snapshot(
            cameras=[camera_a, camera_b],
            calibrations={"CAM-A": calibration_a, "CAM-B": calibration_b},
            assets_by_floor={"floor-1": (("Stair", stair),)},
        )

        self.assertEqual(snapshot.cameras_observing("S1"), ("CAM-A", "CAM-B"))
        self.assertEqual(snapshot.assets_observed_by("CAM-A"), ("S1",))
        self.assertEqual(snapshot.assets_observed_by("CAM-B"), ("S1",))

    def test_10_cameras_observing_empty_for_uncovered_asset(self):

        stair = make_stair("S1", region=StairObservableRegion(center_x=-10.0, center_y=0.0, width=2.0, depth=2.0))
        camera = make_camera()

        snapshot = compute_camera_coverage_snapshot(
            cameras=[camera],
            calibrations={"CAM-1": make_calibration()},
            assets_by_floor={"floor-1": (("Stair", stair),)},
        )

        self.assertEqual(snapshot.cameras_observing("S1"), ())

    def test_11_uncalibrated_camera_still_appears_in_snapshot_as_unknown(self):

        stair = make_stair("S1", region=StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0))
        camera = make_camera(camera_id="CAM-UNCAL")

        snapshot = compute_camera_coverage_snapshot(
            cameras=[camera],
            calibrations={},
            assets_by_floor={"floor-1": (("Stair", stair),)},
        )

        coverage = snapshot.coverage_for_camera("CAM-UNCAL")
        self.assertEqual(coverage.coverage_for("S1").state, CoverageState.UNKNOWN)
        self.assertEqual(snapshot.assets_observed_by("CAM-UNCAL"), ())

    def test_12_unknown_camera_id_and_asset_id_default_gracefully(self):

        snapshot = CameraCoverageSnapshot()

        self.assertEqual(snapshot.cameras_observing("nope"), ())
        self.assertEqual(snapshot.assets_observed_by("nope"), ())
        self.assertEqual(snapshot.coverage_for_camera("nope").camera_id, "nope")


class BuildingStateIntegrationTests(unittest.TestCase):

    def test_13_camera_coverage_defaults_to_none(self):

        estimator = BuildingStateEstimator()

        state = estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
        )

        self.assertIsNone(state.camera_coverage)

    def test_14_camera_coverage_snapshot_passes_through_unchanged(self):

        estimator = BuildingStateEstimator()

        region = StairObservableRegion(center_x=10.0, center_y=0.0, width=2.0, depth=2.0)
        stair = make_stair("S1", region=region)
        camera = make_camera()

        coverage_snapshot = compute_camera_coverage_snapshot(
            cameras=[camera],
            calibrations={"CAM-1": make_calibration()},
            assets_by_floor={"floor-1": (("Stair", stair),)},
            timestamp=10.0,
        )

        state = estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            camera_coverage_snapshot=coverage_snapshot,
        )

        self.assertIs(state.camera_coverage, coverage_snapshot)
        self.assertEqual(state.camera_coverage.cameras_observing("S1"), ("CAM-1",))


class ManualAssetCoverageModelTests(unittest.TestCase):

    def test_15_is_covered_property(self):

        self.assertTrue(AssetCoverage("A", "Stair", CoverageState.FULLY_VISIBLE).is_covered)
        self.assertTrue(AssetCoverage("A", "Stair", CoverageState.PARTIALLY_VISIBLE).is_covered)
        self.assertFalse(AssetCoverage("A", "Stair", CoverageState.NOT_VISIBLE).is_covered)
        self.assertFalse(AssetCoverage("A", "Stair", CoverageState.UNKNOWN).is_covered)

    def test_16_camera_coverage_mapping_is_immutable(self):

        coverage = CameraCoverage(camera_id="CAM-1", assets={"S1": AssetCoverage("S1", "Stair")})

        with self.assertRaises(TypeError):
            coverage.assets["S2"] = AssetCoverage("S2", "Stair")


if __name__ == "__main__":
    unittest.main()

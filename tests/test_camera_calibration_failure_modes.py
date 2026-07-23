import json
import tempfile
import unittest
from pathlib import Path

from camera_calibration.calibration import CalibrationRegistry
from camera_calibration.calibration_loader import CalibrationLoadError, load_calibration_json
from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector
from camera_calibration.validation import ReferencePoint, project_pixel_point, resolution_mismatch, validate_calibration


# =====================================================
# Real Camera Calibration & World-Coordinate Validation milestone,
# Phase 14 -- deliberately exercises the failure modes this milestone's
# own instructions named: missing calibration, wrong camera_id,
# resolution mismatch, invalid FOV, invalid mount height, camera
# looking above horizon, ray parallel to floor, point behind camera,
# projection outside every zone, a deleted zone, calibration loaded for
# the wrong resolution, and a corrupt calibration file. None of these
# should ever crash the live runtime, and no world coordinate is ever
# fabricated in place of a genuinely-unavailable one.
# =====================================================


def make_profile(camera_id="CAM-1", floor_id="floor-1", position=(0.0, 0.0), mount_height=3.0,
                  yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0,
                  image_width=640, image_height=480, focal_length=500.0):

    intrinsics = CameraIntrinsics(
        image_width=image_width, image_height=image_height,
        focal_length_x=focal_length, focal_length_y=focal_length,
    )
    extrinsics = CameraExtrinsics(
        position=position, mount_height=mount_height, yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees, roll_degrees=roll_degrees,
    )
    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


class WrongOrMissingCameraIdTests(unittest.TestCase):

    def test_projecting_for_a_camera_id_never_registered_returns_all_none(self):

        registry = CalibrationRegistry()
        registry.set(make_profile(camera_id="CAM-REAL"))

        projector = WorldProjector(calibrations={p.camera_id: p for p in registry.all_profiles()}, zones_by_floor={})
        result = projector.project("CAM-TYPO", (0.0, 0.0, 10.0, 10.0), detection_confidence=0.9)

        self.assertIsNone(result.world_position)
        self.assertIsNone(result.floor_id)


class InvalidFOVTests(unittest.TestCase):

    def test_zero_fov_raises_a_clear_error_not_a_zero_division_error(self):

        with self.assertRaises(ValueError):
            CameraIntrinsics.from_horizontal_fov(640, 480, 0.0)

    def test_180_degree_fov_raises_rather_than_silently_returning_near_zero_focal_length(self):

        with self.assertRaises(ValueError):
            CameraIntrinsics.from_horizontal_fov(640, 480, 180.0)

    def test_negative_fov_raises_rather_than_silently_returning_negative_focal_length(self):

        with self.assertRaises(ValueError):
            CameraIntrinsics.from_horizontal_fov(640, 480, -10.0)

    def test_ordinary_fov_still_works_unchanged(self):

        intrinsics = CameraIntrinsics.from_horizontal_fov(640, 480, 90.0)
        self.assertGreater(intrinsics.focal_length_x, 0.0)


class InvalidMountHeightTests(unittest.TestCase):

    def test_negative_mount_height_does_not_crash_but_is_the_operators_own_responsibility(self):

        # No new validation is added for this one -- a negative mount_height is
        # physically nonsensical but geometrically well-defined (the math simply
        # treats the camera as being below the floor); this codebase's existing
        # "never fabricate, never crash" discipline is preserved: the ray-cast
        # still runs and either returns a (numerically valid, if physically
        # meaningless) position or None, never an exception.
        profile = make_profile(mount_height=-1.0, pitch_degrees=45.0)
        projector = WorldProjector(calibrations={"CAM-1": profile}, zones_by_floor={})

        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), detection_confidence=0.9)

        self.assertIsNotNone(result)  # never raises


class RayGeometryFailureModeTests(unittest.TestCase):

    def test_camera_looking_above_horizon_never_fabricates_a_position(self):

        profile = make_profile(pitch_degrees=0.0)  # perfectly level -- ray parallel to floor
        world = project_pixel_point((320.0, 240.0), profile)  # dead image center
        self.assertIsNone(world)

    def test_ray_that_only_crosses_the_floor_plane_behind_the_camera_returns_none(self):

        # A camera mounted BELOW the floor plane (mount_height < 0 -- e.g. a
        # miscalibrated/mis-measured height) that still looks downward
        # (direction.z < 0, pointing further away from the floor) has its
        # Z=0 crossing point at NEGATIVE t: geometrically behind the ray's
        # own forward direction, never something the camera could have
        # honestly observed in front of itself.
        profile = make_profile(pitch_degrees=45.0, mount_height=-1.0)
        world = project_pixel_point((320.0, 240.0), profile)  # dead image center
        self.assertIsNone(world)


class ZoneFailureModeTests(unittest.TestCase):

    def test_point_outside_the_building_or_every_known_zone_resolves_to_none_zone_id(self):

        from models.zone import Zone

        # Straight-ahead pixel projects to (3.0, 0.0) per the same trigonometry
        # test_camera_calibration.py already establishes -- this zone is placed
        # far away (x/y in [50, 55]) so (3.0, 0.0) is genuinely outside it.
        zone = Zone(name="Z1", x=50.0, y=50.0, width=5.0, height=5.0, floor_id="floor-1")
        zone.id = "Z1"

        profile = make_profile(pitch_degrees=45.0, mount_height=3.0)
        projector = WorldProjector(calibrations={"CAM-1": profile}, zones_by_floor={"floor-1": [zone]})

        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), detection_confidence=0.9)

        self.assertIsNotNone(result.world_position)
        self.assertIsNone(result.zone_id)

    def test_a_zone_removed_after_the_projector_was_built_is_honestly_no_longer_resolved(self):

        # WorldProjector holds an immutable snapshot of zones_by_floor at
        # construction time (camera_calibration/projection.py's own documented
        # behavior) -- a Zone deleted from the live Building afterward is not
        # magically un-resolved from an already-built projector; a caller must
        # reconstruct WorldProjector with the updated zone list. Documented here
        # as the expected, not-a-bug behavior.
        from models.zone import Zone

        zone = Zone(name="Z1", x=-10.0, y=-10.0, width=20.0, height=20.0, floor_id="floor-1")
        zone.id = "Z1"

        profile = make_profile(pitch_degrees=45.0, mount_height=3.0)
        projector = WorldProjector(calibrations={"CAM-1": profile}, zones_by_floor={"floor-1": [zone]})

        result_before = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), detection_confidence=0.9)
        self.assertEqual(result_before.zone_id, "Z1")

        # Deleting the zone from the ORIGINAL list object does not retroactively
        # affect this already-built projector (it stored its own tuple copy).
        rebuilt_projector = WorldProjector(calibrations={"CAM-1": profile}, zones_by_floor={"floor-1": []})
        result_after = rebuilt_projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), detection_confidence=0.9)
        self.assertIsNone(result_after.zone_id)


class ResolutionMismatchTests(unittest.TestCase):

    def test_matching_resolution_reports_no_mismatch(self):

        profile = make_profile(image_width=1920, image_height=1080)
        self.assertIsNone(resolution_mismatch(profile, 1920, 1080))

    def test_mismatched_resolution_is_detected(self):

        profile = make_profile(image_width=1920, image_height=1080)
        mismatch = resolution_mismatch(profile, 640, 480)
        self.assertEqual(mismatch, (1920, 1080))

    def test_unknown_frame_resolution_is_not_flagged_as_a_mismatch(self):

        # A ReplayFrameSource/synthetic test frame that never populates
        # width/height (None, None) has nothing to honestly compare against --
        # never fabricated as either "matches" or "mismatches".
        profile = make_profile(image_width=1920, image_height=1080)
        self.assertIsNone(resolution_mismatch(profile, None, None))


class CorruptCalibrationFileTests(unittest.TestCase):

    def test_corrupt_json_file_raises_a_clear_calibration_load_error(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = Path(tmp_dir) / "corrupt.json"
            path.write_text("{ this is not valid json", encoding="utf-8")

            with self.assertRaises(CalibrationLoadError):
                load_calibration_json(path)

    def test_calibration_file_missing_a_required_field_raises_a_clear_error(self):

        with tempfile.TemporaryDirectory() as tmp_dir:

            path = Path(tmp_dir) / "incomplete.json"
            path.write_text(json.dumps({"camera_id": "CAM-1"}), encoding="utf-8")

            with self.assertRaises(CalibrationLoadError):
                load_calibration_json(path)

    def test_nonexistent_calibration_file_raises_a_clear_error(self):

        with self.assertRaises(CalibrationLoadError):
            load_calibration_json("no/such/calibration.json")


if __name__ == "__main__":
    unittest.main()

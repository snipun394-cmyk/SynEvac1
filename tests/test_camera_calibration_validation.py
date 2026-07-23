import math
import unittest

from camera_calibration.camera_model import CalibrationProfile, CalibrationQuality, CameraExtrinsics, CameraIntrinsics
from camera_calibration.calibration_loader import calibration_from_dict, calibration_to_dict
from camera_calibration.calibration_solver import (
    MINIMUM_CORRESPONDENCES, CalibrationSolveError, solve_calibration_from_correspondences,
)
from camera_calibration.validation import ReferencePoint, project_pixel_point, validate_calibration


# =====================================================
# Real Camera Calibration & World-Coordinate Validation milestone --
# deterministic, offline unit tests for the two new modules
# (calibration_solver.py, validation.py) and the additive
# CalibrationQuality field. Every ground-truth pose/pixel/expected
# world position is hand-chosen (same discipline as
# tests/test_camera_calibration.py) -- no randomness, no real weights,
# no real video required. These tests prove the MATH is correct; they
# are explicitly NOT a claim of real-world metric accuracy (that
# requires a genuinely measured scene -- see docs/architecture/
# camera_calibration_and_world_projection.md §6 for why no such scene
# was available in this environment).
# =====================================================


def make_profile(
    camera_id="CAM-1", floor_id="floor-1", position=(0.0, 0.0), mount_height=3.0,
    yaw_degrees=15.0, pitch_degrees=40.0, roll_degrees=0.0,
    image_width=1280, image_height=720, focal_length=900.0,
):

    intrinsics = CameraIntrinsics(
        image_width=image_width, image_height=image_height,
        focal_length_x=focal_length, focal_length_y=focal_length,
    )
    extrinsics = CameraExtrinsics(
        position=position, mount_height=mount_height, yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees, roll_degrees=roll_degrees,
    )

    return CalibrationProfile(camera_id=camera_id, floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)


class ProjectPixelPointTests(unittest.TestCase):

    def test_center_pixel_projects_to_the_same_distance_a_bounding_box_ground_contact_point_would(self):

        # pitch=45, height=3 -> straight ahead distance is height/tan(pitch) = 3.0m,
        # the same hand-computed trigonometry test_camera_calibration.py already uses
        # for WorldProjector's own ground-contact-point path -- project_pixel_point()
        # must agree exactly since it reuses the identical ray-cast.
        profile = make_profile(pitch_degrees=45.0, mount_height=3.0, yaw_degrees=0.0)

        world = project_pixel_point((640.0, 360.0), profile)  # dead image center

        self.assertIsNotNone(world)
        self.assertAlmostEqual(world[0], 3.0, places=3)
        self.assertAlmostEqual(world[1], 0.0, places=3)

    def test_level_camera_pixel_never_reaches_the_floor(self):

        profile = make_profile(pitch_degrees=0.0, yaw_degrees=0.0)

        world = project_pixel_point((640.0, 360.0), profile)

        self.assertIsNone(world)


class ValidateCalibrationTests(unittest.TestCase):

    def test_perfect_calibration_reports_zero_error_against_its_own_ground_truth(self):

        profile = make_profile()

        reference_points = [
            ReferencePoint(pixel=(640.0, 400.0), world=project_pixel_point((640.0, 400.0), profile)),
            ReferencePoint(pixel=(500.0, 500.0), world=project_pixel_point((500.0, 500.0), profile)),
            ReferencePoint(pixel=(800.0, 450.0), world=project_pixel_point((800.0, 450.0), profile)),
        ]

        report = validate_calibration(profile, reference_points)

        self.assertEqual(report.reference_point_count, 3)
        self.assertEqual(report.validated_point_count, 3)
        self.assertAlmostEqual(report.mean_error_m, 0.0, places=6)
        self.assertAlmostEqual(report.rmse_m, 0.0, places=6)
        self.assertAlmostEqual(report.max_error_m, 0.0, places=6)

    def test_known_offset_produces_the_hand_computed_error(self):

        profile = make_profile(pitch_degrees=45.0, mount_height=3.0, yaw_degrees=0.0)

        true_world = project_pixel_point((640.0, 360.0), profile)  # (3.0, 0.0)
        # Deliberately claim the "measured" ground truth is 0.5m further away than
        # the calibration actually projects -- a hand-computable 0.5m error.
        offset_world = (true_world[0] + 0.5, true_world[1])

        report = validate_calibration(profile, [ReferencePoint(pixel=(640.0, 360.0), world=offset_world)])

        self.assertAlmostEqual(report.mean_error_m, 0.5, places=3)
        self.assertAlmostEqual(report.rmse_m, 0.5, places=3)

    def test_unprojectable_point_is_counted_in_reference_point_count_but_not_in_error_stats(self):

        profile = make_profile(pitch_degrees=0.0, yaw_degrees=0.0)  # level camera -- never reaches the floor

        report = validate_calibration(profile, [ReferencePoint(pixel=(640.0, 360.0), world=(1.0, 1.0))])

        self.assertEqual(report.reference_point_count, 1)
        self.assertEqual(report.validated_point_count, 0)
        self.assertIsNone(report.mean_error_m)
        self.assertIsNone(report.rmse_m)
        self.assertIsNone(report.point_results[0].error_m)

    def test_mixed_projectable_and_unprojectable_points_only_averages_the_projectable_ones(self):

        profile = make_profile(pitch_degrees=45.0, mount_height=3.0, yaw_degrees=0.0)
        true_world = project_pixel_point((640.0, 360.0), profile)

        # A second, deliberately-impossible pixel: with pitch=45 and this
        # intrinsics/resolution, a pixel far enough above center produces a
        # ray with direction.z >= 0 (looking above the horizon) -- unprojectable.
        report = validate_calibration(
            profile,
            [
                ReferencePoint(pixel=(640.0, 360.0), world=true_world, label="good"),
                ReferencePoint(pixel=(640.0, -5000.0), world=(0.0, 0.0), label="above-horizon"),
            ],
        )

        self.assertEqual(report.reference_point_count, 2)
        self.assertEqual(report.validated_point_count, 1)
        self.assertAlmostEqual(report.mean_error_m, 0.0, places=6)


class SolveCalibrationFromCorrespondencesTests(unittest.TestCase):

    def test_solver_recovers_known_yaw_and_pitch_from_synthetic_correspondences(self):

        true_profile = make_profile(yaw_degrees=15.0, pitch_degrees=40.0, mount_height=3.0, position=(0.0, 0.0))

        pixels = [(640.0, 400.0), (500.0, 500.0), (800.0, 450.0), (640.0, 600.0), (300.0, 420.0)]
        correspondences = [
            ReferencePoint(pixel=px, world=project_pixel_point(px, true_profile)) for px in pixels
        ]

        solved = solve_calibration_from_correspondences(
            camera_id="CAM-1", floor_id="floor-1", intrinsics=true_profile.intrinsics,
            camera_position=(0.0, 0.0), mount_height=3.0, correspondences=correspondences,
        )

        self.assertAlmostEqual(solved.profile.extrinsics.yaw_degrees, 15.0, places=2)
        self.assertAlmostEqual(solved.profile.extrinsics.pitch_degrees, 40.0, places=2)
        self.assertLess(solved.residual_rmse_m, 1e-3)

    def test_too_few_correspondences_raises_rather_than_guessing(self):

        with self.assertRaises(CalibrationSolveError):
            solve_calibration_from_correspondences(
                camera_id="CAM-1", floor_id="floor-1",
                intrinsics=CameraIntrinsics(image_width=1280, image_height=720, focal_length_x=900.0, focal_length_y=900.0),
                camera_position=(0.0, 0.0), mount_height=3.0,
                correspondences=[ReferencePoint(pixel=(1.0, 1.0), world=(1.0, 1.0))] * (MINIMUM_CORRESPONDENCES - 1),
            )

    def test_solved_profile_composes_with_validate_calibration(self):

        true_profile = make_profile(yaw_degrees=-20.0, pitch_degrees=35.0, mount_height=2.5, position=(1.0, 2.0))

        fit_pixels = [(640.0, 400.0), (500.0, 500.0), (800.0, 450.0), (640.0, 600.0)]
        held_out_pixel = (300.0, 420.0)

        fit_points = [ReferencePoint(pixel=px, world=project_pixel_point(px, true_profile)) for px in fit_pixels]
        held_out_points = [ReferencePoint(pixel=held_out_pixel, world=project_pixel_point(held_out_pixel, true_profile))]

        solved = solve_calibration_from_correspondences(
            camera_id="CAM-1", floor_id="floor-1", intrinsics=true_profile.intrinsics,
            camera_position=(1.0, 2.0), mount_height=2.5, correspondences=fit_points,
        )

        report = validate_calibration(solved.profile, held_out_points)

        self.assertEqual(report.validated_point_count, 1)
        self.assertLess(report.rmse_m, 1e-2)


class CalibrationQualitySerializationTests(unittest.TestCase):

    def test_calibration_profile_defaults_to_unvalidated(self):

        profile = make_profile()
        self.assertIsNone(profile.quality)

    def test_quality_round_trips_through_dict_serialization(self):

        profile = make_profile()
        quality = CalibrationQuality(
            reference_point_count=5, validated_point_count=5,
            mean_error_m=0.1, median_error_m=0.09, max_error_m=0.2, rmse_m=0.12,
            validation_timestamp="2026-01-01T00:00:00+00:00",
        )
        validated_profile = CalibrationProfile(
            camera_id=profile.camera_id, floor_id=profile.floor_id,
            intrinsics=profile.intrinsics, extrinsics=profile.extrinsics, quality=quality,
        )

        data = calibration_to_dict(validated_profile)
        self.assertIn("quality", data)

        restored = calibration_from_dict(data)

        self.assertIsNotNone(restored.quality)
        self.assertEqual(restored.quality.reference_point_count, 5)
        self.assertAlmostEqual(restored.quality.rmse_m, 0.12)

    def test_calibration_without_quality_round_trips_with_quality_absent(self):

        profile = make_profile()

        data = calibration_to_dict(profile)
        self.assertNotIn("quality", data)

        restored = calibration_from_dict(data)
        self.assertIsNone(restored.quality)


if __name__ == "__main__":
    unittest.main()

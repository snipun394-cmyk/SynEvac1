import unittest

from camera_calibration.camera_model import (
    CalibrationProfile, CalibrationQuality, CameraExtrinsics, CameraIntrinsics,
    WORLD_POSITION_PROVENANCE_NONE, WORLD_POSITION_PROVENANCE_UNVALIDATED,
    WORLD_POSITION_PROVENANCE_VALIDATED, calibration_provenance,
)
from camera_calibration.projection import WorldProjector

from live_camera_pipeline.human_detector import RawHumanDetection
from live_camera_pipeline.identity_resolver import SimulationIdentityResolver


# =====================================================
# CCTV Connection & Calibration Readiness milestone, Phase 7/8 -- proves
# the three-way world-position provenance distinction this milestone
# added (a genuine gap: before this, an UNVALIDATED calibration's
# world_position was indistinguishable from a VALIDATED one, or from
# "no calibration at all", everywhere downstream of WorldProjector).
# =====================================================


def make_calibration(quality=None, position=(0.0, 0.0), mount_height=3.0, pitch_degrees=45.0):

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=position, mount_height=mount_height, yaw_degrees=0.0, pitch_degrees=pitch_degrees)

    return CalibrationProfile(camera_id="CAM-1", floor_id="floor-1", intrinsics=intrinsics, extrinsics=extrinsics, quality=quality)


VALIDATED_QUALITY = CalibrationQuality(
    reference_point_count=5, validated_point_count=5,
    mean_error_m=0.1, median_error_m=0.1, max_error_m=0.2, rmse_m=0.12,
    validation_timestamp="2026-01-01T00:00:00",
)

ATTEMPTED_BUT_UNPROJECTABLE_QUALITY = CalibrationQuality(
    reference_point_count=3, validated_point_count=0,
    mean_error_m=None, median_error_m=None, max_error_m=None, rmse_m=None,
    validation_timestamp="2026-01-01T00:00:00",
)


class CalibrationProvenanceFunctionTests(unittest.TestCase):

    def test_no_profile_is_none(self):
        self.assertEqual(calibration_provenance(None), WORLD_POSITION_PROVENANCE_NONE)

    def test_profile_with_no_quality_is_unvalidated(self):
        self.assertEqual(calibration_provenance(make_calibration(quality=None)), WORLD_POSITION_PROVENANCE_UNVALIDATED)

    def test_profile_with_attempted_but_unprojectable_validation_is_still_unvalidated(self):
        # A validation attempt that could not project ANY reference
        # point earns no more trust than never validating at all.
        self.assertEqual(
            calibration_provenance(make_calibration(quality=ATTEMPTED_BUT_UNPROJECTABLE_QUALITY)),
            WORLD_POSITION_PROVENANCE_UNVALIDATED,
        )

    def test_profile_with_genuine_rmse_is_validated(self):
        self.assertEqual(
            calibration_provenance(make_calibration(quality=VALIDATED_QUALITY)),
            WORLD_POSITION_PROVENANCE_VALIDATED,
        )


class WorldProjectorProvenanceTests(unittest.TestCase):

    def test_no_calibration_for_camera_reports_none_provenance(self):

        projector = WorldProjector(calibrations={}, zones_by_floor={})
        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), 0.9)

        self.assertIsNone(result.world_position)
        self.assertEqual(result.provenance, WORLD_POSITION_PROVENANCE_NONE)

    def test_no_bounding_box_reports_none_provenance_even_with_a_calibration(self):

        projector = WorldProjector(calibrations={"CAM-1": make_calibration(quality=VALIDATED_QUALITY)}, zones_by_floor={})
        result = projector.project("CAM-1", None, 0.9)

        self.assertIsNone(result.world_position)
        self.assertEqual(result.provenance, WORLD_POSITION_PROVENANCE_NONE)

    def test_unvalidated_calibration_still_produces_a_world_position_marked_unvalidated(self):

        projector = WorldProjector(calibrations={"CAM-1": make_calibration(quality=None)}, zones_by_floor={})
        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), 0.9)

        self.assertIsNotNone(result.world_position)
        self.assertEqual(result.provenance, WORLD_POSITION_PROVENANCE_UNVALIDATED)

    def test_validated_calibration_produces_a_world_position_marked_validated(self):

        projector = WorldProjector(calibrations={"CAM-1": make_calibration(quality=VALIDATED_QUALITY)}, zones_by_floor={})
        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), 0.9)

        self.assertIsNotNone(result.world_position)
        self.assertEqual(result.provenance, WORLD_POSITION_PROVENANCE_VALIDATED)

    def test_geometrically_undefined_ray_still_reports_the_calibrations_own_provenance(self):

        # A camera pointed above the horizon never reaches the floor --
        # world_position is honestly None, but the provenance still
        # reflects the CALIBRATION's own validated/unvalidated status
        # (a real attempt was made with a specific, identifiable
        # calibration), not the generic "nothing to project" NONE.
        calibration = make_calibration(quality=VALIDATED_QUALITY, pitch_degrees=0.0)
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        result = projector.project("CAM-1", (315.0, 200.0, 325.0, 240.0), 0.9)

        self.assertIsNone(result.world_position)
        self.assertEqual(result.provenance, WORLD_POSITION_PROVENANCE_VALIDATED)


class ProvenancePropagationThroughPipelineTypesTests(unittest.TestCase):

    # Mechanical proof the field survives RawHumanDetection -> Detection
    # -- no computation, just "does the plumbing actually carry it."

    def test_raw_detection_defaults_to_none_and_survives_into_detection(self):

        raw = RawHumanDetection(
            camera_id="CAM-1", local_track_id="1", timestamp=0.0,
            world_position_provenance=WORLD_POSITION_PROVENANCE_VALIDATED,
        )

        resolver = SimulationIdentityResolver()
        detections = resolver.resolve((raw,), time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].world_position_provenance, WORLD_POSITION_PROVENANCE_VALIDATED)

    def test_default_provenance_is_none_when_never_supplied(self):

        raw = RawHumanDetection(camera_id="CAM-1", local_track_id="1", timestamp=0.0)
        resolver = SimulationIdentityResolver()
        detections = resolver.resolve((raw,), time=0.0)

        self.assertIsNone(detections[0].world_position_provenance)


if __name__ == "__main__":
    unittest.main()

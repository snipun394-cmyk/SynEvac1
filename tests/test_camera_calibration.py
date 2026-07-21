import unittest

from models.zone import Zone

from camera_calibration.calibration import CalibrationRegistry
from camera_calibration.calibration_loader import (
    CalibrationLoadError, calibration_from_camera, calibration_from_dict, calibration_to_dict,
)
from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.geometry import camera_basis_vectors, intersect_ray_with_floor, pixel_ray_direction
from camera_calibration.projection import WorldProjector, nearest_navigation_node


# =====================================================
# Camera Calibration & World Coordinate Projection milestone, Phase 8
# -- deterministic, offline unit tests. No randomness anywhere in this
# file: every camera pose/pixel/expected world position is hand-chosen
# and independently verifiable by the ray-plane trigonometry each test
# documents.
# =====================================================


def make_calibration(
    camera_id="CAM-1", floor_id="floor-1", position=(0.0, 0.0), mount_height=3.0,
    yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0,
    image_width=640, image_height=480, focal_length=500.0,
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


def make_zone(zone_id, x, y, width, height, floor_id="floor-1"):

    zone = Zone(name=zone_id, x=x, y=y, width=width, height=height, floor_id=floor_id)
    zone.id = zone_id
    return zone


class SingleCameraProjectionTests(unittest.TestCase):

    def test_1_single_camera_projects_center_pixel_to_expected_world_distance(self):

        # pitch=45deg, mount_height=3.0 -> straight-ahead ground point
        # is exactly height/tan(pitch) = 3.0 meters away (standard
        # right-triangle trigonometry, independently verifiable).
        calibration = make_calibration()
        projector = WorldProjector(
            calibrations={"CAM-1": calibration}, zones_by_floor={},
        )

        box = (315.0, 200.0, 325.0, 240.0)  # bottom-center lands near the exact image center (320, 240)
        result = projector.project("CAM-1", box, detection_confidence=0.9)

        self.assertIsNotNone(result.world_position)
        self.assertAlmostEqual(result.world_position[0], 3.0, places=3)
        self.assertAlmostEqual(result.world_position[1], 0.0, places=3)
        self.assertEqual(result.floor_id, "floor-1")


class MultipleCameraTests(unittest.TestCase):

    def test_2_multiple_cameras_project_independently(self):

        cal_a = make_calibration(camera_id="CAM-A", position=(0.0, 0.0), pitch_degrees=45.0)
        cal_b = make_calibration(camera_id="CAM-B", position=(100.0, 100.0), pitch_degrees=90.0)

        projector = WorldProjector(calibrations={"CAM-A": cal_a, "CAM-B": cal_b}, zones_by_floor={})

        box = (315.0, 200.0, 325.0, 240.0)

        result_a = projector.project("CAM-A", box, 0.9)
        result_b = projector.project("CAM-B", box, 0.9)

        self.assertAlmostEqual(result_a.world_position[0], 3.0, places=3)
        # Straight down from (100, 100) -- center pixel lands directly
        # below the camera regardless of yaw.
        self.assertAlmostEqual(result_b.world_position[0], 100.0, places=3)
        self.assertAlmostEqual(result_b.world_position[1], 100.0, places=3)


class ProjectionAccuracyTests(unittest.TestCase):

    def test_3_off_center_pixel_shifts_world_position_along_camera_right_axis(self):

        calibration = make_calibration(pitch_degrees=45.0)
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        center_box = (315.0, 200.0, 325.0, 240.0)
        right_box = (415.0, 200.0, 425.0, 240.0)  # 100px further right

        center_result = projector.project("CAM-1", center_box, 0.9)
        right_result = projector.project("CAM-1", right_box, 0.9)

        # x (straight-ahead distance) is unaffected by a purely
        # horizontal pixel shift when yaw=0; y (camera's right/left
        # axis) changes.
        self.assertAlmostEqual(center_result.world_position[0], right_result.world_position[0], places=3)
        self.assertNotAlmostEqual(center_result.world_position[1], right_result.world_position[1], places=3)

    def test_projection_is_a_pure_function_reproducible_across_calls(self):

        calibration = make_calibration()
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        box = (300.0, 210.0, 340.0, 250.0)

        first = projector.project("CAM-1", box, 0.8)
        second = projector.project("CAM-1", box, 0.8)

        self.assertEqual(first, second)


class CameraRotationTests(unittest.TestCase):

    def test_4_yaw_90_degrees_projects_along_the_y_axis_instead_of_x(self):

        calibration = make_calibration(yaw_degrees=90.0, pitch_degrees=45.0)
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertAlmostEqual(result.world_position[0], 0.0, places=3)
        self.assertAlmostEqual(result.world_position[1], 3.0, places=3)

    def test_4_yaw_180_degrees_projects_along_negative_x(self):

        calibration = make_calibration(yaw_degrees=180.0, pitch_degrees=45.0)
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertAlmostEqual(result.world_position[0], -3.0, places=3)
        self.assertAlmostEqual(result.world_position[1], 0.0, places=3)


class CameraHeightTests(unittest.TestCase):

    def test_5_taller_mount_height_projects_further_away_at_the_same_pitch(self):

        short = make_calibration(camera_id="CAM-SHORT", mount_height=2.0, pitch_degrees=45.0)
        tall = make_calibration(camera_id="CAM-TALL", mount_height=6.0, pitch_degrees=45.0)

        projector = WorldProjector(calibrations={"CAM-SHORT": short, "CAM-TALL": tall}, zones_by_floor={})

        box = (315.0, 200.0, 325.0, 240.0)

        result_short = projector.project("CAM-SHORT", box, 0.9)
        result_tall = projector.project("CAM-TALL", box, 0.9)

        self.assertAlmostEqual(result_short.world_position[0], 2.0, places=3)
        self.assertAlmostEqual(result_tall.world_position[0], 6.0, places=3)

    def test_5_straight_down_pixel_always_lands_at_the_camera_footprint_regardless_of_height(self):

        for height in (2.0, 3.0, 10.0):

            calibration = make_calibration(camera_id="CAM-X", position=(7.0, 4.0), mount_height=height, pitch_degrees=90.0)
            projector = WorldProjector(calibrations={"CAM-X": calibration}, zones_by_floor={})

            box = (315.0, 200.0, 325.0, 240.0)
            result = projector.project("CAM-X", box, 0.9)

            self.assertAlmostEqual(result.world_position[0], 7.0, places=3)
            self.assertAlmostEqual(result.world_position[1], 4.0, places=3)


class DifferentResolutionTests(unittest.TestCase):

    def test_6_center_pixel_result_is_independent_of_image_resolution(self):

        # A different resolution + a re-derived focal length via
        # from_horizontal_fov (same FOV) should still project the
        # exact image-center pixel to the same world point.
        low_res = CameraIntrinsics.from_horizontal_fov(image_width=640, image_height=480, horizontal_fov_degrees=60.0)
        high_res = CameraIntrinsics.from_horizontal_fov(image_width=1920, image_height=1080, horizontal_fov_degrees=60.0)

        extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)

        calibration_low = CalibrationProfile(camera_id="CAM-LOW", floor_id="floor-1", intrinsics=low_res, extrinsics=extrinsics)
        calibration_high = CalibrationProfile(camera_id="CAM-HIGH", floor_id="floor-1", intrinsics=high_res, extrinsics=extrinsics)

        projector = WorldProjector(calibrations={"CAM-LOW": calibration_low, "CAM-HIGH": calibration_high}, zones_by_floor={})

        box_low = (315.0, 235.0, 325.0, 240.0)  # bottom-center = (320, 240) = exact low-res image center
        box_high = (955.0, 535.0, 965.0, 540.0)  # bottom-center = (960, 540) = exact high-res image center

        result_low = projector.project("CAM-LOW", box_low, 0.9)
        result_high = projector.project("CAM-HIGH", box_high, 0.9)

        self.assertAlmostEqual(result_low.world_position[0], result_high.world_position[0], places=3)
        self.assertAlmostEqual(result_low.world_position[1], result_high.world_position[1], places=3)


class MissingAndInvalidCalibrationTests(unittest.TestCase):

    def test_7_missing_calibration_returns_all_none_never_raises(self):

        projector = WorldProjector(calibrations={}, zones_by_floor={})

        result = projector.project("CAM-UNKNOWN", (0.0, 0.0, 10.0, 20.0), 0.9)

        self.assertIsNone(result.world_position)
        self.assertIsNone(result.floor_id)
        self.assertIsNone(result.zone_id)
        self.assertIsNone(result.projection_confidence)

    def test_7_missing_bounding_box_returns_all_none(self):

        calibration = make_calibration()
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        result = projector.project("CAM-1", None, 0.9)

        self.assertIsNone(result.world_position)

    def test_8_level_camera_never_produces_a_fabricated_position(self):

        # pitch=0 (perfectly horizontal) -- the center-pixel ray never
        # reaches the floor plane at all.
        calibration = make_calibration(pitch_degrees=0.0)
        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertIsNone(result.world_position)
        self.assertIsNone(result.zone_id)
        self.assertEqual(result.floor_id, "floor-1")  # still honestly known -- the calibration itself is valid

    def test_8_invalid_calibration_json_raises_a_clear_error(self):

        with self.assertRaises(CalibrationLoadError):
            calibration_from_dict({"camera_id": "CAM-1"})  # missing floor_id/intrinsics/extrinsics


class ZoneLookupTests(unittest.TestCase):

    def test_9_world_position_inside_a_zone_resolves_its_zone_id(self):

        calibration = make_calibration(pitch_degrees=90.0, position=(5.0, 5.0))  # straight down -> lands at (5,5)
        zone = make_zone("zone-a", x=0.0, y=0.0, width=10.0, height=10.0)

        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={"floor-1": [zone]})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertEqual(result.zone_id, "zone-a")

    def test_9_world_position_outside_every_zone_resolves_to_none(self):

        calibration = make_calibration(pitch_degrees=90.0, position=(500.0, 500.0))
        zone = make_zone("zone-a", x=0.0, y=0.0, width=10.0, height=10.0)

        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={"floor-1": [zone]})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertIsNone(result.zone_id)

    def test_9_irregular_polygon_zone_is_checked_via_point_in_polygon(self):

        calibration = make_calibration(pitch_degrees=90.0, position=(1.0, 1.0))
        triangle_zone = make_zone("zone-triangle", x=0.0, y=0.0, width=0.0, height=0.0)
        triangle_zone.polygon = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]  # (1,1) is inside this triangle

        projector = WorldProjector(calibrations={"CAM-1": calibration}, zones_by_floor={"floor-1": [triangle_zone]})

        box = (315.0, 200.0, 325.0, 240.0)
        result = projector.project("CAM-1", box, 0.9)

        self.assertEqual(result.zone_id, "zone-triangle")


class NavigationLookupTests(unittest.TestCase):

    def test_10_nearest_navigation_node_resolves_to_the_closest_zone_node(self):

        from navigation.graph import NavigationGraph
        from navigation.node import Node

        graph = NavigationGraph()

        near_zone = make_zone("zone-near", x=0.0, y=0.0, width=2.0, height=2.0)  # center (1,1)
        far_zone = make_zone("zone-far", x=100.0, y=100.0, width=2.0, height=2.0)  # center (101,101)

        graph.add_node(Node(id="zone-near", name="Near", floor_id="floor-1", node_type=Node.ZONE, reference=near_zone))
        graph.add_node(Node(id="zone-far", name="Far", floor_id="floor-1", node_type=Node.ZONE, reference=far_zone))

        nearest = nearest_navigation_node(graph, "floor-1", world_position=(1.5, 1.5))

        self.assertEqual(nearest, "zone-near")

    def test_10_navigation_lookup_ignores_nodes_on_a_different_floor(self):

        from navigation.graph import NavigationGraph
        from navigation.node import Node

        graph = NavigationGraph()

        other_floor_zone = make_zone("zone-other-floor", x=0.0, y=0.0, width=2.0, height=2.0, floor_id="floor-2")

        graph.add_node(Node(id="zone-other-floor", name="Other", floor_id="floor-2", node_type=Node.ZONE, reference=other_floor_zone))

        nearest = nearest_navigation_node(graph, "floor-1", world_position=(1.0, 1.0))

        self.assertIsNone(nearest)


class ProjectionConsistencyTests(unittest.TestCase):

    def test_11_calibration_round_trips_through_dict_serialization(self):

        original = make_calibration(pitch_degrees=30.0, roll_degrees=5.0)

        restored = calibration_from_dict(calibration_to_dict(original))

        self.assertEqual(restored, original)

    def test_11_calibration_from_camera_reuses_existing_camera_fields(self):

        from models.camera import Camera

        camera = Camera(name="Lobby Cam", floor_id="floor-9", position=(12.0, 34.0), rotation=45.0, horizontal_fov=90.0, mount_height=2.5)

        profile = calibration_from_camera(camera, pitch_degrees=20.0)

        self.assertEqual(profile.camera_id, camera.id)
        self.assertEqual(profile.floor_id, "floor-9")
        self.assertEqual(profile.extrinsics.position, (12.0, 34.0))
        self.assertEqual(profile.extrinsics.mount_height, 2.5)
        self.assertEqual(profile.extrinsics.yaw_degrees, 45.0)
        self.assertEqual(profile.extrinsics.pitch_degrees, 20.0)

    def test_11_basis_vectors_are_orthonormal_for_arbitrary_pose(self):

        extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=37.0, pitch_degrees=22.0, roll_degrees=15.0)

        forward, right, down = camera_basis_vectors(extrinsics)

        def length(v):
            return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5

        def dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        self.assertAlmostEqual(length(forward), 1.0, places=6)
        self.assertAlmostEqual(length(right), 1.0, places=6)
        self.assertAlmostEqual(length(down), 1.0, places=6)
        self.assertAlmostEqual(dot(forward, right), 0.0, places=6)
        self.assertAlmostEqual(dot(forward, down), 0.0, places=6)
        self.assertAlmostEqual(dot(right, down), 0.0, places=6)


class CalibrationRegistryTests(unittest.TestCase):

    def test_registry_stores_and_retrieves_by_camera_id(self):

        registry = CalibrationRegistry()
        profile = make_calibration()

        self.assertFalse(registry.has("CAM-1"))

        registry.set(profile)

        self.assertTrue(registry.has("CAM-1"))
        self.assertEqual(registry.get("CAM-1"), profile)
        self.assertIsNone(registry.get("CAM-UNKNOWN"))
        self.assertEqual(len(registry), 1)

        registry.remove("CAM-1")
        self.assertEqual(len(registry), 0)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.staircase import Staircase
from models.zone import Zone

from visibility.engine import CameraVisibility, VisibilityEngine
from visibility.coverage import FloorCoverage

from camera_validation.metrics import compute_camera_placement_metrics
from camera_validation.network import compute_network_analysis
from camera_validation.recommendations import (
    generate_camera_recommendations,
    generate_network_recommendations,
)
from camera_validation.validator import validate_building, validate_floor


def make_zone(name, x=0.0, y=0.0, width=10.0, height=10.0, floor_id=""):

    return Zone(name=name, x=x, y=y, width=width, height=height, floor_id=floor_id)


class CameraPlacementMetricsTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", width=10.0, height=10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.camera = Camera(
            name="Cam", floor_id=self.floor.id, position=(1.0, 1.0),
            rotation=45.0, horizontal_fov=90.0, max_range=15.0, mount_height=2.5,
        )

    def test_engineering_fields_pass_through_from_the_camera_asset(self):

        visibility = CameraVisibility(camera_id=self.camera.id)

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        self.assertEqual(metrics.camera_id, self.camera.id)
        self.assertEqual(metrics.mount_height, 2.5)
        self.assertEqual(metrics.rotation, 45.0)
        self.assertEqual(metrics.horizontal_fov, 90.0)
        self.assertEqual(metrics.max_range, 15.0)

    def test_zero_coverage_gives_zero_placement_score(self):

        visibility = CameraVisibility(camera_id=self.camera.id, zone_coverage={self.zone.id: 0.0})

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        self.assertEqual(metrics.zone_coverage_percentage, 0.0)
        self.assertEqual(metrics.placement_score, 0.0)

    def test_full_coverage_with_no_assets_gives_coverage_weighted_score(self):

        visibility = CameraVisibility(camera_id=self.camera.id, zone_coverage={self.zone.id: 1.0})

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        self.assertEqual(metrics.zone_coverage_percentage, 100.0)
        # COVERAGE_WEIGHT(0.7) * 100 + ASSET_WEIGHT(0.3) * 0
        self.assertAlmostEqual(metrics.placement_score, 70.0)

    def test_visible_assets_increase_the_placement_score(self):

        visibility = CameraVisibility(
            camera_id=self.camera.id,
            zone_coverage={self.zone.id: 1.0},
            visible_door_ids=("door-1",),
            visible_exit_ids=("exit-1",),
        )

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        # 0.7*100 + 0.3*min(100, 20*2) = 70 + 0.3*40 = 82
        self.assertAlmostEqual(metrics.placement_score, 82.0)
        self.assertEqual(metrics.visible_door_ids, ("door-1",))
        self.assertEqual(metrics.visible_exit_ids, ("exit-1",))

    def test_blind_zone_ids_pass_through(self):

        visibility = CameraVisibility(camera_id=self.camera.id, hidden_zone_ids=(self.zone.id,))

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        self.assertEqual(metrics.blind_zone_ids, (self.zone.id,))

    def test_partial_coverage_is_area_weighted_across_multiple_zones(self):

        zone_2 = make_zone("Room 2", x=10.0, width=10.0, height=10.0, floor_id=self.floor.id)
        self.floor.add_zone(zone_2)

        # zone (10x10=100 area) fully covered; zone_2 (10x10=100 area)
        # not covered at all -> 50% area-weighted overall.
        visibility = CameraVisibility(
            camera_id=self.camera.id, zone_coverage={self.zone.id: 1.0, zone_2.id: 0.0},
        )

        metrics = compute_camera_placement_metrics(self.camera, self.floor, visibility)

        self.assertAlmostEqual(metrics.zone_coverage_percentage, 50.0)


class NetworkAnalysisTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("Zone A", floor_id=self.floor.id)
        self.zone_b = make_zone("Zone B", x=10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        self.exit_obj = Exit(name="Ex", zone_id=self.zone_b.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

    def test_uncovered_exit_is_reported(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={
                "cam-1": CameraVisibility(camera_id="cam-1", zone_coverage={self.zone_a.id: 1.0}),
            },
            zone_combined_coverage={self.zone_a.id: 1.0, self.zone_b.id: 0.0},
            uncovered_zone_ids=(self.zone_b.id,),
            total_floor_coverage_fraction=0.5,
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertEqual(analysis.uncovered_exit_ids, (self.exit_obj.id,))
        self.assertEqual(analysis.uncovered_zone_ids, (self.zone_b.id,))

    def test_covered_exit_is_not_reported(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={
                "cam-1": CameraVisibility(
                    camera_id="cam-1",
                    zone_coverage={self.zone_a.id: 1.0, self.zone_b.id: 1.0},
                    visible_exit_ids=(self.exit_obj.id,),
                ),
            },
            zone_combined_coverage={self.zone_a.id: 1.0, self.zone_b.id: 1.0},
            total_floor_coverage_fraction=1.0,
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertEqual(analysis.uncovered_exit_ids, ())

    def test_camera_with_no_coverage_is_flagged(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={"cam-1": CameraVisibility(camera_id="cam-1", zone_coverage={self.zone_a.id: 0.0})},
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertEqual(analysis.no_coverage_camera_ids, ("cam-1",))
        self.assertEqual(analysis.isolated_camera_ids, ())
        self.assertEqual(analysis.redundant_camera_ids, ())
        self.assertEqual(analysis.excessive_overlap_camera_ids, ())

    def test_sole_camera_covering_a_zone_is_isolated(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={"cam-1": CameraVisibility(camera_id="cam-1", zone_coverage={self.zone_a.id: 1.0})},
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertEqual(analysis.isolated_camera_ids, ("cam-1",))

    def test_two_cameras_fully_overlapping_are_excessive_overlap(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={
                "cam-1": CameraVisibility(camera_id="cam-1", zone_coverage={self.zone_a.id: 1.0}),
                "cam-2": CameraVisibility(camera_id="cam-2", zone_coverage={self.zone_a.id: 1.0}),
            },
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertEqual(set(analysis.excessive_overlap_camera_ids), {"cam-1", "cam-2"})
        self.assertEqual(analysis.isolated_camera_ids, ())

    def test_partial_overlap_is_redundant_not_excessive(self):

        zone_c = make_zone("Zone C", x=20.0, floor_id=self.floor.id)
        self.floor.add_zone(zone_c)

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={
                "cam-1": CameraVisibility(
                    camera_id="cam-1",
                    zone_coverage={self.zone_a.id: 1.0, self.zone_b.id: 1.0, zone_c.id: 1.0},
                ),
                "cam-2": CameraVisibility(camera_id="cam-2", zone_coverage={self.zone_a.id: 1.0}),
            },
        )

        analysis = compute_network_analysis(self.floor, floor_coverage)

        # cam-1 covers 3 zones, only 1 backed up by cam-2 -> 1/3 overlap fraction, "redundant".
        self.assertIn("cam-1", analysis.redundant_camera_ids)
        # cam-2 covers 1 zone, fully backed up by cam-1 -> "excessive overlap".
        self.assertIn("cam-2", analysis.excessive_overlap_camera_ids)

    def test_network_score_penalized_by_uncovered_critical_assets(self):

        covered_floor_coverage = FloorCoverage(
            floor_id=self.floor.id, per_camera={}, total_floor_coverage_fraction=1.0,
        )
        uncovered_analysis = compute_network_analysis(self.floor, covered_floor_coverage)

        # 100% area coverage but the one Exit on this floor is
        # unmonitored (no camera in per_camera at all) -> penalized.
        self.assertLess(uncovered_analysis.network_score, 100.0)
        self.assertEqual(uncovered_analysis.network_score, 90.0)  # 100 - EXIT_PENALTY(10)

    def test_network_score_never_goes_negative(self):

        # Many uncovered doors driving the penalty below zero.
        for i in range(20):
            self.floor.add_door(Door(name=f"D{i}", floor_id=self.floor.id))

        floor_coverage = FloorCoverage(floor_id=self.floor.id, per_camera={}, total_floor_coverage_fraction=0.0)

        analysis = compute_network_analysis(self.floor, floor_coverage)

        self.assertGreaterEqual(analysis.network_score, 0.0)


class CameraRecommendationTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Room", width=10.0, height=10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

    def test_well_covered_camera_gets_no_recommendation(self):

        camera = Camera(name="Cam", floor_id=self.floor.id, zone_ids=(self.zone.id,))
        visibility = CameraVisibility(camera_id=camera.id, zone_coverage={self.zone.id: 1.0})

        from camera_validation.metrics import compute_camera_placement_metrics
        metrics = compute_camera_placement_metrics(camera, self.floor, visibility)

        recommendations = generate_camera_recommendations(camera, self.floor, visibility, metrics)

        self.assertEqual(recommendations, ())

    def test_camera_facing_away_gets_a_rotate_recommendation(self):

        # Camera at the zone's own corner, facing directly away
        # (rotation=180 while the zone extends in +x/+y) -- a large,
        # rotatable gap.
        camera = Camera(
            name="Cam", floor_id=self.floor.id, position=(0.5, 0.5),
            rotation=180.0, horizontal_fov=60.0, max_range=20.0, zone_ids=(self.zone.id,),
        )

        engine = VisibilityEngine()
        self.floor.add_camera(camera)
        visibility = engine.compute_camera_visibility(camera, self.building)

        from camera_validation.metrics import compute_camera_placement_metrics
        metrics = compute_camera_placement_metrics(camera, self.floor, visibility)

        recommendations = generate_camera_recommendations(camera, self.floor, visibility, metrics)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].category, "rotate")
        self.assertIn("Rotate", recommendations[0].message)

    def test_camera_facing_correctly_but_short_range_gets_a_range_recommendation(self):

        camera = Camera(
            name="Cam", floor_id=self.floor.id, position=(0.5, 5.0),
            rotation=0.0, horizontal_fov=170.0, max_range=2.0, zone_ids=(self.zone.id,),
        )

        engine = VisibilityEngine()
        self.floor.add_camera(camera)
        visibility = engine.compute_camera_visibility(camera, self.building)

        from camera_validation.metrics import compute_camera_placement_metrics
        metrics = compute_camera_placement_metrics(camera, self.floor, visibility)

        recommendations = generate_camera_recommendations(camera, self.floor, visibility, metrics)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].category, "increase_range")

    def test_camera_facing_correctly_but_narrow_fov_gets_a_fov_recommendation(self):

        camera = Camera(
            name="Cam", floor_id=self.floor.id, position=(0.5, 5.0),
            rotation=0.0, horizontal_fov=10.0, max_range=20.0, zone_ids=(self.zone.id,),
        )

        engine = VisibilityEngine()
        self.floor.add_camera(camera)
        visibility = engine.compute_camera_visibility(camera, self.building)

        from camera_validation.metrics import compute_camera_placement_metrics
        metrics = compute_camera_placement_metrics(camera, self.floor, visibility)

        recommendations = generate_camera_recommendations(camera, self.floor, visibility, metrics)

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].category, "increase_fov")


class NetworkRecommendationTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Zone", floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.exit_obj = Exit(name="Main Exit", zone_id=self.zone.id, floor_id=self.floor.id)
        self.floor.add_exit(self.exit_obj)

        self.stair = Staircase(name="Stair", from_floor_id=self.floor.id, from_zone_id=self.zone.id)
        self.floor.add_stair(self.stair)

        self.door = Door(name="Door", floor_id=self.floor.id)
        self.floor.add_door(self.door)

    def test_unmonitored_exit_stair_and_door_are_each_flagged(self):

        # uncovered_zone_ids must be set explicitly here -- this is a
        # hand-built FloorCoverage fixture, not one produced by
        # compute_floor_coverage(), so it does not compute this itself.
        floor_coverage = FloorCoverage(
            floor_id=self.floor.id, per_camera={}, uncovered_zone_ids=(self.zone.id,),
        )
        analysis = compute_network_analysis(self.floor, floor_coverage)

        recommendations = generate_network_recommendations(self.floor, analysis)

        categories_by_subject = {r.subject_id: r.category for r in recommendations}

        self.assertEqual(categories_by_subject[self.exit_obj.id], "unmonitored")
        self.assertEqual(categories_by_subject[self.stair.id], "unmonitored")
        self.assertEqual(categories_by_subject[self.door.id], "unmonitored")
        self.assertEqual(categories_by_subject[self.zone.id], "add_camera")

    def test_camera_categories_produce_matching_recommendation_categories(self):

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={
                "cam-1": CameraVisibility(camera_id="cam-1", zone_coverage={self.zone.id: 1.0}),
            },
        )
        analysis = compute_network_analysis(self.floor, floor_coverage)

        recommendations = generate_network_recommendations(self.floor, analysis)

        camera_recommendations = {r.subject_id: r.category for r in recommendations if r.subject_type == "camera"}
        self.assertEqual(camera_recommendations["cam-1"], "isolated")

    def test_camera_label_uses_the_camera_name_when_provided(self):

        camera = Camera(name="Lobby Cam", floor_id=self.floor.id)

        floor_coverage = FloorCoverage(
            floor_id=self.floor.id,
            per_camera={camera.id: CameraVisibility(camera_id=camera.id, zone_coverage={self.zone.id: 1.0})},
        )
        analysis = compute_network_analysis(self.floor, floor_coverage)

        recommendations = generate_network_recommendations(
            self.floor, analysis, camera_by_id={camera.id: camera},
        )

        camera_rec = next(r for r in recommendations if r.subject_id == camera.id)
        self.assertIn("Lobby Cam", camera_rec.message)


class ValidatorIntegrationTests(unittest.TestCase):

    # Phase 6: "Different placements produce different scores", "Poor
    # layouts are detected", "Blind areas are identified" -- all
    # through the real VisibilityEngine, real Building/Camera/Zone/
    # Exit objects, not hand-built fixtures.

    def _build(self, camera_rotation):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        zone = make_zone("Room", width=10.0, height=10.0, floor_id=floor.id)
        floor.add_zone(zone)

        exit_obj = Exit(name="Ex", zone_id=zone.id, floor_id=floor.id)
        floor.add_exit(exit_obj)

        camera = Camera(
            name="Cam", floor_id=floor.id, position=(0.5, 5.0),
            rotation=camera_rotation, horizontal_fov=90.0, max_range=20.0, zone_ids=(zone.id,),
        )
        floor.add_camera(camera)

        return building, floor, zone, exit_obj, camera

    def test_different_placements_produce_different_scores(self):

        good_building, good_floor, _, _, good_camera = self._build(camera_rotation=0.0)
        bad_building, bad_floor, _, _, bad_camera = self._build(camera_rotation=180.0)

        good_report = validate_floor([good_camera], good_building, good_floor)
        bad_report = validate_floor([bad_camera], bad_building, bad_floor)

        good_score = good_report.camera_metrics[good_camera.id].placement_score
        bad_score = bad_report.camera_metrics[bad_camera.id].placement_score

        self.assertGreater(good_score, bad_score)

    def test_poor_layout_is_detected_via_low_network_score_and_recommendations(self):

        _, _, _, _, bad_camera = self._build(camera_rotation=180.0)
        bad_building, bad_floor, _, exit_obj, bad_camera = self._build(camera_rotation=180.0)

        report = validate_floor([bad_camera], bad_building, bad_floor)

        self.assertLess(report.network_analysis.network_score, 50.0)
        self.assertIn(exit_obj.id, report.network_analysis.uncovered_exit_ids)

        categories = {r.category for r in report.recommendations}
        self.assertTrue({"rotate", "relocate", "increase_fov", "increase_range"} & categories)

    def test_blind_areas_are_identified(self):

        building, floor, zone, _, camera = self._build(camera_rotation=180.0)

        report = validate_floor([camera], building, floor)

        self.assertIn(zone.id, report.camera_metrics[camera.id].blind_zone_ids)

    def test_validate_building_aggregates_every_floor(self):

        building = Building(name="B")
        floor_1 = building.create_floor(name="Ground Floor")
        floor_2 = building.create_floor(name="Floor 1", height=3.0)

        zone_1 = make_zone("Zone 1", width=10.0, height=10.0, floor_id=floor_1.id)
        zone_2 = make_zone("Zone 2", width=10.0, height=10.0, floor_id=floor_2.id)
        floor_1.add_zone(zone_1)
        floor_2.add_zone(zone_2)

        camera_1 = Camera(
            name="Cam 1", floor_id=floor_1.id, position=(0.5, 5.0),
            rotation=0.0, horizontal_fov=170.0, max_range=20.0,
        )
        camera_2 = Camera(
            name="Cam 2", floor_id=floor_2.id, position=(0.5, 5.0),
            rotation=180.0, horizontal_fov=170.0, max_range=20.0,
        )
        floor_1.add_camera(camera_1)
        floor_2.add_camera(camera_2)

        report = validate_building([camera_1, camera_2], building)

        self.assertEqual(set(report.per_floor.keys()), {floor_1.id, floor_2.id})
        self.assertIsNotNone(report.floor_report(floor_1.id))
        # Floor 1's well-aimed camera should outscore Floor 2's badly-aimed one.
        self.assertGreater(
            report.per_floor[floor_1.id].network_analysis.network_score,
            report.per_floor[floor_2.id].network_analysis.network_score,
        )

    def test_validate_building_with_no_building_returns_an_empty_report(self):

        report = validate_building([], None)

        self.assertEqual(report.per_floor, {})
        self.assertEqual(report.overall_network_score, 0.0)


class BackwardCompatibilityTests(unittest.TestCase):

    # A legacy-shaped (pre-Camera-Coverage-Engine) saved Camera must
    # validate without error -- Camera.from_dict() already defaults
    # every new field honestly (see tests.test_camera.CameraModelTests.
    # test_loading_a_pre_framework_camera_dict_still_works).

    def test_validates_a_legacy_shaped_camera_without_error(self):

        from models.floor import Floor

        floor_data = {
            "id": "floor-legacy",
            "name": "Ground Floor",
            "display_order": 0,
            "height": 3.0,
            "floor_plan": "",
            "visible": True,
            "locked": False,
            "zones": [
                {
                    "id": "zone-legacy",
                    "name": "Room",
                    "object_type": "Zone",
                    "properties": {},
                    "created_at": "",
                    "modified_at": "",
                    "x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0,
                    "polygon": [], "floor_id": "floor-legacy",
                    "zone_type": "Generic", "max_occupancy": 0,
                }
            ],
            "exits": [], "stairs": [], "elevators": [],
            "cameras": [
                {
                    "id": "cam-legacy", "name": "Legacy Cam", "object_type": "Camera",
                    "properties": {}, "created_at": "", "modified_at": "",
                    "position": (1.0, 1.0), "floor_id": "floor-legacy", "rotation": 0.0,
                    "horizontal_fov": 90.0, "max_range": 25.0, "mount_height": 3.0, "active": True,
                }
            ],
            "detectors": [], "assembly_points": [], "obstacles": [], "doors": [],
        }

        floor = Floor.from_dict(floor_data)

        building = Building(name="Legacy Building")
        building.floors.append(floor)

        report = validate_building(list(floor.cameras), building)

        self.assertIn(floor.id, report.per_floor)
        floor_report = report.per_floor[floor.id]
        self.assertIn("cam-legacy", floor_report.camera_metrics)
        self.assertIsInstance(floor_report.camera_metrics["cam-legacy"].placement_score, float)


class CameraValidationPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase enforces -- camera_validation/ is
    # explicitly allowed (and expected) to import visibility/ (Phase 1's
    # own instruction to reuse it), but never simulator/ground_truth/
    # behavior*/virtual_camera/camera_manager/multi_camera_fusion/etc.

    def test_never_imports_simulation_or_camera_runtime_internals(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "camera_validation"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|ground_truth|behavior|behavior_library|behaviour_profile_resolver|"
            r"simulation_runtime|hazard_evolution|ai_training|rl_training|advisory_system|"
            r"command_center|designer|virtual_camera|camera_manager|multi_camera_fusion)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"camera_validation/{path.name} imports a simulation/camera-runtime module "
                f"directly -- it must only reuse Building/Camera geometry and the Visibility "
                f"Engine",
            )


if __name__ == "__main__":
    unittest.main()

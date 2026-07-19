import unittest

from dataclasses import fields

from perception.models.building_observation import ObservationState
from perception.models.camera_observation import CameraFrameObservation

from perception.fusion.occupancy_estimation import EstimatedOccupancy, OccupancyEstimator


class SingleCameraTests(unittest.TestCase):

    def test_single_camera_reading_becomes_the_zones_estimate(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimates = estimator.estimate(
            [CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=4.0, confidence=0.9)],
        )

        estimate = estimates["zone-a"]
        self.assertEqual(estimate.observation_state, ObservationState.OBSERVED)
        self.assertEqual(estimate.estimated_count, 4.0)
        self.assertEqual(estimate.confidence, 0.9)
        self.assertEqual(estimate.contributing_camera_ids, ["cam1"])


class MultipleCameraTests(unittest.TestCase):

    def test_cameras_assigned_to_different_zones_do_not_cross_contaminate(self):

        estimator = OccupancyEstimator(
            camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-b"},
        )

        estimates = estimator.estimate(
            [
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=2.0),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, estimated_occupant_count=9.0),
            ],
        )

        self.assertEqual(estimates["zone-a"].estimated_count, 2.0)
        self.assertEqual(estimates["zone-b"].estimated_count, 9.0)
        self.assertEqual(estimates["zone-a"].contributing_camera_ids, ["cam1"])
        self.assertEqual(estimates["zone-b"].contributing_camera_ids, ["cam2"])


class DuplicateObservationTests(unittest.TestCase):

    def test_two_cameras_covering_the_same_zone_never_have_their_counts_summed(self):

        estimator = OccupancyEstimator(
            camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-a"},
        )

        estimates = estimator.estimate(
            [
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=5.0),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, estimated_occupant_count=3.0),
            ],
        )

        estimate = estimates["zone-a"]
        # 5.0 + 3.0 would be 8.0 -- summing would double-count anyone
        # visible to both cameras. The higher single reading (5.0) is
        # used instead.
        self.assertEqual(estimate.estimated_count, 5.0)
        self.assertEqual(estimate.contributing_camera_ids, ["cam1", "cam2"])

    def test_tie_is_broken_by_confidence(self):

        estimator = OccupancyEstimator(
            camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-a"},
        )

        estimates = estimator.estimate(
            [
                CameraFrameObservation(
                    camera_id="cam1", timestamp=0.0, estimated_occupant_count=5.0, confidence=0.4,
                ),
                CameraFrameObservation(
                    camera_id="cam2", timestamp=0.0, estimated_occupant_count=5.0, confidence=0.95,
                ),
            ],
        )

        self.assertEqual(estimates["zone-a"].confidence, 0.95)


class UnobservedZoneTests(unittest.TestCase):

    def test_zone_with_no_camera_assigned_is_unobserved(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimate = estimator.estimated_occupancy_for("zone-never-assigned", camera_observations=[])

        self.assertEqual(estimate.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(estimate.estimated_count)
        self.assertIsNone(estimate.confidence)
        self.assertEqual(estimate.contributing_camera_ids, [])

    def test_zone_with_assigned_camera_but_no_reading_this_cycle_is_unobserved(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimates = estimator.estimate(camera_observations=[])

        self.assertEqual(estimates["zone-a"].observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(estimates["zone-a"].estimated_count)

    def test_unobserved_default_is_never_zero_occupants(self):

        default = EstimatedOccupancy()

        self.assertEqual(default.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(default.estimated_count)
        self.assertNotEqual(default.estimated_count, 0.0)


class ConfidencePropagationTests(unittest.TestCase):

    def test_confidence_passes_through_unmodified_for_a_single_camera(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimates = estimator.estimate(
            [CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=1.0, confidence=0.42)],
        )

        self.assertEqual(estimates["zone-a"].confidence, 0.42)

    def test_missing_camera_confidence_stays_none_not_fabricated(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimates = estimator.estimate(
            [CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=1.0)],
        )

        self.assertIsNone(estimates["zone-a"].confidence)


class UnknownPropagationTests(unittest.TestCase):

    def test_camera_with_no_reading_does_not_contribute(self):

        estimator = OccupancyEstimator(
            camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-a"},
        )

        estimates = estimator.estimate(
            [
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=None),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, estimated_occupant_count=6.0),
            ],
        )

        estimate = estimates["zone-a"]
        self.assertEqual(estimate.estimated_count, 6.0)
        self.assertEqual(estimate.contributing_camera_ids, ["cam2"])

    def test_all_cameras_unknown_makes_the_zone_unobserved_not_zero(self):

        estimator = OccupancyEstimator(
            camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-a"},
        )

        estimates = estimator.estimate(
            [
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=None),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, estimated_occupant_count=None),
            ],
        )

        estimate = estimates["zone-a"]
        self.assertEqual(estimate.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(estimate.estimated_count)


class GroundTruthNeverAppearsTests(unittest.TestCase):

    def test_estimated_occupancy_fields_are_all_plain_python_values(self):

        estimator = OccupancyEstimator(camera_zone_assignments={"cam1": "zone-a"})

        estimates = estimator.estimate(
            [CameraFrameObservation(camera_id="cam1", timestamp=0.0, estimated_occupant_count=2.0, confidence=0.7)],
        )

        allowed_field_types = (ObservationState, float, int, str, list, type(None))

        for estimate in estimates.values():
            for estimate_field in fields(estimate):
                value = getattr(estimate, estimate_field.name)
                if isinstance(value, list):
                    for item in value:
                        self.assertIsInstance(item, str)
                else:
                    self.assertIsInstance(value, allowed_field_types)

    def test_module_never_imports_ground_truth_hazard_or_rl_packages(self):

        import pathlib
        import re

        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "perception" / "fusion" / "occupancy_estimation.py"
        )

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(hazard|hazard_evolution|occupancy|fire_growth|smoke_propagation|"
            r"ai_decision|simulator|sandbox|behavior|designer|models|sensors|"
            r"gymnasium|gym|numpy|torch)\b"
        )

        text = path.read_text()

        self.assertIsNone(
            re.search(forbidden, text, re.MULTILINE),
            "perception/fusion/occupancy_estimation.py imports a Ground Truth, "
            "downstream, or AI-framework module -- Occupancy Estimation must stay "
            "independent of all of them, consuming only CameraFrameObservation and "
            "already-resolved camera-to-zone topology",
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from dataclasses import FrozenInstanceError, fields

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from perception.models.building_observation import (
    BuildingObservation,
    ObservationState,
    ObservedEdgeState,
    ObservedNodeState,
    ObservedOccupancy,
    PerceptionSeverity,
)
from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading

from perception.fusion.occupancy_estimation import EstimatedOccupancy
from perception.fusion.sensor_fusion import SensorFusion


GROUND_TRUTH_TYPES = (HazardSnapshot, HazardNodeState, OccupancySnapshot, OccupancyObservation)


def make_fusion(edge_zone_endpoints=None):

    return SensorFusion(
        smoke_detector_zone_assignments={"smoke1": "zone-a"},
        heat_detector_zone_assignments={"heat1": "zone-a"},
        camera_zone_assignments={"cam1": "zone-a", "cam2": "zone-a"},
        edge_zone_endpoints=edge_zone_endpoints,
    )


class OccupancyAndSmokeFusionTests(unittest.TestCase):

    def test_smoke_alarm_and_occupancy_are_both_reflected_for_the_same_zone(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=10.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(
                    observation_state=ObservationState.OBSERVED,
                    estimated_count=5.0, confidence=0.8,
                ),
            },
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=True, confidence=1.0),
            ],
        )

        self.assertEqual(result.occupancy_observation("zone-a").estimated_count, 5.0)
        self.assertEqual(result.occupancy_observation("zone-a").confidence, 0.8)

        node = result.node_observation("zone-a")
        self.assertEqual(node.observation_state, ObservationState.OBSERVED)
        self.assertTrue(node.alarm_active)
        self.assertEqual(node.alarm_source_types, ["Smoke"])
        self.assertEqual(node.estimated_severity, PerceptionSeverity.HIGH)
        self.assertEqual(node.last_observed_time, 10.0)


class OccupancyAndHeatFusionTests(unittest.TestCase):

    def test_heat_alarm_and_occupancy_are_both_reflected_for_the_same_zone(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=10.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(
                    observation_state=ObservationState.OBSERVED, estimated_count=2.0, confidence=0.6,
                ),
            },
            heat_detector_readings=[
                HeatDetectorReading(detector_id="heat1", timestamp=10.0, alarm_active=True, confidence=1.0),
            ],
        )

        node = result.node_observation("zone-a")
        self.assertTrue(node.alarm_active)
        self.assertEqual(node.alarm_source_types, ["Heat"])
        self.assertEqual(node.estimated_severity, PerceptionSeverity.HIGH)

    def test_smoke_and_heat_both_alarming_the_same_zone_is_critical(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=10.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=True),
            ],
            heat_detector_readings=[
                HeatDetectorReading(detector_id="heat1", timestamp=10.0, alarm_active=True),
            ],
        )

        node = result.node_observation("zone-a")
        self.assertEqual(node.alarm_source_types, ["Heat", "Smoke"])
        self.assertEqual(node.estimated_severity, PerceptionSeverity.CRITICAL)


class UnknownPropagationTests(unittest.TestCase):

    def test_zone_with_no_input_at_all_is_unobserved(self):

        fusion = make_fusion()

        result = fusion.fuse(timestamp=0.0, occupancy_estimates={})

        node = result.node_observation("zone-never-mentioned")
        self.assertEqual(node.observation_state, ObservationState.UNOBSERVED)
        self.assertIsNone(node.alarm_active)
        self.assertIsNone(node.estimated_severity)

        occupancy = result.occupancy_observation("zone-never-mentioned")
        self.assertIsNone(occupancy.estimated_count)

    def test_unobserved_estimated_occupancy_is_never_written_through_as_zero(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={"zone-a": EstimatedOccupancy()},  # UNOBSERVED default
        )

        self.assertEqual(len(result.occupancy_observations), 0)
        self.assertIsNone(result.occupancy_observation("zone-a").estimated_count)

    def test_detector_not_in_any_zone_assignment_is_ignored_not_guessed_at(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="unassigned-smoke", timestamp=0.0, alarm_active=True),
            ],
        )

        self.assertEqual(len(result.node_observations), 0)


class MissingSensorsTests(unittest.TestCase):

    def test_fuse_with_only_occupancy_and_no_sensor_lists_does_not_crash(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=5.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(observation_state=ObservationState.OBSERVED, estimated_count=1.0),
            },
        )

        self.assertEqual(result.occupancy_observation("zone-a").estimated_count, 1.0)
        self.assertEqual(len(result.node_observations), 0)
        self.assertEqual(result.system_status.active_camera_count, 0)
        self.assertEqual(result.system_status.active_detector_count, 0)

    def test_missing_camera_data_leaves_visibility_estimate_none(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=False),
            ],
        )

        self.assertIsNone(result.node_observation("zone-a").visibility_estimate)


class PartialObservationTests(unittest.TestCase):

    def test_only_heat_detector_present_leaves_smoke_type_out_of_alarm_sources(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            heat_detector_readings=[
                HeatDetectorReading(detector_id="heat1", timestamp=0.0, alarm_active=True),
            ],
        )

        node = result.node_observation("zone-a")
        self.assertEqual(node.alarm_source_types, ["Heat"])

    def test_camera_only_zone_is_observed_via_visibility_with_no_alarm_opinion(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=3.0,
            occupancy_estimates={},
            camera_observations=[
                CameraFrameObservation(camera_id="cam1", timestamp=3.0, visibility_estimate="reduced"),
            ],
        )

        node = result.node_observation("zone-a")
        self.assertEqual(node.observation_state, ObservationState.OBSERVED)
        self.assertIsNone(node.alarm_active)
        self.assertEqual(node.alarm_source_types, [])
        self.assertEqual(node.visibility_estimate, "reduced")
        self.assertEqual(node.estimated_severity, PerceptionSeverity.MODERATE)

    def test_disagreeing_camera_visibility_takes_the_worst_reading(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            camera_observations=[
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, visibility_estimate="clear"),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, visibility_estimate="heavy"),
            ],
        )

        self.assertEqual(result.node_observation("zone-a").visibility_estimate, "heavy")
        self.assertEqual(result.node_observation("zone-a").estimated_severity, PerceptionSeverity.HIGH)

    def test_confirmed_clear_detector_with_no_visibility_data_is_none_severity(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=False),
            ],
        )

        node = result.node_observation("zone-a")
        self.assertFalse(node.alarm_active)
        self.assertEqual(node.estimated_severity, PerceptionSeverity.NONE)


class ConfidencePropagationTests(unittest.TestCase):

    def test_occupancy_confidence_passes_through_unmodified(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(
                    observation_state=ObservationState.OBSERVED, estimated_count=3.0, confidence=0.37,
                ),
            },
        )

        self.assertEqual(result.occupancy_observation("zone-a").confidence, 0.37)

    def test_missing_occupancy_confidence_stays_none_not_fabricated(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(observation_state=ObservationState.OBSERVED, estimated_count=3.0),
            },
        )

        self.assertIsNone(result.occupancy_observation("zone-a").confidence)


class BuildingObservationCorrectnessTests(unittest.TestCase):

    def test_returns_a_frozen_building_observation_with_the_given_timestamp(self):

        fusion = make_fusion()

        result = fusion.fuse(timestamp=42.0, occupancy_estimates={})

        self.assertIsInstance(result, BuildingObservation)
        self.assertEqual(result.timestamp, 42.0)
        self.assertEqual(result.schema_version, 1)

        with self.assertRaises(FrozenInstanceError):
            result.timestamp = 0.0

    def test_edge_observations_are_empty_without_topology(self):

        fusion = make_fusion()

        result = fusion.fuse(timestamp=0.0, occupancy_estimates={})

        self.assertEqual(len(result.edge_observations), 0)

    def test_edge_is_blocked_when_either_endpoint_is_a_confirmed_hazard(self):

        fusion = make_fusion(edge_zone_endpoints={"edge1": ("zone-a", "zone-b")})

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=True),
            ],
        )

        self.assertTrue(result.edge_observation("edge1").blocked_estimate)

    def test_edge_is_clear_only_when_both_endpoints_are_observed_and_clear(self):

        fusion = SensorFusion(
            smoke_detector_zone_assignments={"smoke1": "zone-a", "smoke2": "zone-b"},
            heat_detector_zone_assignments={},
            camera_zone_assignments={},
            edge_zone_endpoints={"edge1": ("zone-a", "zone-b")},
        )

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=False),
                SmokeDetectorReading(detector_id="smoke2", timestamp=0.0, alarm_active=False),
            ],
        )

        self.assertFalse(result.edge_observation("edge1").blocked_estimate)

    def test_edge_is_unknown_when_one_endpoint_is_unobserved_and_the_other_is_clear(self):

        fusion = make_fusion(edge_zone_endpoints={"edge1": ("zone-a", "zone-b")})

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=False),
            ],
        )

        self.assertIsNone(result.edge_observation("edge1").blocked_estimate)

    def test_system_status_counts_distinct_reporting_devices(self):

        fusion = make_fusion()

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={},
            camera_observations=[
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, visibility_estimate="clear"),
                CameraFrameObservation(camera_id="cam2", timestamp=0.0, visibility_estimate="clear"),
            ],
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=False),
            ],
            heat_detector_readings=[
                HeatDetectorReading(detector_id="heat1", timestamp=0.0, alarm_active=False),
            ],
        )

        self.assertEqual(result.system_status.active_camera_count, 2)
        self.assertEqual(result.system_status.active_detector_count, 2)
        self.assertTrue(result.system_status.panel_communication_ok)


class NoGroundTruthAppearsTests(unittest.TestCase):

    def test_no_field_anywhere_in_building_observation_is_a_ground_truth_object(self):

        fusion = make_fusion(edge_zone_endpoints={"edge1": ("zone-a", "zone-b")})

        result = fusion.fuse(
            timestamp=0.0,
            occupancy_estimates={
                "zone-a": EstimatedOccupancy(
                    observation_state=ObservationState.OBSERVED, estimated_count=4.0, confidence=0.5,
                ),
            },
            camera_observations=[
                CameraFrameObservation(camera_id="cam1", timestamp=0.0, visibility_estimate="heavy"),
            ],
            smoke_detector_readings=[
                SmokeDetectorReading(detector_id="smoke1", timestamp=0.0, alarm_active=True),
            ],
            heat_detector_readings=[
                HeatDetectorReading(detector_id="heat1", timestamp=0.0, alarm_active=True),
            ],
        )

        self._assert_clean(result)
        for node in result.node_observations.values():
            self._assert_clean(node)
        for occupancy in result.occupancy_observations.values():
            self._assert_clean(occupancy)
        for edge in result.edge_observations.values():
            self._assert_clean(edge)
        self._assert_clean(result.system_status)

    def _assert_clean(self, obj):

        self.assertNotIsInstance(obj, GROUND_TRUTH_TYPES)

        for obj_field in fields(obj):
            value = getattr(obj, obj_field.name)
            self.assertNotIsInstance(value, GROUND_TRUTH_TYPES)

    def test_module_never_imports_ground_truth_or_downstream_packages(self):

        import pathlib
        import re

        path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "perception" / "fusion" / "sensor_fusion.py"
        )

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|hazard|hazard_evolution|occupancy|designer|behavior|"
            r"ai_decision|navigation|sandbox|models|sensors|"
            r"gymnasium|gym|numpy|torch)\b"
        )

        text = path.read_text()

        self.assertIsNone(
            re.search(forbidden, text, re.MULTILINE),
            "perception/fusion/sensor_fusion.py imports a Ground Truth, downstream, or "
            "AI-framework module -- Sensor Fusion must stay within perception.* only",
        )


if __name__ == "__main__":
    unittest.main()

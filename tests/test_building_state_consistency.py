import unittest

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from camera_manager.status import CameraStatus
from sensor_manager.status import SensorStatus

from perception.models.camera_observation import CameraFrameObservation
from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.models.heat_detector_observation import HeatDetectorReading

from building_state.consistency import ConsistencyWarningType, check_consistency
from building_state.estimator import BuildingStateEstimator


def make_camera_status(camera_id="cam1", zone_ids=("zone-a",), active=True):

    return CameraStatus(
        camera_id=camera_id, name=camera_id, floor_id="floor1",
        zone_ids=zone_ids, active=active, mode="Simulation",
        has_detection_provider=True,
    )


def make_sensor_status(sensor_id, sensor_type, zone_ids=("zone-a",), active=True):

    return SensorStatus(
        sensor_id=sensor_id, sensor_type=sensor_type, name=sensor_id, floor_id="floor1",
        zone_ids=zone_ids, active=active, mode="Simulation", health_status="OK",
    )


class ConsistencyCheckTests(unittest.TestCase):

    def setUp(self):

        self.estimator = BuildingStateEstimator()

    def _warning_types(self, state):

        return {warning.warning_type for warning in check_consistency(state)}

    def test_clean_state_produces_no_warnings(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0, observations={"zone-a": OccupancyObservation(occupant_count=1.0)}),
            camera_statuses=[make_camera_status()],
            camera_observations=[CameraFrameObservation(camera_id="cam1", timestamp=10.0, estimated_occupant_count=1.0)],
        )

        self.assertEqual(check_consistency(state), ())

    def test_camera_reports_occupants_but_occupancy_is_zero(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0, observations={"zone-a": OccupancyObservation(occupant_count=0.0)}),
            camera_statuses=[make_camera_status()],
            camera_observations=[CameraFrameObservation(camera_id="cam1", timestamp=10.0, estimated_occupant_count=4.0)],
        )

        self.assertIn(ConsistencyWarningType.CAMERA_REPORTS_OCCUPANTS_BUT_ZONE_EMPTY, self._warning_types(state))

    def test_smoke_detector_alarm_with_no_nearby_hazard(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0, node_states={"zone-a": HazardNodeState(hazard_score=0.0)}),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector")],
            smoke_detector_readings=[SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=True)],
        )

        self.assertIn(ConsistencyWarningType.SMOKE_ALARM_WITHOUT_NEARBY_HAZARD, self._warning_types(state))

    def test_heat_detector_alarm_with_no_nearby_hazard(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0, node_states={"zone-a": HazardNodeState(hazard_score=0.0)}),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            heat_detector_statuses=[make_sensor_status("heat1", "HeatDetector")],
            heat_detector_readings=[HeatDetectorReading(detector_id="heat1", timestamp=10.0, alarm_active=True)],
        )

        self.assertIn(ConsistencyWarningType.HEAT_ALARM_WITHOUT_NEARBY_HAZARD, self._warning_types(state))

    def test_alarming_detector_with_real_hazard_produces_no_warning(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0, node_states={"zone-a": HazardNodeState(hazard_score=0.9)}),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector")],
            smoke_detector_readings=[SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=True)],
        )

        self.assertNotIn(ConsistencyWarningType.SMOKE_ALARM_WITHOUT_NEARBY_HAZARD, self._warning_types(state))

    def test_hazard_present_but_covering_detector_inactive(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0, node_states={"zone-a": HazardNodeState(hazard_score=0.9)}),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector", active=False)],
        )

        self.assertIn(ConsistencyWarningType.HAZARD_PRESENT_BUT_DETECTOR_INACTIVE, self._warning_types(state))

    def test_hazard_present_with_no_detector_coverage_is_not_flagged_as_inactive(self):

        # A coverage gap (no detector assigned to the zone at all) is
        # a different problem from an assigned-but-inactive detector --
        # see check_consistency's own _check_hazard_without_active_detector.
        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0, node_states={"zone-a": HazardNodeState(hazard_score=0.9)}),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
        )

        self.assertNotIn(ConsistencyWarningType.HAZARD_PRESENT_BUT_DETECTOR_INACTIVE, self._warning_types(state))

    def test_offline_camera_and_sensor_are_flagged(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=HazardSnapshot(timestamp=10.0),
            occupancy_snapshot=OccupancySnapshot(timestamp=10.0),
            camera_statuses=[make_camera_status(active=False)],
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector", active=False)],
        )

        warning_types = self._warning_types(state)

        self.assertIn(ConsistencyWarningType.CAMERA_OFFLINE, warning_types)
        self.assertIn(ConsistencyWarningType.SENSOR_OFFLINE, warning_types)


if __name__ == "__main__":
    unittest.main()

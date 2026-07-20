import unittest

from dataclasses import FrozenInstanceError

from hazard.node_state import HazardNodeState
from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from models.sensor_asset import DetectorState

from camera_manager.status import CameraStatus
from sensor_manager.status import SensorStatus

from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading
from perception.models.human_observation import HumanClassification, HumanState

from multi_camera_fusion.track import FusedTrack, FusionResult, TrackHistory

from building_state.estimator import BuildingStateEstimator
from building_state.models import ActiveAssetsSummary, BuildingState, HazardSummary


def make_camera_status(camera_id="cam1", zone_ids=("zone-a",), active=True):

    return CameraStatus(
        camera_id=camera_id, name=camera_id, floor_id="floor1",
        zone_ids=zone_ids, active=active, mode="Simulation",
        has_detection_provider=True,
    )


def make_sensor_status(sensor_id, sensor_type, zone_ids=("zone-a",), active=True, health_status="OK"):

    return SensorStatus(
        sensor_id=sensor_id, sensor_type=sensor_type, name=sensor_id, floor_id="floor1",
        zone_ids=zone_ids, active=active, mode="Simulation", health_status=health_status,
    )


def make_track(track_id="t1", zone_id="zone-a"):

    return FusedTrack(
        track_id=track_id, source_camera_ids=("cam1",), floor_id="floor1", zone_id=zone_id,
        classification=HumanClassification.ADULT, human_state=HumanState.WALKING,
        confidence=0.9, timestamp=10.0,
        history=TrackHistory(
            track_id=track_id, previous_camera_id=None, current_camera_id="cam1",
            last_observation_time=10.0, current_zone_id=zone_id, current_floor_id="floor1",
        ),
    )


class BuildingStateImmutabilityTests(unittest.TestCase):

    def test_state_is_frozen(self):

        state = BuildingState(timestamp=1.0)

        with self.assertRaises(FrozenInstanceError):
            state.timestamp = 2.0

    def test_mapping_fields_are_read_only(self):

        state = BuildingState(timestamp=1.0, occupant_tracks={"t1": make_track()})

        with self.assertRaises(TypeError):
            state.occupant_tracks["t2"] = make_track("t2")

    def test_total_accessors_return_none_for_unknown_ids(self):

        state = BuildingState(timestamp=1.0)

        self.assertIsNone(state.occupant_track("unknown"))
        self.assertIsNone(state.camera_observation("unknown"))
        self.assertIsNone(state.smoke_detector_state("unknown"))
        self.assertIsNone(state.heat_detector_state("unknown"))

    def test_zone_severity_defaults_to_none_for_unreported_zone(self):

        state = BuildingState(timestamp=1.0)

        self.assertEqual(state.zone_severity("unreported-zone"), HazardSeverity.NONE)

    def test_default_zone_occupancy_is_empty_snapshot(self):

        state = BuildingState(timestamp=1.0)

        self.assertEqual(state.zone_occupancy.observation_at("zone-a"), OccupancyObservation())


class BuildingStateEstimatorTests(unittest.TestCase):

    def setUp(self):

        self.estimator = BuildingStateEstimator()

        self.hazard_snapshot = HazardSnapshot(
            timestamp=10.0,
            node_states={
                "zone-a": HazardNodeState(hazard_score=0.9),
                "zone-b": HazardNodeState(hazard_score=0.1),
            },
        )

        self.occupancy_snapshot = OccupancySnapshot(
            timestamp=10.0,
            observations={"zone-a": OccupancyObservation(occupant_count=3.0)},
        )

    def test_never_mutates_inputs(self):

        hazard_before = dict(self.hazard_snapshot.node_states)
        occupancy_before = dict(self.occupancy_snapshot.observations)

        self.estimator.estimate(
            10.0,
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
        )

        self.assertEqual(dict(self.hazard_snapshot.node_states), hazard_before)
        self.assertEqual(dict(self.occupancy_snapshot.observations), occupancy_before)

    def test_estimate_is_deterministic_aside_from_state_id(self):

        kwargs = dict(
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
            camera_statuses=[make_camera_status()],
            camera_observations=[CameraFrameObservation(camera_id="cam1", timestamp=10.0, estimated_occupant_count=2.0)],
        )

        first = self.estimator.estimate(10.0, **kwargs)
        second = self.estimator.estimate(10.0, **kwargs)

        self.assertNotEqual(first.state_id, second.state_id)

        self.assertEqual(first.timestamp, second.timestamp)
        self.assertEqual(dict(first.camera_observations), dict(second.camera_observations))
        self.assertEqual(first.hazard_summary, second.hazard_summary)
        self.assertEqual(first.building_alarm_status, second.building_alarm_status)
        self.assertEqual(first.active_assets, second.active_assets)

    def test_hazard_summary_picks_worst_zone(self):

        state = self.estimator.estimate(
            10.0, hazard_snapshot=self.hazard_snapshot, occupancy_snapshot=self.occupancy_snapshot,
        )

        self.assertEqual(state.hazard_summary.overall_severity, HazardSeverity.CRITICAL)
        self.assertEqual(state.hazard_summary.worst_zone_id, "zone-a")
        self.assertEqual(state.zone_severity("zone-b"), HazardSeverity.LOW)

    def test_camera_observation_composed_from_status_and_reading(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
            camera_statuses=[make_camera_status(active=False)],
        )

        asset = state.camera_observation("cam1")

        self.assertIsNotNone(asset)
        self.assertFalse(asset.status.active)
        self.assertIsNone(asset.frame_observation)
        self.assertIn("cam1", state.active_assets.offline_camera_ids)

    def test_occupant_tracks_come_from_fusion_result(self):

        fusion_result = FusionResult(timestamp=10.0, tracks=(make_track(),))

        state = self.estimator.estimate(
            10.0, hazard_snapshot=self.hazard_snapshot, occupancy_snapshot=self.occupancy_snapshot,
            fusion_result=fusion_result,
        )

        track = state.occupant_track("t1")

        self.assertIsNotNone(track)
        self.assertEqual(track.zone_id, "zone-a")

    def test_alarm_status_defaults_to_normal_with_no_detectors(self):

        state = self.estimator.estimate(
            10.0, hazard_snapshot=self.hazard_snapshot, occupancy_snapshot=self.occupancy_snapshot,
        )

        self.assertEqual(state.building_alarm_status, DetectorState.NORMAL)

    def test_alarm_status_is_alarm_when_any_detector_alarms(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector")],
            smoke_detector_readings=[SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=True)],
            heat_detector_statuses=[make_sensor_status("heat1", "HeatDetector")],
            heat_detector_readings=[HeatDetectorReading(detector_id="heat1", timestamp=10.0, alarm_active=False)],
        )

        self.assertEqual(state.building_alarm_status, DetectorState.ALARM)

    def test_alarm_status_is_fault_when_a_healthy_flag_is_off_and_no_alarm(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
            smoke_detector_statuses=[make_sensor_status("smoke1", "SmokeDetector", health_status="Fault")],
            smoke_detector_readings=[SmokeDetectorReading(detector_id="smoke1", timestamp=10.0, alarm_active=False)],
        )

        self.assertEqual(state.building_alarm_status, DetectorState.FAULT)

    def test_active_assets_summary_sorts_ids_into_active_and_offline(self):

        state = self.estimator.estimate(
            10.0,
            hazard_snapshot=self.hazard_snapshot,
            occupancy_snapshot=self.occupancy_snapshot,
            camera_statuses=[make_camera_status("cam1", active=True), make_camera_status("cam2", active=False)],
            smoke_detector_statuses=[
                make_sensor_status("smoke1", "SmokeDetector", active=True),
                make_sensor_status("smoke2", "SmokeDetector", active=False),
            ],
        )

        self.assertEqual(state.active_assets, ActiveAssetsSummary(
            active_camera_ids=("cam1",),
            offline_camera_ids=("cam2",),
            active_sensor_ids=("smoke1",),
            offline_sensor_ids=("smoke2",),
        ))


# =====================================================
# Canonical Live BuildingState Runtime Assembly milestone -- Phase 10
# architecture guards.
# =====================================================


class BuildingStatePackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase enforces. building_state must never
    # depend on live_system (the dependency runs the other direction --
    # see live_system/building_state_gateway.py) or on any AI/Advisory/
    # Command Center package: BuildingStateEstimator "performs no AI
    # reasoning, no reinforcement learning, and no decision-making of
    # any kind" (its own docstring) and must stay that way regardless
    # of what live_system now composes it into.
    #
    # Separately, BuildingState "is observational. It does not
    # acknowledge, silence, reset, or control the panel" (Phase 5 of
    # the Canonical Live BuildingState Runtime Assembly milestone) --
    # enforced here by forbidding the CONTROL-CAPABLE facp/building_control
    # submodules (facp.engine's SimulatedFACP, facp.provider's
    # FACPEventProvider, building_control.controller's
    # BuildingControlController, building_control.providers) while still
    # allowing the read-only value types this package already legitimately
    # imports and passes through unchanged (facp.models.FACPSnapshot,
    # building_control.snapshot.ControlStateSnapshot -- see
    # building_state/models.py's own facp_status/control_status fields).

    def test_never_imports_live_system_ai_advisory_or_command_center(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "building_state"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(live_system|ai_inference|ai_decision|advisory_system|command_center|"
            r"decision_policy|rl_training)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"building_state/{path.name} imports a live-runtime, AI, decision-policy, "
                f"or advisory/command-center module directly -- BuildingState must remain "
                f"a pure observational fusion of already-computed values",
            )

    def test_never_imports_control_capable_facp_or_building_control_internals(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "building_state"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(facp\.engine|facp\.provider|building_control\.controller|"
            r"building_control\.providers)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"building_state/{path.name} imports a control-capable FACP/Building "
                f"Control class directly -- BuildingState is observational and must never "
                f"acknowledge, silence, reset, or control anything itself; only the "
                f"read-only facp.models/building_control.snapshot value types are allowed",
            )


if __name__ == "__main__":
    unittest.main()

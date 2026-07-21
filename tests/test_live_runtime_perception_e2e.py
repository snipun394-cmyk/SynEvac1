import unittest

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy

from scenario.scenario import Scenario

from tests.test_advisory_system import make_metadata

from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.heat_detector import HeatDetector
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from live_runtime.factory import build_offline_demo_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Perception -> BuildingState Integration Bridge milestone, Phase
# 11 -- one deterministic, fully offline end-to-end demonstration of
# the ENTIRE chain named in this milestone's own brief:
#
#   Replay frames -> YOLOHumanDetector -> SingleCameraTracker ->
#   WorldProjection -> BehaviorRecognizer -> CrossCameraIdentityResolver
#   -> LiveOccupantManager -> SmokeDetector/HeatDetector readings ->
#   SimulatedFACP -> SensorFusionEngine -> BuildingStateEstimator ->
#   LiveOrchestrator -> Live AI (unconfigured -- no new AI model built,
#   per this milestone's own explicit "DO NOT implement new AI models"
#   instruction, matching tests/test_live_runtime_e2e.py's own existing
#   precedent) -> AdvisoryReport.
#
# Zero physical CCTV, zero network access, zero hardware anywhere in
# this file.
# =====================================================


CAMERA_ID = "CAM-1"
FLOOR_ID = "floor-1"
ZONE_ID = "zone-1"


def make_building():

    floor = Floor(
        id=FLOOR_ID, name="Ground Floor",
        zones=[Zone(id=ZONE_ID, name="Lobby", x=0.0, y=-10.0, width=20.0, height=20.0, floor_id=FLOOR_ID)],
        cameras=[Camera(id=CAMERA_ID, name="Lobby Camera", floor_id=FLOOR_ID, zone_ids=(ZONE_ID,))],
        smoke_detectors=[SmokeDetector(id="SD-1", name="Smoke 1", floor_id=FLOOR_ID, zone_ids=(ZONE_ID,))],
        heat_detectors=[HeatDetector(id="HD-1", name="Heat 1", floor_id=FLOOR_ID, zone_ids=(ZONE_ID,))],
    )

    return Building(id="e2e-building", name="E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id=FLOOR_ID, intrinsics=intrinsics, extrinsics=extrinsics)

    zone = Zone(name=ZONE_ID, x=0.0, y=-10.0, width=20.0, height=20.0, floor_id=FLOOR_ID)
    zone.id = ZONE_ID

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={FLOOR_ID: [zone]})


def make_advisory_gateway(building):

    scenario = Scenario(metadata=make_metadata("e2e-scn"), occupants=(), firefighters=())

    ground_truth = GroundTruth(
        scenario_id="e2e-scn", definition_id="e2e-def",
        total_evacuation_time=60.0, building_cleared=False,
        reachable_occupants=1, unreachable_occupants=0,
        people_trapped=0, people_evacuated=1,
        worst_exit=None, zone_route_stats=[], maximum_hazard_zone=ZONE_ID,
        hazard_spread_order=(ZONE_ID,), first_hazardous_zone=ZONE_ID,
        doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
        stairs_exceeding_capacity=(), zone_risk_scores=[], stair_risk_scores=[],
        recommendations=[],
        helping_group_count=0, fallen_count=0, possible_injury_count=0,
    )

    decision_policy = DecisionPolicy(
        scenario_id="e2e-scn",
        zone_decisions=[{"zone_id": ZONE_ID, "recommended_exit": None, "recommended_stair": None, "action": "EVACUATE_IMMEDIATELY"}],
        exit_decisions=(), stair_decisions=(), announcements=(),
        rescue_priorities=[{"zone_id": ZONE_ID, "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 1}],
        rescue_order=(),
    )

    return ReplayCompatibleAdvisoryGateway(
        building=building, scenario=scenario, ground_truth=ground_truth,
        decision_policy_provider=lambda time: decision_policy,
    )


class FullOfflinePerceptionToAdvisoryChainTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(310.0, 200.0, 330.0, 260.0)))

        frame_sources = {CAMERA_ID: ReplayFrameSource(camera_id=CAMERA_ID, frames=[(0.0, "frame-0")])}
        for source in frame_sources.values():
            source.start()

        topology = CameraTopology()
        registry = IdentityRegistry()
        transition_model = TransitionModel(topology, timeout_seconds=30.0)
        cross_camera_resolver = RuleBasedCrossCameraIdentityResolver(topology=topology, registry=registry, transition_model=transition_model)

        self.facp = SimulatedFACP(panel_id="FACP-1")

        def smoke_reading_provider(time):

            class _Reading:
                detector_id = "SD-1"
                timestamp = time
                alarm_active = True
                confidence = 0.9

            return [_Reading()]

        def heat_reading_provider(time):

            class _Reading:
                detector_id = "HD-1"
                timestamp = time
                alarm_active = False
                confidence = 0.9

            return [_Reading()]

        self.runtime = build_offline_demo_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=YOLOHumanDetector(backend),
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            cross_camera_identity_resolver=cross_camera_resolver,
            world_projector=make_world_projector(),
            smoke_detector_reading_provider=smoke_reading_provider,
            heat_detector_reading_provider=heat_reading_provider,
            facp=self.facp,
            live_advisory_gateway=make_advisory_gateway(self.building),
        )

    def tearDown(self):
        self.runtime.stop()

    def test_full_chain_from_replay_frames_to_advisory_report(self):

        self.runtime.start()

        # Evaluate FACP with this cycle's own detector condition report
        # BEFORE run_cycle() so facp.current_snapshot() (already wired
        # into build_offline_demo_runtime() via `facp=`) reflects it --
        # the exact same read-only-passthrough discipline docs/
        # architecture/live_runtime_composition.md already documents
        # ("this factory never calls facp.evaluate() itself").
        smoke_status = next(s for s in self.runtime.sensor_manager.all_statuses() if s.sensor_id == "SD-1")
        condition = DetectorConditionReport.from_status_and_reading(
            smoke_status, type("R", (), {"alarm_active": True, "confidence": 0.9})(),
        )
        self.facp.evaluate({"SD-1": condition}, time=0.0)

        self.runtime.run_cycle(0.0)

        # PERCEPTION -- a person was detected, tracked, projected,
        # behavior-recognized, cross-camera-identity-resolved, and is
        # now a LiveOccupant.
        active_occupants = self.runtime.live_occupant_manager.active_occupants()
        self.assertEqual(len(active_occupants), 1)
        occupant = active_occupants[0]
        self.assertEqual(occupant.current_zone_id, ZONE_ID)
        self.assertIsNotNone(occupant.world_position)

        # FUSION -- exactly one shared SensorFusionEngine, real fused
        # SMOKE/OCCUPANCY observations.
        snapshot = self.runtime.perception_fusion_coordinator.collect(0.0)
        fused_kinds = {f.kind.name for f in snapshot.fused_observations}
        self.assertIn("SMOKE", fused_kinds)
        self.assertIn("OCCUPANCY", fused_kinds)

        # STATE -- BuildingState reflects both the fused hazard and the
        # tracked occupant, through the UNMODIFIED BuildingStateEstimator.
        building_state = self.runtime.orchestrator.latest_building_state
        self.assertEqual(len(building_state.occupant_tracks), 1)
        self.assertGreater(building_state.hazard_summary.zone_severities[ZONE_ID].value, 0)
        self.assertEqual(building_state.zone_occupancy.observation_at(ZONE_ID).occupant_count, 1.0)
        self.assertIsNotNone(building_state.facp_status)
        self.assertIn("SD-1", building_state.facp_status.active_alarm_source_ids)

        # ADVISORY -- a real AdvisoryReport was produced from the SAME
        # cycle's AI Decision Evidence (Live AI itself left unconfigured
        # -- no new AI model built this milestone).
        snapshot = self.runtime.command_center_data_source.current_snapshot()
        self.assertIsNotNone(snapshot.advisory_report)

    def test_zero_network_and_zero_hardware_access(self):

        for source in self.runtime.frame_sources.values():
            self.assertEqual(type(source).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

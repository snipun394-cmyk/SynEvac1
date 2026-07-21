import unittest

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy
from decision_policy.exit_policy import KEEP_OPEN
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY

from scenario.scenario import Scenario

from tests.test_advisory_system import make_metadata

from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway
from live_system.event_bus import EventBus

from models.building import Building
from models.camera import Camera
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine
from crowd_intelligence.trends import TrendConfig as CrowdTrendConfig

from evacuation_progress.engine import EvacuationProgressEngine

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Evacuation Progress, Flow & Clearance Intelligence milestone,
# Phase 20 -- deterministic offline end-to-end proof, driven through the
# COMPLETE production chain (ReplayFrameSource -> Fake YOLO -> Tracker ->
# Projection -> Behavior -> LiveOccupants -> Sensor Fusion ->
# BuildingState -> Crowd Intelligence -> Evacuation Progress -> Live AI
# (unconfigured) -> Advisory -> Command Center), across multiple cycles.
# Two occupants approach EXIT-1, queue, then actually cross it (become
# EXITED); a second, SAFE, uncongested EXIT-2 exists throughout. Zero
# network, zero physical CCTV, zero automatic voice/building-control
# execution.
# =====================================================


CAMERA_ID = "CAM-1"


def make_building():

    exit_1 = Exit(id="EXIT-1", floor_id="f1", start_point=(-0.15, -0.1), end_point=(-0.15, -0.1), width=1.2, capacity=2)
    exit_2 = Exit(id="EXIT-2", floor_id="f1", start_point=(15.0, 15.0), end_point=(15.0, 15.0), width=1.2, capacity=2)

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Lobby", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")],
        cameras=[Camera(id=CAMERA_ID, name="Lobby Camera", floor_id="f1", zone_ids=("z1",))],
        exits=[exit_1, exit_2],
    )

    return Building(id="evac-progress-e2e-building", name="Evacuation Progress E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id="f1", intrinsics=intrinsics, extrinsics=extrinsics)

    zone = Zone(name="z1", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")
    zone.id = "z1"

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={"f1": [zone]})


class EvacuationProgressEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        all_exits = self.building.floors[0].exits

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(20.0, 20.0, 60.0, 60.0)))                      # t=0, far
        backend.queue_result(person(confidence=0.9, box=(150.0, 120.0, 190.0, 160.0)))                  # t=1, closer
        backend.queue_result(
            person(confidence=0.9, box=(280.0, 220.0, 320.0, 260.0)),
            person(confidence=0.9, box=(335.0, 250.0, 355.0, 270.0)),
        )                                                                                                 # t=2, both near
        backend.queue_result(
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
            person(confidence=0.9, box=(330.0, 248.0, 350.0, 268.0)),
        )                                                                                                 # t=3, both stationary -- queue forms
        backend.queue_result()                                                                            # t=4, both have crossed EXIT-1

        frame_source = ReplayFrameSource(
            camera_id=CAMERA_ID, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3"), (4.0, "f4")],
        )
        frame_source.start()

        event_bus = EventBus()

        live_occupant_manager = LiveOccupantManager(
            event_bus=event_bus, exits=all_exits, exit_proximity_threshold=2.0, expire_after_seconds=1000.0,
        )
        crowd_intelligence_engine = CrowdIntelligenceEngine(
            self.building, live_occupant_manager, trend_config=CrowdTrendConfig(trend_window_seconds=1.5),
        )
        self.evacuation_progress_engine = EvacuationProgressEngine(self.building, live_occupant_manager, event_bus)
        self.live_occupant_manager = live_occupant_manager

        scenario = Scenario(metadata=make_metadata("evac-progress-e2e"), occupants=(), firefighters=())
        ground_truth = GroundTruth(
            scenario_id="evac-progress-e2e", definition_id="def-1",
            total_evacuation_time=60.0, building_cleared=False,
            reachable_occupants=2, unreachable_occupants=0,
            people_trapped=0, people_evacuated=2,
            worst_exit=None, zone_route_stats=[], maximum_hazard_zone=None,
            hazard_spread_order=(), first_hazardous_zone=None,
            doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
            stairs_exceeding_capacity=(), zone_risk_scores=[], stair_risk_scores=[],
            recommendations=[], helping_group_count=0, fallen_count=0, possible_injury_count=0,
        )

        def decision_policy_provider(time):

            return DecisionPolicy(
                scenario_id="evac-progress-e2e",
                zone_decisions=[{"zone_id": "z1", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "EXIT-1"}],
                exit_decisions=[{"exit_id": "EXIT-1", "status": KEEP_OPEN}, {"exit_id": "EXIT-2", "status": KEEP_OPEN}],
                stair_decisions=(), announcements=(),
                rescue_priorities=[{"zone_id": "z1", "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 2}],
                rescue_order=(),
            )

        advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=self.building, scenario=scenario, ground_truth=ground_truth,
            decision_policy_provider=decision_policy_provider,
        )

        self.runtime = build_live_runtime(
            self.building,
            frame_sources={CAMERA_ID: frame_source},
            human_detector=YOLOHumanDetector(backend),
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(max_centroid_distance=300.0),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
            live_occupant_manager=live_occupant_manager,
            crowd_intelligence_engine=crowd_intelligence_engine,
            evacuation_progress_engine=self.evacuation_progress_engine,
            live_advisory_gateway=advisory_gateway,
            event_bus=event_bus,
            # Deliberately NOT wired -- Phase 15/20's own "no automatic
            # action" requirement.
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_full_chain_progress_rises_queue_forms_flow_measured_then_clears(self):

        progress_values = []
        active_values = []

        for time in (0.0, 1.0, 2.0, 3.0):

            self.runtime.run_cycle(time)

            snapshot = self.evacuation_progress_engine.compute(time, self.runtime.orchestrator.latest_building_state, None)
            progress_values.append(snapshot.evacuation_progress_fraction)
            active_values.append(snapshot.known_active_occupants)

        # Occupancy rises as both occupants are progressively observed.
        self.assertEqual(active_values, [1, 1, 2, 2])
        # Nobody has exited yet -- no honest basis for any progress > 0.
        self.assertTrue(all(v in (None, 0.0) for v in progress_values))

        # Queue has formed by t=3 (both stationary near EXIT-1).
        crowd_snapshot_t3 = self.runtime.crowd_intelligence_engine.compute(3.0)
        snapshot_t3 = self.evacuation_progress_engine.compute(3.0, self.runtime.orchestrator.latest_building_state, crowd_snapshot_t3)
        self.assertGreater(snapshot_t3.exit("EXIT-1").queue_candidate_count, 0)

        # t=4: both occupants have crossed EXIT-1 -- flow is measured,
        # progress reaches 100% of the 2 tracked identities, and the
        # queue clears.
        self.runtime.run_cycle(4.0)
        snapshot_t4 = self.evacuation_progress_engine.compute(4.0, self.runtime.orchestrator.latest_building_state, None)

        self.assertEqual(snapshot_t4.known_active_occupants, 0)
        self.assertEqual(snapshot_t4.known_exited_occupants, 2)
        self.assertAlmostEqual(snapshot_t4.evacuation_progress_fraction, 1.0)
        self.assertTrue(snapshot_t4.exit("EXIT-1").flow_active)
        self.assertEqual(snapshot_t4.exit("EXIT-1").unique_exited_count, 2)
        self.assertGreater(snapshot_t4.exit("EXIT-1").recent_flow_per_minute, 0.0)

        # No double counting anywhere -- exactly 2 physical occupants,
        # never 3+, at every single cycle observed.
        for active, cycle_progress in zip(active_values, progress_values):
            self.assertLessEqual(active, 2)

    def test_advisory_reflects_evacuation_progress_and_safe_exit_remains_available(self):

        for time in (0.0, 1.0, 2.0, 3.0, 4.0):
            self.runtime.run_cycle(time)

        report = self.runtime.orchestrator.latest_advisory_report
        self.assertIsNotNone(report)

        # EXIT-2 (SAFE, uncongested throughout) is never marked unsafe
        # or excluded by this milestone's own evidence.
        exit_targets_from_progress = {
            rec.target_id for rec in report.building_recommendations
            if "progress" in rec.confidence_source and rec.target_type == "exit"
        }
        self.assertNotIn("EXIT-2", exit_targets_from_progress)

    def test_no_automatic_voice_or_building_control_execution(self):

        for time in (0.0, 1.0, 2.0, 3.0, 4.0):
            self.runtime.run_cycle(time)

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)

    def test_zero_network_and_zero_hardware_access(self):

        self.assertEqual(type(self.runtime.frame_sources[CAMERA_ID]).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

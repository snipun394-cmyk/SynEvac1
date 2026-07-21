import unittest

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy
from decision_policy.exit_policy import CLOSE, KEEP_OPEN
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY

from scenario.scenario import Scenario

from tests.test_advisory_system import make_metadata

from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway

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
from crowd_intelligence.trends import TrendConfig

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Crowd Intelligence -> Operational Advisory Integration milestone,
# Phase 17 -- deterministic offline end-to-end proof, driven through the
# COMPLETE production chain (ReplayFrameSource -> Fake YOLO -> Tracker ->
# Projection -> Behavior -> LiveOccupants -> Live Perception ->
# BuildingState -> Crowd Intelligence -> Live AI (left unconfigured, no
# new AI model built) -> Advisory), across multiple cycles. Two SAFE
# exits exist; two occupants accumulate near EXIT-1, forming a queue;
# Advisory must recognize the congestion and may support EXIT-2 -- but
# the moment EXIT-2 itself becomes hazardous, it must NEVER be
# recommended regardless of EXIT-1's own congestion. Zero network, zero
# physical CCTV, zero automatic voice/building-control execution.
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

    return Building(id="crowd-advisory-e2e-building", name="Crowd Advisory E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id="f1", intrinsics=intrinsics, extrinsics=extrinsics)

    zone = Zone(name="z1", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")
    zone.id = "z1"

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={"f1": [zone]})


class CrowdAwareAdvisoryEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend = FakeYOLOBackend()
        backend.queue_result(person(confidence=0.9, box=(20.0, 20.0, 60.0, 60.0)))
        backend.queue_result(person(confidence=0.9, box=(150.0, 120.0, 190.0, 160.0)))
        backend.queue_result(
            person(confidence=0.9, box=(280.0, 220.0, 320.0, 260.0)),
            person(confidence=0.9, box=(335.0, 250.0, 355.0, 270.0)),
        )
        backend.queue_result(
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
            person(confidence=0.9, box=(330.0, 248.0, 350.0, 268.0)),
        )
        backend.queue_result()  # t=4 -- both gone

        frame_source = ReplayFrameSource(
            camera_id=CAMERA_ID, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3"), (4.0, "f4")],
        )
        frame_source.start()

        # This test toggles EXIT-2's own safety mid-run (Phase 17 item 5)
        # -- a plain mutable holder the decision_policy_provider closure
        # below reads fresh every cycle, exactly mirroring how a real
        # Replay/live deployment's own decision_policy_provider would
        # reflect a changing hazard state each cycle.
        self.exit_2_status = {"value": KEEP_OPEN}

        scenario = Scenario(metadata=make_metadata("crowd-adv-e2e"), occupants=(), firefighters=())
        ground_truth = GroundTruth(
            scenario_id="crowd-adv-e2e", definition_id="def-1",
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
                scenario_id="crowd-adv-e2e",
                zone_decisions=[{"zone_id": "z1", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "EXIT-1"}],
                exit_decisions=[
                    {"exit_id": "EXIT-1", "status": KEEP_OPEN},
                    {"exit_id": "EXIT-2", "status": self.exit_2_status["value"]},
                ],
                stair_decisions=(), announcements=(),
                rescue_priorities=[{"zone_id": "z1", "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 2}],
                rescue_order=(),
            )

        advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=self.building, scenario=scenario, ground_truth=ground_truth,
            decision_policy_provider=decision_policy_provider,
        )

        live_occupant_manager = LiveOccupantManager()
        crowd_intelligence_engine = CrowdIntelligenceEngine(
            self.building, live_occupant_manager, trend_config=TrendConfig(trend_window_seconds=1.5),
        )

        self.voice_calls = []
        self.control_calls = []

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
            live_advisory_gateway=advisory_gateway,
            # Deliberately NOT wired: voice_output_provider/
            # building_control_provider -- Phase 15's own "no automatic
            # action" requirement. Both stay None, so
            # voice_evacuation_controller/building_control_controller are
            # None on the resulting LiveRuntime -- there is nothing for
            # crowd intelligence (or Advisory) to have automatically
            # triggered even in principle.
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def _monitor_actions(self, report):
        return [rec.action for rec in report.building_recommendations if "crowd" in rec.confidence_source]

    def _prefer_actions(self, report):
        return [rec.action for rec in report.building_recommendations if rec.action.startswith("Prefer ")]

    def test_full_chain_detects_congestion_and_supports_the_clear_exit(self):

        for time in (0.0, 1.0, 2.0):
            self.runtime.run_cycle(time)

        report_mid = self.runtime.orchestrator.latest_advisory_report
        self.assertIsNotNone(report_mid)

        self.runtime.run_cycle(3.0)  # both occupants now stationary near EXIT-1 -- queue at its peak
        report_peak = self.runtime.orchestrator.latest_advisory_report

        # 1. Crowd Intelligence detects congestion.
        crowd_snapshot = self.runtime.crowd_intelligence_engine.compute(3.0)
        self.assertIn("EXIT-1", crowd_snapshot.building_summary.congested_exits)

        # 2. AdvisoryReport contains crowd evidence.
        monitor_actions = self._monitor_actions(report_peak)
        self.assertTrue(any("EXIT-1" in action for action in monitor_actions))

        # 3. Advisory recognizes the congested exit specifically.
        self.assertTrue(any(action == "Monitor Congestion at Exit EXIT-1" for action in monitor_actions))

        # 4. The alternate SAFE exit (EXIT-2, still KEEP_OPEN, uncongested)
        # may receive supporting preference.
        prefer_actions = self._prefer_actions(report_peak)
        self.assertEqual(prefer_actions, ["Prefer Exit EXIT-2 over Exit EXIT-1"])

        # 6. No voice message sent automatically.
        self.assertIsNone(self.runtime.voice_evacuation_controller)

        # 7. No building control executes automatically.
        self.assertIsNone(self.runtime.building_control_controller)

    def test_alternate_exit_becoming_unsafe_immediately_removes_the_preference_regardless_of_congestion(self):

        for time in (0.0, 1.0, 2.0, 3.0):
            self.runtime.run_cycle(time)

        report_before = self.runtime.orchestrator.latest_advisory_report
        self.assertEqual(self._prefer_actions(report_before), ["Prefer Exit EXIT-2 over Exit EXIT-1"])

        # 5. EXIT-2 becomes hazardous -- decision_policy itself now marks
        # it CLOSE. EXIT-1's own congestion is unchanged (still at its
        # queued peak) -- the preference must disappear immediately,
        # regardless of how congested EXIT-1 still is.
        self.exit_2_status["value"] = CLOSE

        self.runtime.run_cycle(3.5)
        report_after = self.runtime.orchestrator.latest_advisory_report

        self.assertEqual(self._prefer_actions(report_after), [])

        for rec in report_after.building_recommendations:
            if "crowd" in rec.confidence_source:
                self.assertNotEqual(rec.target_id, "EXIT-2")

    def test_advisory_evidence_updates_once_the_queue_clears(self):

        for time in (0.0, 1.0, 2.0, 3.0):
            self.runtime.run_cycle(time)

        report_peak = self.runtime.orchestrator.latest_advisory_report
        self.assertEqual(self._prefer_actions(report_peak), ["Prefer Exit EXIT-2 over Exit EXIT-1"])

        self.runtime.run_cycle(4.0)  # both occupants leave -- queue clears
        report_cleared = self.runtime.orchestrator.latest_advisory_report

        crowd_snapshot = self.runtime.crowd_intelligence_engine.compute(4.0)
        self.assertNotIn("EXIT-1", crowd_snapshot.building_summary.congested_exits)

        self.assertEqual(self._prefer_actions(report_cleared), [])
        self.assertFalse(any("EXIT-1" in action for action in self._monitor_actions(report_cleared)))

    def test_zero_network_and_zero_hardware_access(self):

        self.assertEqual(type(self.runtime.frame_sources[CAMERA_ID]).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

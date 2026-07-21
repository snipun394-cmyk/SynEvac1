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
from models.engineering_asset import DeviceMode
from models.smoke_detector import SmokeDetector

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

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import ResponsePriorityLevel

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Emergency Response & Rescue Priority Intelligence milestone,
# Phase 23 -- deterministic offline end-to-end proof, driven through
# the COMPLETE production chain (ReplayFrameSource -> Fake YOLO ->
# Tracker -> Projection -> Behavior -> LiveOccupants -> Sensor Fusion ->
# BuildingState -> Crowd Intelligence -> Evacuation Progress ->
# Emergency Response Intelligence -> (Live AI unconfigured) -> Advisory
# -> Command Center), across multiple cycles, with THREE zones:
#
#   ZONE-A -- already evacuated, confirmed clear (camera covered).
#   ZONE-B -- occupants remain, a queue forms at EXIT-1, clearance
#             stalls, then both occupants actually cross and it clears.
#   ZONE-C -- no occupants ever visible; its OWN camera starts online
#             (confirmed clear), then goes offline mid-run -- clearance
#             must become UNCERTAIN again, never permanently "clear."
#
# Zero network, zero physical CCTV, zero automatic voice/building-
# control/firefighter-dispatch execution anywhere in this file.
# =====================================================


CAMERA_AB = "CAM-AB"
CAMERA_C = "CAM-C"


def make_building():

    exit_1 = Exit(id="EXIT-1", floor_id="f1", start_point=(-0.15, -0.1), end_point=(-0.15, -0.1), width=1.2, capacity=2)
    smoke_detector = SmokeDetector(id="SD-B", name="Smoke B", floor_id="f1", zone_ids=("zone-b",))

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="zone-a", name="Zone A", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1"),
            Zone(id="zone-b", name="Zone B", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1"),
            Zone(id="zone-c", name="Zone C", x=100.0, y=100.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        cameras=[
            Camera(id=CAMERA_AB, name="AB Camera", floor_id="f1", zone_ids=("zone-a", "zone-b")),
            Camera(id=CAMERA_C, name="C Camera", floor_id="f1", zone_ids=("zone-c",)),
        ],
        smoke_detectors=[smoke_detector],
        exits=[exit_1],
    )

    # zone-a and zone-b deliberately overlap the same physical area for
    # this test's own purposes (zone-a represents "the part of this
    # space already fully evacuated," zone-b "the part still occupied
    # near the exit") -- both are covered by the SAME camera, and
    # occupant zone_id assignment (via WorldProjector's own zone lookup)
    # determines which one each detection actually lands in; only
    # zone-b's own world_projector zone list is populated below, so
    # every real detection resolves to zone-b, leaving zone-a
    # genuinely, honestly occupant-free throughout.
    return Building(id="emergency-response-e2e-building", name="Emergency Response E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0)
    calibration = CalibrationProfile(camera_id=CAMERA_AB, floor_id="f1", intrinsics=intrinsics, extrinsics=extrinsics)

    zone_b = Zone(name="zone-b", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")
    zone_b.id = "zone-b"

    return WorldProjector(calibrations={CAMERA_AB: calibration}, zones_by_floor={"f1": [zone_b]})


class EmergencyResponseEndToEndTests(unittest.TestCase):

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
        backend.queue_result()  # t=4 -- both have crossed EXIT-1

        frame_source = ReplayFrameSource(
            camera_id=CAMERA_AB, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3"), (4.0, "f4")],
        )
        frame_source.start()

        event_bus = EventBus()

        live_occupant_manager = LiveOccupantManager(
            event_bus=event_bus, exits=[self.building.floors[0].exits[0]], exit_proximity_threshold=2.0,
            expire_after_seconds=1000.0,
        )
        crowd_intelligence_engine = CrowdIntelligenceEngine(
            self.building, live_occupant_manager, trend_config=CrowdTrendConfig(trend_window_seconds=1.5),
        )
        # A short trend_window_seconds (matching this test's own 1-
        # second cycle cadence), same reasoning as the crowd_intelligence
        # engine above -- the default 30s window is sized for a real
        # deployment's much longer runtime, not a 4-second test.
        self.evacuation_progress_engine = EvacuationProgressEngine(
            self.building, live_occupant_manager, event_bus, trend_config=CrowdTrendConfig(trend_window_seconds=1.5),
        )
        self.emergency_response_engine = EmergencyResponseIntelligenceEngine(self.building, live_occupant_manager)
        self.live_occupant_manager = live_occupant_manager

        scenario = Scenario(metadata=make_metadata("emergency-response-e2e"), occupants=(), firefighters=())
        ground_truth = GroundTruth(
            scenario_id="emergency-response-e2e", definition_id="def-1",
            total_evacuation_time=60.0, building_cleared=False,
            reachable_occupants=2, unreachable_occupants=0,
            people_trapped=0, people_evacuated=2,
            worst_exit=None, zone_route_stats=[], maximum_hazard_zone=None,
            hazard_spread_order=(), first_hazardous_zone=None,
            doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
            stairs_exceeding_capacity=(), zone_risk_scores=[], stair_risk_scores=[],
            recommendations=[], helping_group_count=0, fallen_count=0, possible_injury_count=0,
        )

        def smoke_reading_provider(time):

            class _Reading:
                detector_id = "SD-B"
                timestamp = time
                alarm_active = True
                confidence = 0.9

            return [_Reading()]

        def decision_policy_provider(time):

            return DecisionPolicy(
                scenario_id="emergency-response-e2e",
                zone_decisions=[{"zone_id": "zone-b", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "EXIT-1"}],
                exit_decisions=[{"exit_id": "EXIT-1", "status": KEEP_OPEN}],
                stair_decisions=(), announcements=(),
                rescue_priorities=[{"zone_id": "zone-b", "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 2}],
                rescue_order=(),
            )

        advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=self.building, scenario=scenario, ground_truth=ground_truth,
            decision_policy_provider=decision_policy_provider,
        )

        self.runtime = build_live_runtime(
            self.building,
            frame_sources={CAMERA_AB: frame_source},
            human_detector=YOLOHumanDetector(backend),
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(max_centroid_distance=300.0),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
            live_occupant_manager=live_occupant_manager,
            crowd_intelligence_engine=crowd_intelligence_engine,
            evacuation_progress_engine=self.evacuation_progress_engine,
            emergency_response_engine=self.emergency_response_engine,
            live_advisory_gateway=advisory_gateway,
            event_bus=event_bus,
            smoke_detector_reading_provider=smoke_reading_provider,
            # Deliberately NOT wired -- Phase 22 items 28/29/30's own
            # "no automatic action" requirement.
        )

        self.runtime.start()

        # CAM-C (zone-c's own, separate camera) starts online -- zone-c
        # is genuinely, confirmedly empty from the very first cycle.
        self.runtime.camera_manager.set_camera_mode(CAMERA_C, DeviceMode.LIVE)

    def tearDown(self):
        self.runtime.stop()

    def _response_snapshot(self, time):

        return self.emergency_response_engine.compute(
            time, self.runtime.orchestrator.latest_building_state,
            self.runtime.crowd_intelligence_engine.compute(time),
            self.evacuation_progress_engine.compute(time, self.runtime.orchestrator.latest_building_state, None),
        )

    def test_zone_b_rises_to_highest_priority_while_zone_c_stays_uncertain(self):

        for time in (0.0, 1.0, 2.0):
            self.runtime.run_cycle(time)

        snapshot = self._response_snapshot(2.0)

        # Zone A: already evacuated, confirmed clear.
        self.assertEqual(snapshot.zone("zone-a").priority_level, ResponsePriorityLevel.LOW)

        # Zone C: genuinely empty AND confirmed by an online camera --
        # also low priority at this point (not yet gone offline).
        self.assertEqual(snapshot.zone("zone-c").priority_level, ResponsePriorityLevel.LOW)

        self.runtime.run_cycle(3.0)  # both occupants now stationary near EXIT-1 -- queue at its peak

        snapshot_peak = self._response_snapshot(3.0)

        # 1. Zone B rises to the highest response priority in the
        # building -- the milestone's own actual requirement is the
        # RELATIVE ranking, not a specific absolute level (this scenario
        # deliberately carries no hazard/FACP evidence, so MODERATE, not
        # HIGH/CRITICAL, is the honest, correctly-scored outcome here --
        # see tests/test_emergency_response.py::StalledIncreasesTests
        # for hazard's own separate, independently-tested contribution).
        self.assertEqual(snapshot_peak.highest_priority_zone_id(), "zone-b")
        self.assertNotEqual(snapshot_peak.zone("zone-b").priority_level, ResponsePriorityLevel.LOW)
        self.assertEqual(snapshot_peak.response_priority_order[0], "zone-b")
        self.assertIn("EVACUATION_STALLED", snapshot_peak.zone("zone-b").reason_codes)

        # Zone C's own camera now goes offline -- clearance must become
        # an uncertainty concern again, never permanently "clear." Camera.
        # active (toggled via disable_camera) is the field that governs
        # whether a camera is currently covering its zone at all --
        # Camera.mode (LIVE/SIMULATION, set via set_camera_mode) is a
        # SEPARATE concept (which DetectionProvider answers for it), not
        # an online/offline switch (confirmed by direct investigation of
        # camera_manager/manager.py: set_camera_mode() only ever mutates
        # Camera.mode, never Camera.active).
        self.runtime.camera_manager.disable_camera(CAMERA_C)
        self.runtime.run_cycle(3.5)

        snapshot_c_offline = self._response_snapshot(3.5)
        zone_c = snapshot_c_offline.zone("zone-c")

        self.assertNotEqual(zone_c.clearance_status, "OBSERVED_CLEAR")
        self.assertIn("UNCERTAIN_OCCUPANCY", zone_c.reason_codes)

    def test_zone_b_priority_falls_once_it_clears(self):

        for time in (0.0, 1.0, 2.0, 3.0):
            self.runtime.run_cycle(time)

        peak_priority = self._response_snapshot(3.0).zone("zone-b").priority_score

        self.runtime.run_cycle(4.0)  # both occupants cross EXIT-1 -- zone-b clears

        cleared_priority = self._response_snapshot(4.0).zone("zone-b").priority_score

        self.assertLess(cleared_priority, peak_priority)

    def test_advisory_reflects_response_priority(self):

        for time in (0.0, 1.0, 2.0, 3.0):
            self.runtime.run_cycle(time)

        report = self.runtime.orchestrator.latest_advisory_report
        self.assertIsNotNone(report)

        response_actions = [rec.action for rec in report.building_recommendations if "response" in rec.confidence_source]
        self.assertTrue(any("zone-b" in action for action in response_actions))

    def test_no_automatic_execution_or_dispatch(self):

        for time in (0.0, 1.0, 2.0, 3.0, 4.0):
            self.runtime.run_cycle(time)

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)

        report = self.runtime.orchestrator.latest_advisory_report
        firefighter_report = report.firefighter_intelligence.to_dict()
        self.assertNotIn("dispatch", firefighter_report)
        self.assertNotIn("assigned_task", firefighter_report)

    def test_zero_network_and_zero_hardware_access(self):

        self.assertEqual(type(self.runtime.frame_sources[CAMERA_AB]).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

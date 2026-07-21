import unittest
from typing import Dict, List, Optional, Tuple

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy
from decision_policy.exit_policy import KEEP_OPEN
from decision_policy.zone_policy import WAIT

from scenario.scenario import Scenario

from tests.test_advisory_system import make_metadata

from live_system.live_advisory_gateway import ReplayCompatibleAdvisoryGateway
from live_system.event_bus import EventBus, EventType

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.zone import Zone

from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource
from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine
from crowd_intelligence.trends import TrendConfig as CrowdTrendConfig

from evacuation_progress.engine import EvacuationProgressEngine

from emergency_response.engine import EmergencyResponseIntelligenceEngine
from emergency_response.models import ResponsePriorityLevel

from live_runtime.factory import build_live_runtime


# =====================================================
# Live Human State & Assistance Perception Bridge milestone, Phase 24 --
# deterministic offline end-to-end proof, driven through the COMPLETE
# production chain (a fake, but honest, HumanDetector -> Tracker ->
# Behavior Recognition -> Identity -> LiveOccupants -> Sensor Fusion ->
# BuildingState -> Crowd Intelligence -> Evacuation Progress ->
# Emergency Response -> Advisory -> Command Center), proving genuine
# classification/state evidence survives every stage without being lost
# or fabricated.
#
# CAMERA_ID's own fake detector never pretends to be YOLO -- it is a
# test-only stand-in that directly asserts classification_evidence/
# state_evidence exactly the way a genuinely richer future detector
# would (mirrors tests/test_live_camera_pipeline.py::FakeHumanDetector's
# own established pattern). RuleBasedBehaviorRecognizer (real,
# production code) supplies the ONLY genuinely live POSSIBLY_FALLEN
# heuristic signal, from real tracking geometry -- never fabricated by
# this test.
#
#   OCC-1 (zone-1): WALKING -> RUNNING -> genuine FALLEN evidence ->
#                   BEING_ASSISTED (all via the fake detector's own
#                   state_evidence field).
#   OCC-2 (zone-2): a low, wide, stationary bounding box -- the REAL
#                   RuleBasedBehaviorRecognizer heuristic (enabled)
#                   genuinely concludes POSSIBLY_FALLEN from geometry
#                   alone; the fake detector supplies NO state_evidence
#                   for this occupant at all.
#   OCC-3 (zone-3): no classification/state evidence ever supplied --
#                   must never receive a fabricated classification.
#
# Zero network, zero physical CCTV, zero automatic voice/building-
# control/firefighter-dispatch execution anywhere in this file.
# =====================================================


CAMERA_ID = "CAM-1"


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="zone-1", name="Zone 1", x=0.0, y=0.0, width=50.0, height=50.0, floor_id="f1"),
            Zone(id="zone-2", name="Zone 2", x=100.0, y=0.0, width=50.0, height=50.0, floor_id="f1"),
            Zone(id="zone-3", name="Zone 3", x=200.0, y=0.0, width=50.0, height=50.0, floor_id="f1"),
        ],
        cameras=[Camera(id=CAMERA_ID, name="Camera 1", floor_id="f1", zone_ids=("zone-1", "zone-2", "zone-3"))],
    )

    return Building(id="human-evidence-e2e-building", name="Human Evidence E2E Building", floors=[floor])


class _CycleSpec:

    def __init__(self, local_track_id, zone_id, bounding_box, classification=None, state=None):
        self.local_track_id = local_track_id
        self.zone_id = zone_id
        self.bounding_box = bounding_box
        self.classification = classification
        self.state = state


class FakeStatefulHumanDetector(HumanDetector):

    # A test-only stand-in for a real, richer vision model -- honestly
    # asserts classification_evidence/state_evidence only where THIS
    # test explicitly configures it (never claims YOLO discovered
    # anything), mirroring tests/test_live_camera_pipeline.py's own
    # FakeHumanDetector pattern exactly.

    def __init__(self):
        self._queue: List[List[_CycleSpec]] = []

    def queue_cycle(self, specs: List[_CycleSpec]) -> None:
        self._queue.append(specs)

    def detect(self, frame) -> Tuple[RawHumanDetection, ...]:

        specs = self._queue.pop(0) if self._queue else []

        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id, local_track_id=spec.local_track_id, timestamp=frame.timestamp,
                bounding_box=spec.bounding_box, confidence=0.9,
                classification_evidence=spec.classification, state_evidence=spec.state,
                floor_id="f1", zone_id=spec.zone_id,
            )
            for spec in specs
        )


class HumanEvidenceEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        self.detector = FakeStatefulHumanDetector()

        # OCC-1 -- walks then runs (real geometry-driven behavior), then
        # the fake detector directly supplies genuine FALLEN evidence,
        # then BEING_ASSISTED -- all via state_evidence, never behavior.
        occ1_boxes = [(90, 95, 110, 115), (95, 95, 115, 115), (100, 95, 120, 115), (100, 95, 120, 115), (100, 95, 120, 115)]
        occ1_states = [None, None, HumanState.FALLEN, HumanState.BEING_ASSISTED, HumanState.BEING_ASSISTED]

        # OCC-2 -- a low, wide, perfectly stationary box (aspect ratio
        # 10/50 = 0.2 < the recognizer's own 0.6 threshold) -- POSSIBLY_
        # FALLEN must emerge from REAL tracking geometry, not from any
        # evidence this test injects directly.
        occ2_box = (290, 140, 340, 150)

        # OCC-3 -- ordinary, upright, stationary -- no evidence of any
        # kind ever supplied.
        occ3_box = (290, 95, 310, 115)

        for t in range(5):

            self.detector.queue_cycle([
                _CycleSpec("t1", "zone-1", occ1_boxes[t], state=occ1_states[t]),
                _CycleSpec("t2", "zone-2", occ2_box),
                _CycleSpec("t3", "zone-3", occ3_box),
            ])

        frame_source = ReplayFrameSource(
            camera_id=CAMERA_ID,
            frames=[(float(t), f"frame-{t}") for t in range(5)],
        )
        frame_source.start()

        event_bus = EventBus()

        live_occupant_manager = LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=1000.0)
        crowd_intelligence_engine = CrowdIntelligenceEngine(
            self.building, live_occupant_manager, trend_config=CrowdTrendConfig(trend_window_seconds=1.5),
        )
        evacuation_progress_engine = EvacuationProgressEngine(
            self.building, live_occupant_manager, event_bus, trend_config=CrowdTrendConfig(trend_window_seconds=1.5),
        )
        self.emergency_response_engine = EmergencyResponseIntelligenceEngine(self.building, live_occupant_manager)
        self.live_occupant_manager = live_occupant_manager

        scenario = Scenario(metadata=make_metadata("human-evidence-e2e"), occupants=(), firefighters=())
        ground_truth = GroundTruth(
            scenario_id="human-evidence-e2e", definition_id="def-1",
            total_evacuation_time=60.0, building_cleared=False,
            reachable_occupants=3, unreachable_occupants=0,
            people_trapped=0, people_evacuated=0,
            worst_exit=None, zone_route_stats=[], maximum_hazard_zone=None,
            hazard_spread_order=(), first_hazardous_zone=None,
            doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
            stairs_exceeding_capacity=(), zone_risk_scores=[], stair_risk_scores=[],
            recommendations=[], helping_group_count=0, fallen_count=0, possible_injury_count=0,
        )

        def decision_policy_provider(time):

            return DecisionPolicy(
                scenario_id="human-evidence-e2e",
                zone_decisions=[
                    {"zone_id": zid, "action": WAIT, "recommended_exit": None}
                    for zid in ("zone-1", "zone-2", "zone-3")
                ],
                exit_decisions=[], stair_decisions=(), announcements=(),
                rescue_priorities=[], rescue_order=(),
            )

        advisory_gateway = ReplayCompatibleAdvisoryGateway(
            building=self.building, scenario=scenario, ground_truth=ground_truth,
            decision_policy_provider=decision_policy_provider,
        )

        self.runtime = build_live_runtime(
            self.building,
            frame_sources={CAMERA_ID: frame_source},
            human_detector=self.detector,
            identity_resolver=SimulationIdentityResolver(),
            tracker=SimpleSingleCameraTracker(max_centroid_distance=100.0),
            behavior_recognizer=RuleBasedBehaviorRecognizer(enable_possibly_fallen_heuristic=True),
            live_occupant_manager=live_occupant_manager,
            crowd_intelligence_engine=crowd_intelligence_engine,
            evacuation_progress_engine=evacuation_progress_engine,
            emergency_response_engine=self.emergency_response_engine,
            live_advisory_gateway=advisory_gateway,
            event_bus=event_bus,
            # Deliberately NOT wired -- no automatic action.
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_occ1_receives_confirmed_assistance_after_fallen(self):

        for t in range(3):
            self.runtime.run_cycle(float(t))

        occ1 = self.live_occupant_manager.occupants_in_zone("zone-1")[0]
        self.assertEqual(occ1.human_state, HumanState.FALLEN)

        snapshot = self.emergency_response_engine.compute(2.0, None, None, None)
        zone1 = snapshot.zone("zone-1")

        self.assertEqual(zone1.confirmed_assistance_count, 1)
        self.assertIn("CONFIRMED_ASSISTANCE_REQUIRED", zone1.reason_codes)

    def test_occ1_transitions_to_being_assisted_and_is_distinguishable(self):

        for t in range(4):
            self.runtime.run_cycle(float(t))

        occ1 = self.live_occupant_manager.occupants_in_zone("zone-1")[0]
        self.assertEqual(occ1.human_state, HumanState.BEING_ASSISTED)

        snapshot = self.emergency_response_engine.compute(3.0, None, None, None)
        zone1 = snapshot.zone("zone-1")

        self.assertEqual(zone1.being_assisted_count, 1)
        self.assertEqual(zone1.confirmed_assistance_count, 0)
        self.assertIn("ASSISTANCE_IN_PROGRESS", zone1.reason_codes)

    def test_occ2_remains_possible_assistance_only(self):

        for t in range(5):
            self.runtime.run_cycle(float(t))

        occ2 = self.live_occupant_manager.occupants_in_zone("zone-2")[0]
        self.assertIsNone(occ2.human_state)

        snapshot = self.emergency_response_engine.compute(4.0, None, None, None)
        zone2 = snapshot.zone("zone-2")

        self.assertEqual(zone2.possible_assistance_count, 1)
        self.assertEqual(zone2.confirmed_assistance_count, 0)
        self.assertIn("POSSIBLE_ASSISTANCE_REQUIRED", zone2.reason_codes)

    def test_occ3_never_receives_a_fabricated_classification(self):

        for t in range(5):
            self.runtime.run_cycle(float(t))

        occ3 = self.live_occupant_manager.occupants_in_zone("zone-3")[0]
        self.assertEqual(occ3.human_classification, HumanClassification.UNKNOWN)
        self.assertIsNone(occ3.human_state)

    def test_no_automatic_execution_or_dispatch(self):

        for t in range(5):
            self.runtime.run_cycle(float(t))

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)

        report = self.runtime.orchestrator.latest_advisory_report
        self.assertIsNotNone(report)
        firefighter_report = report.firefighter_intelligence.to_dict()
        self.assertNotIn("dispatch", firefighter_report)
        self.assertNotIn("assigned_task", firefighter_report)

    def test_zero_network_and_zero_hardware_access(self):

        self.assertEqual(type(self.runtime.frame_sources[CAMERA_ID]).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

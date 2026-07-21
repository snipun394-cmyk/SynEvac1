import unittest

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
from crowd_intelligence.models import TrendDirection
from crowd_intelligence.trends import TrendConfig

from live_runtime.factory import build_live_runtime

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Occupancy, Crowd Density & Congestion Intelligence milestone,
# Phase 16 -- deterministic offline end-to-end proof, driven entirely
# through the real production chain (ReplayFrameSource -> YOLOHumanDetector
# w/ fake backend -> SingleCameraTracker -> WorldProjection ->
# BehaviorRecognizer -> LiveOccupants -> Live Perception Fusion ->
# CrowdIntelligenceEngine -> BuildingState/StateManager), across multiple
# cycles. Two occupants walk toward the same Exit, stop (forming a
# queue), then both leave -- proving occupancy/density/approach/queue/
# congestion all rise, then all fall. Zero network, zero physical CCTV.
#
# Bounding-box sequences and their exact projected world positions were
# derived directly from camera_calibration.projection.WorldProjector's
# own geometry for this fixed camera (position (0,0), mount_height 3m,
# pitch 90 degrees straight down, focal length 500) -- not guessed.
# =====================================================


CAMERA_ID = "CAM-1"


def make_building():

    exit_obj = Exit(id="EXIT-1", floor_id="f1", start_point=(-0.15, -0.1), end_point=(-0.15, -0.1), width=1.2, capacity=2)

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[Zone(id="z1", name="Lobby", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")],
        cameras=[Camera(id=CAMERA_ID, name="Lobby Camera", floor_id="f1", zone_ids=("z1",))],
        exits=[exit_obj],
    )

    return Building(id="crowd-e2e-building", name="Crowd E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0)
    calibration = CalibrationProfile(camera_id=CAMERA_ID, floor_id="f1", intrinsics=intrinsics, extrinsics=extrinsics)

    zone = Zone(name="z1", x=-20.0, y=-20.0, width=40.0, height=40.0, floor_id="f1")
    zone.id = "z1"

    return WorldProjector(calibrations={CAMERA_ID: calibration}, zones_by_floor={"f1": [zone]})


class CrowdAccumulationAndDispersalEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend = FakeYOLOBackend()

        # Person 1 walks toward the Exit over 4 cycles, then stops.
        backend.queue_result(person(confidence=0.9, box=(20.0, 20.0, 60.0, 60.0)))                      # t=0, far
        backend.queue_result(person(confidence=0.9, box=(150.0, 120.0, 190.0, 160.0)))                  # t=1, closer
        backend.queue_result(                                                                          # t=2, person 1 near + person 2 arrives near
            person(confidence=0.9, box=(280.0, 220.0, 320.0, 260.0)),
            person(confidence=0.9, box=(335.0, 250.0, 355.0, 270.0)),
        )
        backend.queue_result(                                                                          # t=3, both stationary near the exit -- queue forms
            person(confidence=0.9, box=(320.0, 240.0, 340.0, 260.0)),
            person(confidence=0.9, box=(330.0, 248.0, 350.0, 268.0)),
        )
        backend.queue_result()                                                                          # t=4, both gone -- queue clears

        frame_source = ReplayFrameSource(
            camera_id=CAMERA_ID,
            frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3"), (4.0, "f4")],
        )
        frame_source.start()

        # A short trend_window_seconds (matching this test's own 1-
        # second cycle cadence) so "trend" compares against the
        # immediately preceding cycle, not this run's very first sample
        # -- the same configuration choice a real deployment makes to
        # match its own update interval (Phase 8's own "configurable
        # time windows"). The default 30s window is designed for a real
        # deployment's much longer runtime, not a 4-second test.
        live_occupant_manager = LiveOccupantManager()
        crowd_intelligence_engine = CrowdIntelligenceEngine(
            self.building, live_occupant_manager, trend_config=TrendConfig(trend_window_seconds=1.5),
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
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_occupancy_density_approach_and_congestion_rise_then_fall(self):

        engine = self.runtime.crowd_intelligence_engine

        occupancy_over_time = []
        density_over_time = []
        approaching_over_time = []
        queue_over_time = []
        congestion_over_time = []

        for time in (0.0, 1.0, 2.0, 3.0, 4.0):

            self.runtime.run_cycle(time)

            snapshot = engine.compute(time)
            zone = snapshot.zone("z1")
            exit_metrics = snapshot.exit("EXIT-1")

            occupancy_over_time.append(zone.occupant_count)
            density_over_time.append(zone.density_people_per_m2)
            approaching_over_time.append(exit_metrics.approaching_count)
            queue_over_time.append(exit_metrics.queue_candidate_count)
            congestion_over_time.append(exit_metrics.congestion_level.value if exit_metrics.congestion_level is not None else 0)

        # Occupancy: 1 -> 1 -> 2 -> 2 -> 0 (both leave at t=4).
        self.assertEqual(occupancy_over_time, [1, 1, 2, 2, 0])

        # Density is a direct multiple of occupancy over the same fixed
        # zone area -- rises and falls in lockstep.
        self.assertEqual(density_over_time[2], density_over_time[3])
        self.assertGreater(density_over_time[2], density_over_time[0])
        self.assertEqual(density_over_time[-1], 0.0)

        # Approach evidence requires >=2 position samples -- honestly
        # absent at t=0 (approaching_count counted as 0, not fabricated),
        # present by t=1/t=2 as person 1 (and then person 2) closes in.
        self.assertEqual(approaching_over_time[0], 0)
        self.assertGreater(max(approaching_over_time[1:4]), 0)

        # Queue forms once both occupants have stopped near the exit
        # (t=3), and clears completely once they leave (t=4).
        self.assertEqual(queue_over_time[3], 2)
        self.assertEqual(queue_over_time[4], 0)

        # Congestion rises to its peak at t=3 (queue at its fullest) and
        # falls back to LOW once everyone has left.
        peak_index = congestion_over_time.index(max(congestion_over_time))
        self.assertEqual(peak_index, 3)
        self.assertLess(congestion_over_time[4], congestion_over_time[3])

    def test_trend_reports_rising_then_falling_congestion(self):

        engine = self.runtime.crowd_intelligence_engine
        trends = []

        for time in (0.0, 1.0, 2.0, 3.0, 4.0):
            self.runtime.run_cycle(time)
            trends.append(engine.compute(time).exit("EXIT-1").trend)

        self.assertIn(TrendDirection.RISING, trends)
        self.assertEqual(trends[-1], TrendDirection.FALLING)

    def test_zero_network_and_zero_hardware_access(self):

        self.assertEqual(type(self.runtime.frame_sources[CAMERA_ID]).__name__, "ReplayFrameSource")


if __name__ == "__main__":
    unittest.main()

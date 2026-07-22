import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.replay_frame_source import ReplayFrameSource

from human_detection.yolo_human_detector import YOLOHumanDetector

from tracking.simple_tracker import SimpleSingleCameraTracker

from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.projection import WorldProjector

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from crowd_intelligence.models import AssetApproachMetrics, CrowdIntelligenceSnapshot, IntensityLevel

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from live_runtime.factory import build_live_runtime

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.models import RecommendationStatus, RecommendationWeights

from tests.human_detection_fixtures import FakeYOLOBackend, person


# =====================================================
# Live Dynamic Evacuation Recommendation Engine milestone, Phase 14 --
# deterministic offline end-to-end proof, driven through the COMPLETE
# production chain (ReplayFrameSource -> Fake YOLO -> Tracker -> World
# Projection -> Behavior Recognition -> LiveOccupants -> BuildingState
# -> Crowd Intelligence -> Evacuation Progress -> Trajectory
# Intelligence -> Emergency Response -> Recommendation Engine ->
# Advisory -> Command Center).
#
# Topology (single floor "f1", central hub z2 with THREE exits reachable
# through it): z2 -- DOOR-1 -- z1 (EXIT-1, shortest) ; z2 -- DOOR-3 --
# z5 (EXIT-3, middle) ; z2 -- DOOR-2 -- z4 (EXIT-2, longest).
#
# Scenario: EXIT-1's own zone (z1) becomes hazardous -> recommendation
# migrates to EXIT-3 (next-shortest safe exit). EXIT-3 then congests
# heavily -> recommendation migrates again to EXIT-2, the one exit that
# remains both safe AND uncongested throughout -- proving the engine
# recomputes automatically while an unsafe exit (EXIT-1) is NEVER
# recommended again for as long as it stays hazardous.
# =====================================================


CAMERA_Z2 = "CAM-Z2"


def make_building():

    floor = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="z1", name="Lobby", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z2", name="Hub", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z4", name="Annex", x=40.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
            Zone(id="z5", name="Wing", x=20.0, y=20.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        cameras=[
            Camera(id=CAMERA_Z2, name="Hub Camera", floor_id="f1", zone_ids=("z2",), position=(25.0, 5.0), mount_height=3.0),
        ],
        doors=[
            Door(id="DOOR-1", name="Hub-Lobby Door", floor_id="f1", zone_a_id="z1", zone_b_id="z2"),
            Door(id="DOOR-2", name="Hub-Annex Door", floor_id="f1", zone_a_id="z2", zone_b_id="z4"),
            Door(id="DOOR-3", name="Hub-Wing Door", floor_id="f1", zone_a_id="z2", zone_b_id="z5"),
        ],
        exits=[
            Exit(id="EXIT-1", name="Lobby Exit", floor_id="f1", zone_id="z1"),
            Exit(id="EXIT-2", name="Annex Exit", floor_id="f1", zone_id="z4"),
            Exit(id="EXIT-3", name="Wing Exit", floor_id="f1", zone_id="z5"),
        ],
    )

    return Building(id="recommendation-e2e-building", name="Recommendation E2E Building", floors=[floor])


def make_world_projector():

    intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)

    zone = Zone(name="z2", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f1")
    zone.id = "z2"

    calibrations = {
        CAMERA_Z2: CalibrationProfile(
            camera_id=CAMERA_Z2, floor_id="f1", intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=(25.0, 5.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=90.0),
        ),
    }

    return WorldProjector(calibrations=calibrations, zones_by_floor={"f1": [zone]})


_CENTER_BOX = (310.0, 230.0, 330.0, 250.0)


class RecommendationMigrationEndToEndTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()

        backend = FakeYOLOBackend()
        for _ in range(4):
            backend.queue_result(person(confidence=0.9, box=_CENTER_BOX))

        frame_sources = {
            CAMERA_Z2: ReplayFrameSource(camera_id=CAMERA_Z2, frames=[(0.0, "f0"), (1.0, "f1"), (2.0, "f2"), (3.0, "f3")]),
        }
        for source in frame_sources.values():
            source.start()

        identity_resolver = MappingIdentityResolver({(CAMERA_Z2, f"{CAMERA_Z2}-T1"): "OCC-1"})

        # Built explicitly (rather than left to build_live_runtime()'s
        # own defaults) so the SAME shared live_occupant_manager/graph
        # this test's own EvacuationRecommendationEngine reads is also
        # the one every other live stage populates.
        event_bus = EventBus()
        building_exits = [exit_obj for floor in self.building.ordered_floors() for exit_obj in floor.exits]
        live_occupant_manager = LiveOccupantManager(event_bus=event_bus, exits=building_exits)
        navigation_graph = NavigationGraphGenerator().build(self.building)

        # Weighs congestion heavily enough to matter against a ~35m
        # distance gap between EXIT-2/EXIT-3 (Phase 4's own "Weights
        # must be configurable" requirement, exercised end-to-end here)
        # -- default weights are proven separately by the engine-level
        # unit tests (tests/test_evacuation_recommendation.py).
        weights = RecommendationWeights(route_distance_weight=0.20, congestion_weight=0.50, queue_weight=0.20)
        recommendation_engine = EvacuationRecommendationEngine(
            self.building, navigation_graph, live_occupant_manager, weights=weights,
        )

        self.runtime = build_live_runtime(
            self.building,
            frame_sources=frame_sources,
            human_detector=YOLOHumanDetector(backend),
            identity_resolver=identity_resolver,
            tracker=SimpleSingleCameraTracker(),
            behavior_recognizer=RuleBasedBehaviorRecognizer(),
            world_projector=make_world_projector(),
            event_bus=event_bus,
            live_occupant_manager=live_occupant_manager,
            evacuation_recommendation_engine=recommendation_engine,
        )

        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_recommendation_migrates_through_all_three_exits(self):

        # Cycle 0 -- everything safe and clear: EXIT-1 (shortest) wins.
        self.runtime.run_cycle(0.0)

        baseline = self.runtime.orchestrator.latest_evacuation_recommendation.zone("z2")
        self.assertEqual(baseline.status, RecommendationStatus.RECOMMENDED)
        self.assertEqual(baseline.recommended_exit_id, "EXIT-1")

        # z1 (EXIT-1's own zone) becomes hazardous.
        def hazard_snapshot_provider(time):

            if time < 1.0:
                return HazardSnapshot()

            return HazardSnapshot(node_states={"z1": HazardNodeState(hazard_score=0.9)})

        self.runtime.orchestrator.building_state_gateway._hazard_snapshot_provider = hazard_snapshot_provider

        self.runtime.run_cycle(1.0)

        after_hazard = self.runtime.orchestrator.latest_evacuation_recommendation.zone("z2")
        self.assertNotEqual(after_hazard.recommended_exit_id, "EXIT-1")
        self.assertEqual(after_hazard.recommended_exit_id, "EXIT-3")
        self.assertNotIn("EXIT-1", after_hazard.ranked_exit_ids)

        # EXIT-3 now congests heavily -- computed directly against this
        # cycle's own real BuildingState (crowd_intelligence's own real
        # engine isn't fed real congestion-producing detections in this
        # scenario; injecting a hand-built CrowdIntelligenceSnapshot is
        # the same established precedent tests/test_emergency_response.
        # py's own StalledIncreasesTests already uses for evacuation_
        # progress evidence).
        self.runtime.run_cycle(2.0)

        crowd_snapshot = CrowdIntelligenceSnapshot(exit_metrics={
            "EXIT-3": AssetApproachMetrics(
                asset_id="EXIT-3", asset_type="Exit", position_available=True,
                congestion_level=IntensityLevel.CRITICAL, queue_candidate_count=15,
            ),
        })

        building_state = self.runtime.orchestrator.latest_building_state

        migrated = self.runtime.evacuation_recommendation_engine.compute(
            2.0, building_state, crowd_snapshot=crowd_snapshot,
        ).zone("z2")

        self.assertEqual(migrated.recommended_exit_id, "EXIT-2")
        self.assertNotIn("EXIT-1", migrated.ranked_exit_ids)  # still excluded -- hazard never relaxed mid-scenario

    def test_no_automatic_execution_or_dispatch(self):

        self.runtime.run_cycle(0.0)
        self.runtime.run_cycle(1.0)

        self.assertIsNone(self.runtime.voice_evacuation_controller)
        self.assertIsNone(self.runtime.building_control_controller)
        self.assertIsNone(self.runtime.facp)


if __name__ == "__main__":
    unittest.main()

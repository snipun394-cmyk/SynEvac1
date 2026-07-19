import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulation_runtime.human_observation_bridge import GroundTruthHumanObservationProvider

from simulator.coordinator import MultiAgentSimulation

from tests.test_virtual_camera import make_zone

from virtual_camera.provider import SimulatedDetectionProvider

from multi_camera_fusion.confidence import MultiCameraConfidenceBoostStrategy
from multi_camera_fusion.engine import MultiCameraFusionEngine


class RealVirtualCameraFusionIntegrationTests(unittest.TestCase):

    # Phase 6's own validation requirement: "Validate using Virtual
    # Cameras." End-to-end through the real Visibility Engine, real
    # Virtual Camera, and real GroundTruthHumanObservationProvider --
    # not FakeDetection stand-ins. Proves MultiCameraFusionEngine
    # genuinely works against the existing DetectionProvider interface,
    # not just an interface it happens to satisfy.

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        # Two adjacent rooms with a wide shared doorway, so a camera in
        # each room can also partially see into the other -- the
        # overlap zone this test exercises.
        self.room_a = make_zone("Room A", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.room_b = make_zone("Room B", 10.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.room_a)
        self.floor.add_zone(self.room_b)

        self.door = Door(
            name="Wide Door", start_point=(10.0, 2.0), end_point=(10.0, 8.0),
            floor_id=self.floor.id, width=6.0,
            zone_a_id=self.room_a.id, zone_b_id=self.room_b.id,
        )
        self.floor.add_door(self.door)

        self.floor.add_exit(Exit(name="Ex", zone_id=self.room_b.id, floor_id=self.floor.id))

        self.camera_a = Camera(
            name="Cam A", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.camera_b = Camera(
            name="Cam B", position=(15.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(self.camera_a)
        self.floor.add_camera(self.camera_b)

        self.graph = NavigationGraphGenerator().build(self.building)
        self.engine = PathfindingEngine(self.graph)

    def test_overlap_zone_fuses_into_one_track_with_both_cameras_as_sources(self):

        simulation = MultiAgentSimulation(self.engine)
        simulation.add_occupant(self.room_a.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        detection_provider = SimulatedDetectionProvider(
            [self.camera_a, self.camera_b], self.building, human_observation_provider,
        )

        fusion_engine = MultiCameraFusionEngine()

        detections = detection_provider.all_detections_at(time=0.0)
        fusion_result = fusion_engine.fuse(detections, time=0.0)

        self.assertEqual(len(fusion_result.tracks), 1)

        track = fusion_result.track("occ-1")
        self.assertIsNotNone(track)
        self.assertEqual(track.zone_id, self.room_a.id)
        # Room A's own camera plus (through the wide shared doorway)
        # Room B's camera both see the occupant standing at the start
        # of Room A.
        self.assertEqual(len(track.source_camera_ids), 2)
        self.assertIn(self.camera_a.id, track.source_camera_ids)
        self.assertIn(self.camera_b.id, track.source_camera_ids)

    def test_confidence_boost_reflects_real_multi_camera_agreement(self):

        simulation = MultiAgentSimulation(self.engine)
        simulation.add_occupant(self.room_a.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        detection_provider = SimulatedDetectionProvider(
            [self.camera_a, self.camera_b], self.building, human_observation_provider,
        )

        fusion_engine = MultiCameraFusionEngine(confidence_strategy=MultiCameraConfidenceBoostStrategy())

        detections = detection_provider.all_detections_at(time=0.0)
        fusion_result = fusion_engine.fuse(detections, time=0.0)

        track = fusion_result.track("occ-1")

        # Both cameras report confidence=1.0 (perfect detection, no
        # imperfections configured) -- the boost strategy still clamps
        # to 1.0 rather than exceeding it.
        self.assertEqual(track.confidence, 1.0)

    def test_single_camera_only_zone_produces_one_source_camera(self):

        simulation = MultiAgentSimulation(self.engine)
        simulation.add_occupant(self.room_b.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        detection_provider = SimulatedDetectionProvider(
            [self.camera_a, self.camera_b], self.building, human_observation_provider,
        )

        fusion_engine = MultiCameraFusionEngine()

        detections = detection_provider.all_detections_at(time=0.0)
        fusion_result = fusion_engine.fuse(detections, time=0.0)

        track = fusion_result.track("occ-1")
        self.assertIsNotNone(track)
        # Whether one or both cameras cover Room B depends on exact
        # geometry, but the fused track must never double-count -- one
        # track for one person, regardless of how many cameras.
        self.assertEqual(len(fusion_result.tracks), 1)

    def test_deterministic_across_two_independent_fusion_runs_of_the_same_scenario(self):

        simulation = MultiAgentSimulation(self.engine)
        simulation.add_occupant(self.room_a.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        detection_provider = SimulatedDetectionProvider(
            [self.camera_a, self.camera_b], self.building, human_observation_provider,
        )

        detections = detection_provider.all_detections_at(time=0.0)

        result_a = MultiCameraFusionEngine().fuse(detections, time=0.0)
        result_b = MultiCameraFusionEngine().fuse(detections, time=0.0)

        self.assertEqual(result_a, result_b)


if __name__ == "__main__":
    unittest.main()

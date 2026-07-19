import unittest

from models.building import Building
from models.camera import Camera
from models.engineering_asset import DeviceMode
from models.exit import Exit

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulation_runtime.human_observation_bridge import GroundTruthHumanObservationProvider

from simulator.coordinator import MultiAgentSimulation

from tests.test_virtual_camera import make_zone

from virtual_camera.provider import SimulatedDetectionProvider

from camera_manager.manager import CameraManager


class CameraManagerSimulationAdapterIntegrationTests(unittest.TestCase):

    # End-to-end through the REAL Phase 6 registration path: a real
    # SimulatedDetectionProvider (virtual_camera/provider.py, built on
    # the real Visibility Engine and a real GroundTruthHumanObservation
    # Provider) registers itself with CameraManager for Simulation mode
    # exactly the way this milestone's own Architecture section
    # describes -- "Simulation Adapter... The Camera Manager should not
    # know how detections are generated. It only receives
    # DetectionProviders." CameraManager itself never imports
    # virtual_camera (enforced by
    # tests.test_camera_manager.CameraManagerPackageDependencyDirectionTests)
    # -- this test proves the duck-typed seam still works against the
    # real implementation, not just a hand-built fake.

    def test_simulation_adapter_registers_and_routes_real_detections(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", 0.0, 0.0, 10.0, 10.0, floor_id=floor.id)
        floor.add_zone(lobby)
        floor.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=floor.id))

        camera = Camera(
            name="Lobby Cam", position=(5.0, 5.0), floor_id=floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        floor.add_camera(camera)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        simulation = MultiAgentSimulation(engine)
        simulation.add_occupant(lobby.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        simulation_adapter = SimulatedDetectionProvider(
            [camera], building, human_observation_provider,
        )

        manager = CameraManager()
        manager.discover_cameras(building)

        # Before registration -- Simulation mode is the default, but no
        # adapter has registered itself yet, so this is still an
        # (unpopulated) architecture placeholder.
        self.assertEqual(manager.detections_for_camera(camera.id, time=0.0), ())

        manager.register_detection_provider(DeviceMode.SIMULATION, simulation_adapter)

        detections = manager.detections_for_camera(camera.id, time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].occupant_id, "occ-1")
        self.assertEqual(detections[0].camera_id, camera.id)

        # Grouped views route the same real Detection objects.
        self.assertEqual(manager.detections_by_camera(time=0.0), {camera.id: detections})
        self.assertEqual(manager.detections_by_zone(time=0.0), {lobby.id: detections})
        self.assertEqual(manager.detections_by_floor(time=0.0), {floor.id: detections})

    def test_switching_the_camera_out_of_simulation_mode_stops_the_adapter(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", 0.0, 0.0, 10.0, 10.0, floor_id=floor.id)
        floor.add_zone(lobby)
        floor.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=floor.id))

        camera = Camera(
            name="Lobby Cam", position=(5.0, 5.0), floor_id=floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        floor.add_camera(camera)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        simulation = MultiAgentSimulation(engine)
        simulation.add_occupant(lobby.id, occupant_id="occ-1", depart_time=0.0)
        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        simulation_adapter = SimulatedDetectionProvider(
            [camera], building, human_observation_provider,
        )

        manager = CameraManager()
        manager.discover_cameras(building)
        manager.register_detection_provider(DeviceMode.SIMULATION, simulation_adapter)

        self.assertEqual(len(manager.detections_for_camera(camera.id, time=0.0)), 1)

        # Live has no registered adapter in this milestone -- switching
        # a camera to it must not require touching the Camera Asset or
        # the already-registered Simulation adapter at all.
        manager.set_camera_mode(camera.id, DeviceMode.LIVE)

        self.assertEqual(manager.detections_for_camera(camera.id, time=0.0), ())
        self.assertTrue(manager.has_detection_provider(DeviceMode.SIMULATION))


if __name__ == "__main__":
    unittest.main()

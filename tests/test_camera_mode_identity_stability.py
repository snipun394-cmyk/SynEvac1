import unittest

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider

from virtual_camera.provider import SimulatedDetectionProvider


class CameraModeIdentityStabilityTests(unittest.TestCase):

    # Phase 11: the Camera Asset's Digital Twin identity -- id, floor,
    # zones, position, FOV, orientation -- must survive a mode switch
    # unchanged. Only the registered DetectionProvider changes.
    #
    # Simulation's own SimulatedDetectionProvider requires a real
    # VirtualCamera/Building/HumanObservationProvider to construct, so
    # this test only needs an empty one registered for SIMULATION to
    # prove the routing contract; it never runs a tick through it.

    def setUp(self):

        self.camera = Camera(
            name="Lobby Cam",
            floor_id="floor-1",
            zone_ids=("zone-1", "zone-2"),
            position=(4.0, 7.5),
            rotation=45.0,
            horizontal_fov=90.0,
            max_range=25.0,
        )

        self.manager = CameraManager()
        self.manager.register_camera(self.camera)

    def test_camera_asset_identity_unchanged_across_simulation_to_live_switch(self):

        original_id = self.camera.id
        original_floor_id = self.camera.floor_id
        original_zone_ids = self.camera.zone_ids
        original_position = self.camera.position
        original_rotation = self.camera.rotation
        original_fov = self.camera.horizontal_fov

        self.manager.register_detection_provider(
            DeviceMode.SIMULATION, SimulatedDetectionProvider([], None, {}),
        )

        self.assertEqual(self.manager.camera_mode(self.camera.id), DeviceMode.SIMULATION)

        live_provider = LiveCameraPipelineDetectionProvider()
        self.manager.register_detection_provider(DeviceMode.LIVE, live_provider)
        self.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)

        camera_after = self.manager.get_camera(self.camera.id)

        # Same object, not a replacement -- and every Digital Twin
        # field is untouched by the mode switch.
        self.assertIs(camera_after, self.camera)
        self.assertEqual(camera_after.id, original_id)
        self.assertEqual(camera_after.floor_id, original_floor_id)
        self.assertEqual(camera_after.zone_ids, original_zone_ids)
        self.assertEqual(camera_after.position, original_position)
        self.assertEqual(camera_after.rotation, original_rotation)
        self.assertEqual(camera_after.horizontal_fov, original_fov)

        # Only the provider that answers for this camera has changed:
        # CameraManager needed zero code changes to route Live mode to
        # a brand-new provider type it has never seen before.
        self.assertEqual(self.manager.camera_mode(self.camera.id), DeviceMode.LIVE)
        live_provider.publish(["CAM-X"], [])
        self.assertEqual(
            self.manager.detections_for_camera(self.camera.id, time=0.0), (),
        )


if __name__ == "__main__":
    unittest.main()

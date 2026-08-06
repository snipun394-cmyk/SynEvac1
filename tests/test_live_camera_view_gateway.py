import unittest

from models.camera import Camera

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from command_center.live_camera_view_gateway import LiveCameraViewGateway

from live_camera_pipeline.frame_source import CameraFrame
from live_camera_pipeline.human_detector import RawHumanDetection


def make_camera(name, floor_id="floor-1"):

    return Camera(name=name, floor_id=floor_id)


class _FakePipeline:

    # A test-only stand-in exposing just the two read methods the
    # gateway calls -- never a real LiveCameraPipeline/decoder/detector.

    def __init__(self, frames_by_camera=None, detections_by_camera=None):

        self._frames = frames_by_camera or {}
        self._detections = detections_by_camera or {}

    def latest_frame(self, camera_id):
        return self._frames.get(camera_id)

    def latest_detections(self, camera_id):
        return self._detections.get(camera_id, ())


class LiveCameraViewGatewayTests(unittest.TestCase):

    # Live CCTV Dashboard milestone -- proves CameraTileData derivation
    # is purely a read of already-existing CameraManager/LiveCameraPipeline
    # state (never a new connection, never a second detector call).

    def setUp(self):

        self.camera_manager = CameraManager()

        self.cam_online = make_camera("Camera 1")
        self.cam_not_configured = make_camera("Camera 2")

        self.camera_manager.register_camera(self.cam_online)
        self.camera_manager.register_camera(self.cam_not_configured)

        self.camera_manager.set_connection_status(self.cam_online.id, CameraConnectionState.ONLINE)

        self.frame = CameraFrame(camera_id=self.cam_online.id, timestamp=1.0, frame_sequence=1, payload_ref=[])
        self.detection = RawHumanDetection(
            camera_id=self.cam_online.id, local_track_id="t1", timestamp=1.0, bounding_box=(0.0, 0.0, 10.0, 10.0),
        )

    def _gateway(self, camera_pipeline=None):

        frame_sources = {self.cam_online.id: object()}

        return LiveCameraViewGateway(self.camera_manager, camera_pipeline, frame_sources)

    def test_a_configured_online_camera_carries_its_frame_and_detections(self):

        pipeline = _FakePipeline({self.cam_online.id: self.frame}, {self.cam_online.id: (self.detection,)})
        gateway = self._gateway(camera_pipeline=pipeline)

        tiles = {tile.camera_id: tile for tile in gateway.camera_tiles()}
        online_tile = tiles[self.cam_online.id]

        self.assertTrue(online_tile.configured)
        self.assertEqual(online_tile.connection_status, CameraConnectionState.ONLINE)
        self.assertIs(online_tile.frame, self.frame)
        self.assertEqual(online_tile.detections, (self.detection,))

    def test_a_camera_with_no_active_frame_source_is_reported_not_configured(self):

        gateway = self._gateway(camera_pipeline=None)

        tiles = {tile.camera_id: tile for tile in gateway.camera_tiles()}
        not_configured_tile = tiles[self.cam_not_configured.id]

        self.assertFalse(not_configured_tile.configured)
        self.assertIsNone(not_configured_tile.frame)
        self.assertEqual(not_configured_tile.detections, ())

    def test_every_camera_manager_camera_produces_exactly_one_tile(self):

        gateway = self._gateway(camera_pipeline=None)

        tiles = gateway.camera_tiles()

        self.assertEqual(
            {tile.camera_id for tile in tiles}, {self.cam_online.id, self.cam_not_configured.id},
        )

    def test_configured_but_not_online_camera_still_carries_its_frame(self):

        # The gateway never filters on connection status, only on
        # frame_sources membership -- deciding whether to DISPLAY a
        # frame for a non-Online camera is the tile widget's job, not
        # this gateway's.

        self.camera_manager.set_connection_status(self.cam_online.id, CameraConnectionState.CONFIGURED)

        pipeline = _FakePipeline({self.cam_online.id: self.frame})
        gateway = self._gateway(camera_pipeline=pipeline)

        tiles = {tile.camera_id: tile for tile in gateway.camera_tiles()}
        online_tile = tiles[self.cam_online.id]

        self.assertEqual(online_tile.connection_status, CameraConnectionState.CONFIGURED)
        self.assertIs(online_tile.frame, self.frame)

    def test_no_camera_pipeline_yields_no_frame_or_detections_even_when_configured(self):

        gateway = self._gateway(camera_pipeline=None)

        tiles = {tile.camera_id: tile for tile in gateway.camera_tiles()}
        online_tile = tiles[self.cam_online.id]

        self.assertTrue(online_tile.configured)
        self.assertIsNone(online_tile.frame)
        self.assertEqual(online_tile.detections, ())


if __name__ == "__main__":
    unittest.main()

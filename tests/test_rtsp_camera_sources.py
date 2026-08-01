import unittest

from models.building import Building
from models.floor import Floor
from models.camera import Camera
from models.engineering_asset import ConnectionInfo, DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from live_runtime_launcher.rtsp_camera_sources import build_rtsp_frame_sources
from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


# =====================================================
# CP PLUS NVR -> SynEvac1 Live Runtime Integration milestone -- proves
# the ONE new composition seam this milestone adds: a configured Camera
# Asset becomes a real RTSPFrameSource, one-to-one, only when actually
# configured for Live mode with an rtsp_address. No real NVR, no real
# credentials, no network I/O anywhere in this file -- FakeRTSPBackend
# (tests/live_camera_pipeline_fixtures.py, already used by
# tests/test_rtsp_camera_manager_status_integration.py for exactly this
# kind of proof) stands in for OpenCVFrameDecoderBackend throughout.
# =====================================================


class _FakeCredentialStore:

    def __init__(self, passwords=None):
        self._passwords = dict(passwords or {})

    def get_credential(self, reference_id):
        return self._passwords.get(reference_id)

    def save_credential(self, reference_id, password):
        self._passwords[reference_id] = password

    def delete_credential(self, reference_id):
        self._passwords.pop(reference_id, None)

    def has_credential(self, reference_id):
        return reference_id in self._passwords


def _building_with_camera(camera):

    floor = Floor(id="floor-1", name="Ground Floor", cameras=[camera])
    return Building(id="b1", name="Test Building", floors=[floor])


class BuildRTSPFrameSourcesTests(unittest.TestCase):

    def setUp(self):
        self.camera_manager = CameraManager()
        self.credential_store = _FakeCredentialStore({"CAM-1": "correct-password"})

    def _sources(self, building):
        return build_rtsp_frame_sources(
            building, self.camera_manager, self.credential_store,
            decoder_backend_factory=FakeRTSPBackend,
        )

    def test_no_building_yields_no_sources(self):

        self.assertEqual(self._sources(None), {})

    def test_camera_left_in_simulation_mode_is_never_wired(self):

        camera = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1",
            mode=DeviceMode.SIMULATION,
            connection=ConnectionInfo(rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0"),
        )
        building = _building_with_camera(camera)

        self.assertEqual(self._sources(building), {})

    def test_live_camera_with_no_rtsp_address_is_never_wired(self):

        camera = Camera(id="CAM-1", name="Cam 1", floor_id="floor-1", mode=DeviceMode.LIVE)
        building = _building_with_camera(camera)

        self.assertEqual(self._sources(building), {})

    def test_live_camera_with_rtsp_address_produces_exactly_one_rtsp_frame_source(self):

        camera = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1",
            mode=DeviceMode.LIVE,
            connection=ConnectionInfo(
                rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0",
                username="synevac_svc",
                credential_ref="CAM-1",
            ),
        )
        building = _building_with_camera(camera)

        sources = self._sources(building)

        self.assertEqual(set(sources.keys()), {"CAM-1"})
        source = sources["CAM-1"]

        self.assertIsInstance(source, RTSPFrameSource)
        self.assertEqual(source.endpoint, "rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0")
        self.assertEqual(source.username, "synevac_svc")
        self.assertEqual(source.credential_ref, "CAM-1")

    def test_password_is_resolved_from_credential_store_never_stored_on_the_source(self):

        camera = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1",
            mode=DeviceMode.LIVE,
            connection=ConnectionInfo(
                rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0",
                username="synevac_svc",
                credential_ref="CAM-1",
            ),
        )
        building = _building_with_camera(camera)

        source = self._sources(building)["CAM-1"]

        # RTSPFrameSource never carries a resolved password as a plain
        # attribute -- only credential_ref (a reference id, never a
        # secret). Prove password resolution actually reaches the
        # right value by starting the (fake) connection and inspecting
        # what the backend's open() call actually received.
        source.start()

        backend = source._decoder
        self.assertEqual(len(backend.open_calls), 1)
        endpoint, username, password = backend.open_calls[0]
        self.assertEqual(username, "synevac_svc")
        self.assertEqual(password, "correct-password")

        source.stop()

    def test_wrong_credential_ref_never_crashes_wiring_only_fails_at_connect_time(self):

        camera = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1",
            mode=DeviceMode.LIVE,
            connection=ConnectionInfo(
                rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0",
                username="synevac_svc",
                credential_ref="NO-SUCH-REF",
            ),
        )
        building = _building_with_camera(camera)

        source = self._sources(building)["CAM-1"]
        source.start()  # never raises -- RTSPFrameSource converts this to an honest status

        self.assertEqual(source.status, "Stream Unavailable")
        self.assertIsNotNone(source.last_error)

        source.stop()

    def test_status_callback_bridges_into_camera_manager_connection_status(self):

        camera = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1",
            mode=DeviceMode.LIVE,
            connection=ConnectionInfo(
                rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0",
                username="synevac_svc",
                credential_ref="CAM-1",
            ),
        )
        building = _building_with_camera(camera)
        self.camera_manager.discover_cameras(building)

        source = self._sources(building)["CAM-1"]

        self.assertEqual(
            self.camera_manager.connection_status("CAM-1"), CameraConnectionState.CONFIGURED,
        )

        source.start()

        self.assertEqual(
            self.camera_manager.connection_status("CAM-1"), CameraConnectionState.ONLINE,
        )

        source.stop()

    def test_multiple_cameras_each_get_their_own_independent_source(self):

        cam1 = Camera(
            id="CAM-1", name="Cam 1", floor_id="floor-1", mode=DeviceMode.LIVE,
            connection=ConnectionInfo(rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=1&subtype=0", credential_ref="CAM-1"),
        )
        cam2 = Camera(
            id="CAM-2", name="Cam 2", floor_id="floor-1", mode=DeviceMode.LIVE,
            connection=ConnectionInfo(rtsp_address="rtsp://192.168.1.248:554/cam/realmonitor?channel=2&subtype=0", credential_ref="CAM-2"),
        )
        cam3_sim = Camera(id="CAM-3", name="Cam 3", floor_id="floor-1", mode=DeviceMode.SIMULATION)

        floor = Floor(id="floor-1", name="Ground Floor", cameras=[cam1, cam2, cam3_sim])
        building = Building(id="b1", name="Test Building", floors=[floor])

        sources = self._sources(building)

        self.assertEqual(set(sources.keys()), {"CAM-1", "CAM-2"})
        self.assertIsNot(sources["CAM-1"], sources["CAM-2"])
        self.assertIsNot(sources["CAM-1"]._decoder, sources["CAM-2"]._decoder)


if __name__ == "__main__":
    unittest.main()

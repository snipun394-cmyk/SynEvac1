import unittest

from models.camera import Camera
from models.engineering_asset import DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


# RTSP Frame Source milestone, Phase 8: prove the smallest clean seam
# between RTSPFrameSource's own status and CameraManager's pre-existing
# (but, until now, never actually called -- see docs/architecture/
# cctv_integration_readiness.md Sec 18.1/18.4) connection-status
# registry. This file lives in tests/, importing BOTH camera_manager
# and live_camera_pipeline, deliberately -- the guard that forbids
# live_camera_pipeline/ from importing camera_manager only scans
# production source files (tests/test_live_camera_pipeline.py's
# LiveCameraPipelineDependencyDirectionTests, tests/
# test_rtsp_frame_source.py's own copy of the same scan), never test
# modules. A caller wiring a real deployment together does exactly
# what this file's setUp does: build the bridge as a plain closure at
# the composition root, not inside either package.


def _bridge(manager: CameraManager):

    # The "on_status_changed(camera_id, status, detail)" callback the
    # milestone spec describes, expressed as the one-line adapter a
    # composition root would actually write. `CameraConnectionState
    # (status)` works because RTSPFrameSource's plain status strings
    # were deliberately chosen to match CameraConnectionState's own
    # `.value` strings exactly (see rtsp_frame_source.py's own
    # docstring).

    def on_status_changed(camera_id, status, detail):
        manager.set_connection_status(camera_id, CameraConnectionState(status))

    return on_status_changed


def no_delay(_seconds):
    pass


class RTSPStatusUpdatesCameraManagerTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()
        self.camera = Camera(id="CAM-001", name="Lobby Cam", floor_id="floor-1")
        self.manager.register_camera(self.camera)
        self.manager.set_camera_mode("CAM-001", DeviceMode.LIVE)

        self.backend = FakeRTSPBackend()

    def _make_source(self, **kwargs):

        return RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=self.backend,
            sleep_fn=no_delay,
            status_callback=_bridge(self.manager),
            **kwargs,
        )

    def test_default_status_before_any_activity_is_configured(self):

        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.CONFIGURED,
        )

    def test_successful_connect_reports_online_to_camera_manager(self):

        source = self._make_source()
        source.start()

        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.ONLINE,
        )

    def test_failed_connect_reports_stream_unavailable_to_camera_manager(self):

        self.backend.fail_open_with(ConnectionError("down"))

        source = self._make_source(max_retries=0)
        source.start()

        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.STREAM_UNAVAILABLE,
        )

    def test_stop_reports_configured_to_camera_manager(self):

        source = self._make_source()
        source.start()
        source.stop()

        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.CONFIGURED,
        )

    def test_drop_and_reconnect_transitions_through_degraded_back_to_online(self):

        source = self._make_source()
        source.start()
        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.ONLINE,
        )

        seen_states = []
        original_set = self.manager.set_connection_status

        def spy(camera_id, state):
            seen_states.append(state)
            original_set(camera_id, state)

        self.manager.set_connection_status = spy

        self.backend.fail_read_once_with(ConnectionError("dropped"))
        source.read_frame()

        self.assertIn(CameraConnectionState.DEGRADED, seen_states)
        self.assertEqual(
            self.manager.connection_status("CAM-001"), CameraConnectionState.ONLINE,
        )

    def test_camera_manager_status_survives_camera_manager_never_understanding_rtsp(self):

        # Command Center consumes CameraManager.connection_status()
        # (and, transitively, the Camera Manager Panel's own status
        # text -- designer/widgets/camera_manager_panel.py) without any
        # of them importing or knowing anything about RTSPFrameSource,
        # FrameDecoderBackend, or "reconnect" -- the milestone's own
        # "Command Center must remain transport-agnostic" requirement.
        # Proven structurally: nothing above imports RTSPFrameSource
        # except this test file's own composition code.

        import pathlib
        import re

        repo_root = pathlib.Path(__file__).resolve().parent.parent

        forbidden = r"^\s*(from|import)\s+live_camera_pipeline\b"

        for path in (
            repo_root / "camera_manager" / "manager.py",
            repo_root / "designer" / "widgets" / "camera_manager_panel.py",
        ):

            text = path.read_text(encoding="utf-8")

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path} imports live_camera_pipeline directly -- Command Center/CameraManager "
                f"must stay transport-agnostic, wiring belongs at a composition root instead",
            )


if __name__ == "__main__":
    unittest.main()

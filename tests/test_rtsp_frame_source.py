import unittest

from credential_store.local_file_store import LocalFileCredentialStore

from live_camera_pipeline.rtsp_backend import FrameDecoderError
from live_camera_pipeline.rtsp_frame_source import (
    RTSPFrameSource,
    STATUS_CONFIGURED,
    STATUS_CONNECTING,
    STATUS_ONLINE,
    redact_endpoint,
)

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


def no_delay(_seconds):
    # Injected in place of time.sleep for every test in this file --
    # the RTSP Frame Source milestone's own "tests must not actually
    # sleep for long durations" requirement (Phase 7).
    pass


class NoNetworkIOOnConstructionTests(unittest.TestCase):

    # Phase 2's hard requirement: constructing RTSPFrameSource must
    # never attempt a connection.

    def test_construction_never_calls_backend_open(self):

        backend = FakeRTSPBackend()

        RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=backend,
            username="operator",
            password="hunter2",
            sleep_fn=no_delay,
        )

        self.assertEqual(backend.open_calls, [])
        self.assertFalse(backend.is_open)

    def test_construction_leaves_status_configured_and_not_running(self):

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            sleep_fn=no_delay,
        )

        self.assertEqual(source.status, STATUS_CONFIGURED)
        self.assertFalse(source.is_running)


class LifecycleTests(unittest.TestCase):

    def setUp(self):

        self.backend = FakeRTSPBackend()
        self.source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=self.backend,
            sleep_fn=no_delay,
        )

    def test_start_opens_the_backend_and_reports_online(self):

        self.source.start()

        self.assertTrue(self.source.is_running)
        self.assertEqual(self.source.status, STATUS_ONLINE)
        self.assertEqual(len(self.backend.open_calls), 1)

    def test_repeated_start_while_connected_does_not_reopen(self):

        self.source.start()
        self.source.start()

        self.assertEqual(len(self.backend.open_calls), 1)

    def test_stop_closes_the_backend(self):

        self.source.start()
        self.source.stop()

        self.assertFalse(self.source.is_running)
        self.assertFalse(self.backend.is_open)
        self.assertEqual(self.source.status, STATUS_CONFIGURED)

    def test_repeated_stop_is_safe(self):

        self.source.start()
        self.source.stop()
        self.source.stop()  # must not raise

        self.assertFalse(self.source.is_running)

    def test_stop_before_start_is_safe(self):

        self.source.stop()  # must not raise
        self.assertFalse(self.source.is_running)

    def test_read_frame_before_start_returns_none_without_touching_backend(self):

        result = self.source.read_frame()

        self.assertIsNone(result)
        self.assertEqual(self.backend.open_calls, [])

    def test_read_frame_after_stop_returns_none(self):

        self.source.start()
        self.source.stop()

        self.assertIsNone(self.source.read_frame())


class FramePassthroughTests(unittest.TestCase):

    def setUp(self):

        self.backend = FakeRTSPBackend()
        self.source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=self.backend,
            sleep_fn=no_delay,
            clock_fn=lambda: 42.0,
        )
        self.source.start()

    def test_no_frame_available_returns_none(self):

        self.assertIsNone(self.source.read_frame())

    def test_decoded_frame_becomes_a_camera_frame_with_camera_id(self):

        self.backend.queue_frame(payload_ref={"raw": "bytes"})

        frame = self.source.read_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(frame.camera_id, "CAM-001")
        self.assertEqual(frame.payload_ref, {"raw": "bytes"})
        self.assertEqual(frame.timestamp, 42.0)

    def test_metadata_passes_through_when_present(self):

        self.backend.queue_frame(payload_ref=None, width=1920, height=1080, codec="H264")

        frame = self.source.read_frame()

        self.assertEqual(frame.width, 1920)
        self.assertEqual(frame.height, 1080)
        self.assertEqual(frame.codec, "H264")

    def test_metadata_defaults_to_none_when_backend_does_not_report_it(self):

        self.backend.queue_frame(payload_ref=None)

        frame = self.source.read_frame()

        self.assertIsNone(frame.width)
        self.assertIsNone(frame.height)
        self.assertIsNone(frame.codec)

    def test_frame_sequence_increments_locally_per_source(self):

        self.backend.queue_frame()
        self.backend.queue_frame()

        first = self.source.read_frame()
        second = self.source.read_frame()

        self.assertEqual(first.frame_sequence, 0)
        self.assertEqual(second.frame_sequence, 1)


class RedactEndpointTests(unittest.TestCase):

    def test_redacts_inline_credentials(self):

        redacted = redact_endpoint("rtsp://admin:hunter2@10.0.0.20/stream")

        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("admin", redacted)
        self.assertEqual(redacted, "rtsp://***:***@10.0.0.20/stream")

    def test_endpoint_without_credentials_is_unchanged(self):

        self.assertEqual(
            redact_endpoint("rtsp://10.0.0.20/stream"), "rtsp://10.0.0.20/stream",
        )

    def test_none_and_empty_are_handled(self):

        self.assertEqual(redact_endpoint(None), "")
        self.assertEqual(redact_endpoint(""), "")


class CredentialSafetyTests(unittest.TestCase):

    # Phase 9: a real-looking password/credential must never appear in
    # repr(), exceptions, or status detail, no matter how it entered
    # the source (a directly-supplied password, an inline-in-endpoint
    # credential, or a resolved-from-store secret that leaks into a
    # backend's own exception text).

    def test_repr_never_contains_the_password(self):

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            username="operator",
            password="hunter2",
            sleep_fn=no_delay,
        )

        text = repr(source)

        self.assertNotIn("hunter2", text)
        self.assertIn("<redacted>", text)

    def test_repr_with_no_password_shows_unset(self):

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            sleep_fn=no_delay,
        )

        self.assertIn("<unset>", repr(source))

    def test_repr_redacts_credentials_embedded_in_the_endpoint(self):

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://admin:hunter2@10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            sleep_fn=no_delay,
        )

        text = repr(source)

        self.assertNotIn("hunter2", text)

    def test_open_failure_exception_message_containing_password_is_redacted_in_status(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(ValueError("auth failed for password hunter2"))

        statuses = []

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=backend,
            password="hunter2",
            max_retries=0,
            sleep_fn=no_delay,
            status_callback=lambda cam, status, detail: statuses.append((cam, status, detail)),
        )

        source.start()

        self.assertNotIn("hunter2", source.last_error)
        self.assertTrue(all("hunter2" not in (detail or "") for _, _, detail in statuses))

    def test_resolved_store_password_never_leaks_into_status_detail(self):

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:

            store = LocalFileCredentialStore(path=Path(tmp_dir) / "credentials.json")
            store.save_credential("CAM-001", "super-secret-store-value")

            backend = FakeRTSPBackend()
            backend.fail_open_with(ValueError("failed with super-secret-store-value"))

            source = RTSPFrameSource(
                camera_id="CAM-001",
                endpoint="rtsp://10.0.0.5/stream",
                decoder_backend=backend,
                credential_ref="CAM-001",
                credential_store=store,
                max_retries=0,
                sleep_fn=no_delay,
            )

            source.start()

            self.assertNotIn("super-secret-store-value", source.last_error)

    def test_missing_credential_error_never_contains_a_secret(self):

        backend = FakeRTSPBackend()

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=backend,
            credential_ref="CAM-001",
            credential_store=None,
            max_retries=0,
            sleep_fn=no_delay,
        )

        source.start()

        self.assertIn("CAM-001", source.last_error)
        self.assertIsInstance(source.last_error, str)


class StatusCallbackTests(unittest.TestCase):

    def test_callback_is_invoked_with_camera_id_status_and_detail(self):

        events = []

        source = RTSPFrameSource(
            camera_id="CAM-042",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            sleep_fn=no_delay,
            status_callback=lambda cam, status, detail: events.append((cam, status, detail)),
        )

        source.start()

        self.assertIn(("CAM-042", STATUS_CONNECTING, None), events)
        self.assertIn(("CAM-042", STATUS_ONLINE, None), events)

    def test_a_raising_callback_never_crashes_the_source(self):

        def bad_callback(cam, status, detail):
            raise RuntimeError("boom")

        source = RTSPFrameSource(
            camera_id="CAM-001",
            endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=FakeRTSPBackend(),
            sleep_fn=no_delay,
            status_callback=bad_callback,
        )

        source.start()  # must not raise

        self.assertEqual(source.status, STATUS_ONLINE)


class CameraIdGuaranteeTests(unittest.TestCase):

    # Phase 4: CameraFrame.camera_id must always equal the camera_id
    # RTSPFrameSource was constructed with -- the RTSP endpoint itself
    # (or anything a backend reports) must never be able to influence
    # SynEvac's Digital Twin camera identity.

    def test_frame_camera_id_is_always_the_configured_one_regardless_of_endpoint(self):

        for endpoint in (
            "rtsp://old-camera.local/stream",
            "rtsp://10.0.0.99:554/completely/different/path",
        ):

            backend = FakeRTSPBackend()
            backend.queue_frame(payload_ref={"local_track_id": "1"})

            source = RTSPFrameSource(
                camera_id="CAM-001", endpoint=endpoint, decoder_backend=backend, sleep_fn=no_delay,
            )
            source.start()

            frame = source.read_frame()

            self.assertEqual(frame.camera_id, "CAM-001")

    def test_decoded_frame_metadata_cannot_smuggle_a_different_camera_id(self):

        # DecodedFrame structurally has no camera_id field at all --
        # even a misbehaving/malicious backend has nothing to smuggle
        # identity through. Proven by trying to abuse payload_ref as an
        # attempted vector and confirming it has zero effect on
        # CameraFrame.camera_id.

        backend = FakeRTSPBackend()
        backend.queue_frame(payload_ref={"camera_id": "CAM-999-SPOOFED"})

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://10.0.0.5/stream",
            decoder_backend=backend, sleep_fn=no_delay,
        )
        source.start()

        frame = source.read_frame()

        self.assertEqual(frame.camera_id, "CAM-001")
        self.assertEqual(frame.payload_ref, {"camera_id": "CAM-999-SPOOFED"})


class DependencyDirectionTests(unittest.TestCase):

    # Same regex-scan convention as every other package boundary guard
    # in this codebase (tests/test_live_camera_pipeline.py's own
    # LiveCameraPipelineDependencyDirectionTests already covers every
    # *.py in live_camera_pipeline/, including these two new files, for
    # camera_manager/multi_camera_fusion/building_state -- re-asserted
    # here, scoped to just the RTSP files, for a fast, obvious failure
    # if that boundary is ever crossed).

    def test_rtsp_files_never_import_camera_manager(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_camera_pipeline"

        forbidden = r"^\s*(from|import)\s+(camera_manager|multi_camera_fusion|building_state)\b"

        for name in ("rtsp_frame_source.py", "rtsp_backend.py"):

            text = (package_dir / name).read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"live_camera_pipeline/{name} imports a downstream package directly",
            )


if __name__ == "__main__":
    unittest.main()

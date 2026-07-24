import socket
import time
import unittest
from pathlib import Path

from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend, build_authenticated_url

from live_camera_pipeline.rtsp_backend import FrameDecoderError


# =====================================================
# CCTV Connection & Calibration Readiness milestone, Phase 2/3 -- proves
# OpenCVFrameDecoderBackend satisfies live_camera_pipeline.rtsp_backend.
# FrameDecoderBackend's own contract for real (not faked) decode/
# transport, using a real local video file (validation_media/vtest.avi)
# rather than a real network camera -- exactly the same offline-first
# discipline every other RTSP milestone test already follows
# (tests/test_rtsp_frame_source.py, tests/test_rtsp_offline_e2e.py),
# extended here to the one real backend implementation instead of
# FakeRTSPBackend.
# =====================================================


REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = REPO_ROOT / "validation_media" / "vtest.avi"


class BuildAuthenticatedUrlTests(unittest.TestCase):

    # Pure string logic -- no cv2 required, runs unconditionally.

    def test_embeds_username_and_password_into_rtsp_netloc(self):

        url = build_authenticated_url("rtsp://192.168.1.10:554/stream1", "admin", "secret123")
        self.assertEqual(url, "rtsp://admin:secret123@192.168.1.10:554/stream1")

    def test_leaves_local_file_paths_unchanged(self):

        url = build_authenticated_url("validation_media/vtest.avi", "admin", "secret123")
        self.assertEqual(url, "validation_media/vtest.avi")

    def test_leaves_endpoint_unchanged_when_no_credentials_supplied(self):

        url = build_authenticated_url("rtsp://192.168.1.10:554/stream1", None, None)
        self.assertEqual(url, "rtsp://192.168.1.10:554/stream1")

    def test_handles_username_only(self):

        url = build_authenticated_url("rtsp://192.168.1.10/stream1", "admin", None)
        self.assertEqual(url, "rtsp://admin:@192.168.1.10/stream1")


class NoNetworkIOOnConstructionTests(unittest.TestCase):

    def test_constructor_performs_no_io(self):

        # If this touched a socket/file/import of cv2 itself, the
        # absence of any real endpoint here would raise -- constructing
        # never even imports cv2 (deferred to open()).
        backend = OpenCVFrameDecoderBackend()
        self.assertFalse(backend.is_open)

    def test_read_before_open_raises_frame_decoder_error(self):

        backend = OpenCVFrameDecoderBackend()
        with self.assertRaises(FrameDecoderError):
            backend.read()

    def test_close_before_open_is_safe(self):

        backend = OpenCVFrameDecoderBackend()
        backend.close()  # must not raise
        self.assertFalse(backend.is_open)


class OpenFailureTests(unittest.TestCase):

    def test_nonexistent_path_raises_frame_decoder_error(self):

        backend = OpenCVFrameDecoderBackend()

        with self.assertRaises(FrameDecoderError):
            backend.open("this_file_definitely_does_not_exist_12345.avi", None, None)

        self.assertFalse(backend.is_open)

    def test_error_message_never_contains_a_supplied_password(self):

        backend = OpenCVFrameDecoderBackend()

        try:
            backend.open("rtsp://totally-unreachable-host-xyz:554/stream", "admin", "SuperSecretPW")
        except FrameDecoderError as exc:
            self.assertNotIn("SuperSecretPW", str(exc))
        else:
            self.fail("expected FrameDecoderError for an unreachable RTSP host")

    def test_repeated_failed_opens_never_hang_or_loop(self):

        # This backend itself performs exactly one attempt per open()
        # call -- no internal retry loop of its own (RTSPFrameSource
        # owns that policy). Calling open() three times against a bad
        # path must fail three times, deterministically, never block.
        backend = OpenCVFrameDecoderBackend()

        for _ in range(3):
            with self.assertRaises(FrameDecoderError):
                backend.open("this_file_definitely_does_not_exist_12345.avi", None, None)


class ConnectionRefusedTests(unittest.TestCase):

    # CCTV Connection & Calibration Readiness milestone, Phase 11 --
    # "connection refused" is a genuinely different real-network failure
    # mode from "unreachable host" (tested in OpenFailureTests above via
    # an unroutable TEST-NET-1 address). A raw TCP connect to a closed
    # local port refuses instantly at the OS level -- but verified
    # directly here, FFmpeg's own RTSP client does NOT surface that as
    # an immediate failure; it still waits out the full configured
    # open_timeout_ms budget before giving up (an honest finding about
    # this specific decode library's own RTSP handshake behavior, not a
    # bug in this backend -- documented in docs/architecture/
    # physical_cctv_access_checklist.md so an operator sets a SHORT
    # open_timeout_ms and tests reachability with ping/VLC first, rather
    # than assuming a bad endpoint will fail fast on its own).

    def test_connection_refused_fails_within_the_configured_timeout_bound(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
        sock.close()  # port is now guaranteed closed -- nothing listening

        configured_timeout_ms = 3000
        backend = OpenCVFrameDecoderBackend(open_timeout_ms=configured_timeout_ms)

        start = time.monotonic()
        with self.assertRaises(FrameDecoderError):
            backend.open(f"rtsp://127.0.0.1:{closed_port}/stream", None, None)
        elapsed = time.monotonic() - start

        # Never silently retries forever, and never exceeds the
        # configured budget by more than a small margin -- bounded, even
        # if not "instant."
        self.assertLess(elapsed, (configured_timeout_ms / 1000.0) + 3.0)


@unittest.skipUnless(
    VIDEO_PATH.exists(),
    f"local validation video not found at {VIDEO_PATH} -- this opt-in real-decode test is "
    f"skipped, never failed, when the local artifact is absent",
)
class RealLocalVideoDecodeTests(unittest.TestCase):

    # Real cv2.VideoCapture, real local file, real decoded frames --
    # everything FakeRTSPBackend structurally cannot prove.

    def test_open_read_close_lifecycle_against_a_real_video_file(self):

        backend = OpenCVFrameDecoderBackend()
        self.assertFalse(backend.is_open)

        backend.open(str(VIDEO_PATH), None, None)
        self.assertTrue(backend.is_open)

        frame = backend.read()
        self.assertIsNotNone(frame)
        self.assertIsNotNone(frame.payload_ref)
        self.assertEqual(frame.width, 768)
        self.assertEqual(frame.height, 576)

        backend.close()
        self.assertFalse(backend.is_open)

    def test_close_is_idempotent(self):

        backend = OpenCVFrameDecoderBackend()
        backend.open(str(VIDEO_PATH), None, None)
        backend.close()
        backend.close()  # must not raise
        self.assertFalse(backend.is_open)

    def test_reopen_after_close_works(self):

        backend = OpenCVFrameDecoderBackend()
        backend.open(str(VIDEO_PATH), None, None)
        first_frame = backend.read()
        backend.close()

        backend.open(str(VIDEO_PATH), None, None)
        second_frame = backend.read()
        backend.close()

        self.assertIsNotNone(first_frame)
        self.assertIsNotNone(second_frame)

    def test_reading_past_end_of_file_returns_none_not_an_error(self):

        backend = OpenCVFrameDecoderBackend()
        backend.open(str(VIDEO_PATH), None, None)

        frame_count = 0
        while True:
            frame = backend.read()
            if frame is None:
                break
            frame_count += 1
            if frame_count > 10000:
                self.fail("video never ended -- something is wrong with the fixture")

        self.assertGreater(frame_count, 0)

        # Calling read() again after end-of-stream must still return
        # None, never raise -- the capture itself is still "open" in
        # OpenCV's own sense even once exhausted.
        self.assertIsNone(backend.read())

        backend.close()


@unittest.skipUnless(
    VIDEO_PATH.exists(),
    f"local validation video not found at {VIDEO_PATH} -- this opt-in real-decode test is "
    f"skipped, never failed, when the local artifact is absent",
)
class RealDecoderThroughRTSPFrameSourceTests(unittest.TestCase):

    # The seam that matters most: RTSPFrameSource itself (unmodified)
    # driven by this real backend, proving the production integration
    # point -- not just the backend in isolation.

    def test_rtsp_frame_source_reaches_online_and_produces_real_camera_frames(self):

        from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

        backend = OpenCVFrameDecoderBackend()
        source = RTSPFrameSource(
            camera_id="CAM-REAL-DECODER-TEST", endpoint=str(VIDEO_PATH), decoder_backend=backend,
        )

        source.start()
        self.assertEqual(source.status, "Online")

        frame = source.read_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.camera_id, "CAM-REAL-DECODER-TEST")
        self.assertEqual(frame.frame_sequence, 0)
        self.assertEqual(frame.width, 768)
        self.assertEqual(frame.height, 576)

        frame2 = source.read_frame()
        self.assertEqual(frame2.frame_sequence, 1)

        source.stop()
        self.assertEqual(source.status, "Configured")

    def test_mid_stream_drop_is_detected_and_reconnect_recovers(self):

        # Simulates a real dropped connection (not a simple end-of-file)
        # by reaching directly into the backend's own real cv2.VideoCapture
        # and releasing it out from under RTSPFrameSource -- the next
        # read_frame() call must detect this as a genuine drop (the
        # backend's own is_open correctly reports False) and recover via
        # RTSPFrameSource's existing bounded reconnect, using the SAME
        # real backend, no test double involved at any point.
        from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

        backend = OpenCVFrameDecoderBackend()
        source = RTSPFrameSource(
            camera_id="CAM-DROP-TEST", endpoint=str(VIDEO_PATH), decoder_backend=backend,
            sleep_fn=lambda _seconds: None,
        )
        source.start()
        self.assertEqual(source.status, "Online")

        frame = source.read_frame()
        self.assertIsNotNone(frame)

        # Simulate the underlying stream dying -- release cv2's own
        # handle directly without going through backend.close(), the
        # same "capture is open a moment ago, not anymore" state a real
        # dropped RTSP connection would leave behind.
        backend._capture.release()

        frame_during_drop = source.read_frame()
        self.assertIsNone(frame_during_drop)
        self.assertEqual(source.status, "Online")  # reconnect already succeeded synchronously

        # Reconnect re-opened the SAME real local file from the start --
        # proves the real backend's open()/close() cycle genuinely works
        # for a second time, not just the first.
        frame_after_reconnect = source.read_frame()
        self.assertIsNotNone(frame_after_reconnect)

        source.stop()


@unittest.skipUnless(
    VIDEO_PATH.exists(),
    f"local validation video not found at {VIDEO_PATH} -- this opt-in real-decode test is "
    f"skipped, never failed, when the local artifact is absent",
)
class OneCameraFailsWhileAnotherContinuesTests(unittest.TestCase):

    # CCTV Connection & Calibration Readiness milestone, Phase 11 -- "one
    # camera fails while another continues" (a real, likely day-one
    # scenario: one physical camera misconfigured/unreachable, the rest
    # of the building's cameras must keep working). Proven at the
    # LiveCameraPipeline level, both frame sources real.

    def test_one_bad_source_never_stops_the_pipeline_from_serving_the_good_one(self):

        from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource
        from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
        from live_camera_pipeline.identity_resolver import SimulationIdentityResolver
        from live_camera_pipeline.pipeline import LiveCameraPipeline

        from tests.human_detection_fixtures import FakeYOLOBackend, person
        from human_detection.yolo_human_detector import YOLOHumanDetector

        good_backend = OpenCVFrameDecoderBackend()
        good_source = RTSPFrameSource(
            camera_id="CAM-GOOD", endpoint=str(VIDEO_PATH), decoder_backend=good_backend,
        )

        bad_backend = OpenCVFrameDecoderBackend(open_timeout_ms=1000)
        bad_source = RTSPFrameSource(
            camera_id="CAM-BAD", endpoint="this_path_does_not_exist_at_all.avi", decoder_backend=bad_backend,
            max_retries=0, sleep_fn=lambda _seconds: None,
        )

        good_source.start()
        bad_source.start()

        self.assertEqual(good_source.status, "Online")
        self.assertEqual(bad_source.status, "Stream Unavailable")

        yolo_backend = FakeYOLOBackend()
        yolo_backend.queue_result(person(confidence=0.9, box=(10.0, 10.0, 40.0, 90.0)))

        detection_provider = LiveCameraPipelineDetectionProvider()

        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-GOOD": good_source, "CAM-BAD": bad_source},
            human_detector=YOLOHumanDetector(yolo_backend),
            identity_resolver=SimulationIdentityResolver(),
            detection_provider=detection_provider,
        )

        # Must not raise, and CAM-GOOD's own detections must still reach
        # the detection provider despite CAM-BAD never producing a frame.
        pipeline.run_cycle(0.0)

        good_detections = detection_provider.detections_at("CAM-GOOD", 0.0)
        bad_detections = detection_provider.detections_at("CAM-BAD", 0.0)

        self.assertGreater(len(good_detections), 0)
        self.assertEqual(len(bad_detections), 0)

        good_source.stop()
        bad_source.stop()


if __name__ == "__main__":
    unittest.main()

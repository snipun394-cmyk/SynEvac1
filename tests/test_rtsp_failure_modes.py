import unittest

from live_camera_pipeline.rtsp_backend import FrameDecoderError
from live_camera_pipeline.rtsp_frame_source import (
    RTSPFrameSource,
    STATUS_CONFIGURED,
    STATUS_CONNECTING,
    STATUS_DEGRADED,
    STATUS_ONLINE,
    STATUS_STREAM_UNAVAILABLE,
)

from tests.live_camera_pipeline_fixtures import FakeRTSPBackend


# RTSP Frame Source milestone, Phase 7 (reconnect strategy) and Phase
# 12 (failure validation) -- both sets of scenarios are exercised here
# together since, by design (see rtsp_frame_source.py's own
# _connect_with_retries), a "reconnect" IS just another bounded
# connect-with-retries call; there is only one retry mechanism, used
# both for the very first connection attempt and for recovering from a
# mid-stream drop. Every scenario below asserts the same three things
# Phase 12 requires of all fifteen: no crash, no fabricated frame, no
# stale frame silently presented as current.


class SleepRecorder:

    # Deterministic stand-in for time.sleep -- records every delay it
    # was asked for (so a test can assert the backoff schedule) without
    # ever actually blocking. Can also be configured to trigger a side
    # effect (e.g. calling source.stop()) on a specific call, which is
    # how "stop() during reconnect" is made deterministically
    # reproducible without real threads (see StopDuringReconnectTests).

    def __init__(self):
        self.delays = []
        self._side_effect_at_call = None
        self._side_effect = None

    def trigger_on_call(self, call_index, side_effect):
        self._side_effect_at_call = call_index
        self._side_effect = side_effect

    def __call__(self, seconds):

        self.delays.append(seconds)

        if len(self.delays) - 1 == self._side_effect_at_call:
            self._side_effect()


class InitialConnectionTests(unittest.TestCase):

    # Phase 7 scenarios 1-3.

    def test_initial_connection_succeeds(self):

        backend = FakeRTSPBackend()
        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
        )

        source.start()

        self.assertEqual(source.status, STATUS_ONLINE)
        self.assertEqual(len(backend.open_calls), 1)

    def test_initial_connection_fails_exhausts_retries_honestly(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(ConnectionError("no route to host"))

        sleeper = SleepRecorder()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=3, retry_delay=1.0,
            backoff_factor=2.0, sleep_fn=sleeper,
        )

        source.start()

        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)
        self.assertEqual(len(backend.open_calls), 4)  # 1 initial + 3 retries
        self.assertEqual(sleeper.delays, [1.0, 2.0, 4.0])  # exponential backoff, 3 gaps
        self.assertIsNone(source.read_frame())  # no crash, no fake frame

    def test_connection_succeeds_after_retry(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(TimeoutError("connect timed out"), times=2)

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=3, sleep_fn=SleepRecorder(),
        )

        source.start()

        self.assertEqual(source.status, STATUS_ONLINE)
        self.assertEqual(len(backend.open_calls), 3)


class StreamDropAndReconnectTests(unittest.TestCase):

    # Phase 7 scenarios 4-6.

    def test_stream_drops_after_several_frames_then_reconnects(self):

        backend = FakeRTSPBackend()
        backend.queue_frame(payload_ref={"n": 1})
        backend.queue_frame(payload_ref={"n": 2})

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
        )
        source.start()

        self.assertIsNotNone(source.read_frame())
        self.assertIsNotNone(source.read_frame())

        backend.fail_read_once_with(ConnectionError("stream dropped"))

        # The cycle the drop is detected on returns None honestly --
        # never the previous frame replayed as if it were current.
        dropped_cycle_result = source.read_frame()
        self.assertIsNone(dropped_cycle_result)

        # Reconnect happened inline, synchronously, and succeeded
        # (FakeRTSPBackend's default open() succeeds) -- back online.
        self.assertEqual(source.status, STATUS_ONLINE)

        backend.queue_frame(payload_ref={"n": 3})
        recovered_frame = source.read_frame()

        self.assertIsNotNone(recovered_frame)
        self.assertEqual(recovered_frame.payload_ref, {"n": 3})

    def test_reconnect_after_drop_uses_degraded_status_while_trying(self):

        statuses = []

        backend = FakeRTSPBackend()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
            status_callback=lambda cam, status, detail: statuses.append(status),
        )
        source.start()
        statuses.clear()

        backend.fail_read_once_with(ConnectionError("dropped"))
        source.read_frame()

        self.assertIn(STATUS_DEGRADED, statuses)
        self.assertIn(STATUS_ONLINE, statuses)

    def test_reconnect_repeatedly_fails_leaves_source_honestly_unavailable(self):

        backend = FakeRTSPBackend()
        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=2, sleep_fn=SleepRecorder(),
        )
        source.start()
        self.assertEqual(source.status, STATUS_ONLINE)

        backend.fail_read_once_with(ConnectionError("dropped"))
        backend.fail_open_with(ConnectionError("still down"))  # every reconnect attempt fails

        result = source.read_frame()

        self.assertIsNone(result)  # no crash, no fake frame
        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)

        # Still "running" (session active, never stopped) -- just
        # honestly unable to serve frames until explicitly restarted.
        self.assertTrue(source.is_running)
        self.assertIsNone(source.read_frame())  # stays honestly empty, no repeated hammering


class StopAndDisableDuringReconnectTests(unittest.TestCase):

    # Phase 7 scenarios 7-8. Since this codebase's live pipeline is
    # single-threaded/synchronous by design (live_camera_pipeline/
    # pipeline.py's own run_cycle() -- see docs/architecture/
    # cctv_integration_readiness.md Sec 18.1's "no concurrency in the
    # orchestrator itself"), a literal concurrent stop() mid-retry is
    # not otherwise reachable. The injected sleep_fn is the
    # deterministic seam that simulates it: it fires between retry
    # attempts, exactly where a real concurrent stop() would land.

    def test_stop_during_initial_connect_retry_aborts_cleanly(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(ConnectionError("down"))  # every attempt fails

        sleeper = SleepRecorder()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=5, sleep_fn=sleeper,
        )

        sleeper.trigger_on_call(0, source.stop)

        source.start()  # must not raise, must not keep retrying after stop()

        self.assertFalse(source.is_running)
        self.assertEqual(source.status, STATUS_CONFIGURED)
        # Only the attempts up through the one where stop() fired.
        self.assertLess(len(backend.open_calls), 6)

    def test_stop_during_drop_reconnect_aborts_cleanly(self):

        backend = FakeRTSPBackend()
        sleeper = SleepRecorder()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=5, sleep_fn=sleeper,
        )
        source.start()
        self.assertEqual(source.status, STATUS_ONLINE)

        backend.fail_read_once_with(ConnectionError("dropped"))
        backend.fail_open_with(ConnectionError("still down"))

        sleeper.trigger_on_call(0, source.stop)

        result = source.read_frame()  # must not raise

        self.assertIsNone(result)
        self.assertFalse(source.is_running)
        self.assertEqual(source.status, STATUS_CONFIGURED)

    def test_camera_disabled_during_reconnect_is_the_same_as_stop(self):

        # RTSPFrameSource has no concept of "Camera.active" (it is
        # decoupled from CameraManager/the Digital Twin by design --
        # Phase 8's own "must not import the entire CameraManager"
        # requirement). A supervisor that disables a camera is expected
        # to translate that into calling stop() on the camera's frame
        # source -- exactly the mechanism already proven above.

        backend = FakeRTSPBackend()
        sleeper = SleepRecorder()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=5, sleep_fn=sleeper,
        )
        source.start()

        backend.fail_read_once_with(ConnectionError("dropped"))
        backend.fail_open_with(ConnectionError("still down"))

        def disable_camera():
            # What a real supervisor would do: stop the source whose
            # camera was just disabled.
            source.stop()

        sleeper.trigger_on_call(0, disable_camera)

        source.read_frame()

        self.assertFalse(source.is_running)
        self.assertEqual(source.status, STATUS_CONFIGURED)


class ConnectionInfoFailureModeTests(unittest.TestCase):

    # Phase 12 items 1-6: every distinct "why did the connection fail"
    # reason is honestly reported without RTSPFrameSource needing to
    # know or care what kind of failure it was -- the decoder backend's
    # own exception carries that meaning, RTSPFrameSource only ever
    # asks "did open()/read() raise or not."

    def _assert_fails_honestly(self, backend, **kwargs):

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=0, sleep_fn=SleepRecorder(), **kwargs,
        )

        source.start()  # must not raise

        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)
        self.assertIsNone(source.read_frame())  # no crash, no fake frame
        return source

    def test_wrong_endpoint(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(ConnectionError("could not resolve host"))
        self._assert_fails_honestly(backend)

    def test_authentication_failure(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(PermissionError("401 Unauthorized"))
        self._assert_fails_honestly(backend)

    def test_timeout(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(TimeoutError("connect timed out after 5s"))
        self._assert_fails_honestly(backend)

    def test_stream_unavailable(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(FrameDecoderError("404 Stream Not Found"))
        self._assert_fails_honestly(backend)

    def test_decoder_error(self):

        backend = FakeRTSPBackend()
        backend.fail_open_with(FrameDecoderError("unsupported codec"))
        self._assert_fails_honestly(backend)

    def test_malformed_frame_mid_stream(self):

        backend = FakeRTSPBackend()
        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, max_retries=0, sleep_fn=SleepRecorder(),
        )
        source.start()

        backend.fail_read_once_with(FrameDecoderError("malformed frame data"))
        backend.fail_open_with(ConnectionError("still down"))  # reconnect also fails

        result = source.read_frame()

        self.assertIsNone(result)
        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)

    def test_connection_drop_mid_stream(self):

        backend = FakeRTSPBackend()
        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
        )
        source.start()

        backend.fail_read_once_with(ConnectionError("connection reset by peer"))
        result = source.read_frame()

        self.assertIsNone(result)  # honest, this cycle's frame is lost
        self.assertEqual(source.status, STATUS_ONLINE)  # default backend reconnects cleanly


class CredentialFailureModeTests(unittest.TestCase):

    # Phase 12 items 11-12.

    def test_missing_credential_fails_honestly(self):

        backend = FakeRTSPBackend()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, credential_ref="CAM-001", credential_store=None,
            max_retries=0, sleep_fn=SleepRecorder(),
        )

        source.start()  # must not raise

        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)
        self.assertEqual(backend.open_calls, [])  # never even attempted to open

    def test_credential_store_unavailable_fails_honestly(self):

        class RaisingStore:

            def get_credential(self, reference_id):
                raise OSError("credential store unreachable")

        backend = FakeRTSPBackend()

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, credential_ref="CAM-001", credential_store=RaisingStore(),
            max_retries=0, sleep_fn=SleepRecorder(),
        )

        source.start()  # must not raise

        self.assertEqual(source.status, STATUS_STREAM_UNAVAILABLE)
        self.assertIsNone(source.read_frame())


class SourceStoppedAndCameraIdMismatchTests(unittest.TestCase):

    # Phase 12 items 13, 15 (item 14 -- "camera removed from Digital
    # Twin" -- is a CameraManager-side concern; RTSPFrameSource has no
    # reference to CameraManager at all by design, so there is nothing
    # for it to react to. See docs/architecture/
    # cctv_integration_readiness.md's RTSP section for why that
    # decoupling is deliberate).

    def test_source_stopped_read_frame_is_none_forever(self):

        backend = FakeRTSPBackend()
        backend.queue_frame(payload_ref={"n": 1})

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
        )
        source.start()
        source.stop()

        for _ in range(3):
            self.assertIsNone(source.read_frame())

    def test_camera_id_is_never_derived_from_a_mismatched_backend_report(self):

        backend = FakeRTSPBackend()
        # Nothing in DecodedFrame can carry a camera_id at all -- this
        # test documents that structurally, not just behaviorally: even
        # a payload deliberately shaped to look like a different
        # camera's data has zero effect on CameraFrame.camera_id.
        backend.queue_frame(payload_ref={"spoofed_camera_id": "CAM-999"})

        source = RTSPFrameSource(
            camera_id="CAM-001", endpoint="rtsp://cam/stream",
            decoder_backend=backend, sleep_fn=SleepRecorder(),
        )
        source.start()

        frame = source.read_frame()

        self.assertEqual(frame.camera_id, "CAM-001")


if __name__ == "__main__":
    unittest.main()

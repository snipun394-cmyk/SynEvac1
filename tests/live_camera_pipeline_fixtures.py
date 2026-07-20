from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.human_detector import HumanDetector, RawHumanDetection
from live_camera_pipeline.rtsp_backend import DecodedFrame, FrameDecoderBackend


class MockHumanDetector(HumanDetector):

    # A deterministic stand-in for a real vision model (YOLO + a local
    # tracker) -- CCTV Pipeline End-to-End Offline Validation milestone's
    # own "do not implement YOLO, use deterministic mocks" requirement.
    # A CameraFrame's payload_ref is a plain, test-controlled sequence
    # of dicts (no image processing of any kind); each entry becomes one
    # RawHumanDetection, still namespaced by frame.camera_id exactly the
    # way a real per-camera local tracker would produce it. Reuses
    # HumanClassification/HumanState as-is -- no new vocabulary.

    def detect(self, frame):

        if frame.payload_ref is None:
            return ()

        return tuple(
            RawHumanDetection(
                camera_id=frame.camera_id,
                local_track_id=entry["local_track_id"],
                timestamp=frame.timestamp,
                confidence=entry.get("confidence", 0.9),
                classification_evidence=entry.get(
                    "classification_evidence", HumanClassification.ADULT,
                ),
                state_evidence=entry.get("state_evidence", HumanState.WALKING),
                floor_id=entry.get("floor_id", "floor-1"),
                zone_id=entry.get("zone_id", "zone-1"),
            )
            for entry in frame.payload_ref
        )


class FakeRTSPBackend(FrameDecoderBackend):

    # RTSP Frame Source milestone: a fully deterministic, offline stand-
    # in for a real RTSP decoder (ffmpeg/OpenCV/gstreamer or whatever a
    # real implementation eventually wraps) -- lives in tests/, not
    # live_camera_pipeline/, same "production package stays pure
    # interface, test doubles live in the test module/shared fixtures"
    # convention as MockHumanDetector above and FakeFrameSource/
    # FakeHumanDetector in tests/test_live_camera_pipeline.py. Nothing
    # here touches a socket, a file, or any real media library -- a
    # test configures exactly what open()/read() do via plain method
    # calls, deterministically, with zero timing dependency of its own.

    def __init__(self):

        self._is_open = False
        self.open_calls = []

        self._open_exception = None
        self._open_fail_remaining = None

        self._read_queue = []
        self._read_exception = None

    # --- test configuration ------------------------------------------

    def fail_open_with(self, exception: Exception, times: "int | None" = None) -> None:

        # times=None fails every subsequent open() call. times=N fails
        # the next N calls, then the (N+1)th succeeds -- "connection
        # succeeds after retry".
        self._open_exception = exception
        self._open_fail_remaining = times

    def stop_failing_open(self) -> None:

        self._open_exception = None
        self._open_fail_remaining = None

    def queue_frame(self, payload_ref=None, width=None, height=None, codec=None) -> None:

        self._read_queue.append(DecodedFrame(payload_ref, width, height, codec))

    def fail_read_once_with(self, exception: Exception) -> None:

        # The *next* read() call raises this exception exactly once
        # (simulating a mid-stream drop/decoder error/malformed frame),
        # then read() resumes serving whatever is queued.
        self._read_exception = exception

    # --- FrameDecoderBackend interface --------------------------------

    def open(self, endpoint, username, password) -> None:

        self.open_calls.append((endpoint, username, password))

        if self._open_exception is not None:

            if self._open_fail_remaining is None:
                raise self._open_exception

            if self._open_fail_remaining > 0:
                self._open_fail_remaining -= 1
                raise self._open_exception

        self._is_open = True

    def read(self):

        if self._read_exception is not None:

            exc = self._read_exception
            self._read_exception = None
            raise exc

        if self._read_queue:
            return self._read_queue.pop(0)

        return None

    def close(self) -> None:

        self._is_open = False

    @property
    def is_open(self) -> bool:

        return self._is_open

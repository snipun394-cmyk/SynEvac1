from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DecodedFrame:

    # What one FrameDecoderBackend.read() call honestly hands back --
    # deliberately the same "opaque payload plus optional metadata"
    # shape CameraFrame itself uses (live_camera_pipeline/frame_source.
    # py), so RTSPFrameSource.read_frame() is a pure, no-interpretation
    # copy from one to the other. payload_ref stays untyped (Any) for
    # exactly the reason CameraFrame.payload_ref does -- nothing in this
    # package may ever inspect it as an image (tests/
    # test_no_cv_dependencies.py enforces no cv2/torch/ultralytics/onvif
    # import anywhere in live_camera_pipeline/). width/height/codec are
    # honestly-optional -- a real backend that cannot report one leaves
    # it None, never a guess.

    payload_ref: Optional[Any] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None


class FrameDecoderError(Exception):

    # The one error type this seam's own contract establishes -- a
    # backend is free to raise any exception from open()/read() (a real
    # implementation will naturally raise whatever its underlying
    # decode/transport library raises), but RTSPFrameSource never
    # branches on a specific backend exception type; it catches
    # Exception broadly and converts every failure into honest
    # status/detail information (RTSP Frame Source milestone, Phase 6).
    # FrameDecoderError exists only for RTSPFrameSource's own internal
    # failures that are not the backend's fault (e.g. an unresolved
    # credential, Phase 9) and for fakes/tests that want a clearly-named
    # exception rather than a bare ValueError.

    pass


class FrameDecoderBackend(ABC):

    # The transport/decode seam RTSPFrameSource (live_camera_pipeline/
    # rtsp_frame_source.py) delegates to -- exists solely so
    # RTSPFrameSource itself can be fully unit-tested without a real
    # RTSP server, real network I/O, or a real decode library (no
    # cv2/ffmpeg-python/gstreamer import anywhere in this package, same
    # guard as every other live_camera_pipeline module). A future real
    # implementation (once physical CCTV access exists) wraps whatever
    # concrete decode/transport library is chosen behind exactly this
    # interface; nothing in RTSPFrameSource, CameraManager,
    # MultiCameraFusionEngine, or BuildingState needs to change when
    # that implementation is written.
    #
    # Deliberately minimal -- this is not a generic media framework, it
    # is the smallest seam that makes RTSPFrameSource's own connect/
    # read/reconnect/close logic testable in isolation from "how do you
    # actually open an RTSP stream."

    @abstractmethod
    def open(self, endpoint: str, username: Optional[str], password: Optional[str]) -> None:

        # Must raise on failure (wrong endpoint, auth failure, timeout,
        # stream unavailable, ...) rather than returning falsy/None --
        # RTSPFrameSource's connect/reconnect loop distinguishes success
        # from failure only by whether this raises.
        ...

    @abstractmethod
    def read(self) -> Optional[DecodedFrame]:

        # None means "no frame currently available" (honest, not an
        # error -- RTSPFrameSource must never fabricate a frame for
        # this). A raised exception means the stream itself failed
        # (dropped connection, decoder error, malformed frame) --
        # RTSPFrameSource treats this as a drop and begins its bounded
        # reconnect strategy.
        ...

    @abstractmethod
    def close(self) -> None:

        # Must be safe to call even when not open (RTSPFrameSource.stop()
        # may call this defensively) and must never raise.
        ...

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

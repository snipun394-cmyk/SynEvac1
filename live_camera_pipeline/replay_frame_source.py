from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from live_camera_pipeline.frame_source import CameraFrame, CameraFrameSource


class ReplayFrameSource(CameraFrameSource):

    # The offline/replay counterpart to a future RTSPFrameSource --
    # concrete, not a test double, living in production code so a real
    # Replay deployment (recorded video, per docs/architecture/
    # cctv_integration_readiness.md's own data-flow diagram) has
    # somewhere to plug in today. Deliberately does NOT decode a real
    # video file itself: doing so needs a real image-decoding library
    # (OpenCV or equivalent), and this package is guarded against ever
    # importing one (tests/test_no_cv_dependencies.py) -- video-frame
    # decoding is exactly the kind of computer-vision work this
    # milestone explicitly leaves for later. Instead, this class takes
    # an already-materialized, ordered, in-memory sequence of frames --
    # today that means synthetic/generated frames a test builds
    # directly; tomorrow it could just as easily be a lightweight
    # loader that reads a manifest and calls this constructor, without
    # this class itself changing.
    #
    # `source_path` is optional provenance only -- if given, it is
    # never opened or read (no I/O in this class beyond what
    # `is_source_available` needs to answer honestly), only checked for
    # existence, so a Replay source pointed at a recording that was
    # never actually captured yet reports "missing" truthfully instead
    # of silently producing zero frames indistinguishable from "no
    # people were ever visible."

    def __init__(
        self,
        camera_id: str,
        frames: Sequence[Tuple[float, Any]] = (),
        source_path: Optional[Path] = None,
    ):

        self.camera_id = camera_id
        self.source_path = Path(source_path) if source_path is not None else None

        self._frames = list(frames)
        self._next_index = 0
        self._running = False

    # =====================================================
    # Status (Phase 8 truthfully-derived Replay status)
    # =====================================================

    @property
    def is_source_available(self) -> bool:

        # A source is available if it was handed real frames directly,
        # or if it points at a source_path that genuinely exists on
        # disk. A ReplayFrameSource constructed with neither is exactly
        # the honest "Source Missing" case -- never a crash, never a
        # silently-empty stream indistinguishable from "nobody there."

        if self._frames:
            return True

        if self.source_path is not None:
            return self.source_path.exists()

        return False

    @property
    def has_frames_remaining(self) -> bool:

        return self._next_index < len(self._frames)

    @property
    def frame_count(self) -> int:

        return len(self._frames)

    # =====================================================
    # CameraFrameSource interface
    # =====================================================

    @property
    def is_running(self) -> bool:

        return self._running

    def start(self) -> None:

        # No thread, no connection, no I/O -- "started" only gates
        # whether read_frame() will hand out frames, the same
        # before-start()/after-stop() discipline a real capture device
        # would need, proven here without needing a real device.
        self._running = True

    def stop(self) -> None:

        self._running = False

    def read_frame(self) -> Optional[CameraFrame]:

        if not self._running:
            return None

        if self._next_index >= len(self._frames):
            return None

        timestamp, payload_ref = self._frames[self._next_index]

        frame = CameraFrame(
            camera_id=self.camera_id,
            timestamp=timestamp,
            frame_sequence=self._next_index,
            payload_ref=payload_ref,
        )

        self._next_index += 1

        return frame

    # =====================================================

    def reset(self) -> None:

        # Deterministic re-stepping for tests that need to replay the
        # same fixed sequence more than once.
        self._next_index = 0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CameraFrame:

    # The smallest honest unit a real camera stream can hand
    # upstream -- one image, from one camera, at one instant. Nothing
    # in this codebase interprets `payload_ref` (deliberately typed
    # Any, not e.g. a numpy array) -- doing so is a future
    # HumanDetector's job, not this seam's. Keeping this type free of
    # any particular image representation is what lets
    # CameraFrameSource stay independent of OpenCV/any specific
    # capture library (Phase 3's own requirement).

    camera_id: str
    timestamp: float
    frame_sequence: int
    payload_ref: Optional[Any] = None

    # RTSP Frame Source milestone, Phase 5: additive, backward-compatible
    # metadata a real stream can honestly report and a synthetic/replayed
    # one (ReplayFrameSource, every existing test's FakeFrameSource)
    # simply never sets -- defaulting to None rather than a fabricated
    # value. HumanDetector must never depend on these (see
    # live_camera_pipeline/human_detector.py); they exist purely for a
    # diagnostic/debug view (docs/architecture/cctv_integration_readiness.
    # md Sec 18.3) to have something honest to report once a real source
    # exists to populate them.
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None


class CameraFrameSource(ABC):

    # The seam a real camera connection plugs into. Today: no
    # implementation at all. Tomorrow: SimulationFrameSource,
    # ReplayFrameSource, and -- once real CCTV access exists --
    # RTSPFrameSource, each a concrete subclass of exactly this
    # interface, changing nothing about HumanDetector, IdentityResolver,
    # LiveCameraPipelineDetectionProvider, CameraManager,
    # MultiCameraFusionEngine, or BuildingState.
    #
    # Deliberately no OpenCV/RTSP-client import here or anywhere else
    # in this package -- see tests/test_no_cv_dependencies.py.

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def read_frame(self) -> Optional[CameraFrame]:
        ...

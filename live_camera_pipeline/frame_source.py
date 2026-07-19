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

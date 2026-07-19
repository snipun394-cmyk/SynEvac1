from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraStatus:

    # A read-only snapshot of one Camera Asset's management state --
    # never stored anywhere, always produced fresh by CameraManager.
    # camera_status() from whatever the live Camera object (and this
    # manager's own detection-provider registry) currently show. Same
    # "derived, never persisted" discipline as CameraVisibility/
    # FloorCoverage (see visibility/engine.py, visibility/coverage.py).

    camera_id: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    mode: str

    # Whether CameraManager currently has a DetectionProvider
    # registered for this camera's own mode -- True for Simulation
    # once a Simulation Adapter is registered, always False for
    # Replay/Live in this milestone (Phase 4's own "architecture
    # placeholder, no working implementation yet" requirement) until a
    # future adapter registers itself (Phase 6).
    has_detection_provider: bool

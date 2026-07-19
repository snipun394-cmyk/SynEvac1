from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpeakerStatus:

    # A read-only snapshot of one Speaker asset's management state --
    # same "derived, never stored anywhere, produced fresh by the
    # manager" role as camera_manager.status.CameraStatus/sensor_manager.
    # status.SensorStatus, independently restated here rather than
    # imported (SpeakerManager must not couple to either -- see
    # SpeakerManager's own docstring).

    speaker_id: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    mode: str
    health_status: str

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EmergencyLightStatus:

    # A read-only snapshot of one Emergency Light asset's management
    # state -- same "derived, never stored, produced fresh by the
    # manager" role as sign_manager.status.SignStatus/speaker_manager.
    # status.SpeakerStatus, independently restated here rather than
    # imported (EmergencyLightManager must not couple to either -- see
    # EmergencyLightManager's own docstring).

    light_id: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    health_status: str
    light_type: str

    # The one thing this asset actually contributes to the Digital Twin
    # this milestone -- see models.emergency_light.EmergencyLight.
    # compute_availability()'s own docstring for what this honestly
    # does and does not mean (never route safety evidence).
    availability: str

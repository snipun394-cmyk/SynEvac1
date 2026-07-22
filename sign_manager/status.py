from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SignStatus:

    # A read-only snapshot of one DynamicEvacuationSign asset's
    # management state -- same "derived, never stored, produced fresh
    # by the manager" role as speaker_manager.status.SpeakerStatus,
    # independently restated here rather than imported (SignManager
    # must not couple to speaker_manager -- see SignManager's own
    # docstring).

    sign_id: str
    name: str
    floor_id: str
    zone_ids: Tuple[str, ...]

    active: bool
    orientation: float
    supported_indications: Tuple[str, ...]

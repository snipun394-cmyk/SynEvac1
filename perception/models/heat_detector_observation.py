from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HeatDetectorReading:

    # One heat detector's report at a single instant -- same
    # native-output-only convention as SmokeDetectorReading.
    #
    # SynEvac V1 supports Fixed Temperature heat detectors only --
    # Rate-of-Rise heat detectors are deliberately out of scope. This
    # type therefore carries no rate-of-rise field at all: a dormant,
    # always-None placeholder field would misrepresent what this
    # device class supports. If Rate-of-Rise support is added later,
    # it should arrive as a new detector type (or an explicit
    # extension of this one), not as a field that silently sits unused
    # until then.

    detector_id: str
    timestamp: float
    alarm_active: bool

    confidence: Optional[float] = None

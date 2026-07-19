from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmokeDetectorReading:

    # One smoke detector's report at a single instant. alarm_active is
    # the device's own native output -- a threshold-crossing bit, not
    # a continuous smoke value. A real smoke detector does not measure
    # and report a smoke level the way Ground Truth's HazardNodeState
    # does; reporting anything more precise than a bool here would
    # misrepresent what the device actually tells you, not simplify
    # it. "No reading available" is expressed by a provider simply not
    # producing a SmokeDetectorReading for this detector this cycle,
    # never by a third state on alarm_active itself.

    detector_id: str
    timestamp: float
    alarm_active: bool

    confidence: Optional[float] = None

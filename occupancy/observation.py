from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OccupancyObservation:

    # One node's estimated headcount at a point in time -- the
    # Occupancy pipeline's own equivalent of HazardNodeState,
    # deliberately not that class or a subclass of it: Hazard
    # describes the building, Occupancy describes people, and the two
    # are meant to be combinable later (inside a future AI Decision
    # Engine / Firefighter Decision Support system) without either
    # package importing or depending on the other. Keyed externally by
    # node_id (see OccupancySnapshot), never stored on the object
    # itself -- same convention Node/Edge/HazardNodeState already
    # follow.
    #
    # occupant_count is whatever a producer (CCTV, a vision model, an
    # entry counter, BLE/UWB positioning, WiFi localization, or a
    # simulated replay) reports -- None means "no reading available",
    # never "zero people". confidence is optional provenance metadata
    # a future AI Decision Engine can use to weigh disagreeing sources
    # against each other; None means the producer does not report a
    # confidence value at all.

    occupant_count: Optional[float] = None
    confidence: Optional[float] = None

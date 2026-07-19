from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CameraFrameObservation:

    # One camera's typed report for a single frame/instant -- device-
    # scoped, not node-scoped: which Node.id(s) this frame corresponds
    # to is resolved later, by a future Occupancy Estimation stage via
    # Camera.coverage_polygon(), never by this type itself. One level
    # less opaque than sensors.reading.SensorReading (estimated
    # occupant count and visibility are already named fields, not a
    # raw payload), but still says nothing about which part of the
    # building they apply to, and still carries no ground-truth-shaped
    # value.
    #
    # estimated_occupant_count/visibility_estimate/confidence are each
    # independently Optional -- a camera implementation that only does
    # people-counting (or only visibility estimation) leaves the field
    # it doesn't produce as None rather than fabricating a value for it.

    camera_id: str
    timestamp: float

    estimated_occupant_count: Optional[float] = None
    visibility_estimate: Optional[str] = None
    confidence: Optional[float] = None

from typing import Optional, Sequence

from sensor_fusion.observation import ObservationKind


# Sensor Fusion Engine milestone, Phase 7 -- conflict is measured
# PER (location, kind) group, never across different kinds (a smoke
# detector staying silent is never "in conflict with" a camera
# reporting the zone occupied -- they are two entirely different
# measurements, fused into two separate FusedObservations; see Phase 7's
# own worked example in docs/architecture/sensor_fusion.md Sec 6).
# A single contributing source is NEVER a conflict (nothing to disagree
# with) -- this is what lets confidence.fuse_confidence()'s own
# single-source case "retain confidence honestly" rather than being
# penalized for a disagreement that does not exist.

DEFAULT_NUMERIC_CONFLICT_TOLERANCE = {
    ObservationKind.OCCUPANCY: 2.0,     # people -- minor headcount disagreement is expected, not a real conflict
    ObservationKind.TEMPERATURE: 10.0,  # degrees -- sensor placement/calibration variance is normal
    ObservationKind.VISIBILITY: 5.0,    # meters
}

# Any numeric kind not listed above (including a future one) falls
# back to this -- conservative (any measurable disagreement counts),
# since this package has no honest basis to guess a tolerance for a
# kind it does not specifically know about.
DEFAULT_GENERIC_NUMERIC_TOLERANCE = 0.0

_BOOLEAN_KINDS = (ObservationKind.SMOKE, ObservationKind.HEAT, ObservationKind.FIRE, ObservationKind.ALARM)


def detect_conflict(kind: ObservationKind, measurements: Sequence, tolerance: Optional[float] = None) -> bool:

    if len(measurements) < 2:
        return False

    if kind in _BOOLEAN_KINDS:
        return len({bool(measurement) for measurement in measurements}) > 1

    if kind == ObservationKind.BEHAVIOR:
        return len(set(measurements)) > 1

    # Numeric kinds (OCCUPANCY/TEMPERATURE/VISIBILITY, and any future
    # numeric kind) -- disagreement is a spread beyond tolerance, not
    # any difference at all (real sensors never agree to the decimal).
    effective_tolerance = (
        tolerance if tolerance is not None
        else DEFAULT_NUMERIC_CONFLICT_TOLERANCE.get(kind, DEFAULT_GENERIC_NUMERIC_TOLERANCE)
    )

    return (max(measurements) - min(measurements)) > effective_tolerance

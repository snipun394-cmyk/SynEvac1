# Smoke-to-visibility conversion, isolated in its own module so the
# formula is swappable (e.g. for a future optical-density-based model)
# without touching SmokePropagationModel's propagation/growth logic --
# same "one small, documented placeholder, independently replaceable"
# pattern as HazardSeverity's score cutoffs and FireGrowthCurve's
# t-squared shape.
#
# A simple linear falloff between clear-room visibility and a near-
# zero floor -- not a validated smoke-optics model. Never reaches
# exactly 0.0: a fully-logged space is still documented as MIN_
# VISIBILITY_M, not nothing, so nothing downstream needs to special-
# case a zero denominator.

CLEAR_VISIBILITY_M = 30.0
MIN_VISIBILITY_M = 0.5


def visibility_from_smoke_level(smoke_level: float) -> float:

    clamped_smoke_level = max(0.0, min(smoke_level, 1.0))

    return CLEAR_VISIBILITY_M - (
        (CLEAR_VISIBILITY_M - MIN_VISIBILITY_M) * clamped_smoke_level
    )

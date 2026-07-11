from enum import Enum, auto


class HazardSeverity(Enum):

    # A read-only classification of "how bad is it here" -- derived,
    # never authored directly (see HazardNodeState.severity). Every
    # consumer (Human Behavior, reporting, UI, and later Fire/Smoke/
    # Detector/RL) reads severity through from_score() below rather
    # than inventing its own thresholds, so there is exactly one place
    # in the whole codebase that maps a hazard_score to a label.

    NONE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()

    # =====================================================
    # Centralized hazard_score -> HazardSeverity mapping. hazard_score
    # is a normalized float in [0.0, 1.0] (0.0 clear, 1.0 untenable).
    # These cutoffs are a placeholder classification, not a validated
    # life-safety threshold model -- same honesty as CongestionModel's
    # documented linear degradation and Building Analysis's "no
    # hardcoded life-safety thresholds". They live only here; nothing
    # downstream re-derives severity from smoke_level/visibility/
    # temperature, or keeps its own copy of these cutoffs.
    # =====================================================

    @classmethod
    def from_score(cls, hazard_score):

        if hazard_score is None:
            return cls.NONE

        if hazard_score >= 0.85:
            return cls.CRITICAL

        if hazard_score >= 0.60:
            return cls.HIGH

        if hazard_score >= 0.30:
            return cls.MODERATE

        if hazard_score >= 0.05:
            return cls.LOW

        return cls.NONE

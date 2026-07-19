from dataclasses import dataclass

from behaviour_profile_resolver.category import OccupantCategory


# =====================================================
# Phase 4 -- Dynamic Priority Model. A pure, deterministic function of
# already-known factors -- same "pure, deterministic function of an
# already-built input, the only place allowed to produce a value"
# convention perception.human_inference.derive_inference_flags()
# already establishes. Every weight below is a fixed, documented
# constant (never learned, never randomized) -- the same "illustrative,
# not validated" honesty this codebase applies to every other scoring
# threshold (HazardSeverity.from_score, FALL_HAZARD_SCORE_THRESHOLD).
# Two calls with identical RescuePriorityFactors always return the
# identical score -- "Priority must be deterministic" (Phase 4's own
# rule), satisfied structurally, not by convention alone.
# =====================================================

WEIGHT_WHEELCHAIR = 0.15
WEIGHT_POSSIBLE_INJURY = 0.30
WEIGHT_FALLEN = 0.35
WEIGHT_SMOKE_EXPOSURE = 0.25
WEIGHT_DISTANCE_FROM_HAZARD = 0.15
WEIGHT_ACCESSIBILITY = 0.10
WEIGHT_ISOLATION = 0.15

# Age-category baseline -- vulnerable categories start with a nonzero
# floor even before any hazard/injury signal is known; ordinary mobile
# adults/staff/firefighters/unknowns start at zero (they are never a
# rescue target purely for existing -- see FirefighterDecisionEngine's
# own minimum-urgency floor).
_CATEGORY_BASE_WEIGHT = {
    OccupantCategory.CHILD: 0.25,
    OccupantCategory.ELDERLY: 0.20,
    OccupantCategory.WHEELCHAIR_USER: 0.20,
    OccupantCategory.VISITOR: 0.05,
}


@dataclass(frozen=True)
class RescuePriorityFactors:

    category: OccupantCategory
    is_wheelchair_user: bool = False
    possible_injury: bool = False
    fallen: bool = False

    # All three below are normalized to [0.0, 1.0] by the caller before
    # being passed in -- this function clamps defensively but never
    # reinterprets an out-of-range input as anything but its clamped
    # bound.
    smoke_exposure: float = 0.0
    distance_from_hazard: float = 1.0  # 1.0 = far/safe, 0.0 = at the hazard
    accessibility_score: float = 1.0  # 1.0 = fully able to self-evacuate unaided
    isolation_score: float = 0.0  # 1.0 = no capable helper nearby


def _clamp(value: float) -> float:

    return max(0.0, min(1.0, value))


def compute_rescue_priority(factors: RescuePriorityFactors) -> float:

    score = _CATEGORY_BASE_WEIGHT.get(factors.category, 0.0)

    if factors.is_wheelchair_user:
        score += WEIGHT_WHEELCHAIR

    if factors.possible_injury:
        score += WEIGHT_POSSIBLE_INJURY

    if factors.fallen:
        score += WEIGHT_FALLEN

    score += WEIGHT_SMOKE_EXPOSURE * _clamp(factors.smoke_exposure)
    score += WEIGHT_DISTANCE_FROM_HAZARD * _clamp(1.0 - factors.distance_from_hazard)
    score += WEIGHT_ACCESSIBILITY * _clamp(1.0 - factors.accessibility_score)
    score += WEIGHT_ISOLATION * _clamp(factors.isolation_score)

    return round(score, 6)

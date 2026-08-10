import math

# =====================================================
# Kinateder & Warren (2021) Herding Evidence -- PUBLISHED DATA ONLY.
#
# Kinateder, M. & Warren, W.H. (2021), "Exit choice during evacuation
# is influenced by both the size and proportion of the egressing
# crowd," Physica A 569, 125746.
# DOI: 10.1016/j.physa.2021.125746
#
# This module holds ONLY the published coefficients (Table 4, a binary
# logistic mixed model, fitted pooled across all 3 of the paper's VR
# experiments) and a pure function that derives a fixed-effects-only
# probability from them via the standard logistic transform. It is
# deliberately separate from any SynEvac RouteChoiceStrategy -- see
# behavior_library/route_choice_strategies.py's own
# EmpiricalProportionHerdingRouteChoiceStrategy, the one and only
# consumer of this module.
#
# Table 4 codes crowd proportion as a CATEGORICAL factor (five
# discrete tested levels: 60/70/80/90/100%), not a continuous
# predictor -- the published model has no functional form to
# interpolate between those levels, so none is invented here. Crowd
# size 10 is the table's own implicit reference level (no size-10
# interaction term exists at all); crowd size 20 is expressed as an
# interaction relative to it, reported only at 80/90/100%.
#
# PUBLISHED COEFFICIENTS vs. MODEL-DERIVED PROBABILITIES: the
# constants below (INTERCEPT, PROPORTION_COEFFICIENTS,
# SIZE_20_INTERACTION_COEFFICIENTS) are the paper's own log-odds
# values, transcribed exactly. model_implied_probability() and
# follow_probability_for() COMPUTE a derived probability from those
# coefficients on demand -- this is a fixed-effects-only
# approximation (the paper's own mixed model also carries a
# participant-level random-effects term this module does not, and
# cannot, reproduce) -- it is NEVER a raw observed participant
# frequency, and is never cached/stored as if it were.
# =====================================================

CITATION = (
    "Kinateder, M. & Warren, W.H. (2021), \"Exit choice during evacuation is "
    "influenced by both the size and proportion of the egressing crowd,\" "
    "Physica A 569, 125746."
)
DOI = "10.1016/j.physa.2021.125746"

# Reference/baseline category for the crowd-proportion factor.
REFERENCE_PROPORTION_PERCENT = 60

# Model intercept at the 60%-proportion, size-10 reference condition.
# SE 0.240, p = .882 -- not significantly different from chance at
# this baseline.
INTERCEPT = -0.036

# Log-odds coefficient, relative to the 60% reference, keyed by tested
# crowd-proportion percentage. 60% itself carries an explicit 0.0
# entry (its own reference category) so every lookup below can treat
# all five tested levels uniformly, with no special case.
PROPORTION_COEFFICIENTS = {
    60: 0.0,
    70: 0.063,   # p = .753 (not significant)
    80: 0.551,   # p < .001
    90: 0.740,   # p < .001
    100: 1.190,  # p < .001
}

# Crowd-size=20 interaction coefficient (relative to crowd-size=10),
# keyed by the SAME tested proportion percentages. No interaction term
# was reported at 60% or 70% in the published table -- per this
# milestone's own governing instruction, an unreported interaction is
# treated as exactly 0.0 (i.e. no evidence of a size-20 effect at
# those levels), never guessed at any other value.
SIZE_20_INTERACTION_COEFFICIENTS = {
    60: 0.0,
    70: 0.0,
    80: -0.593,  # p < .05
    90: -0.751,  # p < .001
    100: -0.269,  # p = .354 (not significant)
}

SUPPORTED_CROWD_SIZES = (10, 20)
SUPPORTED_PROPORTION_LEVELS_PERCENT = (60, 70, 80, 90, 100)


# =====================================================


def nearest_supported_proportion_level(proportion_percent):

    # No interpolation is ever performed -- this is discrete category
    # selection over the five published levels only. Tie-break rule,
    # explicit and deterministic: on an exact tie (e.g. 75%, equidistant
    # between 70 and 80), the LOWER category wins -- the more
    # conservative choice (a smaller implied herding effect), consistent
    # with this whole investigation's own "err toward the legacy/
    # unenhanced behavior whenever the empirical structure doesn't
    # clearly apply" discipline. min()'s own tuple comparison already
    # implements this: for equal distance, the smaller `level` value in
    # the second tuple slot wins.

    return min(
        SUPPORTED_PROPORTION_LEVELS_PERCENT,
        key=lambda level: (abs(level - proportion_percent), level),
    )


def model_implied_probability(crowd_size, proportion_percent):

    # Returns the fixed-effects-only probability implied by the
    # published logistic mixed model, evaluated at the nearest
    # supported proportion category -- or None if crowd_size itself is
    # not exactly 10 or 20 (crowd size is NEVER snapped/interpolated;
    # only proportion is, and only within this function's own already-
    # size-gated caller).

    if crowd_size not in SUPPORTED_CROWD_SIZES:
        return None

    level = nearest_supported_proportion_level(proportion_percent)

    logit = INTERCEPT + PROPORTION_COEFFICIENTS[level]

    if crowd_size == 20:
        logit += SIZE_20_INTERACTION_COEFFICIENTS[level]

    return 1.0 / (1.0 + math.exp(-logit))


def follow_probability_for(crowd_size, majority_proportion):

    # The one public domain-policy entry point. `majority_proportion`
    # is a fraction in (0, 1], exactly SynEvac's own
    # majority_count / total_observed_decisions construction (see
    # EmpiricalProportionHerdingRouteChoiceStrategy). Returns the
    # empirical, model-derived probability when both crowd_size and
    # majority_proportion fall within the evidence's own supported
    # domain; returns None in every other case -- the caller is
    # responsible for falling back to legacy constant follow_probability
    # behavior whenever this returns None, never for fabricating a
    # value of its own.
    #
    # Domain policy, in order:
    #   1. crowd_size not in {10, 20}          -> None (no size evidence elsewhere)
    #   2. proportion below the 60% reference  -> None (nothing brackets it)
    #   3. proportion above 100% (structurally
    #      impossible -- majority_count can
    #      never exceed total_observed_decisions
    #      by construction) -> None, fail safe
    #      rather than fabricate, never raise
    #      (every RouteChoiceStrategy must always
    #      produce a RouteChoice)
    #   4. otherwise -> nearest-category lookup, no interpolation

    if crowd_size not in SUPPORTED_CROWD_SIZES:
        return None

    proportion_percent = majority_proportion * 100.0

    if proportion_percent < REFERENCE_PROPORTION_PERCENT:
        return None

    if proportion_percent > 100.0 + 1e-9:
        return None

    return model_implied_probability(crowd_size, min(proportion_percent, 100.0))

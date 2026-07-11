class FireGrowthCurve:

    # The growth-model abstraction FireGrowthModel delegates to for
    # "how intense is the fire after this much elapsed time" -- mirrors
    # HazardMergeStrategy's role for HazardEvolutionEngine: the model
    # that consumes this embeds no growth math of its own, so a future
    # curve (NFPA-categorized, empirically fit, ML-based) plugs in
    # without touching FireGrowthModel at all.
    #
    # Must be a pure function: the same elapsed_time always produces
    # the same result, no side effects, no dependency on anything but
    # its own input. FireGrowthModel never calls this with a negative
    # elapsed_time (it withholds any opinion before ignition itself),
    # so implementations do not need to guard against that case.

    def intensity_at(self, elapsed_time: float) -> float:

        # Returns a normalized value in [0.0, 1.0] -- 0.0 unignited/
        # negligible, 1.0 fully developed. Directly usable as
        # HazardNodeState.hazard_score.

        raise NotImplementedError


class TSquaredFireGrowthCurve(FireGrowthCurve):

    # V1's default -- the classic "t-squared" design-fire curve used in
    # fire engineering as a first-approximation growth shape (heat
    # release rate proportional to elapsed_time squared), not a
    # validated physics/CFD model -- same documented-simplification
    # honesty as CongestionModel's linear degradation and
    # HazardSeverity's threshold buckets. growth_time is the elapsed
    # time (seconds) at which intensity saturates at 1.0; smaller
    # values model a faster-growing fire, matching how the NFPA
    # slow/medium/fast/ultra-fast growth-rate categories are
    # distinguished by a single time-to-reference-intensity parameter.

    def __init__(self, growth_time: float):

        self.growth_time = growth_time

    # =====================================================

    def intensity_at(self, elapsed_time: float) -> float:

        return min((elapsed_time / self.growth_time) ** 2, 1.0)

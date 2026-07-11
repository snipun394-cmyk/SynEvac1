class SmokeGrowthCurve:

    # The growth-model abstraction SmokePropagationModel delegates to
    # for "how dense is the smoke this far after it arrived at a
    # node" -- mirrors fire_growth.growth_curve.FireGrowthCurve's role
    # for FireGrowthModel, deliberately re-declared as its own
    # interface rather than reused, so smoke_propagation and
    # fire_growth stay two independently replaceable HazardSource
    # packages that happen to share a shape, not two things coupled
    # through a shared base class.
    #
    # Must be a pure function: the same elapsed_time always produces
    # the same result, no side effects, no dependency on anything but
    # its own input. SmokePropagationModel never calls this with a
    # negative elapsed_time (it withholds any opinion for a node the
    # smoke front hasn't reached yet), so implementations do not need
    # to guard against that case.

    def intensity_at(self, elapsed_time: float) -> float:

        # Returns a normalized value in [0.0, 1.0] -- 0.0 negligible,
        # 1.0 fully smoke-logged. Directly usable as both
        # HazardNodeState.smoke_level and HazardNodeState.hazard_score.

        raise NotImplementedError


class TSquaredSmokeGrowthCurve(SmokeGrowthCurve):

    # V1's default -- the same t-squared shape
    # fire_growth.growth_curve.TSquaredFireGrowthCurve uses, since
    # smoke buildup at a point (once the front has arrived) is, like
    # fire intensity, a first-approximation "starts slow, accelerates"
    # curve rather than a validated physics/CFD model. growth_time is
    # the elapsed time (seconds) after the smoke front's arrival at
    # which local smoke_level saturates at 1.0 -- independent of, and
    # typically shorter than, FireGrowthModel's own growth_time, since
    # smoke logging a space plausibly happens faster than the fire
    # itself reaching full development there.

    def __init__(self, growth_time: float):

        self.growth_time = growth_time

    # =====================================================

    def intensity_at(self, elapsed_time: float) -> float:

        return min((elapsed_time / self.growth_time) ** 2, 1.0)

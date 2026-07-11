import math
import random

from behavior.pre_movement import PreMovementDelayStrategy


class ProbabilisticPreMovementDelay(PreMovementDelayStrategy):

    # A lognormal pre-movement delay -- lognormal is the shape commonly
    # used in evacuation literature for human response/pre-movement
    # time, but this is a simple, two-parameter draw, not a validated
    # RSET pre-movement-time model -- same documented-simplification
    # honesty as every other Default*/Strategy in this codebase.
    #
    # median_delay (seconds) is the distribution's median -- the most
    # directly interpretable parameter for a scenario author ("half of
    # occupants start moving within this many seconds"). spread is the
    # underlying normal distribution's sigma, controlling how variable
    # individual response times are; larger spread means a longer,
    # heavier tail of very late movers. rng is constructor-injected
    # for the same reason as ComplianceDecisionStrategy's.

    def __init__(self, median_delay, spread=0.5, rng=None):

        if median_delay <= 0:
            raise ValueError(
                f"ProbabilisticPreMovementDelay.median_delay must be > 0, "
                f"got {median_delay!r}."
            )

        self.median_delay = median_delay
        self.spread = spread
        self.rng = rng or random.Random()

    # =====================================================

    def delay(self, context) -> float:

        mu = math.log(self.median_delay)

        return self.rng.lognormvariate(mu, self.spread)

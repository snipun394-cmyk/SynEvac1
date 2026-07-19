from typing import Tuple


class ConfidenceFusionStrategy:

    # Phase 4 -- interchangeable confidence-fusion strategies. Same
    # "one interface, raises NotImplementedError, plus a documented
    # default" shape as every other Strategy in this codebase
    # (CostModel, CapacityModel, CongestionModel) -- a future strategy
    # is a new subclass, never a branch added here.
    #
    # Contract: fuse(confidences) receives every individual Detection's
    # own confidence for one fused track at one instant (always at
    # least one element -- MultiCameraFusionEngine never calls this
    # with an empty group) and returns a single value in [0.0, 1.0].

    def fuse(self, confidences: Tuple[float, ...]) -> float:

        raise NotImplementedError


class HighestConfidenceStrategy(ConfidenceFusionStrategy):

    # The simplest, most conservative choice, and MultiCameraFusion
    # Engine's own default: trust whichever single camera was most
    # confident, ignore the rest. Never produces a value higher than
    # any individual camera actually reported.

    def fuse(self, confidences: Tuple[float, ...]) -> float:

        return max(confidences)


class WeightedAverageConfidenceStrategy(ConfidenceFusionStrategy):

    # An unweighted mean today (every source counted equally) --
    # named "weighted" because it is the natural extension point for
    # a future per-camera reliability weight (e.g. a camera with a
    # documented lower detection_probability counting for less), not
    # because unequal weights are implemented yet.

    def fuse(self, confidences: Tuple[float, ...]) -> float:

        return sum(confidences) / len(confidences)


class MultiCameraConfidenceBoostStrategy(ConfidenceFusionStrategy):

    # Wraps another strategy (default HighestConfidenceStrategy) and
    # adds a small, documented, non-validated boost for every
    # additional camera simultaneously confirming the same track --
    # same "wrap the base model, special-case only your own concern"
    # shape as StairCapacityModel/StairAwareCongestionModel elsewhere
    # in this codebase. The engineering judgment: two independent
    # cameras agreeing on the same person is stronger evidence than
    # either alone, so confidence should reflect that agreement, not
    # just repeat one camera's own number. Always clamped to 1.0 --
    # confidence can never exceed certainty regardless of how many
    # cameras agree.

    BOOST_PER_EXTRA_CAMERA = 0.05

    def __init__(self, base_strategy: ConfidenceFusionStrategy = None):

        self.base_strategy = base_strategy or HighestConfidenceStrategy()

    def fuse(self, confidences: Tuple[float, ...]) -> float:

        base = self.base_strategy.fuse(confidences)

        extra_cameras = len(confidences) - 1

        boosted = base + extra_cameras * self.BOOST_PER_EXTRA_CAMERA

        return min(1.0, boosted)

from hazard.node_state import HazardNodeState


class TenabilityScorer:

    # The scoring-policy abstraction TenabilityModel delegates to for
    # "given one node's currently known hazard readings, how survivable
    # is it there" -- mirrors FireGrowthCurve/SmokeGrowthCurve's role
    # for their own models: the model that consumes this embeds no
    # scoring math of its own, so a future FED, toxicity, or thermal-
    # aware scorer plugs in without touching TenabilityModel at all.
    #
    # Must be a pure function of its input: the same HazardNodeState
    # always produces the same result, no side effects, no dependency
    # on anything but the fields already present on that state.

    def score(self, node_state: HazardNodeState) -> float:

        # Returns a normalized value in [0.0, 1.0] -- 0.0 fully
        # tenable, 1.0 untenable. Directly usable as
        # HazardNodeState.hazard_score.

        raise NotImplementedError


class SmokeVisibilityTenabilityScorer(TenabilityScorer):

    # V1's whole scoring policy -- derives tenability from exactly the
    # two fields the Smoke Propagation Model produces (smoke_level,
    # visibility), nothing else. Deliberately not a real FED
    # (Fractional Effective Dose) calculation -- no CO, CO2, radiant
    # heat, or oxygen depletion terms exist yet for it to combine, and
    # this is documented as a placeholder simplification, the same
    # honesty as FireGrowthModel's t-squared curve and Smoke
    # Propagation's linear visibility falloff.
    #
    # score = max(smoke_level, visibility_impairment) -- worst-of-the-
    # two, not a compounding/multiplicative combination. smoke_level
    # and visibility are two readings of the same underlying smoke at
    # the same node (Smoke Propagation derives visibility directly
    # from smoke_level), so treating them as independent, compounding
    # exposures would double-count one physical cause as if it were
    # two. max() also keeps this consistent with
    # DefaultHazardMergeStrategy's own worst-case-wins philosophy
    # elsewhere in the Hazard layer.

    DEFAULT_MIN_TENABLE_VISIBILITY_M = 3.0  # An arbitrary, documented
                                              # placeholder -- not a
                                              # validated life-safety
                                              # code threshold. Always
                                              # overridable via the
                                              # constructor.

    def __init__(self, min_tenable_visibility_m: float = DEFAULT_MIN_TENABLE_VISIBILITY_M):

        self.min_tenable_visibility_m = min_tenable_visibility_m

    # =====================================================

    def score(self, node_state: HazardNodeState) -> float:

        smoke_component = (
            node_state.smoke_level
            if node_state.smoke_level is not None
            else 0.0
        )

        return max(smoke_component, self._visibility_impairment(node_state.visibility))

    # =====================================================

    def _visibility_impairment(self, visibility) -> float:

        # None means "no visibility reading at all for this node" --
        # not the same thing as "perfectly clear", but with nothing to
        # go on this contributes no additional impairment beyond
        # whatever smoke_component already reflects.
        if visibility is None or visibility >= self.min_tenable_visibility_m:
            return 0.0

        if visibility <= 0:
            return 1.0

        return (self.min_tenable_visibility_m - visibility) / self.min_tenable_visibility_m

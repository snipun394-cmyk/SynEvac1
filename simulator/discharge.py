from navigation.flow_region import FlowRegion


class DischargeModel:

    # Admission Control V4 -- Dual Constraint Architecture. The new,
    # independent THROUGHPUT constraint MultiAgentSimulation's
    # discharge_model parameter accepts, alongside (never in place of)
    # CapacityModel -- see coordinator.py's own header comment for the
    # full architectural rationale. Mirrors CapacityModel/CongestionModel's
    # own role: the one interface MultiAgentSimulation depends on for
    # "how fast can this edge/region discharge admitted occupants."
    #
    # Returns the maximum sustained rate (people/second) this edge or
    # FlowRegion can discharge -- None means "not derivable", the same
    # convention DefaultCapacityModel/StairCapacityModel already use for
    # an edge whose width can't be determined. MultiAgentSimulation
    # enforces this as a minimum inter-admission gap of
    # 1/discharge_rate; it is NEVER interpreted as an occupancy ceiling
    # -- that remains CapacityModel's own, sole concern, by design.

    def discharge_rate(self, edge_or_region):

        raise NotImplementedError


class DefaultDischargeModel(DischargeModel):

    # A literature-grounded default: the Nelson & Mowrer / SFPE
    # Handbook (Ch. 14, "Emergency Movement") specific-flow figure of
    # 1.316 persons/s per meter of boundary-layer-adjusted effective
    # width -- the same figure the Admission Control V3 investigation's
    # own Candidate C experiment (simulator/capacity_candidate_c.py)
    # already validated for exactly this physical quantity. This is an
    # INDEPENDENT restatement of that same well-established constant,
    # not a shared import -- Candidate C's own file is frozen,
    # experimental, and untouched by this milestone; nothing here reads
    # or modifies it.
    #
    # Dual-accepts a plain Edge or a FlowRegion, the same shape
    # FlowRegionCapacityModel/FlowRegionCapacityModelV2/
    # CapacityModelCandidateC already establish -- a region's own
    # aggregate representative_width stands in for a single edge's own
    # width, with no per-member-edge lookup needed for this quantity.

    SPECIFIC_FLOW_PERS_PER_S_PER_M = 1.316  # Nelson & Mowrer / SFPE Handbook Ch. 14

    BOUNDARY_LAYER_M = 0.15  # SFPE boundary-layer subtraction

    MINIMUM_EFFECTIVE_WIDTH_M = 0.1

    def discharge_rate(self, edge_or_region):

        width = self._width_of(edge_or_region)

        if width is None:
            return None

        effective_width = max(width - self.BOUNDARY_LAYER_M, self.MINIMUM_EFFECTIVE_WIDTH_M)

        return self.SPECIFIC_FLOW_PERS_PER_S_PER_M * effective_width

    def _width_of(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            return edge_or_region.representative_width

        return edge_or_region.width

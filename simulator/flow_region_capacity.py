from navigation.flow_region import FlowRegion
from simulator.capacity import CapacityModel, DefaultCapacityModel, StairCapacityModel


class FlowRegionCapacityModel(CapacityModel):

    # Hybrid Flow Regions (Option D), Milestone 2 -- a region-aware
    # CapacityModel, built entirely alongside today's per-edge models
    # without touching them. NOT yet used by MultiAgentSimulation or
    # scenario_runner (see Milestone 3+): this class exists so it can
    # be unit-tested and integrated later without any further design
    # work, exactly as the Option D design document's Milestone 2
    # requires.
    #
    # capacity() accepts EITHER a plain Edge (today's usual argument,
    # delegated straight through to `base_model` -- StairCapacityModel
    # by default, the same model scenario_runner already wires up
    # today, so nothing changes for any existing caller that never
    # passes a FlowRegion) OR a FlowRegion (the new case: one
    # aggregate capacity for the whole region). This dual acceptance
    # is what makes the model a genuine drop-in replacement candidate
    # for Milestone 3+ -- a future coordinator can hand this model
    # either an Edge or a FlowRegion and get the right answer either
    # way, without needing to know which one it has.
    #
    # The region formula reuses Option E's own area-based reasoning
    # (capacity ~ footprint area x an assumed jam density) applied to
    # the region's own total footprint instead of one edge's --
    # `total_length` and `representative_width` are exactly the two
    # aggregate fields FlowRegionInferencer already computes at
    # inference time for this purpose (see navigation/flow_region.py),
    # so no per-member-edge lookup is needed here at all.

    JAM_DENSITY_PEOPLE_PER_SQUARE_METER = 2.0

    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def __init__(self, base_model: CapacityModel = None):

        self.base_model = base_model or StairCapacityModel()

    def capacity(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            return self._region_capacity(edge_or_region)

        return self.base_model.capacity(edge_or_region)

    def _region_capacity(self, region: FlowRegion):

        # Same "None means not derivable, fall back to the minimum"
        # convention Edge.width/Edge.walking_distance and
        # DefaultCapacityModel already use -- a region whose members'
        # own geometry couldn't be determined gets the same one-
        # person-at-a-time floor as an edge in the same situation,
        # never a fabricated number.
        if (
            region.total_length is None
            or region.representative_width is None
        ):
            return self.MINIMUM_CAPACITY

        area = region.total_length * region.representative_width

        derived_capacity = int(
            area * self.JAM_DENSITY_PEOPLE_PER_SQUARE_METER
        )

        return max(derived_capacity, self.MINIMUM_CAPACITY)

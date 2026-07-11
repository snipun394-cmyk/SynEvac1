from simulator.capacity import CapacityModel, DefaultCapacityModel

from hazard.snapshot import HazardSnapshot


class HazardAwareCapacityModel(CapacityModel):

    # Wraps any existing CapacityModel (typically DefaultCapacityModel)
    # with a HazardSnapshot -- mirrors HazardAwareCostModel's role.
    # MultiAgentSimulation only ever calls capacity_model.capacity(edge);
    # it never needs to know this wrapping exists.
    #
    # Always floored at DefaultCapacityModel.MINIMUM_CAPACITY, same as
    # the base model itself -- MultiAgentSimulation plans each
    # occupant's route once and never reroutes it, so an edge that
    # reaches capacity 0 mid-run would strand whoever is already queued
    # for it, permanently. Full closure belongs at the routing layer
    # (HazardAwareCostModel steering new plans away from a blocked
    # edge before anyone commits to it), never here -- this model only
    # narrows throughput, it never stops it outright.

    def __init__(self, base_model: CapacityModel, hazard_snapshot: HazardSnapshot):

        self.base_model = base_model
        self.hazard_snapshot = hazard_snapshot

    # =====================================================

    def capacity(self, edge):

        base_capacity = self.base_model.capacity(edge)
        modifier = self.hazard_snapshot.edge_state(edge.id).capacity_modifier

        reduced_capacity = int(base_capacity * modifier)

        return max(reduced_capacity, DefaultCapacityModel.MINIMUM_CAPACITY)

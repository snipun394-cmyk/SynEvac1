import math

from navigation.edge import Edge
from navigation.flow_region import FlowRegion
from simulator.capacity import CapacityModel, DefaultCapacityModel


class CapacityModelCandidateC(CapacityModel):

    # Admission Control V3 investigation, Candidate C ("Derived
    # Server-Count Model") -- experimental only. Not wired into
    # scenario_runner, Designer, or any production default;
    # reachable only through the same opt-in
    # calibration_benchmark.simulation_seam.run_with_overrides() path
    # V1/V2 already use.
    #
    # Preserves CapacityModel's exact interface (capacity() still
    # returns one number, still gates admission the same way in
    # MultiAgentSimulation) -- only the FORMULA behind that number
    # changes. Where DefaultCapacityModel/StairCapacityModel derive
    # capacity from an ad hoc "people per meter of width" constant,
    # this model derives it from two independently-grounded physical
    # quantities, combined via the queueing-theory identity
    # (a Little's-Law-consistent server count):
    #
    #   capacity = ceil(discharge_rate * service_time)
    #
    # - discharge_rate (people/second) is the specific flow rate this
    #   edge/region can sustain, per the Nelson & Mowrer / SFPE
    #   Handbook (Ch. 14, "Emergency Movement") figure of 1.316
    #   persons/s per meter of EFFECTIVE width, after subtracting a
    #   0.15 m boundary layer -- the same figures cited in the
    #   Admission Control V3 architecture investigation's survey of
    #   Pathfinder/SFPE.
    # - service_time (seconds) is how long one admitted occupant
    #   occupies a "server slot" -- reused directly from Edge's own
    #   already-computed `traversal_time` (walking_distance /
    #   Edge.ASSUMED_WALK_SPEED_M_PER_S), never a new independent
    #   free parameter. This is deliberate: the architecture
    #   investigation noted storage_capacity/discharge_rate/
    #   service_time are not three independent numbers under
    #   Little's Law -- any two determine the third, so this model
    #   only ever introduces ONE new physical input (discharge_rate)
    #   and derives the rest from data the graph already computes.
    #
    # An explicit authored capacity (Exit.capacity) always wins, same
    # precedence as DefaultCapacityModel/StairCapacityModel -- a
    # derived formula only ever fills in for the unauthored default
    # case, never overrides real engineering data.
    #
    # Same dual-acceptance shape as FlowRegionCapacityModel (V1) and
    # FlowRegionCapacityModelV2 (V2): a plain Edge uses its own
    # width/traversal_time; a FlowRegion uses its own aggregate
    # representative_width/total_length, so this model is directly
    # comparable to V1/V2 in the same Flow-Region-aware validation
    # runs. Unlike V1 (area x jam-density) and V2 (min-cut), this is
    # not a region-topology-specific formula -- CHAIN and MERGE
    # regions are treated identically, both reduced to the region's
    # own two aggregate fields.

    SPECIFIC_FLOW_PERS_PER_S_PER_M = 1.316  # Nelson & Mowrer / SFPE Handbook Ch. 14

    BOUNDARY_LAYER_M = 0.15  # SFPE boundary-layer subtraction

    MINIMUM_EFFECTIVE_WIDTH_M = 0.1

    MINIMUM_CAPACITY = DefaultCapacityModel.MINIMUM_CAPACITY

    def capacity(self, edge_or_region):

        if isinstance(edge_or_region, FlowRegion):
            return self._region_capacity(edge_or_region)

        return self._edge_capacity(edge_or_region)

    def _edge_capacity(self, edge):

        explicit_capacity = edge.capacity

        if explicit_capacity is not None:
            return max(int(explicit_capacity), self.MINIMUM_CAPACITY)

        discharge_rate = self._discharge_rate(edge.width)
        service_time = edge.traversal_time

        return self._derive(discharge_rate, service_time)

    def _region_capacity(self, region: FlowRegion):

        discharge_rate = self._discharge_rate(region.representative_width)
        service_time = self._region_service_time(region.total_length)

        return self._derive(discharge_rate, service_time)

    def _discharge_rate(self, width):

        if width is None:
            return None

        effective_width = max(width - self.BOUNDARY_LAYER_M, self.MINIMUM_EFFECTIVE_WIDTH_M)

        return self.SPECIFIC_FLOW_PERS_PER_S_PER_M * effective_width

    def _region_service_time(self, total_length):

        if total_length is None:
            return None

        return total_length / Edge.ASSUMED_WALK_SPEED_M_PER_S

    def _derive(self, discharge_rate, service_time):

        if discharge_rate is None or service_time is None or service_time <= 0:
            return self.MINIMUM_CAPACITY

        derived_capacity = math.ceil(discharge_rate * service_time)

        return max(derived_capacity, self.MINIMUM_CAPACITY)

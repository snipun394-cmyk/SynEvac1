from navigation.flow_region import FlowRegion
from simulator.congestion import (
    CongestionModel,
    DefaultCongestionModel,
    StairAwareCongestionModel,
)


class FlowRegionCongestionModel(CongestionModel):

    # Hybrid Flow Regions (Option D), Milestone 2 -- a region-aware
    # CongestionModel, built entirely alongside today's per-edge models
    # without touching them. NOT yet used by MultiAgentSimulation or
    # scenario_runner (see Milestone 3+).
    #
    # speed_factor() accepts EITHER a plain Edge (delegated straight
    # through to `base_model` -- StairAwareCongestionModel by default,
    # the same model scenario_runner already wires up today, so nothing
    # changes for any existing caller that never passes a FlowRegion)
    # OR a FlowRegion (the new case: one shared speed factor for every
    # occupant currently entering any member edge of that region).
    #
    # The region formula is deliberately the SAME shape as
    # DefaultCongestionModel's plain linear degradation -- not
    # StairAwareCongestionModel's counterflow-penalised one. A merged
    # region can combine edges of different types (a stair chain plus
    # the door it discharges into), and FlowRegion itself carries no
    # per-member edge-type breakdown to gate a stair-only penalty on;
    # rather than guess, this stays at the same simple, honest
    # occupancy-ratio formula the whole simulator's congestion model
    # started from. DefaultCongestionModel.speed_factor() never
    # actually reads its own `edge` argument, so it is reused directly
    # here rather than re-deriving the same formula a second time.

    MINIMUM_SPEED_FACTOR = DefaultCongestionModel.MINIMUM_SPEED_FACTOR

    def __init__(self, base_model: CongestionModel = None):

        self.base_model = base_model or StairAwareCongestionModel()
        self._region_model = DefaultCongestionModel()

    def speed_factor(self, edge_or_region, other_occupants, capacity, opposing_occupants=0):

        if isinstance(edge_or_region, FlowRegion):

            return self._region_model.speed_factor(
                edge_or_region,
                other_occupants,
                capacity,
                opposing_occupants,
            )

        return self.base_model.speed_factor(
            edge_or_region,
            other_occupants,
            capacity,
            opposing_occupants,
        )

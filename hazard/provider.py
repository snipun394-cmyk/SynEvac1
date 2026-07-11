from hazard.snapshot import HazardSnapshot


class HazardProvider:

    # The one interface every future producer of dynamic environmental
    # state (Fire Growth, Smoke Propagation, Detectors, Live Sensors,
    # Scenario Generation, RL) is meant to implement -- mirrors
    # CostModel/CapacityModel's role as the single seam a whole future
    # subsystem plugs into. Consumers never depend on how a snapshot at
    # a given time was produced, only on this method.

    def snapshot_at(self, time: float) -> HazardSnapshot:

        raise NotImplementedError


class ManualHazardProvider(HazardProvider):

    # V1's whole provider: one fixed, hand-authored (or empty)
    # HazardSnapshot, returned regardless of the requested time. This
    # is what lets a scenario be hand-built and fed through
    # HazardAwareCostModel/HazardAwareCapacityModel/DecisionContext
    # today, with no fire/smoke/sensor logic behind it -- and what a
    # future time-varying provider (e.g. one backed by a fire growth
    # simulation) replaces without any change to those consumers.
    #
    # CostModel.cost(edge) and CapacityModel.capacity(edge) take no
    # time parameter, and MultiAgentSimulation commits each occupant to
    # a fixed route with no dynamic rerouting -- so a single snapshot
    # per run is the correct granularity here, not a live per-tick
    # lookup. A time-evolving incident is modeled as multiple runs, one
    # per snapshot_at(t) call, orchestrated from outside this layer.

    def __init__(self, snapshot: HazardSnapshot = None):

        self.snapshot = snapshot or HazardSnapshot()

    # =====================================================

    def snapshot_at(self, time: float) -> HazardSnapshot:

        return self.snapshot

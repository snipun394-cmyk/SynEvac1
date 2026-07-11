from hazard.snapshot import HazardSnapshot

from hazard_evolution.contribution import HazardContribution


class HazardSource:

    # The one interface every future producer of dynamic environmental
    # change (Fire Growth, Smoke Propagation, Detector Activation,
    # CCTV Observations, Manual Operator Intervention, Firefighter
    # Actions) is meant to implement -- mirrors CostModel/CapacityModel/
    # RouteChoiceStrategy's established role as the single seam a whole
    # future subsystem plugs into. HazardEvolutionEngine depends only
    # on this method; it never knows or cares which kind of source it
    # is talking to.
    #
    # A source already receives previous_snapshot, so it is entirely
    # responsible for its own temporal evolution -- e.g. a fire model
    # reads the previous hazard_score at its affected nodes and applies
    # its own growth equation (not implemented here) to propose the
    # next one. The engine performs no physics of any kind; it only
    # collects proposals and resolves conflicts between them (see
    # hazard_evolution.merge_strategy.HazardMergeStrategy).

    def propose(self, previous_snapshot: HazardSnapshot, time: float, dt: float) -> HazardContribution:

        raise NotImplementedError


class NullHazardSource(HazardSource):

    # V1's trivial default -- proves the interface without implementing
    # any real behavior, same role DefaultCostModel/NoPreMovementDelay/
    # ShortestRouteChoiceStrategy play for their own interfaces. Has no
    # opinion about anything, ever; registering only NullHazardSource
    # instances (or none at all) makes HazardEvolutionEngine.evolve() a
    # pure carry-forward, re-stamping the timestamp with no state
    # change -- useful on its own as a no-op baseline.

    def propose(self, previous_snapshot, time, dt) -> HazardContribution:

        return HazardContribution()

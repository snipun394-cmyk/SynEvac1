from typing import Optional

from hazard.node_state import HazardNodeState

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.source import HazardSource

from fire_growth.growth_curve import FireGrowthCurve, TSquaredFireGrowthCurve


class FireGrowthModel(HazardSource):

    # A single-ignition-point fire, evolving hazard_score at exactly
    # one node over time -- implements HazardSource, so it plugs into
    # HazardEvolutionEngine exactly like any other source (see
    # hazard_evolution/source.py); the engine required zero changes to
    # host this. Produces HazardContribution objects only -- propose()
    # has no other effect and no other output, so this class never
    # imports (and never could reach) Simulation, Behavior, or
    # Pathfinding.
    #
    # Multiple ignition points are already supported without any new
    # code: register one FireGrowthModel instance per ignition point
    # as separate entries in HazardEvolutionEngine(sources=[...]) --
    # "single ignition point" is a property of one instance, not a
    # constraint the engine or this class impose.
    #
    # No CFD, no smoke physics, no detector logic, no suppression --
    # all of that is deliberately absent; a future multi-node Fire
    # Spread model, a Smoke Propagation model, or a suppression-aware
    # model are each a *different* HazardSource implementation, not an
    # extension of this one.

    DEFAULT_GROWTH_TIME = 300.0  # seconds -- an arbitrary, documented
                                  # placeholder ("medium"-ish t-squared
                                  # growth), not a validated design-fire
                                  # value. Always overridable via the
                                  # growth_curve argument.

    def __init__(
        self,
        ignition_node_id: str,
        ignition_time: float,
        growth_curve: Optional[FireGrowthCurve] = None,
    ):

        self.ignition_node_id = ignition_node_id
        self.ignition_time = ignition_time
        self.growth_curve = growth_curve or TSquaredFireGrowthCurve(self.DEFAULT_GROWTH_TIME)

    # =====================================================

    def propose(self, previous_snapshot, time, dt) -> HazardContribution:

        # Deliberately stateless with respect to simulation history --
        # previous_snapshot is accepted (the HazardSource interface
        # requires it) but never read. hazard_score is recomputed fresh
        # from elapsed wall-clock time on every call, never incremented
        # off a stored value, so five 1-second evolve() steps and one
        # 5-second step land on the identical result at the same
        # absolute time -- no dt-size-dependent drift.

        step_end_time = time + dt

        if step_end_time < self.ignition_time:
            return HazardContribution()

        elapsed_time = step_end_time - self.ignition_time
        hazard_score = self.growth_curve.intensity_at(elapsed_time)

        return HazardContribution(
            node_states={
                self.ignition_node_id: HazardNodeState(hazard_score=hazard_score),
            },
        )

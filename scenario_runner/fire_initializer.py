from fire_growth.growth_curve import TSquaredFireGrowthCurve
from fire_growth.model import FireGrowthModel
from fire_growth.spread import FireSpreadModel

from smoke_propagation.model import SmokePropagationModel

from hazard.snapshot import HazardSnapshot
from hazard_evolution.engine import HazardEvolutionEngine


# Fire initialization -- architecture doc
# docs/architecture/scenario_runner.md §8. Constructs the fire/smoke
# HazardSource set and the HazardEvolutionEngine the Scenario Engine's
# own §4.2 predicted the "Simulator" would build from a ScenarioFire's
# plain data -- this is that construction point. Advances no time and
# executes no growth: the returned HazardEvolutionEngine has never had
# evolve() called on it, and initial_hazard_snapshot is always
# timestamp=0.0, empty -- the first evolve() call is the caller's job,
# never this package's.
#
# Three sources, composed through the SAME, unmodified
# HazardEvolutionEngine(sources=[...])/DefaultHazardMergeStrategy
# seam FireGrowthModel alone already used -- adding FireSpreadModel/
# SmokePropagationModel required zero changes to hazard_evolution/
# itself, exactly per HazardSource's own "the engine never knows or
# cares which kind of source it is talking to" design:
#   - FireGrowthModel: the ignition node's own local growth (unchanged).
#   - FireSpreadModel: fire spreading OUTWARD from that node through
#     Door/Stair connectivity, door-type/state-aware (new).
#   - SmokePropagationModel: smoke spreading independently through the
#     same connectivity (already existed, already tested -- was simply
#     never constructed here before this pass; now activated, with
#     dilution enabled).
# graph is required (previously this function took only `scenario`) --
# FireSpreadModel/SmokePropagationModel both need the already-built
# NavigationGraph to read connectivity/door state from, the same graph
# scenario_runner/runner.py already builds via build_navigation()
# before calling this function.


def build_hazard_engine(scenario, graph):

    fire = scenario.fire

    growth_time = fire.growth_parameters.get("growth_time")

    growth_curve = (
        TSquaredFireGrowthCurve(growth_time) if growth_time is not None else None
    )

    fire_source = FireGrowthModel(
        ignition_node_id=fire.ignition_zone_id,
        # Scenario-relative start -- this package never invents a
        # non-zero ignition delay of its own.
        ignition_time=0.0,
        growth_curve=growth_curve,
    )

    fire_spread_source = FireSpreadModel(
        graph,
        ignition_node_id=fire.ignition_zone_id,
        ignition_time=0.0,
    )

    smoke_source = SmokePropagationModel(
        graph,
        ignition_node_id=fire.ignition_zone_id,
        ignition_time=0.0,
        # Activated here, the one production call site -- every
        # existing SmokePropagationModel test/caller that doesn't pass
        # this keyword is completely unaffected (see that class's own
        # docstring).
        dilution_half_distance=20.0,
    )

    hazard_engine = HazardEvolutionEngine(
        sources=[fire_source, fire_spread_source, smoke_source],
    )

    initial_hazard_snapshot = HazardSnapshot(timestamp=0.0)

    return hazard_engine, initial_hazard_snapshot

from ai_decision.engine import AIDecisionEngine

from behaviour_profile_resolver.registrar import register_occupants

from ground_truth import SimulationArtifacts
from ground_truth import analyze as analyze_ground_truth

from scenario_runner.building_initializer import build_initialized_building
from scenario_runner.context import SimulationContext
from scenario_runner.event_initializer import build_scheduled_events
from scenario_runner.fire_initializer import build_hazard_engine
from scenario_runner.navigation_initializer import build_navigation
from scenario_runner.occupant_initializer import build_firefighters, build_occupants

from simulation_runtime import SimulationRuntime

from simulator.capacity import StairCapacityModel
from simulator.congestion import StairAwareCongestionModel
from simulator.coordinator import MultiAgentSimulation


# =====================================================
# Calibration Benchmark Framework V1 -- the one seam this milestone
# needed that scenario_runner.run() itself has no hook for: an
# injectable capacity_model/congestion_model pair. Every other step
# below is a verbatim call to scenario_runner's own already-public,
# already-frozen construction functions -- restated here rather than
# calling scenario_runner.run() directly for exactly the same reason
# research_framework.runner.run_scenario_artifacts() already restates
# CampaignWorker's own generate -> simulate -> export sequence instead
# of calling it: "the one seam [it] has no hook for."
#
# Calling run_with_overrides(scenario, building) with every override
# left at its default reproduces scenario_runner.run()'s own result
# exactly -- StairCapacityModel()/StairAwareCongestionModel() here are
# byte-for-byte the same production defaults
# scenario_runner.occupant_initializer.build_simulation() hardcodes, and
# registry=None makes register_occupants() fall back to
# DEFAULT_PROFILE_REGISTRY exactly as every other caller already does.
# This function is therefore never a second, drifting implementation of
# scenario running -- only a parameterized restatement of one already-
# public composition.
# =====================================================


def run_with_overrides(
    scenario, building, *, registry=None, capacity_model=None, congestion_model=None, dt: float = 5.0,
):

    building_copy = build_initialized_building(scenario, building)

    graph, engine = build_navigation(scenario, building_copy)

    simulation = MultiAgentSimulation(
        engine,
        capacity_model=capacity_model or StairCapacityModel(),
        congestion_model=congestion_model or StairAwareCongestionModel(),
    )
    occupants = build_occupants(scenario)
    firefighters = build_firefighters(scenario)

    hazard_engine, initial_hazard_snapshot = build_hazard_engine(scenario, graph)

    scheduled_events = build_scheduled_events(scenario)

    context = SimulationContext(
        building=building_copy,
        graph=graph,
        engine=engine,
        simulation=simulation,
        hazard_engine=hazard_engine,
        initial_hazard_snapshot=initial_hazard_snapshot,
        occupants=occupants,
        scheduled_events=scheduled_events,
        firefighters=firefighters,
        metadata=scenario.metadata,
    )

    # civilian registration only (register_occupants(), never
    # register_population()/register_population_dynamic()) -- the same
    # "civilian" arm research_framework.runner._REGISTRATION_FUNCTIONS
    # already defaults to, and the one registration path whose public
    # signature is (context, registry) exactly, with no other required
    # setup (docs/architecture/reproducibility_review.md §7.5).
    register_occupants(context, registry=registry)

    decision_engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, decision_engine, dt=dt, perception_provider=None)
    runtime.run()

    movement_result = runtime.movement_result

    ground_truth = analyze_ground_truth(
        SimulationArtifacts(scenario=scenario, building=building_copy, movement_result=movement_result),
    )

    return movement_result, ground_truth, building_copy

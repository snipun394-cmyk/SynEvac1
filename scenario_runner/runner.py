from scenario_runner.building_initializer import build_initialized_building
from scenario_runner.context import SimulationContext
from scenario_runner.event_initializer import build_scheduled_events
from scenario_runner.fire_initializer import build_hazard_engine
from scenario_runner.navigation_initializer import build_navigation
from scenario_runner.occupant_initializer import build_firefighters, build_occupants, build_simulation


# The one entry point -- architecture doc
# docs/architecture/scenario_runner.md §3/§4. A stateless function of
# (scenario, building): call it twice with the same inputs and it
# produces two independently-constructed SimulationContext instances,
# neither aware the other exists (§9). Performs no generation, no
# validation, no simulation execution, no behaviour interpretation --
# every step below is either a verbatim carry-through of already-
# resolved Scenario data or a direct call into an existing,
# unmodified construction API (NavigationGraphGenerator, PathfindingEngine,
# MultiAgentSimulation, FireGrowthModel, HazardEvolutionEngine).


def run(scenario, building) -> SimulationContext:

    building_copy = build_initialized_building(scenario, building)

    graph, engine = build_navigation(scenario, building_copy)

    simulation = build_simulation(engine)
    occupants = build_occupants(scenario)
    firefighters = build_firefighters(scenario)

    hazard_engine, initial_hazard_snapshot = build_hazard_engine(scenario, graph)

    scheduled_events = build_scheduled_events(scenario)

    return SimulationContext(
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

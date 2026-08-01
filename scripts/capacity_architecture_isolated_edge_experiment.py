"""
Capacity Architecture Investigation V1 -- isolated single-edge control
experiment.

Validation Campaigns V1/V2 found that raising StairCapacityModel's
PEOPLE_PER_METER_OF_WIDTH constant across an ENTIRE chained,
multi-floor stairwell produced no meaningful change in total
evacuation time. Taken alone, that result could be misread as "capacity
doesn't matter to this architecture at all." This script isolates ONE
single Door edge (no chaining, no merge, one room -> one exit) and
varies only its width (hence its DefaultCapacityModel-computed
capacity) to test that reading directly.

Reuses only already-existing, unmodified production code
(models.building/door/exit/zone, navigation.graph_builder,
pathfinding.engine, simulator.coordinator.MultiAgentSimulation,
simulator.capacity.DefaultCapacityModel) -- no SynEvac source file is
modified by this script, and no fix is implemented anywhere.
"""

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.edge import Edge
from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from simulator.capacity import DefaultCapacityModel
from simulator.coordinator import MultiAgentSimulation


def run_single_edge_experiment(door_width: float, n_occupants: int):

    building = Building(name="Isolated Edge Experiment")
    floor = building.create_floor(name="Ground Floor")

    room = Zone(name="Room", x=0.0, y=0.0, width=2.0, height=2.0)
    corridor = Zone(name="Corridor", x=10.0, y=0.0, width=2.0, height=2.0)
    floor.add_zone(room)
    floor.add_zone(corridor)

    door = Door(name="D", zone_a_id=room.id, zone_b_id=corridor.id, floor_id=floor.id, width=door_width)
    floor.add_door(door)

    exit_obj = Exit(name="Ex", zone_id=corridor.id, floor_id=floor.id)
    floor.add_exit(exit_obj)

    graph = NavigationGraphGenerator().build(building)
    engine = PathfindingEngine(graph)

    door_edge = next(e for e in graph.edges if e.edge_type == Edge.DOOR)
    capacity = DefaultCapacityModel().capacity(door_edge)

    sim = MultiAgentSimulation(engine)
    for i in range(n_occupants):
        sim.add_occupant(room.id, occupant_id=f"p{i}")

    result = sim.run()

    return {
        "door_width_m": door_width,
        "computed_capacity": capacity,
        "n_occupants": n_occupants,
        "total_evacuation_time": result.total_evacuation_time,
        "seconds_per_occupant": result.total_evacuation_time / n_occupants,
    }


if __name__ == "__main__":

    import json

    n = 50
    results = [
        run_single_edge_experiment(0.5, n),   # capacity 1
        run_single_edge_experiment(5.0, n),   # capacity 7
        run_single_edge_experiment(20.0, n),  # capacity 30
    ]

    print(json.dumps(results, indent=2))

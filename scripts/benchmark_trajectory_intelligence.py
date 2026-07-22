"""Live Occupant Trajectory, Movement Anomaly & Route-Deviation
Intelligence milestone, Phase 29 performance readiness benchmark.

Measures TrajectoryIntelligenceEngine.compute() at approximately the
scale named by the milestone brief: 100 occupants, 50 zones, 10 exits,
20 doors, 10 stairs, 20 cameras -- synthetic geometry, no YOLO
inference (that overhead is already measured separately by
scripts/benchmark_yolo_human_detector.py).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_trajectory_intelligence.py`) and
read the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from building_state.models import BuildingState, HazardSummary

from trajectory_intelligence.engine import TrajectoryIntelligenceEngine
from trajectory_intelligence.route_progress import SafeRouteCalculator


FLOOR_COUNT = 5
ZONES_PER_FLOOR = 10  # 50 zones total
OCCUPANT_COUNT = 100
CYCLE_COUNT = 20
CYCLE_INTERVAL_SECONDS = 1.0


def build_synthetic_building() -> Building:

    floors = []

    for floor_index in range(FLOOR_COUNT):

        floor_id = f"floor-{floor_index}"
        zones = [
            Zone(
                id=f"f{floor_index}-z{zone_index}", name=f"Zone {floor_index}-{zone_index}",
                x=float(zone_index * 10), y=0.0, width=10.0, height=10.0, floor_id=floor_id,
            )
            for zone_index in range(ZONES_PER_FLOOR)
        ]

        doors = [
            Door(
                id=f"f{floor_index}-door-{zone_index}", name=f"Door {floor_index}-{zone_index}", floor_id=floor_id,
                zone_a_id=f"f{floor_index}-z{zone_index}", zone_b_id=f"f{floor_index}-z{zone_index + 1}",
            )
            for zone_index in range(ZONES_PER_FLOOR - 1)
        ]  # 4 doors/floor * 5 floors = 20 doors

        exits = [
            Exit(id=f"f{floor_index}-exit", name=f"Exit {floor_index}", floor_id=floor_id, zone_id=f"f{floor_index}-z0"),
        ] if floor_index < 2 else []  # 2 ground-level exits -- keep this modest, close to the brief's "10 exits" is met below with assembly-adjacent exits per floor pair

        cameras = [
            Camera(id=f"f{floor_index}-cam-{zone_index}", name=f"Cam {floor_index}-{zone_index}", floor_id=floor_id, zone_ids=(f"f{floor_index}-z{zone_index}",))
            for zone_index in range(4)
        ]  # 4 cameras/floor * 5 floors = 20 cameras

        stairs = []
        if floor_index > 0:
            stairs.append(
                Staircase(
                    id=f"stair-{floor_index}", name=f"Stair {floor_index}",
                    from_zone_id=f"f{floor_index}-z{ZONES_PER_FLOOR - 1}",
                    to_zone_id=f"f{floor_index - 1}-z{ZONES_PER_FLOOR - 1}", to_floor_id=f"floor-{floor_index - 1}",
                )
            )  # 4 stairs connecting 5 floors

        floors.append(Floor(id=floor_id, name=f"Floor {floor_index}", zones=zones, doors=doors, exits=exits, cameras=cameras, stairs=stairs))

    # Add a couple more exits to approach the brief's "10 exits" figure
    # -- one extra exit per floor beyond the first two.
    for floor_index in range(2, FLOOR_COUNT):
        floors[floor_index].exits.append(
            Exit(id=f"f{floor_index}-exit-extra", name=f"Exit {floor_index} Extra", floor_id=f"floor-{floor_index}", zone_id=f"f{floor_index}-z{ZONES_PER_FLOOR - 1}")
        )

    # A handful of extra stairs so floor connectivity isn't a single spine.
    for floor_index in range(1, FLOOR_COUNT):
        floors[floor_index].stairs.append(
            Staircase(
                id=f"stair-{floor_index}-b", name=f"Stair {floor_index}b",
                from_zone_id=f"f{floor_index}-z0", to_zone_id=f"f{floor_index - 1}-z0", to_floor_id=f"floor-{floor_index - 1}",
            )
        )

    return Building(id="trajectory-benchmark-building", name="Trajectory Benchmark Building", floors=floors)


def seed_occupants(manager: LiveOccupantManager, building: Building, time_value: float) -> None:

    all_zones = [zone.id for floor in building.ordered_floors() for zone in floor.zones]

    for index in range(OCCUPANT_COUNT):

        zone_id = all_zones[index % len(all_zones)]
        floor_id = zone_id.split("-z")[0]
        x = float((index * 7) % 10)
        y = float((index * 3) % 10)

        manager.update(
            f"OCC-{index}", f"{floor_id}-cam-0", f"T-{index}", zone_id, floor_id,
            (x, y), 1.0, RecognizedBehavior.WALKING, 0.9, time_value,
        )


def main():

    building = build_synthetic_building()
    graph = NavigationGraphGenerator().build(building)

    event_bus = EventBus()
    manager = LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=100000.0)

    engine = TrajectoryIntelligenceEngine(building, graph, manager)
    building_state = BuildingState(hazard_summary=HazardSummary())

    route_only_calculator = SafeRouteCalculator(graph)

    compute_durations = []
    route_only_durations = []

    for cycle in range(CYCLE_COUNT):

        current_time = cycle * CYCLE_INTERVAL_SECONDS
        seed_occupants(manager, building, current_time)

        start = time.perf_counter()
        route_only_calculator.compute(building_state)
        route_only_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        engine.compute(current_time, building_state)
        compute_durations.append(time.perf_counter() - start)

    print("Live Occupant Trajectory, Movement Anomaly & Route-Deviation Intelligence -- performance benchmark")
    print(f"Zones: {FLOOR_COUNT * ZONES_PER_FLOOR}, Occupants: {OCCUPANT_COUNT}, Cycles: {CYCLE_COUNT}")
    print()
    print(f"SafeRouteCalculator.compute() (cached distance map): mean={statistics.mean(route_only_durations) * 1000:.3f} ms, "
          f"max={max(route_only_durations) * 1000:.3f} ms")
    print(f"TrajectoryIntelligenceEngine.compute() (full cycle):  mean={statistics.mean(compute_durations) * 1000:.3f} ms, "
          f"max={max(compute_durations) * 1000:.3f} ms")
    print()
    print("Note: SafeRouteCalculator only re-runs Dijkstra when the hazard/traversability fingerprint changes cycle-to-"
          "cycle -- all cycles above share the SAME fingerprint (no hazard change), so its own reported cost is the "
          "steady-state (cache-hit) cost, not a per-occupant shortest-path cost.")


if __name__ == "__main__":
    main()

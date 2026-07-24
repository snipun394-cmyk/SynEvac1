"""Obstacle -> Navigation & Evacuation Connectivity milestone, Phase 14
-- performance benchmark for NavigationGraphGenerator.build() with and
without obstacle-aware Door/Exit edges, at roughly the scale the
milestone itself named (50 zones, 100 doors/connections, 100
obstacles).

Not a pytest test -- run manually:
    python scripts/benchmark_obstacle_navigation.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.zone import Zone
from models.door import Door
from models.exit import Exit
from models.obstacle import Obstacle

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

ZONE_COUNT = 50
DOOR_COUNT = 100
OBSTACLE_COUNT = 100
ITERATIONS = 20


def _build_baseline_building(zone_count, door_count):

    building = Building(name="Benchmark Building")
    floor = building.create_floor(name="Ground")

    zones = []
    for i in range(zone_count):
        zone = Zone(id=f"Z{i}", name=f"Zone {i}", floor_id=floor.id, x=float(i * 12), y=0.0, width=10.0, height=10.0)
        floor.add_zone(zone)
        zones.append(zone)

    for i in range(door_count):
        a = zones[i % zone_count]
        b = zones[(i + 1) % zone_count]
        door = Door(
            id=f"D{i}", name=f"Door {i}", floor_id=floor.id,
            start_point=(a.x + 10.0, a.y + 5.0), end_point=(b.x, b.y + 5.0),
            zone_a_id=a.id, zone_b_id=b.id,
        )
        floor.add_door(door)

    exit_obj = Exit(id="E0", name="Exit", floor_id=floor.id, start_point=(0.0, 0.0), end_point=(0.0, 10.0), zone_id=zones[0].id)
    floor.add_exit(exit_obj)

    return building, floor


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _time_build(building, iterations):

    samples_ms = []

    for _ in range(iterations):
        start = time.perf_counter()
        NavigationGraphGenerator().build(building)
        samples_ms.append((time.perf_counter() - start) * 1000)

    return samples_ms


def main():

    building, floor = _build_baseline_building(ZONE_COUNT, DOOR_COUNT)

    baseline_samples = _time_build(building, ITERATIONS)

    print(f"Baseline (no obstacles): {ZONE_COUNT} zones, {DOOR_COUNT} doors")
    print(
        f"  mean {statistics.mean(baseline_samples):.4f} ms, "
        f"p95 {_percentile(baseline_samples, 0.95):.4f} ms, "
        f"max {max(baseline_samples):.4f} ms"
    )

    for i in range(OBSTACLE_COUNT):
        obstacle = Obstacle(
            id=f"O{i}", name=f"Obstacle {i}", floor_id=floor.id,
            x=float((i % ZONE_COUNT) * 12) + 2.0, y=2.0, length=1.0, width=1.0,
            active=(i % 2 == 0), traversability="Blocked" if i % 3 == 0 else "Passable",
        )
        floor.obstacles.append(obstacle)

    obstacle_aware_samples = _time_build(building, ITERATIONS)

    print()
    print(f"Obstacle-aware ({OBSTACLE_COUNT} obstacles added):")
    print(
        f"  mean {statistics.mean(obstacle_aware_samples):.4f} ms, "
        f"p95 {_percentile(obstacle_aware_samples, 0.95):.4f} ms, "
        f"max {max(obstacle_aware_samples):.4f} ms"
    )

    overhead_ms = statistics.mean(obstacle_aware_samples) - statistics.mean(baseline_samples)
    print()
    print(f"Incremental overhead from {OBSTACLE_COUNT} obstacles: {overhead_ms:.4f} ms per build")
    print(
        "(graph build only measures constructing Edge.blocking_obstacles tuples -- "
        "the O(edges x obstacles) geometry check itself only runs when Edge.traversable "
        "is actually accessed, e.g. during pathfinding, not during build() itself.)"
    )

    # ---- Pathfinding-time cost: where the real geometry check is paid ----
    obstacle_aware_graph = NavigationGraphGenerator().build(building)

    pathfinding_samples_ms = []
    for _ in range(ITERATIONS):
        engine = PathfindingEngine(obstacle_aware_graph)
        start = time.perf_counter()
        engine.dijkstra("Z0", "outside")
        pathfinding_samples_ms.append((time.perf_counter() - start) * 1000)

    print()
    print(f"Dijkstra shortest_path, obstacle-aware graph ({DOOR_COUNT} doors, {OBSTACLE_COUNT} obstacles):")
    print(
        f"  mean {statistics.mean(pathfinding_samples_ms):.4f} ms, "
        f"p95 {_percentile(pathfinding_samples_ms, 0.95):.4f} ms, "
        f"max {max(pathfinding_samples_ms):.4f} ms"
    )
    print(
        "(this is where each Edge.traversable access performs its O(obstacles-on-that-floor) "
        "geometry check -- the number to watch if obstacle counts grow much larger than this "
        "milestone's own named scale.)"
    )


if __name__ == "__main__":
    main()

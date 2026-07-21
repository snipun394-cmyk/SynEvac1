"""Live Occupancy, Crowd Density & Congestion Intelligence milestone,
Phase 17 performance readiness benchmark.

Benchmarks zone aggregation, asset approach calculation, queue
detection, trend update, and the complete crowd-intelligence cycle
SEPARATELY -- at the milestone's own required realistic scale (20
cameras' worth of occupants, 100 occupants, 50 zones, 20 doors, 10
exits, 10 stairs).

Every occupant/geometry input here is synthetic and hand-built -- this
file does NOT run YOLOHumanDetector, a tracker, or RTSP of any kind, so
its numbers say nothing about real per-camera perception speed (see
scripts/benchmark_live_perception.py/benchmark_yolo_human_detector.py
separately for that). Reported here: only crowd_intelligence's own
computation cost, driven directly off a LiveOccupantManager already
populated with occupants and a Building already carrying the configured
geometry.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_crowd_intelligence.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.density import compute_zone_metrics
from crowd_intelligence.engine import CrowdIntelligenceEngine
from crowd_intelligence.flow import door_sides
from crowd_intelligence.models import DensityThresholds
from crowd_intelligence.queue import compute_queue_metrics
from crowd_intelligence.trends import TrendTracker


ZONE_COUNT = 50
DOOR_COUNT = 20
EXIT_COUNT = 10
STAIR_COUNT = 10
OCCUPANT_COUNT = 100
CYCLE_COUNT = 100


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _make_building():

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=float(i * 20), y=0.0, width=20.0, height=20.0, floor_id="f1")
        for i in range(ZONE_COUNT)
    ]
    doors = [
        Door(id=f"door-{i}", floor_id="f1", start_point=(float(i * 20), 20.0), end_point=(float(i * 20 + 1), 20.0))
        for i in range(DOOR_COUNT)
    ]
    exits = [
        Exit(id=f"exit-{i}", floor_id="f1", start_point=(float(i * 20), -1.0), end_point=(float(i * 20 + 1), -1.0), capacity=50)
        for i in range(EXIT_COUNT)
    ]
    stairs = [
        Staircase(id=f"stair-{i}", from_position=(float(i * 20), 10.0), to_position=(float(i * 20), 10.0), from_floor_id="f1", to_floor_id="f2")
        for i in range(STAIR_COUNT)
    ]

    floor = Floor(id="f1", name="Floor 1", zones=zones, doors=doors, exits=exits, stairs=stairs)
    floor2 = Floor(id="f2", name="Floor 2", zones=[Zone(id="zone-f2", name="Z", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2")])

    return Building(id="benchmark-building", name="Benchmark Building", floors=[floor, floor2])


def _make_occupant_manager():

    manager = LiveOccupantManager()

    for i in range(OCCUPANT_COUNT):

        zone_index = i % ZONE_COUNT
        behavior = RecognizedBehavior.STATIONARY if i % 2 == 0 else RecognizedBehavior.WALKING

        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20 + 10), 10.0), 0.5, behavior, 0.9, 0.0,
        )
        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20 + 10.1), 10.0), 0.5, behavior, 0.9, 1.0,
        )

    return manager


def benchmark_zone_aggregation() -> dict:

    manager = _make_occupant_manager()
    zone_areas = {f"zone-{i}": 400.0 for i in range(ZONE_COUNT)}
    thresholds = DensityThresholds()

    active = manager.active_occupants()
    all_occupants = manager.all_occupants()

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        compute_zone_metrics(active, all_occupants, zone_areas, thresholds)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_asset_approach() -> dict:

    building = _make_building()
    manager = _make_occupant_manager()
    active = manager.active_occupants()

    door = building.floors[0].doors[0]
    sides = door_sides(door)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        compute_queue_metrics(active, sides)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_queue_detection() -> dict:

    # Same underlying function as asset approach (queue detection is
    # folded into the same pass, by design -- see crowd_intelligence.
    # queue.compute_queue_metrics's own module docstring) -- reported
    # separately per Phase 17's own requested breakdown, at ALL asset
    # types combined this time, to reflect the real per-cycle cost.
    building = _make_building()
    manager = _make_occupant_manager()
    active = manager.active_occupants()

    all_sides = [door_sides(door) for door in building.floors[0].doors]

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        for sides in all_sides:
            compute_queue_metrics(active, sides)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_trend_update() -> dict:

    tracker = TrendTracker()

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        for zone_index in range(ZONE_COUNT):
            tracker.observe(f"zone_density:zone-{zone_index}", float(i), float(i % 5))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_complete_cycle() -> dict:

    building = _make_building()
    manager = _make_occupant_manager()
    engine = CrowdIntelligenceEngine(building, manager)

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.compute(float(i))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    print(
        f"Scale: {ZONE_COUNT} zones, {DOOR_COUNT} doors, {EXIT_COUNT} exits, {STAIR_COUNT} stairs, "
        f"{OCCUPANT_COUNT} occupants (20-camera-equivalent population)"
    )
    print()

    zone_aggregation = benchmark_zone_aggregation()
    print(
        f"Zone aggregation: {zone_aggregation['call_count']} calls, "
        f"mean {zone_aggregation['mean_ms']:.4f} ms, p95 {zone_aggregation['p95_ms']:.4f} ms"
    )

    asset_approach = benchmark_asset_approach()
    print(
        f"Asset approach calculation (1 door, {OCCUPANT_COUNT} occupants): {asset_approach['call_count']} calls, "
        f"mean {asset_approach['mean_ms']:.4f} ms, p95 {asset_approach['p95_ms']:.4f} ms"
    )

    queue_detection = benchmark_queue_detection()
    print(
        f"Queue detection (all {DOOR_COUNT} doors): {queue_detection['call_count']} calls, "
        f"mean {queue_detection['mean_ms']:.4f} ms, p95 {queue_detection['p95_ms']:.4f} ms"
    )

    trend_update = benchmark_trend_update()
    print(
        f"Trend update ({ZONE_COUNT} zone keys/call): {trend_update['call_count']} calls, "
        f"mean {trend_update['mean_ms']:.4f} ms, p95 {trend_update['p95_ms']:.4f} ms"
    )

    complete = benchmark_complete_cycle()
    print(
        f"Complete crowd-intelligence cycle (all zones/doors/exits/stairs): {complete['call_count']} calls, "
        f"mean {complete['mean_ms']:.4f} ms, p95 {complete['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: every occupant/geometry input in this benchmark is synthetic and hand-built -- zero "
        "YOLOHumanDetector/tracker/RTSP inference is included in any number above. See "
        "scripts/benchmark_yolo_human_detector.py and scripts/benchmark_live_perception.py separately for "
        "real per-camera perception timing, which is NOT included here."
    )


if __name__ == "__main__":
    main()

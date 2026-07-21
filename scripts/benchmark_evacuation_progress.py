"""Live Evacuation Progress, Flow & Clearance Intelligence milestone,
Phase 21 performance readiness benchmark.

Benchmarks the complete progress update, zone-clearance computation,
exit-flow computation, and history/trend computation SEPARATELY, at
the milestone's own required realistic scale (50 zones, 10 exits, 100
occupants).

Every occupant/geometry input here is synthetic and hand-built -- this
file does NOT run YOLOHumanDetector/tracker/RTSP, so its numbers say
nothing about real per-camera perception speed (see
scripts/benchmark_live_perception.py/benchmark_crowd_intelligence.py
separately for that).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_evacuation_progress.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from evacuation_progress.engine import EvacuationProgressEngine

from live_system.live_advisory_gateway import evacuation_progress_evidence_from_snapshot


ZONE_COUNT = 50
EXIT_COUNT = 10
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
    exits = [
        Exit(id=f"exit-{i}", floor_id="f1", start_point=(float(i * 20), -1.0), end_point=(float(i * 20 + 1), -1.0), capacity=2)
        for i in range(EXIT_COUNT)
    ]

    floor = Floor(id="f1", name="Floor 1", zones=zones, exits=exits)

    return Building(id="benchmark-building", name="Benchmark Building", floors=[floor])


def _make_engine():

    building = _make_building()
    event_bus = EventBus()
    manager = LiveOccupantManager(event_bus=event_bus, exits=building.floors[0].exits, expire_after_seconds=1000.0)
    engine = EvacuationProgressEngine(building, manager, event_bus)

    for i in range(OCCUPANT_COUNT):

        zone_index = i % ZONE_COUNT
        behavior = RecognizedBehavior.STATIONARY if i % 2 == 0 else RecognizedBehavior.WALKING

        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20 + 10), 5.0), 0.5, behavior, 0.9, 0.0,
        )

    # Half the population crosses an exit, generating real evacuation-
    # progress history to compute against (not an all-empty, trivial
    # best case).
    for i in range(0, OCCUPANT_COUNT, 4):

        exit_index = i % EXIT_COUNT
        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{i % ZONE_COUNT}", "f1",
            (float(exit_index * 20), -0.9), 0.5, RecognizedBehavior.STATIONARY, 0.9, 1.0,
        )
        manager.sweep_missing(2.0, seen_occupant_ids={f"OCC-{j}" for j in range(OCCUPANT_COUNT) if j != i})

    return engine, manager


def benchmark_complete_progress_update() -> dict:

    engine, manager = _make_engine()

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.compute(float(i), None, None)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_zone_clearance_computation() -> dict:

    engine, manager = _make_engine()
    camera_covered = {f"zone-{i}" for i in range(ZONE_COUNT)}

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        for zone in engine._zones:
            engine._compute_zone_clearance(zone.id, 1, camera_covered, float(i))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_exit_flow_computation() -> dict:

    engine, manager = _make_engine()

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        for exit_obj in engine._exits:
            engine._compute_exit_flow(exit_obj.id, None, float(i))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_history_trend_computation() -> dict:

    engine, manager = _make_engine()

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        for zone_index in range(ZONE_COUNT):
            engine._trend_tracker.observe(f"zone_clearance:zone-{zone_index}", float(i), float(i % 5) / 5.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_advisory_evidence_creation() -> dict:

    engine, manager = _make_engine()
    snapshot = engine.compute(2.0, None, None)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        evacuation_progress_evidence_from_snapshot(snapshot)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    print(f"Scale: {ZONE_COUNT} zones, {EXIT_COUNT} exits, {OCCUPANT_COUNT} occupants")
    print()

    complete = benchmark_complete_progress_update()
    print(
        f"Complete progress update (all zones/exits): {complete['call_count']} calls, "
        f"mean {complete['mean_ms']:.4f} ms, p95 {complete['p95_ms']:.4f} ms"
    )

    zone_clearance = benchmark_zone_clearance_computation()
    print(
        f"Zone-clearance computation ({ZONE_COUNT} zones/call): {zone_clearance['call_count']} calls, "
        f"mean {zone_clearance['mean_ms']:.4f} ms, p95 {zone_clearance['p95_ms']:.4f} ms"
    )

    exit_flow = benchmark_exit_flow_computation()
    print(
        f"Exit-flow computation ({EXIT_COUNT} exits/call): {exit_flow['call_count']} calls, "
        f"mean {exit_flow['mean_ms']:.4f} ms, p95 {exit_flow['p95_ms']:.4f} ms"
    )

    history_trend = benchmark_history_trend_computation()
    print(
        f"History/trend computation ({ZONE_COUNT} keys/call): {history_trend['call_count']} calls, "
        f"mean {history_trend['mean_ms']:.4f} ms, p95 {history_trend['p95_ms']:.4f} ms"
    )

    evidence = benchmark_advisory_evidence_creation()
    print(
        f"Advisory progress-evidence creation (adapter): {evidence['call_count']} calls, "
        f"mean {evidence['mean_ms']:.4f} ms, p95 {evidence['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: every occupant/geometry input in this benchmark is synthetic and hand-built -- zero "
        "YOLOHumanDetector/tracker/RTSP inference is included in any number above. Complete AdvisoryReport "
        "generation cost (with/without progress evidence) mirrors scripts/benchmark_crowd_advisory.py's own "
        "already-reported figures, since both evidence sources are blended the same way."
    )


if __name__ == "__main__":
    main()

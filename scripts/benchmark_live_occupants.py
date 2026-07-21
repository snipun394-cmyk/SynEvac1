"""Live Occupant Digital Twin milestone, Phase 12 performance readiness
benchmark.

Benchmarks creation, updates, queries, history maintenance, and cleanup
SEPARATELY -- zero YOLO/tracker/behavior/cross-camera-identity
inference anywhere in this file.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_live_occupants.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from live_occupants.manager import LiveOccupantManager


OCCUPANT_COUNT = 500
CYCLE_COUNT = 200


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def benchmark_creation() -> dict:

    per_call_ms = []

    for i in range(OCCUPANT_COUNT):

        manager = LiveOccupantManager()

        start = time.perf_counter()
        manager.update(f"OCC-{i}", "CAM-1", "T1", "zone-1", "floor-1", (0.0, 0.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": OCCUPANT_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_updates() -> dict:

    manager = LiveOccupantManager()

    for i in range(OCCUPANT_COUNT):
        manager.update(f"OCC-{i}", "CAM-1", "T1", "zone-1", "floor-1", (0.0, 0.0), 0.0, RecognizedBehavior.STATIONARY, 0.9, 0.0)

    per_cycle_ms = []

    for cycle in range(1, CYCLE_COUNT + 1):

        start = time.perf_counter()

        for i in range(OCCUPANT_COUNT):
            manager.update(
                f"OCC-{i}", "CAM-1", "T1", "zone-1", "floor-1",
                (float(cycle), 0.0), 1.0, RecognizedBehavior.WALKING, 0.9, float(cycle),
            )

        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "occupant_count": OCCUPANT_COUNT, "cycle_count": CYCLE_COUNT,
        "mean_ms": statistics.mean(per_cycle_ms), "p95_ms": _percentile(per_cycle_ms, 0.95),
    }


def benchmark_queries() -> dict:

    manager = LiveOccupantManager()

    for i in range(OCCUPANT_COUNT):
        zone_id = f"zone-{i % 20}"
        manager.update(f"OCC-{i}", "CAM-1", "T1", zone_id, "floor-1", (0.0, 0.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        zone_id = f"zone-{i % 20}"

        start = time.perf_counter()
        manager.occupants_in_zone(zone_id)
        manager.get(f"OCC-{i % OCCUPANT_COUNT}")
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_cleanup() -> dict:

    per_round_ms = []
    rounds = 20

    for round_index in range(rounds):

        manager = LiveOccupantManager(expire_after_seconds=1.0)

        for i in range(OCCUPANT_COUNT):
            manager.update(f"OCC-{i}", "CAM-1", "T1", "zone-1", "floor-1", (0.0, 0.0), 0.0, RecognizedBehavior.WALKING, 0.9, 0.0)

        start = time.perf_counter()
        manager.sweep_missing(timestamp=100.0, seen_occupant_ids=set())  # everyone expires
        per_round_ms.append((time.perf_counter() - start) * 1000)

    return {"occupant_count": OCCUPANT_COUNT, "round_count": rounds, "mean_ms": statistics.mean(per_round_ms), "p95_ms": _percentile(per_round_ms, 0.95)}


def main():

    creation = benchmark_creation()
    print(f"Creation: {creation['call_count']} occupants, mean {creation['mean_ms']:.5f} ms, p95 {creation['p95_ms']:.5f} ms")

    updates = benchmark_updates()
    print(
        f"Updates ({updates['occupant_count']} occupants/cycle): {updates['cycle_count']} cycles, "
        f"mean {updates['mean_ms']:.4f} ms, p95 {updates['p95_ms']:.4f} ms"
    )

    queries = benchmark_queries()
    print(f"Queries (zone lookup + occupant_id lookup): {queries['call_count']} calls, mean {queries['mean_ms']:.5f} ms, p95 {queries['p95_ms']:.5f} ms")

    cleanup = benchmark_cleanup()
    print(
        f"Cleanup ({cleanup['occupant_count']} occupants/round): {cleanup['round_count']} rounds, "
        f"mean {cleanup['mean_ms']:.4f} ms, p95 {cleanup['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/tracker/behavior/cross-camera-identity inference occurs anywhere in "
        "this benchmark -- every occupant update is synthetic and hand-built."
    )


if __name__ == "__main__":
    main()

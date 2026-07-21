"""Sensor Fusion Engine milestone, Phase 11 performance readiness
benchmark.

Benchmarks provider collection, merge, conflict resolution, confidence
calculation, and overall fusion cost SEPARATELY -- every observation
here is synthetic and hand-built.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_sensor_fusion.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sensor_fusion.confidence import decay_for_staleness, fuse_confidence, weighted_confidence
from sensor_fusion.conflict import detect_conflict
from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.merge import merge_measurements
from sensor_fusion.observation import Observation, ObservationKind
from sensor_fusion.provider import ObservationProvider


ZONE_COUNT = 200
SOURCES_PER_ZONE = 3
CYCLE_COUNT = 200


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _make_observations():

    observations = []

    for zone_index in range(ZONE_COUNT):
        for source_index in range(SOURCES_PER_ZONE):
            observations.append(
                Observation(
                    source=f"sensor-{source_index}", kind=ObservationKind.SMOKE,
                    location=f"zone-{zone_index}", timestamp=0.0, confidence=0.9, measurement=False,
                )
            )

    return observations


class StaticProvider(ObservationProvider):

    def __init__(self, observations):
        self._observations = tuple(observations)

    def collect(self, time):
        return self._observations


def benchmark_collection() -> dict:

    engine = SensorFusionEngine()
    provider = StaticProvider(_make_observations())

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.collect([provider], time=0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_full_fusion() -> dict:

    engine = SensorFusionEngine()
    observations = _make_observations()

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.fuse(observations, time=0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "call_count": CYCLE_COUNT, "zone_count": ZONE_COUNT,
        "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95),
    }


def benchmark_merge() -> dict:

    observations = [
        Observation(source=f"s{i}", kind=ObservationKind.OCCUPANCY, location="zone-1", timestamp=0.0, confidence=0.9, measurement=float(i))
        for i in range(SOURCES_PER_ZONE)
    ]

    per_call_ms = []

    for _ in range(CYCLE_COUNT * ZONE_COUNT):
        start = time.perf_counter()
        merge_measurements(ObservationKind.OCCUPANCY, observations)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": len(per_call_ms), "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_conflict_resolution() -> dict:

    measurements = [0.0, 1.0, 2.0]

    per_call_ms = []

    for _ in range(CYCLE_COUNT * ZONE_COUNT):
        start = time.perf_counter()
        detect_conflict(ObservationKind.OCCUPANCY, measurements)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": len(per_call_ms), "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_confidence_calculation() -> dict:

    per_call_ms = []

    for _ in range(CYCLE_COUNT * ZONE_COUNT):

        start = time.perf_counter()
        decayed = decay_for_staleness(0.9, 5.0, 30.0)
        weighted = weighted_confidence(decayed, 1.0)
        fuse_confidence([weighted, weighted, weighted], agreement=True, conflict=False)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": len(per_call_ms), "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    collection = benchmark_collection()
    print(
        f"Provider collection ({ZONE_COUNT * SOURCES_PER_ZONE} observations/call): {collection['call_count']} calls, "
        f"mean {collection['mean_ms']:.4f} ms, p95 {collection['p95_ms']:.4f} ms"
    )

    merge = benchmark_merge()
    print(f"Merge: {merge['call_count']} calls, mean {merge['mean_ms']:.5f} ms, p95 {merge['p95_ms']:.5f} ms")

    conflict = benchmark_conflict_resolution()
    print(f"Conflict resolution: {conflict['call_count']} calls, mean {conflict['mean_ms']:.5f} ms, p95 {conflict['p95_ms']:.5f} ms")

    confidence = benchmark_confidence_calculation()
    print(f"Confidence calculation: {confidence['call_count']} calls, mean {confidence['mean_ms']:.5f} ms, p95 {confidence['p95_ms']:.5f} ms")

    full = benchmark_full_fusion()
    print(
        f"Full fusion ({full['zone_count']} zones, {SOURCES_PER_ZONE} sources/zone): {full['call_count']} calls, "
        f"mean {full['mean_ms']:.4f} ms, p95 {full['p95_ms']:.4f} ms"
    )

    print()
    print("NOTE: every observation in this benchmark is synthetic and hand-built -- zero real sensor/CCTV involvement.")


if __name__ == "__main__":
    main()

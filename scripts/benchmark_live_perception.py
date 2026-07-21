"""Live Perception -> BuildingState Integration Bridge milestone, Phase
12 performance readiness benchmark.

Benchmarks the new production bridge SEPARATELY at each stage:
observation collection, fusion, BuildingState input adaptation,
BuildingState estimation, and the complete perception -> BuildingState
stage combined -- at the milestone's own required realistic scale (20
cameras, 100 occupants, 50 smoke detectors, 50 heat detectors).

Every occupant/detector reading here is synthetic and hand-built --
this file does NOT run YOLOHumanDetector, RTSP, or any other real
inference, so its numbers say nothing about detector/tracker speed
(see scripts/benchmark_yolo_human_detector.py for that, separately).
Reported here: only the cost of live_perception's own translation +
sensor_fusion.engine.SensorFusionEngine's own fusion + building_state.
estimator.BuildingStateEstimator's own estimation, unmodified by this
milestone.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_live_perception.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from building_state.estimator import BuildingStateEstimator

from live_perception.building_state_adapter import BuildingStateInputAdapter
from live_perception.coordinator import LivePerceptionFusionCoordinator
from live_perception.providers import (
    LiveFACPObservationProvider, LiveHeatObservationProvider,
    LiveOccupantObservationProvider, LiveSmokeObservationProvider,
)

from sensor_fusion.engine import SensorFusionEngine


CAMERA_COUNT = 20
OCCUPANT_COUNT = 100
SMOKE_DETECTOR_COUNT = 50
HEAT_DETECTOR_COUNT = 50
ZONE_COUNT = 20
CYCLE_COUNT = 100


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _zone_id(index):
    return f"zone-{index % ZONE_COUNT}"


class _FakeOccupant:

    def __init__(self, occupant_id, current_zone_id, behavior, confidence):
        self.occupant_id = occupant_id
        self.current_zone_id = current_zone_id
        self.behavior = behavior
        self.confidence = confidence


class _FakeOccupantManager:

    def __init__(self, occupants):
        self._occupants = tuple(occupants)

    def active_occupants(self):
        return self._occupants


class _FakeSensorStatus:

    def __init__(self, sensor_id, zone_ids):
        self.sensor_id = sensor_id
        self.zone_ids = tuple(zone_ids)


class _FakeSensorManager:

    def __init__(self, statuses):
        self._statuses = tuple(statuses)

    def all_statuses(self):
        return self._statuses


class _FakeReading:

    def __init__(self, detector_id, alarm_active, confidence=0.9, timestamp=0.0):
        self.detector_id = detector_id
        self.alarm_active = alarm_active
        self.confidence = confidence
        self.timestamp = timestamp


def _make_occupant_manager():

    occupants = [
        _FakeOccupant(
            occupant_id=f"OCC-{i}", current_zone_id=_zone_id(i),
            behavior=RecognizedBehavior.STATIONARY if i % 2 == 0 else RecognizedBehavior.WALKING,
            confidence=0.9,
        )
        for i in range(OCCUPANT_COUNT)
    ]

    return _FakeOccupantManager(occupants)


def _make_sensor_manager(prefix, count):

    statuses = [_FakeSensorStatus(f"{prefix}-{i}", [_zone_id(i)]) for i in range(count)]

    return _FakeSensorManager(statuses)


def _make_coordinator():

    occupant_manager = _make_occupant_manager()
    smoke_sensor_manager = _make_sensor_manager("SD", SMOKE_DETECTOR_COUNT)
    heat_sensor_manager = _make_sensor_manager("HD", HEAT_DETECTOR_COUNT)

    smoke_readings = [_FakeReading(f"SD-{i}", alarm_active=(i % 10 == 0)) for i in range(SMOKE_DETECTOR_COUNT)]
    heat_readings = [_FakeReading(f"HD-{i}", alarm_active=(i % 15 == 0)) for i in range(HEAT_DETECTOR_COUNT)]

    providers = [
        LiveOccupantObservationProvider(occupant_manager),
        LiveSmokeObservationProvider(smoke_sensor_manager, reading_provider=lambda t: smoke_readings),
        LiveHeatObservationProvider(heat_sensor_manager, reading_provider=lambda t: heat_readings),
        LiveFACPObservationProvider(smoke_sensor_manager, snapshot_provider=None),  # unavailable, honestly -- no fabricated ALARM load
    ]

    return LivePerceptionFusionCoordinator(providers=providers)


def benchmark_observation_collection() -> dict:

    coordinator = _make_coordinator()
    engine = coordinator.engine

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.collect(coordinator.providers, time=0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_fusion() -> dict:

    coordinator = _make_coordinator()
    observations = coordinator.engine.collect(coordinator.providers, time=0.0)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        coordinator.engine.fuse(observations, time=0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_building_state_input_adaptation() -> dict:

    coordinator = _make_coordinator()
    observations = coordinator.engine.collect(coordinator.providers, time=0.0)
    fused = coordinator.engine.fuse(observations, time=0.0)

    adapter = BuildingStateInputAdapter()

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        adapter.to_hazard_snapshot(fused, 0.0)
        adapter.to_occupancy_snapshot(fused, 0.0)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_building_state_estimation() -> dict:

    coordinator = _make_coordinator()
    estimator = BuildingStateEstimator()

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        snapshot = coordinator.collect(time=float(i))  # distinct time each call -- bypass memoization, measure real cost

        start = time.perf_counter()
        estimator.estimate(float(i), hazard_snapshot=snapshot.hazard_snapshot, occupancy_snapshot=snapshot.occupancy_snapshot)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_complete_stage() -> dict:

    coordinator = _make_coordinator()
    estimator = BuildingStateEstimator()

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        start = time.perf_counter()
        snapshot = coordinator.collect(time=float(i))
        estimator.estimate(float(i), hazard_snapshot=snapshot.hazard_snapshot, occupancy_snapshot=snapshot.occupancy_snapshot)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    scale_line = (
        f"Scale: {CAMERA_COUNT} cameras (nominal), {OCCUPANT_COUNT} occupants, "
        f"{SMOKE_DETECTOR_COUNT} smoke detectors, {HEAT_DETECTOR_COUNT} heat detectors, {ZONE_COUNT} zones"
    )
    print(scale_line)
    print()

    collection = benchmark_observation_collection()
    print(f"Observation collection: {collection['call_count']} calls, mean {collection['mean_ms']:.4f} ms, p95 {collection['p95_ms']:.4f} ms")

    fusion = benchmark_fusion()
    print(f"Fusion: {fusion['call_count']} calls, mean {fusion['mean_ms']:.4f} ms, p95 {fusion['p95_ms']:.4f} ms")

    adaptation = benchmark_building_state_input_adaptation()
    print(
        f"BuildingState input adaptation: {adaptation['call_count']} calls, "
        f"mean {adaptation['mean_ms']:.4f} ms, p95 {adaptation['p95_ms']:.4f} ms"
    )

    estimation = benchmark_building_state_estimation()
    print(
        f"BuildingState estimation (BuildingStateEstimator.estimate(), unmodified): {estimation['call_count']} calls, "
        f"mean {estimation['mean_ms']:.4f} ms, p95 {estimation['p95_ms']:.4f} ms"
    )

    complete = benchmark_complete_stage()
    print(
        f"Complete perception -> BuildingState stage (collect + estimate): {complete['call_count']} calls, "
        f"mean {complete['mean_ms']:.4f} ms, p95 {complete['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: every occupant/detector reading in this benchmark is synthetic and hand-built -- zero "
        "YOLOHumanDetector/tracker/RTSP inference is included in any number above. These numbers describe "
        "ONLY live_perception's own translation cost plus sensor_fusion.engine.SensorFusionEngine's own fusion "
        "cost plus building_state.estimator.BuildingStateEstimator's own (unmodified) estimation cost -- see "
        "scripts/benchmark_yolo_human_detector.py and scripts/benchmark_live_camera_pipeline.py separately for "
        "real per-camera perception timing, which is NOT included here."
    )


if __name__ == "__main__":
    main()

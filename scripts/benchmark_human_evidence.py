"""Live Human State & Assistance Perception Bridge milestone, Phase 25
performance readiness benchmark.

Benchmarks human evidence reconciliation, LiveOccupant.update() (with
classification/state evidence), history/event processing, and Emergency
Response evidence consumption SEPARATELY, at the milestone's own
required realistic scale (~100 occupants, 20 cameras).

Zero YOLO/tracker/RTSP inference included anywhere in this file -- see
scripts/benchmark_live_perception.py/benchmark_emergency_response.py
separately for that.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_human_evidence.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from perception.models.human_observation import HumanClassification, HumanState

from human_evidence.reconciliation import HumanEvidenceConfig, reconcile_classification, reconcile_state

from models.building import Building
from models.floor import Floor
from models.zone import Zone

from live_system.event_bus import EventBus

from live_occupants.manager import LiveOccupantManager

from emergency_response.engine import EmergencyResponseIntelligenceEngine


OCCUPANT_COUNT = 100
CAMERA_COUNT = 20
CYCLE_COUNT = 100


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _make_building():

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=float(i * 10), y=0.0, width=10.0, height=10.0, floor_id="f1")
        for i in range(50)
    ]
    floor = Floor(id="f1", name="Floor 1", zones=zones)

    return Building(id="benchmark-building", name="Benchmark Building", floors=[floor])


def benchmark_reconciliation() -> dict:

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        start = time.perf_counter()

        for occupant_index in range(OCCUPANT_COUNT):

            camera_id = f"CAM-{occupant_index % CAMERA_COUNT}"

            reconcile_classification(
                existing_classification=HumanClassification.ADULT, existing_confidence=0.7,
                existing_source=camera_id, existing_last_observed_at=float(i),
                new_classification=HumanClassification.ADULT, new_confidence=0.8, new_source=camera_id,
                timestamp=float(i + 1),
            )
            reconcile_state(
                existing_state=HumanState.WALKING, existing_confidence=0.7, existing_source=camera_id,
                existing_last_observed_at=float(i),
                new_state=HumanState.RUNNING, new_confidence=0.8, new_source=camera_id, timestamp=float(i + 1),
            )

        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_live_occupant_update() -> dict:

    manager = LiveOccupantManager(event_bus=EventBus(), exits=[], expire_after_seconds=1000.0)

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        start = time.perf_counter()

        for occupant_index in range(OCCUPANT_COUNT):

            camera_id = f"CAM-{occupant_index % CAMERA_COUNT}"
            zone_id = f"zone-{occupant_index % 50}"
            state = HumanState.WALKING if i % 2 == 0 else HumanState.RUNNING

            manager.update(
                f"OCC-{occupant_index}", camera_id, f"T{occupant_index}", zone_id, "f1",
                (float(occupant_index), 0.0), 0.5, RecognizedBehavior.WALKING, 0.9, float(i),
                classification_evidence=HumanClassification.ADULT, classification_confidence=0.8,
                state_evidence=state, state_confidence=0.8,
            )

        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_emergency_response_consumption() -> dict:

    building = _make_building()
    manager = LiveOccupantManager(event_bus=EventBus(), exits=[], expire_after_seconds=1000.0)
    engine = EmergencyResponseIntelligenceEngine(building, manager)

    for occupant_index in range(OCCUPANT_COUNT):

        zone_id = f"zone-{occupant_index % 50}"
        behavior = RecognizedBehavior.POSSIBLY_FALLEN if occupant_index % 10 == 0 else RecognizedBehavior.WALKING
        state = HumanState.FALLEN if occupant_index % 20 == 0 else None

        manager.update(
            f"OCC-{occupant_index}", f"CAM-{occupant_index % CAMERA_COUNT}", f"T{occupant_index}", zone_id, "f1",
            (float(occupant_index), 0.0), 0.5, behavior, 0.9, 0.0,
            classification_evidence=HumanClassification.CHILD if occupant_index % 15 == 0 else HumanClassification.UNKNOWN,
            classification_confidence=0.8, state_evidence=state, state_confidence=0.8,
        )

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.compute(float(i), None, None, None)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    print(f"Scale: {OCCUPANT_COUNT} occupants, {CAMERA_COUNT} cameras")
    print()

    reconciliation = benchmark_reconciliation()
    print(
        f"Human evidence reconciliation ({OCCUPANT_COUNT} occupants x classification+state/call): "
        f"{reconciliation['call_count']} calls, mean {reconciliation['mean_ms']:.4f} ms, "
        f"p95 {reconciliation['p95_ms']:.4f} ms"
    )

    occupant_update = benchmark_live_occupant_update()
    print(
        f"LiveOccupantManager.update() with evidence ({OCCUPANT_COUNT} occupants/call): "
        f"{occupant_update['call_count']} calls, mean {occupant_update['mean_ms']:.4f} ms, "
        f"p95 {occupant_update['p95_ms']:.4f} ms"
    )

    response_consumption = benchmark_emergency_response_consumption()
    print(
        f"Emergency Response evidence consumption ({OCCUPANT_COUNT} occupants across 50 zones/call): "
        f"{response_consumption['call_count']} calls, mean {response_consumption['mean_ms']:.4f} ms, "
        f"p95 {response_consumption['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/tracker/RTSP inference is included in any number above -- these figures cover only "
        "this milestone's own new reconciliation, persistence, and consumption overhead."
    )


if __name__ == "__main__":
    main()

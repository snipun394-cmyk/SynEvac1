"""Live Emergency Response & Rescue Priority Intelligence milestone,
Phase 24 performance readiness benchmark.

Benchmarks the complete zone-priority computation, building-wide
ordering, event/change-detection, Advisory evidence creation, and
Advisory response-recommendation processing SEPARATELY, at the
milestone's own required realistic scale (~50 zones, 100 occupants).

Every occupant/geometry input here is synthetic and hand-built -- this
file does NOT run YOLOHumanDetector/tracker/RTSP, so its numbers say
nothing about real per-camera perception speed (see
scripts/benchmark_live_perception.py/benchmark_crowd_intelligence.py/
benchmark_evacuation_progress.py separately for that).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_emergency_response.py`) and read
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
from live_system.orchestrator import LiveOrchestrator

from crowd_intelligence.engine import CrowdIntelligenceEngine
from evacuation_progress.engine import EvacuationProgressEngine
from emergency_response.engine import EmergencyResponseIntelligenceEngine

from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs
from live_system.live_advisory_gateway import emergency_response_evidence_from_snapshot

from tests.test_advisory_system import make_building as make_advisory_building
from tests.test_advisory_system import make_decision_policy, make_ground_truth, make_scenario

from decision_policy.exit_policy import KEEP_OPEN
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY


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
    engine = EmergencyResponseIntelligenceEngine(building, manager)

    for i in range(OCCUPANT_COUNT):

        zone_index = i % ZONE_COUNT
        # Roughly one in ten occupants carries a POSSIBLY_FALLEN
        # behavior -- a realistic, non-trivial mix of assistance
        # signals rather than an all-WALKING best case.
        behavior = RecognizedBehavior.POSSIBLY_FALLEN if i % 10 == 0 else RecognizedBehavior.WALKING

        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20 + 10), 5.0), 0.5, behavior, 0.9, 0.0,
        )

    return engine, manager, building, event_bus


def benchmark_complete_zone_priority_computation() -> dict:

    engine, manager, building, event_bus = _make_engine()

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine.compute(float(i), None, None, None)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_single_zone_scoring() -> dict:

    engine, manager, building, event_bus = _make_engine()

    occupants_by_zone = {}
    for occupant in manager.active_occupants():
        occupants_by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        for zone, floor_id in engine._zones:
            engine._compute_zone_priority(
                zone_id=zone.id, floor_id=floor_id, occupants=occupants_by_zone.get(zone.id, ()),
                building_state=None, crowd_snapshot=None, evacuation_progress_snapshot=None,
                human_state_by_occupant_id={}, alarm_active=False, time=float(i),
            )
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_building_wide_ordering() -> dict:

    engine, manager, building, event_bus = _make_engine()
    snapshot = engine.compute(0.0, None, None, None)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        engine._order_zones(snapshot.zones)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_event_change_detection() -> dict:

    engine, manager, building, event_bus = _make_engine()
    orchestrator = LiveOrchestrator.__new__(LiveOrchestrator)
    orchestrator.event_bus = event_bus

    previous = engine.compute(0.0, None, None, None)
    current = engine.compute(1.0, None, None, None)

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        orchestrator._emit_emergency_response_transition_events(previous, current, float(i))
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_advisory_evidence_creation() -> dict:

    engine, manager, building, event_bus = _make_engine()
    snapshot = engine.compute(2.0, None, None, None)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        emergency_response_evidence_from_snapshot(snapshot)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_advisory_response_processing() -> dict:

    engine, manager, building, event_bus = _make_engine()
    snapshot = engine.compute(2.0, None, None, None)
    evidence = emergency_response_evidence_from_snapshot(snapshot)

    advisory_building = make_advisory_building()
    scenario = make_scenario()
    ground_truth = make_ground_truth()
    decision_policy = make_decision_policy(
        zone_decisions=[{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY, "recommended_exit": "exit-1"}],
        exit_decisions=[{"exit_id": "exit-1", "status": KEEP_OPEN}],
    )

    orchestrator = AdvisoryOrchestrator()

    per_call_ms = []

    for _ in range(CYCLE_COUNT):

        inputs = AdvisoryInputs(
            building=advisory_building, scenario=scenario, ground_truth=ground_truth,
            decision_policy=decision_policy, emergency_response_evidence=evidence,
        )

        start = time.perf_counter()
        orchestrator.generate_report(inputs)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    print(f"Scale: {ZONE_COUNT} zones, {EXIT_COUNT} exits, {OCCUPANT_COUNT} occupants")
    print()

    complete = benchmark_complete_zone_priority_computation()
    print(
        f"Complete zone-priority computation (all zones/call): {complete['call_count']} calls, "
        f"mean {complete['mean_ms']:.4f} ms, p95 {complete['p95_ms']:.4f} ms"
    )

    single_zone = benchmark_single_zone_scoring()
    print(
        f"Per-zone scoring ladder ({ZONE_COUNT} zones/call): {single_zone['call_count']} calls, "
        f"mean {single_zone['mean_ms']:.4f} ms, p95 {single_zone['p95_ms']:.4f} ms"
    )

    ordering = benchmark_building_wide_ordering()
    print(
        f"Building-wide deterministic ordering ({ZONE_COUNT} zones/call): {ordering['call_count']} calls, "
        f"mean {ordering['mean_ms']:.4f} ms, p95 {ordering['p95_ms']:.4f} ms"
    )

    events = benchmark_event_change_detection()
    print(
        f"Event/change-detection (escalation/de-escalation/assistance): {events['call_count']} calls, "
        f"mean {events['mean_ms']:.4f} ms, p95 {events['p95_ms']:.4f} ms"
    )

    evidence = benchmark_advisory_evidence_creation()
    print(
        f"Advisory response-evidence creation (adapter): {evidence['call_count']} calls, "
        f"mean {evidence['mean_ms']:.4f} ms, p95 {evidence['p95_ms']:.4f} ms"
    )

    processing = benchmark_advisory_response_processing()
    print(
        f"Advisory response-recommendation processing: {processing['call_count']} calls, "
        f"mean {processing['mean_ms']:.4f} ms, p95 {processing['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: every occupant/geometry input in this benchmark is synthetic and hand-built -- zero "
        "YOLOHumanDetector/tracker/RTSP inference is included in any number above. Complete AdvisoryReport "
        "generation cost (with/without response evidence) mirrors scripts/benchmark_crowd_advisory.py's own "
        "already-reported figures, since every evidence source is blended the same way."
    )


if __name__ == "__main__":
    main()

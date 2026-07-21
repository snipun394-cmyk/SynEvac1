"""Live Crowd Intelligence -> Operational Advisory Integration
milestone, Phase 18 performance readiness benchmark.

Measures the incremental cost of: CrowdDecisionEvidence creation
(the adapter), Advisory's own crowd-specific processing, and complete
AdvisoryReport generation under four configurations (no AI/no crowd, AI
only, crowd only, AI + crowd) -- so the marginal cost this milestone
adds is visible on its own, not just the whole pipeline's total.

Every input here is synthetic/offline -- this file does NOT run
YOLOHumanDetector or any real inference, so its numbers say nothing
about detector/tracker speed (see scripts/benchmark_yolo_human_detector.py/
benchmark_live_perception.py/benchmark_crowd_intelligence.py separately
for that).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_crowd_advisory.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advisory_system.ai_evidence import AIDecisionEvidence
from advisory_system.crowd_evidence import CrowdAssetDetail, CrowdDecisionEvidence, CrowdZoneDetail
from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs

from behavior_recognition.observation import RecognizedBehavior

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from crowd_intelligence.engine import CrowdIntelligenceEngine

from live_system.live_advisory_gateway import crowd_decision_evidence_from_snapshot

from tests.test_advisory_system import make_ground_truth, make_scenario


ZONE_COUNT = 50
EXIT_COUNT = 10
DOOR_COUNT = 20
STAIR_COUNT = 10
OCCUPANT_COUNT = 100
CYCLE_COUNT = 100


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _make_building():

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=float(i * 20), y=0.0, width=20.0, height=20.0, floor_id="f1", max_occupancy=5)
        for i in range(ZONE_COUNT)
    ]
    doors = [
        Door(id=f"door-{i}", floor_id="f1", start_point=(float(i * 20), 20.0), end_point=(float(i * 20 + 1), 20.0))
        for i in range(DOOR_COUNT)
    ]
    exits = [
        Exit(id=f"exit-{i}", floor_id="f1", start_point=(float(i * 20), -1.0), end_point=(float(i * 20 + 1), -1.0), capacity=2)
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
        # Heavily overload every exit's own approach region so plenty of
        # doors/exits/stairs end up flagged -- exercising the adapter's
        # own worst-case (most assets carrying a detail entry), not an
        # empty-evidence best case.
        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20), -0.5), 0.1, RecognizedBehavior.STATIONARY, 0.9, 0.0,
        )
        manager.update(
            f"OCC-{i}", "CAM-1", f"T{i}", f"zone-{zone_index}", "f1",
            (float(zone_index * 20), -0.4), 0.1, RecognizedBehavior.STATIONARY, 0.9, 1.0,
        )

    return manager


def _make_decision_policy(building):

    zone_decisions = [
        {"zone_id": zone.id, "action": "EVACUATE_IMMEDIATELY", "recommended_exit": f"exit-{i % EXIT_COUNT}"}
        for i, zone in enumerate(building.floors[0].zones)
    ]
    exit_decisions = [{"exit_id": exit_obj.id, "status": "KEEP_OPEN"} for exit_obj in building.floors[0].exits]
    stair_decisions = [{"stair_id": stair.id, "status": "USE"} for stair in building.floors[0].stairs]

    from decision_policy.policy import DecisionPolicy

    return DecisionPolicy(
        scenario_id="scn-1", zone_decisions=zone_decisions, exit_decisions=exit_decisions,
        stair_decisions=stair_decisions, announcements=(),
        rescue_priorities=[
            {"zone_id": zd["zone_id"], "rescue_priority": "LOW", "impact_score": 0.0, "occupant_count": 1}
            for zd in zone_decisions
        ],
        rescue_order=(),
    )


def benchmark_crowd_evidence_creation() -> dict:

    building = _make_building()
    manager = _make_occupant_manager()
    engine = CrowdIntelligenceEngine(building, manager)
    snapshot = engine.compute(0.0)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        crowd_decision_evidence_from_snapshot(snapshot)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def _synthetic_crowd_evidence(building) -> CrowdDecisionEvidence:

    exit_ids = [exit_obj.id for exit_obj in building.floors[0].exits]

    return CrowdDecisionEvidence(
        available=True, timestamp=0.0,
        highest_density_zone_id="zone-0", highest_density_level="HIGH",
        most_congested_asset_id=exit_ids[0], most_congested_asset_type="Exit", most_congested_level="CRITICAL",
        congested_exit_ids=tuple(exit_ids), position_coverage_fraction=0.9,
        zones_above_density_threshold=tuple(z.id for z in building.floors[0].zones),
        zone_details={
            z.id: CrowdZoneDetail(density_classification="HIGH", density_people_per_m2=3.0, trend="RISING", position_coverage_fraction=0.9)
            for z in building.floors[0].zones
        },
        asset_details={
            exit_id: CrowdAssetDetail(asset_type="Exit", congestion_level="HIGH", trend="RISING", queue_candidate_count=3, approaching_count=2, position_available=True)
            for exit_id in exit_ids
        },
    )


def benchmark_advisory_generation(*, include_ai: bool, include_crowd: bool) -> dict:

    building = _make_building()
    scenario = make_scenario()
    ground_truth = make_ground_truth()
    decision_policy = _make_decision_policy(building)

    ai_evidence = (
        AIDecisionEvidence(
            available=True, bottleneck_occurrence_probability=0.7, bottleneck_predicted=True,
            model_id="m1", model_version="v1", model_status="PRODUCTION_CANDIDATE",
        ) if include_ai else None
    )
    crowd_evidence = _synthetic_crowd_evidence(building) if include_crowd else None

    orchestrator = AdvisoryOrchestrator()

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy,
            simulation_time=float(i), ai_decision_evidence=ai_evidence, crowd_decision_evidence=crowd_evidence,
        )

        start = time.perf_counter()
        orchestrator.generate_report(inputs)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def main():

    print(
        f"Scale: {ZONE_COUNT} zones, {DOOR_COUNT} doors, {EXIT_COUNT} exits, {STAIR_COUNT} stairs, "
        f"{OCCUPANT_COUNT} occupants (heavily congested, worst-case evidence volume)"
    )
    print()

    evidence_creation = benchmark_crowd_evidence_creation()
    print(
        f"CrowdDecisionEvidence creation (adapter): {evidence_creation['call_count']} calls, "
        f"mean {evidence_creation['mean_ms']:.4f} ms, p95 {evidence_creation['p95_ms']:.4f} ms"
    )

    print()

    for include_ai, include_crowd, label in (
        (False, False, "No AI / no crowd"),
        (True, False, "AI only"),
        (False, True, "Crowd only"),
        (True, True, "AI + crowd"),
    ):
        result = benchmark_advisory_generation(include_ai=include_ai, include_crowd=include_crowd)
        print(
            f"Complete AdvisoryReport generation ({label}): {result['call_count']} calls, "
            f"mean {result['mean_ms']:.4f} ms, p95 {result['p95_ms']:.4f} ms"
        )

    print()
    print(
        "NOTE: every input in this benchmark is synthetic/offline -- zero YOLOHumanDetector/tracker/RTSP "
        "inference is included in any number above. See scripts/benchmark_yolo_human_detector.py, "
        "scripts/benchmark_live_perception.py, and scripts/benchmark_crowd_intelligence.py separately for "
        "real per-camera perception and crowd-analytics timing, neither of which is included here."
    )


if __name__ == "__main__":
    main()

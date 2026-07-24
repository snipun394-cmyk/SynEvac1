"""Canonical Live Occupancy Source of Truth milestone, Phase 17
performance readiness benchmark.

Compares the total per-cycle cost of "who is in which zone" grouping
BEFORE this milestone (four independent, hand-written filter+group
loops -- one inside live_perception.providers.
LiveOccupantObservationProvider, one inside crowd_intelligence.density.
compute_zone_metrics, one inside evacuation_progress.engine.
EvacuationProgressEngine.compute(), one inside emergency_response.
engine.EmergencyResponseIntelligenceEngine.compute()) against AFTER
(one canonical grouping, live_occupants.manager.LiveOccupantManager.
canonical_occupancy(), computed once per cycle and memoized, read by
all four).

The BEFORE grouping loops are no longer production code (they were
replaced, not merely deprecated) -- this script reproduces their exact
prior logic inline, clearly labeled, purely so a genuine before/after
comparison remains possible. This is measurement scaffolding only, not
a restoration of the old duplication into production.

Scenario: 20 cameras, 100 occupants, 50 zones (Phase 17's own required
shape). YOLO/detection inference is deliberately NOT included -- this
benchmarks only the occupancy-grouping seam itself, exactly as scripts/
benchmark_live_camera_pipeline.py already benchmarks only the identity-
resolution/fusion/estimation seam it owns.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually:
    python scripts/benchmark_canonical_occupancy.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.camera import Camera
from models.floor import Floor
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from live_perception.providers import LiveOccupantObservationProvider
from live_system.event_bus import EventBus

from crowd_intelligence.engine import CrowdIntelligenceEngine
from evacuation_progress.engine import EvacuationProgressEngine
from emergency_response.engine import EmergencyResponseIntelligenceEngine


CAMERA_COUNT = 20
OCCUPANT_COUNT = 100
ZONE_COUNT = 50
CYCLES = 200


def make_building():

    zones = [
        Zone(id=f"zone-{i}", name=f"Zone {i}", x=float(i) * 10.0, y=0.0, width=8.0, height=8.0, floor_id="floor-1")
        for i in range(ZONE_COUNT)
    ]
    cameras = [
        Camera(id=f"CAM-{i}", name=f"Camera {i}", floor_id="floor-1", zone_ids=(f"zone-{i % ZONE_COUNT}",))
        for i in range(CAMERA_COUNT)
    ]

    floor = Floor(id="floor-1", name="Ground Floor", zones=zones, cameras=cameras)

    return Building(id="benchmark-building", name="Benchmark Building", floors=[floor])


def populate(manager: LiveOccupantManager, time_value: float) -> None:

    for i in range(OCCUPANT_COUNT):

        zone_id = f"zone-{i % ZONE_COUNT}"
        camera_id = f"CAM-{i % CAMERA_COUNT}"

        manager.update(
            f"OCC-{i}", camera_id, f"T{i}", zone_id, "floor-1",
            (float(i), float(i)), 1.2, None, 0.9, time_value,
        )


# =====================================================
# BEFORE -- the four independent grouping loops this milestone removed
# from production, reproduced here verbatim (pre-canonicalization git
# history) purely for comparison.
# =====================================================


def before_provider_group(active_occupants, time_value):

    counts = {}
    confidence_sums = {}

    for occupant in active_occupants:

        if occupant.current_zone_id is None:
            continue

        counts[occupant.current_zone_id] = counts.get(occupant.current_zone_id, 0) + 1
        confidence_sums[occupant.current_zone_id] = confidence_sums.get(occupant.current_zone_id, 0.0) + occupant.confidence

    return counts


def before_crowd_group(active_occupants):

    by_zone = {}

    for occupant in active_occupants:

        if occupant.current_zone_id is None:
            continue

        by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

    return by_zone


def before_progress_group(active_occupants):

    active_by_zone = {}

    for occupant in active_occupants:
        if occupant.current_zone_id is not None:
            active_by_zone[occupant.current_zone_id] = active_by_zone.get(occupant.current_zone_id, 0) + 1

    return active_by_zone


def before_emergency_group(active_occupants):

    occupants_by_zone = {}

    for occupant in active_occupants:
        if occupant.current_zone_id is not None:
            occupants_by_zone.setdefault(occupant.current_zone_id, []).append(occupant)

    return occupants_by_zone


def run_before(manager: LiveOccupantManager, cycles: int) -> float:

    start = time.perf_counter()

    for cycle in range(cycles):

        active_occupants = manager.active_occupants()

        before_provider_group(active_occupants, float(cycle))
        before_crowd_group(active_occupants)
        before_progress_group(active_occupants)
        before_emergency_group(active_occupants)

    return time.perf_counter() - start


# =====================================================
# AFTER -- one canonical_occupancy() call per cycle, memoized, read by
# all four consumers via the SAME OccupancyFacts.
# =====================================================


def run_after(manager: LiveOccupantManager, cycles: int) -> float:

    start = time.perf_counter()

    for cycle in range(cycles):

        time_value = float(cycle)

        # All four consumers below call this with the SAME time_value
        # in production, within one orchestrator cycle -- memoized after
        # the first call, exactly as benchmarked here.
        facts_1 = manager.canonical_occupancy(time_value)
        facts_2 = manager.canonical_occupancy(time_value)
        facts_3 = manager.canonical_occupancy(time_value)
        facts_4 = manager.canonical_occupancy(time_value)

        assert facts_1 is facts_2 is facts_3 is facts_4

    return time.perf_counter() - start


# =====================================================
# Full-stage cost -- the real, current production consumers (not the
# reproduced BEFORE loops), for an honest "what does a real cycle cost
# today" number alongside the isolated grouping comparison above.
# =====================================================


def run_full_stage_cost(building, manager: LiveOccupantManager, cycles: int) -> float:

    event_bus = EventBus()

    provider = LiveOccupantObservationProvider(manager)
    crowd_engine = CrowdIntelligenceEngine(building, manager)
    progress_engine = EvacuationProgressEngine(building, manager, event_bus)
    emergency_engine = EmergencyResponseIntelligenceEngine(building, manager)

    start = time.perf_counter()

    for cycle in range(cycles):

        time_value = float(cycle)

        provider.collect(time_value)
        crowd_snapshot = crowd_engine.compute(time_value)
        progress_snapshot = progress_engine.compute(time_value, None, crowd_snapshot)
        emergency_engine.compute(time_value, None, crowd_snapshot, progress_snapshot)

    return time.perf_counter() - start


def main():

    building = make_building()

    manager_before = LiveOccupantManager()
    populate(manager_before, 0.0)
    before_seconds = run_before(manager_before, CYCLES)

    manager_after = LiveOccupantManager()
    populate(manager_after, 0.0)
    after_seconds = run_after(manager_after, CYCLES)

    manager_full = LiveOccupantManager()
    populate(manager_full, 0.0)
    full_stage_seconds = run_full_stage_cost(building, manager_full, CYCLES)

    print()
    print(f"=== Canonical Occupancy Benchmark ({CAMERA_COUNT} cameras, {OCCUPANT_COUNT} occupants, "
          f"{ZONE_COUNT} zones, {CYCLES} cycles) ===")
    print()
    print(f"BEFORE (4 independent grouping loops/cycle):  {before_seconds * 1000:.2f} ms total, "
          f"{before_seconds / CYCLES * 1000:.4f} ms/cycle")
    print(f"AFTER  (1 canonical grouping, memoized):      {after_seconds * 1000:.2f} ms total, "
          f"{after_seconds / CYCLES * 1000:.4f} ms/cycle")
    print(f"Grouping-only speedup: {before_seconds / after_seconds:.2f}x")
    print()
    print(f"Full real-consumer stage cost (provider + crowd + progress + emergency, "
          f"canonical, current production code): {full_stage_seconds * 1000:.2f} ms total, "
          f"{full_stage_seconds / CYCLES * 1000:.4f} ms/cycle")
    print()


if __name__ == "__main__":
    main()

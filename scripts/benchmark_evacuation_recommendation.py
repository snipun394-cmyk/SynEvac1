"""Live Dynamic Evacuation Recommendation Engine milestone, Phase 15
performance readiness benchmark.

Measures EvacuationRecommendationEngine.compute() at approximately the
milestone's own named scale: 100 occupants, 50 zones, 10 exits --
synthetic geometry, no YOLO inference (that overhead is already
measured separately by scripts/benchmark_yolo_human_detector.py).
Reuses the same synthetic building generator as scripts/
benchmark_trajectory_intelligence.py so the two milestones' own
benchmark numbers are directly comparable.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_evacuation_recommendation.py`) and
read the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from behavior_recognition.observation import RecognizedBehavior

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from building_state.models import BuildingState, HazardSummary

from evacuation_recommendation.engine import EvacuationRecommendationEngine
from evacuation_recommendation.ranking import SafeExitDistanceCalculator

from scripts.benchmark_trajectory_intelligence import CYCLE_COUNT, CYCLE_INTERVAL_SECONDS, OCCUPANT_COUNT, build_synthetic_building, seed_occupants


def main():

    building = build_synthetic_building()
    graph = NavigationGraphGenerator().build(building)

    zone_count = sum(len(floor.zones) for floor in building.ordered_floors())
    exit_count = sum(len(floor.exits) for floor in building.ordered_floors())

    event_bus = EventBus()
    manager = LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=100000.0)

    engine = EvacuationRecommendationEngine(building, graph, manager)
    building_state = BuildingState(hazard_summary=HazardSummary())

    candidate_only_calculator = SafeExitDistanceCalculator(graph)

    candidate_durations = []
    compute_durations = []

    for cycle in range(CYCLE_COUNT):

        current_time = cycle * CYCLE_INTERVAL_SECONDS
        seed_occupants(manager, building, current_time)

        start = time.perf_counter()
        candidate_only_calculator.compute(building_state)
        candidate_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        engine.compute(current_time, building_state)
        compute_durations.append(time.perf_counter() - start)

    print("Live Dynamic Evacuation Recommendation Engine -- performance benchmark")
    print(f"Zones: {zone_count}, Exits: {exit_count}, Occupants: {OCCUPANT_COUNT}, Cycles: {CYCLE_COUNT}")
    print()
    print(f"SafeExitDistanceCalculator.compute() (cached, {exit_count} Dijkstra runs on fingerprint change): "
          f"mean={statistics.mean(candidate_durations) * 1000:.3f} ms, max={max(candidate_durations) * 1000:.3f} ms")
    print(f"EvacuationRecommendationEngine.compute() (ranking + explanation, full cycle): "
          f"mean={statistics.mean(compute_durations) * 1000:.3f} ms, max={max(compute_durations) * 1000:.3f} ms")
    print()
    print("Note: SafeExitDistanceCalculator only re-runs its N-exit Dijkstra batch when the hazard/traversability "
          "fingerprint changes cycle-to-cycle -- all cycles above share the SAME fingerprint (no hazard change), so "
          "its own reported cost is the steady-state (cache-hit) cost, not the first-cycle N-Dijkstra cost.")


if __name__ == "__main__":
    main()

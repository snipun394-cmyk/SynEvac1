"""Live Evacuation Guidance & Zoned Message Planning milestone, Phase 27
performance readiness benchmark.

Measures EvacuationGuidanceEngine.compute() at approximately the
milestone's own named scale: 50 occupied zones, 10 exits, 20 doors, 10
stairs -- synthetic geometry, no YOLO inference. Reuses the same
synthetic building generator as scripts/benchmark_trajectory_intelligence.py/
scripts/benchmark_evacuation_recommendation.py so all three milestones'
own benchmark numbers are directly comparable, feeding this engine a
real EvacuationRecommendationSnapshot computed by the previous
milestone's own engine (never a hand-waved stand-in).

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_evacuation_guidance.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from live_system.event_bus import EventBus

from building_state.models import BuildingState, HazardSummary

from evacuation_recommendation.engine import EvacuationRecommendationEngine

from evacuation_guidance.engine import EvacuationGuidanceEngine

from scripts.benchmark_trajectory_intelligence import CYCLE_COUNT, CYCLE_INTERVAL_SECONDS, build_synthetic_building, seed_occupants


class _FakeSpeaker:

    def __init__(self, speaker_id):
        self.id = speaker_id


class _FakeSpeakerManager:

    def active_speakers_in_zone(self, zone_id):
        return (_FakeSpeaker(f"SPK-{zone_id}"),)


def main():

    building = build_synthetic_building()
    graph = NavigationGraphGenerator().build(building)

    zone_count = sum(len(floor.zones) for floor in building.ordered_floors())
    exit_count = sum(len(floor.exits) for floor in building.ordered_floors())
    door_count = sum(len(floor.doors) for floor in building.ordered_floors())
    stair_count = sum(len(floor.stairs) for floor in building.ordered_floors())

    event_bus = EventBus()
    manager = LiveOccupantManager(event_bus=event_bus, exits=[], expire_after_seconds=100000.0)

    recommendation_engine = EvacuationRecommendationEngine(building, graph, manager)
    guidance_engine = EvacuationGuidanceEngine(building, graph)
    speaker_manager = _FakeSpeakerManager()

    building_state = BuildingState(hazard_summary=HazardSummary())

    durations = []

    for cycle in range(CYCLE_COUNT):

        current_time = cycle * CYCLE_INTERVAL_SECONDS
        seed_occupants(manager, building, current_time)

        recommendation_snapshot = recommendation_engine.compute(current_time, building_state)

        start = time.perf_counter()
        guidance_engine.compute(current_time, recommendation_snapshot, building_state, speaker_manager)
        durations.append(time.perf_counter() - start)

    print("Live Evacuation Guidance & Zoned Message Planning -- performance benchmark")
    print(f"Zones: {zone_count}, Exits: {exit_count}, Doors: {door_count}, Stairs: {stair_count}, Cycles: {CYCLE_COUNT}")
    print()
    print(f"EvacuationGuidanceEngine.compute() (route planning + validation + instructions + speaker mapping, "
          f"full cycle): mean={statistics.mean(durations) * 1000:.3f} ms, max={max(durations) * 1000:.3f} ms")
    print()
    print("Note: unlike evacuation_recommendation.ranking.SafeExitDistanceCalculator (one cached Dijkstra batch "
          "shared across every zone), evacuation_guidance.route_planner.resolve_route() runs one FRESH "
          "distances_from() query per OCCUPIED zone every cycle (Phase 5's own 'find a valid safe path to the "
          "exact recommended exit' requirement is inherently point-to-point, not a shared distance map). Still "
          "comfortably sub-cycle at this milestone's own named scale -- add per-zone caching only if a future, "
          "much larger deployment's own measurement shows it is actually needed (Phase 27's own guidance).")


if __name__ == "__main__":
    main()

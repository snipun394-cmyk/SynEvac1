"""
Live Dynamic Evacuation Signage milestone, Phase 29 -- performance
benchmark. Builds a synthetic building with 50 zones (2 floors), 10
exits, 20 doors, 10 stairs, and 100 signs, then times route-to-sign
mapping, direction geometry, validation, conflict detection, and a
complete planning cycle. No assertions, no correctness claims -- purely
informational, run manually: `python scripts/benchmark_dynamic_signage.py`.

No network access, no hardware access -- entirely in-process.
"""

import sys
import time as time_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from evacuation_guidance.engine import EvacuationGuidanceEngine
from evacuation_recommendation.models import EvacuationRecommendationSnapshot, RecommendationStatus, ZoneEvacuationRecommendation

from dynamic_signage.models import DynamicSignageConfig
from dynamic_signage.planner import DynamicSignagePlanner

from tests.dynamic_signage_fixtures import make_sign


ZONES_PER_FLOOR = 25
FLOORS = 2
TOTAL_ZONES = ZONES_PER_FLOOR * FLOORS  # 50
DOORS_PER_FLOOR = 10  # 20 total
EXITS_PER_FLOOR = 5  # 10 total
STAIRS = 10
SIGNS = 100


def build_benchmark_building() -> Building:

    floors = []

    for floor_index in range(FLOORS):

        floor_id = f"floor-{floor_index}"
        zones = []
        doors = []
        exits = []

        for zone_index in range(ZONES_PER_FLOOR):

            zone_id = f"z-{floor_index}-{zone_index}"
            zones.append(Zone(id=zone_id, name=zone_id, x=float(zone_index * 12), y=0.0, width=10.0, height=10.0, floor_id=floor_id))

        # A simple chain of doors linking consecutive zones -- enough to
        # form real, walkable routes without needing a full mesh.
        for door_index in range(DOORS_PER_FLOOR):

            a = zones[door_index % ZONES_PER_FLOOR].id
            b = zones[(door_index + 1) % ZONES_PER_FLOOR].id

            doors.append(Door(
                id=f"door-{floor_index}-{door_index}", name=f"D{floor_index}-{door_index}", floor_id=floor_id,
                zone_a_id=a, zone_b_id=b, start_point=(float(door_index * 12 + 10), 3.0), end_point=(float(door_index * 12 + 10), 7.0),
            ))

        for exit_index in range(EXITS_PER_FLOOR):

            zone_id = zones[exit_index * (ZONES_PER_FLOOR // EXITS_PER_FLOOR)].id

            exits.append(Exit(
                id=f"exit-{floor_index}-{exit_index}", name=f"E{floor_index}-{exit_index}", floor_id=floor_id,
                zone_id=zone_id, start_point=(0.0, 3.0), end_point=(0.0, 7.0),
            ))

        floors.append(Floor(id=floor_id, name=f"Floor {floor_index}", zones=zones, doors=doors, exits=exits))

    stairs = []

    for stair_index in range(STAIRS):

        from_zone = floors[0].zones[stair_index % ZONES_PER_FLOOR].id
        to_zone = floors[1].zones[stair_index % ZONES_PER_FLOOR].id

        stairs.append(Staircase(
            id=f"stair-{stair_index}", name=f"S{stair_index}",
            from_zone_id=from_zone, to_zone_id=to_zone, from_floor_id="floor-0", to_floor_id="floor-1",
            from_position=(float(stair_index * 5), 20.0), to_position=(float(stair_index * 5), 20.0),
        ))

    floors[0].stairs = stairs

    return Building(id="benchmark-building", name="Benchmark Building", floors=floors)


def build_recommendation_snapshot(building: Building) -> EvacuationRecommendationSnapshot:

    zones = {}

    all_zone_ids = [zone.id for floor in building.ordered_floors() for zone in floor.zones]

    for index, zone_id in enumerate(all_zone_ids):

        floor_id = building.ordered_floors()[0 if index < ZONES_PER_FLOOR else 1].id
        exit_id = f"exit-{0 if index < ZONES_PER_FLOOR else 1}-{index % EXITS_PER_FLOOR}"

        zones[zone_id] = ZoneEvacuationRecommendation(
            zone_id=zone_id, floor_id=floor_id, status=RecommendationStatus.RECOMMENDED,
            recommended_exit_id=exit_id, ranked_exit_ids=(exit_id,), confidence=0.8, occupant_count=1, timestamp=0.0,
        )

    return EvacuationRecommendationSnapshot(timestamp=0.0, zones=zones, safe_exit_ids=tuple({z.recommended_exit_id for z in zones.values()}))


def build_signs(building: Building):

    all_zone_ids = [zone.id for floor in building.ordered_floors() for zone in floor.zones]

    signs = []

    for index in range(SIGNS):

        zone_id = all_zone_ids[index % len(all_zone_ids)]
        floor_id = building.ordered_floors()[0 if index % len(all_zone_ids) < ZONES_PER_FLOOR else 1].id

        signs.append(make_sign(
            f"sign-{index}", floor_id=floor_id, zone_ids=(zone_id,),
            position=(float((index % 10) * 12 + 5), 5.0), orientation=float((index * 37) % 360),
        ))

    return signs


def _timed(label, func):

    start = time_module.perf_counter()
    result = func()
    elapsed_ms = (time_module.perf_counter() - start) * 1000.0

    print(f"{label:45s} {elapsed_ms:8.2f} ms")

    return result


def main():

    building = _timed("Build 50-zone/20-door/10-exit/10-stair building", build_benchmark_building)
    graph = _timed("Build navigation graph", lambda: NavigationGraphGenerator().build(building))
    recommendation_snapshot = _timed("Build recommendation snapshot", lambda: build_recommendation_snapshot(building))

    guidance_engine = EvacuationGuidanceEngine(building, graph)
    guidance_snapshot = _timed("Compute EvacuationGuidanceSnapshot (50 zones)", lambda: guidance_engine.compute(0.0, recommendation_snapshot, None))

    signs = _timed("Build 100 signs", lambda: build_signs(building))

    planner = DynamicSignagePlanner(building, DynamicSignageConfig())

    result = _timed("Complete signage planning cycle (100 signs)", lambda: planner.compute(0.0, guidance_snapshot, signs))

    # A second cycle with identical inputs -- measures the steady-state
    # cost once every revision fingerprint is already cached.
    _timed("Second identical cycle (revision cache warm)", lambda: planner.compute(1.0, guidance_snapshot, signs))

    print(f"\nSigns planned: {len(result.instructions)}, conflicts: {len(result.conflicts)}")


if __name__ == "__main__":
    main()

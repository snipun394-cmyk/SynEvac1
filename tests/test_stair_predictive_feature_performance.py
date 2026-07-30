import time
import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from live_occupants.manager import LiveOccupantManager

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor_v2_1 import (
    build_stair_flow_snapshot_for_prediction, extract_live_experimental_candidate_features,
)
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts


# =====================================================
# Stair Predictive-Feature Live Parity milestone, Phase 13 -- incremental
# cost of producing candidate_recent_flow_rate for every Stair candidate
# from the new stair_flow source, at a realistic scale: 20 cameras, 20
# stairs, 100 occupants (each with a completed entry+exit, half observed
# by two simultaneous cameras). Pure Python/CPU cost -- no YOLO/ML
# inference involved anywhere in this path, reported separately by
# construction (this test never imports human_detection/cv2/torch).
# =====================================================


CAMERA_COUNT = 20
STAIR_COUNT = 20
OCCUPANT_COUNT = 100

MAX_SECONDS = 2.0


def make_building_with_stairs(stair_count: int):

    building = Building(name="Predictive Feature Performance Test Building")

    floor_1 = Floor(name="Floor 1", id="floor-1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", id="floor-2", display_order=1, height=3.0)

    stairs = []
    for i in range(stair_count):

        zone_a = Zone(id=f"zone-{i}-a", name=f"Zone {i}A", x=float(i) * 10, y=0.0, width=5.0, height=5.0, floor_id="floor-1")
        zone_b = Zone(id=f"zone-{i}-b", name=f"Zone {i}B", x=float(i) * 10, y=0.0, width=5.0, height=5.0, floor_id="floor-2")
        floor_1.zones.append(zone_a)
        floor_2.zones.append(zone_b)

        stair = Staircase(
            id=f"S{i}", name=f"S{i}", from_floor_id="floor-1", to_floor_id="floor-2",
            from_zone_id=zone_a.id, to_zone_id=zone_b.id, width=1.5,
        )
        floor_1.stairs.append(stair)
        stairs.append(stair)

    building.floors = [floor_1, floor_2]

    return building, floor_1, floor_2, stairs


class StairPredictiveFeaturePerformanceTests(unittest.TestCase):

    def test_realistic_multi_camera_multi_stair_scale(self):

        building, floor_1, floor_2, stairs = make_building_with_stairs(STAIR_COUNT)
        manager = LiveOccupantManager()

        camera_ids = [f"CAM-{i}" for i in range(CAMERA_COUNT)]

        for occupant_index in range(OCCUPANT_COUNT):

            occupant_id = f"OCC-{occupant_index}"
            stair = stairs[occupant_index % STAIR_COUNT]
            camera_a = camera_ids[occupant_index % CAMERA_COUNT]
            camera_b = camera_ids[(occupant_index + 1) % CAMERA_COUNT]

            manager.update(
                occupant_id, camera_a, f"T-{occupant_index}-a", stair.from_zone_id, floor_1.id,
                (1.0, 1.0), 0.0, None, 0.9, 0.0,
            )
            manager.update(
                occupant_id, camera_a, f"T-{occupant_index}-a", None, floor_1.id,
                (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id=stair.id,
            )
            manager.update(
                occupant_id, camera_b, f"T-{occupant_index}-b", None, floor_1.id,
                (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id=stair.id,
            )
            manager.update(
                occupant_id, camera_b, f"T-{occupant_index}-b", stair.to_zone_id, floor_2.id,
                (3.0, 3.0), 0.5, None, 0.9, 2.0,
            )

        occupants = manager.all_occupants()
        self.assertEqual(len(occupants), OCCUPANT_COUNT)

        edges = edges_by_candidate_id(building)
        candidates = {c.candidate_id: c for c in enumerate_candidates(building)}
        alt_counts = build_alternative_route_counts(tuple(candidates.values()))

        start = time.perf_counter()

        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(stairs, occupants, building, 2.0)

        total_flow_rate = 0
        for stair in stairs:

            candidate = candidates[stair.id]
            edge = edges[stair.id]

            features = extract_live_experimental_candidate_features(
                candidate, edge, 2.0,
                building=building, crowd_snapshot=None, occupancy_facts=None,
                alternative_route_counts=alt_counts, evacuation_snapshot=None, occupants=None,
                stair_flow_snapshot=stair_flow_snapshot,
            )
            total_flow_rate += features["candidate_recent_flow_rate"] or 0

        elapsed = time.perf_counter() - start

        print(
            f"\n[stair predictive-feature perf] {CAMERA_COUNT} cameras, {STAIR_COUNT} stairs, "
            f"{OCCUPANT_COUNT} occupants -> stair_flow_snapshot + {STAIR_COUNT} candidate feature "
            f"extractions took {elapsed * 1000:.2f} ms (no ML/YOLO inference in this path)"
        )

        self.assertLess(elapsed, MAX_SECONDS)
        self.assertEqual(total_flow_rate, OCCUPANT_COUNT)


if __name__ == "__main__":
    unittest.main()

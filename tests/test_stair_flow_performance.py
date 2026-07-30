import time
import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from live_occupants.manager import LiveOccupantManager

from stair_flow.compute import compute_stair_flow_snapshot


# =====================================================
# Live Stair Flow & Movement Direction Intelligence milestone -- Phase
# 15. Benchmarks a realistic multi-camera case: ~20 cameras, ~20 stairs,
# ~100 occupants, each occupant having genuinely traversed a stair
# (entry + exit recorded, some observed by 2 cameras at once -- the
# common healthy-coverage case Phase 8 also proves correctness for).
#
# This is a deterministic, offline benchmark (no external service, no
# randomness affecting correctness) -- it asserts a generous wall-clock
# ceiling so it stays a meaningful regression signal without being flaky
# on a slower CI machine, and prints the actual measured cost for a
# human reading test output.
# =====================================================


CAMERA_COUNT = 20
STAIR_COUNT = 20
OCCUPANT_COUNT = 100

MAX_SECONDS = 2.0  # generous ceiling; actual cost is expected to be well under this


def make_building_with_stairs(stair_count: int):

    building = Building(name="Performance Test Building")

    floor_1 = Floor(name="Floor 1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", display_order=1, height=3.0)

    building.add_floor(floor_1)
    building.add_floor(floor_2)

    stairs = []
    for i in range(stair_count):
        stair = Staircase(name=f"S{i}", from_floor_id=floor_1.id, to_floor_id=floor_2.id)
        stair.id = f"S{i}"
        floor_1.add_stair(stair)
        stairs.append(stair)

    return building, floor_1, floor_2, stairs


class StairFlowPerformanceTests(unittest.TestCase):

    def test_realistic_multi_camera_scale(self):

        building, floor_1, floor_2, stairs = make_building_with_stairs(STAIR_COUNT)
        manager = LiveOccupantManager()

        camera_ids = [f"CAM-{i}" for i in range(CAMERA_COUNT)]

        for occupant_index in range(OCCUPANT_COUNT):

            occupant_id = f"OCC-{occupant_index}"
            stair = stairs[occupant_index % STAIR_COUNT]
            camera_a = camera_ids[occupant_index % CAMERA_COUNT]
            camera_b = camera_ids[(occupant_index + 1) % CAMERA_COUNT]

            manager.update(
                occupant_id, camera_a, f"T-{occupant_index}-a", "ZONE-A", floor_1.id,
                (1.0, 1.0), 0.0, None, 0.9, 0.0,
            )
            # Two cameras both observe the entry this cycle.
            manager.update(
                occupant_id, camera_a, f"T-{occupant_index}-a", None, floor_1.id,
                (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id=stair.id,
            )
            manager.update(
                occupant_id, camera_b, f"T-{occupant_index}-b", None, floor_1.id,
                (2.0, 2.0), 0.5, None, 0.9, 1.0, stair_id=stair.id,
            )
            manager.update(
                occupant_id, camera_b, f"T-{occupant_index}-b", "ZONE-B", floor_2.id,
                (3.0, 3.0), 0.5, None, 0.9, 2.0,
            )

        occupants = manager.all_occupants()
        self.assertEqual(len(occupants), OCCUPANT_COUNT)

        start = time.perf_counter()
        snapshot = compute_stair_flow_snapshot(stairs, occupants, building, timestamp=2.0, window_seconds=60.0)
        elapsed = time.perf_counter() - start

        print(
            f"\n[stair_flow perf] {CAMERA_COUNT} cameras, {STAIR_COUNT} stairs, "
            f"{OCCUPANT_COUNT} occupants -> compute_stair_flow_snapshot() took {elapsed * 1000:.2f} ms"
        )

        self.assertLess(elapsed, MAX_SECONDS)

        total_entries = sum(m.entries or 0 for m in snapshot.by_stair.values())
        total_exits = sum(m.exits or 0 for m in snapshot.by_stair.values())

        self.assertEqual(total_entries, OCCUPANT_COUNT)
        self.assertEqual(total_exits, OCCUPANT_COUNT)


if __name__ == "__main__":
    unittest.main()

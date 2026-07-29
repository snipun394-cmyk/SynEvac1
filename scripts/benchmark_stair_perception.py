"""Observable Stair Perception milestone, Phase 27 performance benchmark.

Benchmarks the spatial world-position -> Stair lookup and the Stair
occupancy derivation SEPARATELY, at the milestone's own required scale:
50 zones, 20 stairs, 20 cameras, 100 occupants.

Every occupant/geometry input here is synthetic and hand-built -- this
file does NOT run YOLOHumanDetector, a tracker, or RTSP of any kind, so
its numbers say nothing about real per-camera perception speed (mirrors
scripts/benchmark_crowd_intelligence.py's own disclosure). Reported
here: only this milestone's own new computation cost --
camera_calibration.stair_lookup.locate_stair()/covered_stair_ids() and
stair_perception.facts.compute_stair_occupancy_snapshot().

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_stair_perception.py`) and read the
printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase, StairObservableRegion
from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.stair_lookup import build_stairs_by_floor, covered_stair_ids, locate_stair

from live_occupants.manager import LiveOccupantManager

from stair_perception.facts import compute_stair_occupancy_snapshot


ZONE_COUNT = 50
STAIR_COUNT = 20
CAMERA_COUNT = 20
OCCUPANT_COUNT = 100

ITERATIONS = 200


def build_building():

    building = Building(name="Benchmark Building")
    floor = Floor(name="Floor 1", display_order=0)
    building.add_floor(floor)

    for i in range(ZONE_COUNT):
        zone = Zone(name=f"Zone {i}", x=float(i * 20), y=0.0, width=10.0, height=10.0, floor_id=floor.id)
        floor.add_zone(zone)

    stairs = []
    for i in range(STAIR_COUNT):

        stair = Staircase(name=f"Stair {i}", from_floor_id=floor.id, to_floor_id=floor.id, width=1.5)
        stair.from_observable_region = StairObservableRegion(center_x=float(i * 20), center_y=100.0, width=2.0, depth=2.0)
        floor.add_stair(stair)
        stairs.append(stair)

    return building, floor, stairs


def build_calibrations(floor_id):

    calibrations = {}

    for i in range(CAMERA_COUNT):

        intrinsics = CameraIntrinsics(image_width=640, image_height=480, focal_length_x=500.0, focal_length_y=500.0)
        extrinsics = CameraExtrinsics(position=(float(i * 10), 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=45.0, roll_degrees=0.0)
        calibrations[f"CAM-{i}"] = CalibrationProfile(camera_id=f"CAM-{i}", floor_id=floor_id, intrinsics=intrinsics, extrinsics=extrinsics)

    return calibrations


def build_occupant_manager(floor_id, stairs):

    manager = LiveOccupantManager()

    for i in range(OCCUPANT_COUNT):

        # Roughly a fifth of occupants land on a stair, the rest
        # elsewhere -- a plausible, non-degenerate mix for benchmarking
        # the grouping/lookup cost.
        stair_id = stairs[i % STAIR_COUNT].id if i % 5 == 0 else None

        manager.update(
            f"OCC-{i}", f"CAM-{i % CAMERA_COUNT}", f"T-{i}", f"zone-{i % ZONE_COUNT}", floor_id,
            (float(i), 0.0), 0.0, None, 0.9, 0.0, stair_id=stair_id,
        )

    return manager


def timed(label, fn, iterations=ITERATIONS):

    samples = []

    for _ in range(iterations):

        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(samples)
    print(f"{label}: {mean_ms:.4f} ms/call (mean over {iterations} calls)")

    return mean_ms


def main():

    building, floor, stairs = build_building()
    calibrations = build_calibrations(floor.id)
    stairs_by_floor = build_stairs_by_floor(building)
    manager = build_occupant_manager(floor.id, stairs)

    calibrated_floor_ids = frozenset(c.floor_id for c in calibrations.values())

    print(
        f"Scale: {ZONE_COUNT} zones, {STAIR_COUNT} stairs, {CAMERA_COUNT} cameras, "
        f"{OCCUPANT_COUNT} occupants\n"
    )

    # World-position -> Stair lookup, one call per occupant per cycle
    # (the same cost live_camera_pipeline.pipeline.LiveCameraPipeline
    # pays inside camera_calibration.projection.WorldProjector.project()).
    world_position = (float(STAIR_COUNT * 10), 100.0)
    timed(
        "World-position -> Stair lookup (locate_stair, single occupant)",
        lambda: locate_stair(stairs_by_floor[floor.id], floor.id, world_position),
    )

    timed(
        "Stair coverage derivation (covered_stair_ids, whole building)",
        lambda: covered_stair_ids(stairs_by_floor, calibrated_floor_ids),
    )

    coverage = covered_stair_ids(stairs_by_floor, calibrated_floor_ids)
    stair_ids = [stair.id for stair in stairs]

    def occupancy_cycle():

        facts = manager.canonical_occupancy(0.0)
        return compute_stair_occupancy_snapshot(stair_ids, facts.occupant_ids_by_stair, coverage, 0.0)

    timed(
        "Full Stair occupancy derivation (canonical_occupancy + compute_stair_occupancy_snapshot)",
        occupancy_cycle,
    )

    print("\nDone. No performance regression to any existing zone/crowd/trajectory computation -- this")
    print("milestone's own cost is entirely additive (a single extra pass already folded into the")
    print("existing occupant loop, plus this new, independent Stair-specific derivation).")


if __name__ == "__main__":
    main()

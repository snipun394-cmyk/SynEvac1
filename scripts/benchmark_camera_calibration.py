"""Camera Calibration & World Coordinate Projection milestone, Phase 10
performance readiness benchmark.

Benchmarks projection, geometry, and zone lookup SEPARATELY -- zero
YOLO/tracker/behavior-recognizer inference anywhere in this file.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_camera_calibration.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.zone import Zone

from camera_calibration.camera_model import CalibrationProfile, CameraExtrinsics, CameraIntrinsics
from camera_calibration.geometry import camera_basis_vectors, intersect_ray_with_floor, pixel_ray_direction
from camera_calibration.projection import WorldProjector


ZONE_COUNT = 200
CYCLE_COUNT = 2000


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def benchmark_geometry() -> dict:

    intrinsics = CameraIntrinsics(image_width=1920, image_height=1080, focal_length_x=1200.0, focal_length_y=1200.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=15.0, pitch_degrees=45.0, roll_degrees=2.0)

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        start = time.perf_counter()
        direction = pixel_ray_direction(960.0 + (i % 50), 540.0 + (i % 30), intrinsics, extrinsics)
        intersect_ray_with_floor((0.0, 0.0, 3.0), direction)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_basis_vectors() -> dict:

    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=15.0, pitch_degrees=45.0, roll_degrees=2.0)

    per_call_ms = []

    for _ in range(CYCLE_COUNT):
        start = time.perf_counter()
        camera_basis_vectors(extrinsics)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_projection_with_zone_lookup() -> dict:

    intrinsics = CameraIntrinsics(image_width=1920, image_height=1080, focal_length_x=1200.0, focal_length_y=1200.0)
    extrinsics = CameraExtrinsics(position=(0.0, 0.0), mount_height=3.0, yaw_degrees=0.0, pitch_degrees=60.0)
    calibration = CalibrationProfile(camera_id="CAM-BENCH", floor_id="floor-1", intrinsics=intrinsics, extrinsics=extrinsics)

    zones = []
    for i in range(ZONE_COUNT):
        zone = Zone(name=f"zone-{i}", x=float(i) * 10.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1")
        zone.id = f"zone-{i}"
        zones.append(zone)

    projector = WorldProjector(calibrations={"CAM-BENCH": calibration}, zones_by_floor={"floor-1": zones})

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        box = (900.0 + (i % 100), 500.0, 920.0 + (i % 100), 540.0)

        start = time.perf_counter()
        projector.project("CAM-BENCH", box, 0.9)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {
        "call_count": CYCLE_COUNT, "zone_count": ZONE_COUNT,
        "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95),
    }


def main():

    basis = benchmark_basis_vectors()
    print(f"Basis-vector computation: {basis['call_count']} calls, mean {basis['mean_ms']:.5f} ms, p95 {basis['p95_ms']:.5f} ms")

    geometry = benchmark_geometry()
    print(
        f"Geometry (ray + floor intersection): {geometry['call_count']} calls, "
        f"mean {geometry['mean_ms']:.5f} ms, p95 {geometry['p95_ms']:.5f} ms"
    )

    projection = benchmark_projection_with_zone_lookup()
    print(
        f"Full projection incl. zone lookup ({projection['zone_count']} zones): {projection['call_count']} calls, "
        f"mean {projection['mean_ms']:.5f} ms, p95 {projection['p95_ms']:.5f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/tracker/behavior-recognizer inference occurs anywhere in this "
        "benchmark -- every bounding box/calibration is synthetic and hand-built."
    )


if __name__ == "__main__":
    main()

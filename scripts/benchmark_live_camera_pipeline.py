"""Phase 14 performance readiness benchmark.

Benchmarks the ARCHITECTURE around the future live camera pipeline --
identity resolution, multi-camera fusion, and BuildingState estimation
-- with synthetic detections. There is no computer vision to benchmark
yet (no real detector exists), so this deliberately measures only the
seams this milestone actually built.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_live_camera_pipeline.py`) and read
the printed report.
"""

import sys
import time
from pathlib import Path

# Run directly (`python scripts/benchmark_live_camera_pipeline.py`), not as
# a package -- the repo root must be on sys.path the same way it already
# is for every `python -m unittest` invocation from that directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from perception.models.human_observation import HumanClassification, HumanState

from live_camera_pipeline.human_detector import RawHumanDetection
from live_camera_pipeline.identity_resolver import MappingIdentityResolver

from multi_camera_fusion.engine import MultiCameraFusionEngine

from building_state.estimator import BuildingStateEstimator


PEOPLE_PER_CAMERA = 8


def make_raw_detections(camera_count: int, time_value: float):

    raw_detections = []
    mapping = {}

    for camera_index in range(camera_count):

        camera_id = f"CAM-{camera_index:03d}"

        for person_index in range(PEOPLE_PER_CAMERA):

            local_track_id = str(person_index)
            global_id = f"GLOBAL-{camera_index:03d}-{person_index:03d}"

            mapping[(camera_id, local_track_id)] = global_id

            raw_detections.append(
                RawHumanDetection(
                    camera_id=camera_id,
                    local_track_id=local_track_id,
                    timestamp=time_value,
                    confidence=0.9,
                    classification_evidence=HumanClassification.ADULT,
                    state_evidence=HumanState.WALKING,
                    floor_id=f"floor-{camera_index % 4}",
                    zone_id=f"zone-{camera_index % 4}-{person_index % 3}",
                )
            )

    return raw_detections, mapping


def benchmark_camera_count(camera_count: int) -> dict:

    raw_detections, mapping = make_raw_detections(camera_count, time_value=0.0)

    resolver = MappingIdentityResolver(mapping)
    fusion_engine = MultiCameraFusionEngine()
    estimator = BuildingStateEstimator()

    start = time.perf_counter()
    detections = resolver.resolve(raw_detections, time=0.0)
    resolve_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fusion_result = fusion_engine.fuse(detections, time=0.0)
    fuse_seconds = time.perf_counter() - start

    start = time.perf_counter()
    estimator.estimate(
        0.0,
        hazard_snapshot=HazardSnapshot(),
        occupancy_snapshot=OccupancySnapshot(),
        fusion_result=fusion_result,
    )
    estimate_seconds = time.perf_counter() - start

    return {
        "camera_count": camera_count,
        "detection_count": len(raw_detections),
        "resolve_ms": resolve_seconds * 1000,
        "fuse_ms": fuse_seconds * 1000,
        "estimate_ms": estimate_seconds * 1000,
        "total_ms": (resolve_seconds + fuse_seconds + estimate_seconds) * 1000,
    }


def main():

    print(f"{'Cameras':>8} {'Detections':>11} {'Resolve (ms)':>13} "
          f"{'Fuse (ms)':>10} {'Estimate (ms)':>14} {'Total (ms)':>11}")

    for camera_count in (4, 16, 32):

        result = benchmark_camera_count(camera_count)

        print(
            f"{result['camera_count']:>8} {result['detection_count']:>11} "
            f"{result['resolve_ms']:>13.3f} {result['fuse_ms']:>10.3f} "
            f"{result['estimate_ms']:>14.3f} {result['total_ms']:>11.3f}"
        )


if __name__ == "__main__":
    main()

"""Single-Camera Tracking Framework milestone, Phase 11 performance
readiness benchmark.

Benchmarks SimpleSingleCameraTracker's own overhead -- matching, track
update, track creation, and track deletion -- entirely separate from
YOLO/any detector inference: every RawHumanDetection here is
synthetic, hand-built, never produced by a real or fake detector call.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_single_camera_tracker.py`) and read
the printed report.
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from live_camera_pipeline.human_detector import RawHumanDetection

from tracking.simple_tracker import SimpleSingleCameraTracker


PEOPLE_COUNT = 20
CYCLE_COUNT = 500


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _detection(person_index: int, frame_index: int) -> RawHumanDetection:

    x = person_index * 30.0 + frame_index * 0.5  # slow, steady drift -- always re-matchable
    return RawHumanDetection(
        camera_id="CAM-BENCH", local_track_id="unused", timestamp=float(frame_index),
        bounding_box=(x, 0.0, x + 20.0, 60.0), confidence=0.9,
    )


def benchmark_steady_state_matching_and_update() -> dict:

    # PEOPLE_COUNT tracks already established, then CYCLE_COUNT further
    # cycles of pure matching + track update -- zero creation, zero
    # deletion, isolating matching/update cost specifically.
    tracker = SimpleSingleCameraTracker()

    tracker.update("CAM-BENCH", 0.0, [_detection(i, 0) for i in range(PEOPLE_COUNT)])

    per_cycle_ms = []

    for frame_index in range(1, CYCLE_COUNT + 1):

        detections = [_detection(i, frame_index) for i in range(PEOPLE_COUNT)]

        start = time.perf_counter()
        tracker.update("CAM-BENCH", float(frame_index), detections)
        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "people_count": PEOPLE_COUNT,
        "cycle_count": CYCLE_COUNT,
        "mean_ms": statistics.mean(per_cycle_ms),
        "p95_ms": _percentile(per_cycle_ms, 0.95),
        "max_ms": max(per_cycle_ms),
    }


def benchmark_track_creation() -> dict:

    # A fresh tracker + fresh camera_id every cycle -- every detection
    # is necessarily a brand-new track (no existing track to match),
    # isolating pure creation cost.
    per_cycle_ms = []

    for i in range(CYCLE_COUNT):

        tracker = SimpleSingleCameraTracker()
        detections = [_detection(person_index, 0) for person_index in range(PEOPLE_COUNT)]

        start = time.perf_counter()
        tracker.update(f"CAM-{i}", 0.0, detections)
        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "people_count": PEOPLE_COUNT,
        "cycle_count": CYCLE_COUNT,
        "mean_ms": statistics.mean(per_cycle_ms),
        "p95_ms": _percentile(per_cycle_ms, 0.95),
        "max_ms": max(per_cycle_ms),
    }


def benchmark_track_deletion() -> dict:

    # PEOPLE_COUNT tracks established, then repeatedly let every one of
    # them expire (max_missing_frames=1 -- deleted on the 2nd
    # consecutive empty update), isolating pure deletion/expiry cost.
    # A fresh batch is recreated between measured rounds so every
    # measured cycle is genuinely an expiry-triggering cycle, not a
    # no-op on an already-empty tracker.
    tracker = SimpleSingleCameraTracker(max_missing_frames=1)

    per_cycle_ms = []
    rounds = 100

    for round_index in range(rounds):

        camera_id = f"CAM-DEL-{round_index}"
        tracker.update(camera_id, 0.0, [_detection(i, 0) for i in range(PEOPLE_COUNT)])
        tracker.update(camera_id, 1.0, [])  # 1st miss -- still MISSING

        start = time.perf_counter()
        tracker.update(camera_id, 2.0, [])  # 2nd miss -- triggers deletion of all PEOPLE_COUNT tracks
        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "people_count": PEOPLE_COUNT,
        "round_count": rounds,
        "mean_ms": statistics.mean(per_cycle_ms),
        "p95_ms": _percentile(per_cycle_ms, 0.95),
        "max_ms": max(per_cycle_ms),
    }


def main():

    steady = benchmark_steady_state_matching_and_update()
    print(
        f"Steady-state matching + update ({steady['people_count']} people/camera): "
        f"{steady['cycle_count']} cycles, mean {steady['mean_ms']:.4f} ms, "
        f"p95 {steady['p95_ms']:.4f} ms, max {steady['max_ms']:.4f} ms"
    )

    creation = benchmark_track_creation()
    print(
        f"Track creation ({creation['people_count']} new tracks/cycle): "
        f"{creation['cycle_count']} cycles, mean {creation['mean_ms']:.4f} ms, "
        f"p95 {creation['p95_ms']:.4f} ms, max {creation['max_ms']:.4f} ms"
    )

    deletion = benchmark_track_deletion()
    print(
        f"Track deletion/expiry ({deletion['people_count']} tracks expiring/cycle): "
        f"{deletion['round_count']} rounds, mean {deletion['mean_ms']:.4f} ms, "
        f"p95 {deletion['p95_ms']:.4f} ms, max {deletion['max_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/detector inference occurs anywhere in this benchmark -- "
        "every RawHumanDetection is synthetic and hand-built. See "
        "scripts/benchmark_yolo_human_detector.py for detector-side overhead, "
        "measured completely separately."
    )


if __name__ == "__main__":
    main()

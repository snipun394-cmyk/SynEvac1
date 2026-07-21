"""Human Behavior Recognition Framework milestone, Phase 11 performance
readiness benchmark.

Benchmarks RuleBasedBehaviorRecognizer's own overhead -- history
maintenance, metric computation, and behavior inference -- entirely
separate from YOLO/detector inference and separate from
SingleCameraTracker's own matching overhead: every TrackedHuman here is
synthetic and hand-built, never produced by a real tracker or detector
call.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_behavior_recognition.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from behavior_recognition.behavior_history import BehaviorHistory
from behavior_recognition.metrics import compute_metrics
from behavior_recognition.rule_based_recognizer import RuleBasedBehaviorRecognizer


PEOPLE_COUNT = 20
CYCLE_COUNT = 500


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _tracked(person_index: int, frame_index: int) -> TrackedHuman:

    x = person_index * 30.0 + frame_index * 5.0

    return TrackedHuman(
        track_id=f"T{person_index}", camera_id="CAM-BENCH",
        bounding_box=(x, 0.0, x + 20.0, 60.0), confidence=0.9,
        state=TrackState.TRACKED, age=frame_index + 1, frames_seen=frame_index + 1,
        frames_missing=0, last_timestamp=float(frame_index),
    )


def benchmark_history_maintenance_and_metrics() -> dict:

    history = BehaviorHistory()

    for person_index in range(PEOPLE_COUNT):
        history.append("CAM-BENCH", f"T{person_index}", 0.0, (0.0, 0.0, 20.0, 60.0))

    per_cycle_ms = []

    for frame_index in range(1, CYCLE_COUNT + 1):

        start = time.perf_counter()

        for person_index in range(PEOPLE_COUNT):
            tracked = _tracked(person_index, frame_index)
            history.append("CAM-BENCH", tracked.track_id, float(frame_index), tracked.bounding_box)
            compute_metrics(history.recent("CAM-BENCH", tracked.track_id), tracked.age, stationary_velocity_threshold=5.0)

        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "people_count": PEOPLE_COUNT,
        "cycle_count": CYCLE_COUNT,
        "mean_ms": statistics.mean(per_cycle_ms),
        "p95_ms": _percentile(per_cycle_ms, 0.95),
        "max_ms": max(per_cycle_ms),
    }


def benchmark_full_recognize_call() -> dict:

    recognizer = RuleBasedBehaviorRecognizer()

    recognizer.recognize("CAM-BENCH", 0.0, [_tracked(i, 0) for i in range(PEOPLE_COUNT)])

    per_cycle_ms = []

    for frame_index in range(1, CYCLE_COUNT + 1):

        tracked_humans = [_tracked(i, frame_index) for i in range(PEOPLE_COUNT)]

        start = time.perf_counter()
        recognizer.recognize("CAM-BENCH", float(frame_index), tracked_humans)
        per_cycle_ms.append((time.perf_counter() - start) * 1000)

    return {
        "people_count": PEOPLE_COUNT,
        "cycle_count": CYCLE_COUNT,
        "mean_ms": statistics.mean(per_cycle_ms),
        "p95_ms": _percentile(per_cycle_ms, 0.95),
        "max_ms": max(per_cycle_ms),
    }


def main():

    history_and_metrics = benchmark_history_maintenance_and_metrics()
    print(
        f"History maintenance + metric computation ({history_and_metrics['people_count']} people/camera): "
        f"{history_and_metrics['cycle_count']} cycles, mean {history_and_metrics['mean_ms']:.4f} ms, "
        f"p95 {history_and_metrics['p95_ms']:.4f} ms, max {history_and_metrics['max_ms']:.4f} ms"
    )

    full_call = benchmark_full_recognize_call()
    print(
        f"Full recognize() call, including classification ({full_call['people_count']} people/camera): "
        f"{full_call['cycle_count']} cycles, mean {full_call['mean_ms']:.4f} ms, "
        f"p95 {full_call['p95_ms']:.4f} ms, max {full_call['max_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/detector inference and zero SingleCameraTracker matching occurs "
        "anywhere in this benchmark -- every TrackedHuman is synthetic and hand-built. See "
        "scripts/benchmark_yolo_human_detector.py and scripts/benchmark_single_camera_tracker.py "
        "for detector-side and tracker-side overhead, measured completely separately."
    )


if __name__ == "__main__":
    main()

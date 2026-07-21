"""Cross-Camera Identity Resolution (ReID Framework) milestone,
Phase 11 performance readiness benchmark.

Benchmarks registry lookup, matching, transition evaluation, and
cleanup SEPARATELY -- every TrackedHuman/GlobalIdentityRecord here is
synthetic and hand-built, zero YOLO/tracker/behavior-recognizer
inference anywhere in this file.

Not a pytest test: timing assertions in CI are flaky by nature. Run
manually (`python scripts/benchmark_cross_camera_identity.py`) and read
the printed report.
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.matching import RuleBasedCrossCameraMatcher
from cross_camera_identity.observation import CrossCameraObservation
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology
from cross_camera_identity.transition_model import TransitionModel


IDENTITY_COUNT = 200
CYCLE_COUNT = 500


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _tracked(track_id: str, camera_id: str, age: int) -> TrackedHuman:

    return TrackedHuman(
        track_id=track_id, camera_id=camera_id, bounding_box=(0.0, 0.0, 10.0, 20.0),
        confidence=0.9, state=TrackState.TRACKED, age=age, frames_seen=age,
        frames_missing=0, last_timestamp=0.0,
    )


def benchmark_registry_lookup() -> dict:

    registry = IdentityRegistry()

    for i in range(IDENTITY_COUNT):
        registry.create(f"CAM-{i}", "T1", timestamp=0.0)

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        camera_id = f"CAM-{i % IDENTITY_COUNT}"
        start = time.perf_counter()
        registry.lookup_binding(camera_id, "T1")
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_matching() -> dict:

    topology = CameraTopology()
    for i in range(IDENTITY_COUNT):
        topology.add_transition(f"CAM-{i}", "CAM-ARRIVAL", min_transition_time=0.0, max_transition_time=1000.0)

    registry = IdentityRegistry()
    candidates = []
    for i in range(IDENTITY_COUNT):
        gid = registry.create(f"CAM-{i}", "T1", timestamp=0.0)
        registry.release(f"CAM-{i}", "T1")
        candidates.append(registry.get(gid))

    matcher = RuleBasedCrossCameraMatcher(default_max_transition_time=1000.0)

    per_call_ms = []

    for i in range(CYCLE_COUNT):

        observation = CrossCameraObservation(
            camera_id="CAM-ARRIVAL", track_id="T-NEW", timestamp=10.0,
            track_confidence=0.9, track_age=5, behavior=None,
        )

        start = time.perf_counter()
        matcher.find_match(observation, candidates, topology)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "candidate_count": IDENTITY_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_transition_evaluation() -> dict:

    topology = CameraTopology()
    topology.add_transition("CAM-A", "CAM-B", min_transition_time=1.0, max_transition_time=10.0)
    transition_model = TransitionModel(topology, timeout_seconds=30.0)

    registry = IdentityRegistry()
    gid = registry.create("CAM-A", "T1", timestamp=0.0)
    registry.release("CAM-A", "T1")
    record = registry.get(gid)

    per_call_ms = []

    for i in range(CYCLE_COUNT):
        start = time.perf_counter()
        transition_model.is_expired(record, now=5.0)
        transition_model.pending_transition_for(record)
        per_call_ms.append((time.perf_counter() - start) * 1000)

    return {"call_count": CYCLE_COUNT, "mean_ms": statistics.mean(per_call_ms), "p95_ms": _percentile(per_call_ms, 0.95)}


def benchmark_cleanup() -> dict:

    per_round_ms = []
    rounds = 100

    for round_index in range(rounds):

        registry = IdentityRegistry()
        topology = CameraTopology()
        transition_model = TransitionModel(topology, timeout_seconds=1.0)
        resolver = RuleBasedCrossCameraIdentityResolver(topology=topology, registry=registry, transition_model=transition_model)

        for i in range(IDENTITY_COUNT):
            resolver.resolve(f"CAM-{i}", 0.0, [_tracked("T1", f"CAM-{i}", age=5)], {})
            resolver.resolve(f"CAM-{i}", 0.1, [_tracked("T1", f"CAM-{i}", age=5)], {})  # touch, then...
            resolver.resolve(f"CAM-{i}", 0.2, [], {})  # (a MISSING cycle keeps it bound; use EXPIRED explicitly below)

        start = time.perf_counter()
        resolver.resolve("CAM-CLEANUP-TRIGGER", 100.0, [], {})  # long past every identity's 1s timeout
        per_round_ms.append((time.perf_counter() - start) * 1000)

    return {"round_count": rounds, "identity_count": IDENTITY_COUNT, "mean_ms": statistics.mean(per_round_ms), "p95_ms": _percentile(per_round_ms, 0.95)}


def main():

    lookup = benchmark_registry_lookup()
    print(f"Registry lookup: {lookup['call_count']} calls, mean {lookup['mean_ms']:.5f} ms, p95 {lookup['p95_ms']:.5f} ms")

    matching = benchmark_matching()
    print(
        f"Matching ({matching['candidate_count']} candidates): {matching['call_count']} calls, "
        f"mean {matching['mean_ms']:.4f} ms, p95 {matching['p95_ms']:.4f} ms"
    )

    transition = benchmark_transition_evaluation()
    print(f"Transition evaluation: {transition['call_count']} calls, mean {transition['mean_ms']:.5f} ms, p95 {transition['p95_ms']:.5f} ms")

    cleanup = benchmark_cleanup()
    print(
        f"Cleanup ({cleanup['identity_count']} identities/round): {cleanup['round_count']} rounds, "
        f"mean {cleanup['mean_ms']:.4f} ms, p95 {cleanup['p95_ms']:.4f} ms"
    )

    print()
    print(
        "NOTE: zero YOLO/tracker/behavior-recognizer inference occurs anywhere in this "
        "benchmark -- every TrackedHuman/GlobalIdentityRecord is synthetic and hand-built."
    )


if __name__ == "__main__":
    main()

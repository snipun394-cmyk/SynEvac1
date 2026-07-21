import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Sample = Tuple[float, BoundingBox]  # (timestamp, bounding_box)


@dataclass(frozen=True)
class TemporalMetrics:

    # Human Behavior Recognition Framework milestone, Phase 6 -- reusable
    # temporal metrics, computed ONCE per recognize() call and stored
    # separately from any behavior LABEL (behavior_recognition.observation.
    # BehaviorObservation.supporting_metrics) so a future ML model can
    # reuse these exact numbers as its own feature input instead of
    # recomputing them -- this is the whole reason metrics live in their
    # own module, independent of RuleBasedBehaviorRecognizer's own
    # thresholds/classification logic.
    #
    # Every field is Optional exactly where there is genuinely
    # insufficient history to compute it honestly (never a fabricated
    # 0.0/None-standing-in-for-unknown ambiguity) -- None always means
    # "not enough samples yet", never "measured as zero".

    velocity: Optional[float]           # centroid displacement magnitude, pixels/second
    direction: Optional[float]          # heading of the most recent displacement, radians (atan2 convention)
    distance_travelled: float           # cumulative centroid path length over the retained history window, pixels
    stationary_duration: float          # consecutive seconds, ending now, with velocity below the caller's threshold
    track_age: int                      # passed through from the current TrackedHuman.age, not derived from history
    acceleration: Optional[float] = None  # pixels/second^2, None with fewer than 3 samples


def _centroid(box: BoundingBox) -> Tuple[float, float]:

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:

    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compute_metrics(
    samples: Sequence[Sample],
    track_age: int,
    stationary_velocity_threshold: float,
) -> TemporalMetrics:

    # Pure function of a plain (timestamp, bounding_box) sample sequence
    # (oldest first) -- no camera/tracking/AI dependency of any kind,
    # exactly Phase 12's "depends only on geometry and time" requirement.
    # `samples` is expected to already be whatever bounded window a
    # caller's BehaviorHistory retains (this function never trims/stores
    # anything itself).

    if len(samples) < 2:

        return TemporalMetrics(
            velocity=None, direction=None, distance_travelled=0.0,
            stationary_duration=0.0, track_age=track_age, acceleration=None,
        )

    centroids = [(timestamp, _centroid(box)) for timestamp, box in samples]

    distance_travelled = sum(
        _distance(centroids[i][1], centroids[i + 1][1])
        for i in range(len(centroids) - 1)
    )

    (t_prev, c_prev), (t_last, c_last) = centroids[-2], centroids[-1]
    dt = t_last - t_prev

    if dt > 0:
        velocity = _distance(c_prev, c_last) / dt
        direction = math.atan2(c_last[1] - c_prev[1], c_last[0] - c_prev[0]) if velocity > 0 else None
    else:
        # Two samples sharing an identical timestamp -- no honest rate
        # to report (never divide by zero, never fabricate infinity).
        velocity = None
        direction = None

    acceleration = None

    if len(centroids) >= 3 and velocity is not None:

        (t_prev2, c_prev2) = centroids[-3]
        dt_prev = t_prev - t_prev2

        if dt_prev > 0:
            velocity_prev = _distance(c_prev2, c_prev) / dt_prev
            acceleration = (velocity - velocity_prev) / dt

    stationary_duration = _compute_stationary_duration(centroids, stationary_velocity_threshold)

    return TemporalMetrics(
        velocity=velocity,
        direction=direction,
        distance_travelled=distance_travelled,
        stationary_duration=stationary_duration,
        track_age=track_age,
        acceleration=acceleration,
    )


def _compute_stationary_duration(
    centroids: Sequence[Tuple[float, Tuple[float, float]]],
    stationary_velocity_threshold: float,
) -> float:

    # Walks backward from the most recent sample, summing consecutive
    # segment durations for as long as each segment's own velocity stays
    # at or below the threshold -- stops at the first segment that
    # exceeds it (or when history is exhausted). 0.0 if the most recent
    # segment is already moving.

    duration = 0.0

    for i in range(len(centroids) - 1, 0, -1):

        t_a, c_a = centroids[i - 1]
        t_b, c_b = centroids[i]

        dt = t_b - t_a

        if dt <= 0:
            break

        segment_velocity = _distance(c_a, c_b) / dt

        if segment_velocity > stationary_velocity_threshold:
            break

        duration += dt

    return duration

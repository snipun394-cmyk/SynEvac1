import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Sample = Tuple[float, BoundingBox]  # (timestamp, bounding_box)

Point = Tuple[float, float]
WorldSample = Tuple[float, Point]  # (timestamp, world_position)


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


@dataclass(frozen=True)
class WorldTemporalMetrics:

    # Camera Calibration & World Coordinate Projection milestone, Phase
    # 6 -- the WORLD-space counterpart to TemporalMetrics above, computed
    # from real building-space (meters) positions instead of image
    # pixels, whenever a camera_calibration.projection.WorldProjector
    # has supplied one. Deliberately a SEPARATE type, never a retrofit
    # of TemporalMetrics itself: the two are computed from genuinely
    # different inputs (world meters vs. image pixels) and this package
    # must be able to report "I have pixel-space metrics but no
    # world-space ones yet" (calibration not configured for this
    # camera) without conflating the two unit systems in one struct.
    #
    # Same "Optional means genuinely insufficient history, never a
    # fabricated zero" discipline as TemporalMetrics.

    world_velocity: Optional[float]            # meters/second
    world_direction: Optional[float]           # radians, atan2 convention
    world_distance_travelled: float            # meters, over the retained history window
    world_stationary_duration: float            # consecutive seconds, ending now, below the caller's world-space threshold


def _centroid(box: BoundingBox) -> Point:

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Point, b: Point) -> float:

    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _kinematics(
    points: Sequence[Tuple[float, Point]],
    stationary_velocity_threshold: float,
) -> Tuple[Optional[float], Optional[float], float, float, Optional[float]]:

    # Shared math behind both compute_metrics() (pixel centroids) and
    # compute_world_metrics() (world positions) -- identical formulas,
    # different unit system depending on what `points` actually holds.
    # Returns (velocity, direction, distance_travelled,
    # stationary_duration, acceleration).

    if len(points) < 2:
        return None, None, 0.0, 0.0, None

    distance_travelled = sum(
        _distance(points[i][1], points[i + 1][1])
        for i in range(len(points) - 1)
    )

    (t_prev, c_prev), (t_last, c_last) = points[-2], points[-1]
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

    if len(points) >= 3 and velocity is not None:

        (t_prev2, c_prev2) = points[-3]
        dt_prev = t_prev - t_prev2

        if dt_prev > 0:
            velocity_prev = _distance(c_prev2, c_prev) / dt_prev
            acceleration = (velocity - velocity_prev) / dt

    stationary_duration = _compute_stationary_duration(points, stationary_velocity_threshold)

    return velocity, direction, distance_travelled, stationary_duration, acceleration


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

    centroids = [(timestamp, _centroid(box)) for timestamp, box in samples]

    velocity, direction, distance_travelled, stationary_duration, acceleration = _kinematics(
        centroids, stationary_velocity_threshold,
    )

    return TemporalMetrics(
        velocity=velocity,
        direction=direction,
        distance_travelled=distance_travelled,
        stationary_duration=stationary_duration,
        track_age=track_age,
        acceleration=acceleration,
    )


def compute_world_metrics(
    world_samples: Sequence[WorldSample],
    stationary_velocity_threshold: float,
) -> WorldTemporalMetrics:

    # The world-space equivalent of compute_metrics() -- identical
    # math, applied directly to (timestamp, world_position) samples
    # rather than bounding-box centroids (no centroid step needed --
    # a world_position already IS a single point). Pure function, same
    # "geometry and time only" dependency as compute_metrics().

    velocity, direction, distance_travelled, stationary_duration, _acceleration = _kinematics(
        list(world_samples), stationary_velocity_threshold,
    )

    return WorldTemporalMetrics(
        world_velocity=velocity,
        world_direction=direction,
        world_distance_travelled=distance_travelled,
        world_stationary_duration=stationary_duration,
    )


def _compute_stationary_duration(
    points: Sequence[Tuple[float, Point]],
    stationary_velocity_threshold: float,
) -> float:

    # Walks backward from the most recent sample, summing consecutive
    # segment durations for as long as each segment's own velocity stays
    # at or below the threshold -- stops at the first segment that
    # exceeds it (or when history is exhausted). 0.0 if the most recent
    # segment is already moving.

    duration = 0.0

    for i in range(len(points) - 1, 0, -1):

        t_a, c_a = points[i - 1]
        t_b, c_b = points[i]

        dt = t_b - t_a

        if dt <= 0:
            break

        segment_velocity = _distance(c_a, c_b) / dt

        if segment_velocity > stationary_velocity_threshold:
            break

        duration += dt

    return duration

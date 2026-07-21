from typing import Mapping, Optional, Sequence, Tuple

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from behavior_recognition.behavior_history import BehaviorHistory, DEFAULT_MAX_HISTORY_LENGTH
from behavior_recognition.metrics import BoundingBox, TemporalMetrics, WorldTemporalMetrics, compute_metrics, compute_world_metrics
from behavior_recognition.observation import BehaviorObservation, RecognizedBehavior
from behavior_recognition.recognizer import BehaviorRecognizer


# Human Behavior Recognition Framework milestone, Phase 6 -- named
# defaults, never inline magic numbers. All in pixels/second (velocity),
# seconds (duration), or plain ratios -- tuned for a typical
# 1920x1080-ish camera frame; a caller overrides any of these to match
# its own camera's actual pixel/meter scale once one is known.
DEFAULT_STATIONARY_VELOCITY_THRESHOLD = 5.0
DEFAULT_RUNNING_VELOCITY_THRESHOLD = 80.0
DEFAULT_CONFIDENCE_SATURATION_SAMPLES = 10

DEFAULT_POSSIBLY_FALLEN_ASPECT_RATIO_THRESHOLD = 0.6
DEFAULT_POSSIBLY_FALLEN_MIN_STATIONARY_DURATION = 2.0
DEFAULT_POSSIBLY_FALLEN_CONFIDENCE_FACTOR = 0.5

# Camera Calibration & World Coordinate Projection milestone, Phase 6 --
# the WORLD-space (meters/second) counterpart to the pixel thresholds
# above, used in PREFERENCE to them whenever a world position is
# available for a track. 0.3 m/s is a typical "shifting weight/fidgeting
# while standing" upper bound for genuinely stationary; 2.5 m/s is a
# typical human running threshold (a brisk walk tops out around 2 m/s).
DEFAULT_WORLD_STATIONARY_VELOCITY_THRESHOLD = 0.3
DEFAULT_WORLD_RUNNING_VELOCITY_THRESHOLD = 2.5


class RuleBasedBehaviorRecognizer(BehaviorRecognizer):

    # A clean, deterministic engineering baseline -- plain velocity
    # thresholds over BehaviorHistory's own bounded per-track sample
    # window, computed via behavior_recognition.metrics.compute_metrics.
    # NOT a neural network, NOT pose estimation, NOT OpenPose/MediaPipe/
    # YOLO Pose/MoveNet (Phase 5's explicit exclusion list) -- a future
    # ML-based recognizer can replace this class entirely; nothing
    # importing BehaviorRecognizer (Phase 4's seam) needs to change when
    # that happens.
    #
    # STATIONARY / WALKING / RUNNING are ordered purely by
    # stationary_velocity_threshold / running_velocity_threshold --
    # UNKNOWN is returned whenever there is genuinely not enough history
    # yet to compute a velocity at all (TemporalMetrics.velocity is
    # None), never a guessed default.
    #
    # POSSIBLY_FALLEN is DISABLED BY DEFAULT (enable_possibly_fallen_
    # heuristic=False). The only geometric signal available to a rule-
    # based recognizer for this -- a bounding box becoming wide/low
    # (height/width below possibly_fallen_aspect_ratio_threshold) and
    # staying near-stationary for possibly_fallen_min_stationary_
    # duration seconds -- has real, common false-positive causes this
    # milestone cannot honestly rule out: a person crouching, sitting,
    # bending down, or picking something up up produces an
    # indistinguishable bounding-box shape. This is why Phase 5 calls it
    # "optional only if genuinely reliable" -- it is not genuinely
    # reliable from geometry alone, so it stays opt-in, and even when
    # enabled its confidence is scaled down by
    # possibly_fallen_confidence_factor to reflect that explicitly. It
    # is never remapped onto perception.models.human_observation.
    # HumanState.FALLEN anywhere in this codebase (see live_camera_
    # pipeline/pipeline.py's own behavior-to-HumanState mapping, and
    # docs/architecture/behavior_recognition.md Sec 4, for why).
    #
    # Confidence for STATIONARY/WALKING/RUNNING is a plain, documented
    # "more samples seen -> more confidence in the velocity estimate"
    # formula (min(1.0, sample_count / confidence_saturation_samples)) --
    # an honest engineering confidence-in-measurement, never a claimed
    # ML-derived probability.

    def __init__(
        self,
        history_length: int = DEFAULT_MAX_HISTORY_LENGTH,
        stationary_velocity_threshold: float = DEFAULT_STATIONARY_VELOCITY_THRESHOLD,
        running_velocity_threshold: float = DEFAULT_RUNNING_VELOCITY_THRESHOLD,
        confidence_saturation_samples: int = DEFAULT_CONFIDENCE_SATURATION_SAMPLES,
        enable_possibly_fallen_heuristic: bool = False,
        possibly_fallen_aspect_ratio_threshold: float = DEFAULT_POSSIBLY_FALLEN_ASPECT_RATIO_THRESHOLD,
        possibly_fallen_min_stationary_duration: float = DEFAULT_POSSIBLY_FALLEN_MIN_STATIONARY_DURATION,
        possibly_fallen_confidence_factor: float = DEFAULT_POSSIBLY_FALLEN_CONFIDENCE_FACTOR,
        world_stationary_velocity_threshold: float = DEFAULT_WORLD_STATIONARY_VELOCITY_THRESHOLD,
        world_running_velocity_threshold: float = DEFAULT_WORLD_RUNNING_VELOCITY_THRESHOLD,
    ):

        self.stationary_velocity_threshold = stationary_velocity_threshold
        self.running_velocity_threshold = running_velocity_threshold
        self.confidence_saturation_samples = confidence_saturation_samples

        self.enable_possibly_fallen_heuristic = enable_possibly_fallen_heuristic
        self.possibly_fallen_aspect_ratio_threshold = possibly_fallen_aspect_ratio_threshold
        self.possibly_fallen_min_stationary_duration = possibly_fallen_min_stationary_duration
        self.possibly_fallen_confidence_factor = possibly_fallen_confidence_factor

        self.world_stationary_velocity_threshold = world_stationary_velocity_threshold
        self.world_running_velocity_threshold = world_running_velocity_threshold

        self.history = BehaviorHistory(max_length=history_length)

    # =====================================================

    def recognize(
        self,
        camera_id: str,
        timestamp: float,
        tracked_humans: Sequence[TrackedHuman],
        world_positions_by_track_id: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> Tuple[BehaviorObservation, ...]:

        world_positions_by_track_id = world_positions_by_track_id or {}

        observations = []

        for tracked in tracked_humans:

            if tracked.state == TrackState.EXPIRED:
                # The track is gone for good -- forget its history now,
                # never accumulate a dict entry for a track_id that will
                # never be seen again (Phase 7's "no memory leaks").
                self.history.clear(camera_id, tracked.track_id)
                continue

            if tracked.state == TrackState.MISSING:
                # Not observed this cycle -- nothing honest to report,
                # and no new geometry to append either (coasting on
                # whatever was last actually seen).
                continue

            world_position = world_positions_by_track_id.get(tracked.track_id)

            self.history.append(camera_id, tracked.track_id, timestamp, tracked.bounding_box, world_position=world_position)
            samples = self.history.recent(camera_id, tracked.track_id)

            metrics = compute_metrics(samples, tracked.age, self.stationary_velocity_threshold)

            world_samples = self.history.recent_world(camera_id, tracked.track_id)
            world_metrics = (
                compute_world_metrics(world_samples, self.world_stationary_velocity_threshold)
                if world_samples else None
            )

            behavior, confidence = self._classify(metrics, world_metrics, tracked, sample_count=len(samples))

            observations.append(
                BehaviorObservation(
                    camera_id=camera_id,
                    track_id=tracked.track_id,
                    timestamp=timestamp,
                    recognized_behavior=behavior,
                    confidence=confidence,
                    supporting_metrics=metrics,
                    world_metrics=world_metrics,
                )
            )

        return tuple(observations)

    # =====================================================

    def _classify(
        self,
        metrics: TemporalMetrics,
        world_metrics: Optional[WorldTemporalMetrics],
        tracked: TrackedHuman,
        sample_count: int,
    ) -> Tuple[RecognizedBehavior, float]:

        # Camera Calibration & World Coordinate Projection milestone,
        # Phase 6 -- "operate using world-space motion instead of
        # pixel-space whenever calibration is available, gracefully
        # fall back to image-space if calibration is unavailable":
        # world_metrics.world_velocity (meters/second, against the
        # world-space thresholds) is used whenever it is honestly
        # computable; pixel-space metrics.velocity (against the
        # pixel-space thresholds) is used only when it is not.
        use_world = world_metrics is not None and world_metrics.world_velocity is not None

        velocity = world_metrics.world_velocity if use_world else metrics.velocity
        stationary_duration = world_metrics.world_stationary_duration if use_world else metrics.stationary_duration
        stationary_threshold = self.world_stationary_velocity_threshold if use_world else self.stationary_velocity_threshold
        running_threshold = self.world_running_velocity_threshold if use_world else self.running_velocity_threshold

        if velocity is None:
            return RecognizedBehavior.UNKNOWN, 0.0

        base_confidence = min(1.0, sample_count / self.confidence_saturation_samples)

        if (
            self.enable_possibly_fallen_heuristic
            and velocity <= stationary_threshold
            and stationary_duration >= self.possibly_fallen_min_stationary_duration
            and self._looks_fallen(tracked.bounding_box)
        ):
            return RecognizedBehavior.POSSIBLY_FALLEN, base_confidence * self.possibly_fallen_confidence_factor

        if velocity <= stationary_threshold:
            return RecognizedBehavior.STATIONARY, base_confidence

        if velocity < running_threshold:
            return RecognizedBehavior.WALKING, base_confidence

        return RecognizedBehavior.RUNNING, base_confidence

    # =====================================================

    def _looks_fallen(self, bounding_box: Optional[BoundingBox]) -> bool:

        if bounding_box is None:
            return False

        x1, y1, x2, y2 = bounding_box
        width = x2 - x1
        height = y2 - y1

        if width <= 0:
            return False

        return (height / width) < self.possibly_fallen_aspect_ratio_threshold

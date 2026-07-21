from typing import Optional, Sequence, Tuple

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from behavior_recognition.behavior_history import BehaviorHistory, DEFAULT_MAX_HISTORY_LENGTH
from behavior_recognition.metrics import BoundingBox, TemporalMetrics, compute_metrics
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
    ):

        self.stationary_velocity_threshold = stationary_velocity_threshold
        self.running_velocity_threshold = running_velocity_threshold
        self.confidence_saturation_samples = confidence_saturation_samples

        self.enable_possibly_fallen_heuristic = enable_possibly_fallen_heuristic
        self.possibly_fallen_aspect_ratio_threshold = possibly_fallen_aspect_ratio_threshold
        self.possibly_fallen_min_stationary_duration = possibly_fallen_min_stationary_duration
        self.possibly_fallen_confidence_factor = possibly_fallen_confidence_factor

        self.history = BehaviorHistory(max_length=history_length)

    # =====================================================

    def recognize(
        self,
        camera_id: str,
        timestamp: float,
        tracked_humans: Sequence[TrackedHuman],
    ) -> Tuple[BehaviorObservation, ...]:

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

            self.history.append(camera_id, tracked.track_id, timestamp, tracked.bounding_box)
            samples = self.history.recent(camera_id, tracked.track_id)

            metrics = compute_metrics(samples, tracked.age, self.stationary_velocity_threshold)
            behavior, confidence = self._classify(metrics, tracked, sample_count=len(samples))

            observations.append(
                BehaviorObservation(
                    camera_id=camera_id,
                    track_id=tracked.track_id,
                    timestamp=timestamp,
                    recognized_behavior=behavior,
                    confidence=confidence,
                    supporting_metrics=metrics,
                )
            )

        return tuple(observations)

    # =====================================================

    def _classify(
        self,
        metrics: TemporalMetrics,
        tracked: TrackedHuman,
        sample_count: int,
    ) -> Tuple[RecognizedBehavior, float]:

        if metrics.velocity is None:
            return RecognizedBehavior.UNKNOWN, 0.0

        base_confidence = min(1.0, sample_count / self.confidence_saturation_samples)

        if (
            self.enable_possibly_fallen_heuristic
            and metrics.velocity <= self.stationary_velocity_threshold
            and metrics.stationary_duration >= self.possibly_fallen_min_stationary_duration
            and self._looks_fallen(tracked.bounding_box)
        ):
            return RecognizedBehavior.POSSIBLY_FALLEN, base_confidence * self.possibly_fallen_confidence_factor

        if metrics.velocity <= self.stationary_velocity_threshold:
            return RecognizedBehavior.STATIONARY, base_confidence

        if metrics.velocity < self.running_velocity_threshold:
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

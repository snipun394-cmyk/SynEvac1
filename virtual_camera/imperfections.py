from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectionImperfectionModel:

    # Phase 4 -- optional simulation realism controls, off by default.
    # A plain, immutable parameter bag (same shape as simulator.capacity.
    # CapacityModel/simulator.congestion.CongestionModel's own
    # "documented, tunable constant, not a validated real-world sensor
    # figure" honesty) -- every field here defaults to "perfect
    # detection" so an existing caller that never heard of this class
    # gets byte-identical behavior to before it existed.

    # Probability [0.0, 1.0] that a visible, real occupant is actually
    # reported this tick -- 1.0 (the default) means every visible
    # occupant is always detected. Below 1.0 models a missed detection.
    detection_probability: float = 1.0

    # Probability [0.0, 1.0], evaluated once per detections_at() call,
    # of generating one extra synthetic detection with no corresponding
    # real occupant (see virtual_camera.detection.Detection.
    # is_false_positive). 0.0 (the default) never generates one.
    false_positive_rate: float = 0.0

    # Standard deviation of Gaussian noise added to a real detection's
    # confidence (clamped back to [0.0, 1.0]). 0.0 (the default) always
    # reports exactly 1.0 -- deterministic, no RNG draw at all.
    confidence_variation: float = 0.0

    # Seconds a detection's reported occupant position/classification/
    # state lags behind the real, current simulation time -- models a
    # real detector's processing latency. 0.0 (the default) means the
    # Virtual Camera resolves ground truth at exactly the requested
    # time, no delay at all.
    tracking_delay: float = 0.0

    # Seeds every random draw this model's owner (VirtualCamera) makes
    # -- missed-detection/false-positive/confidence-variation rolls --
    # so a research run comparing imperfection settings can still be
    # reproduced exactly. None (the default) uses fresh, non-
    # deterministic randomness, since is_perfect (see below) already
    # guarantees no RNG is ever touched unless at least one imperfection
    # is actually enabled.
    seed: Optional[int] = None

    # =====================================================

    def __post_init__(self):

        if not (0.0 <= self.detection_probability <= 1.0):

            raise ValueError(
                "detection_probability must be in [0.0, 1.0], got "
                f"{self.detection_probability!r}."
            )

        if not (0.0 <= self.false_positive_rate <= 1.0):

            raise ValueError(
                "false_positive_rate must be in [0.0, 1.0], got "
                f"{self.false_positive_rate!r}."
            )

        if self.confidence_variation < 0.0:

            raise ValueError(
                "confidence_variation must be >= 0.0, got "
                f"{self.confidence_variation!r}."
            )

        if self.tracking_delay < 0.0:

            raise ValueError(
                f"tracking_delay must be >= 0.0, got {self.tracking_delay!r}."
            )

    # =====================================================

    @property
    def is_perfect(self) -> bool:

        # The single switch VirtualCamera checks before touching any
        # RNG at all -- Phase 5's own determinism requirement
        # ("Detection stream remains deterministic when imperfections
        # are disabled") holds trivially true whenever this is True,
        # because no random draw of any kind occurs in that path.

        return (
            self.detection_probability >= 1.0
            and self.false_positive_rate <= 0.0
            and self.confidence_variation <= 0.0
            and self.tracking_delay <= 0.0
        )


# The shared, singleton "nothing configured" instance -- reused as the
# default everywhere in this package rather than each call site
# constructing its own equal-but-distinct DetectionImperfectionModel().
PERFECT_DETECTION = DetectionImperfectionModel()

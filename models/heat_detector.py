from dataclasses import dataclass
from typing import Optional

from models.sensor_asset import DetectorState, HealthStatus, SensorAsset


@dataclass
class HeatDetector(SensorAsset):

    # Phase 4's Heat Detector asset -- reuses the exact same framework
    # and DetectorState model as SmokeDetector (see that class's own
    # module docstring for the full "why this stays pure, no hazard/
    # perception import" reasoning; nothing here repeats it). Fixed-
    # temperature only, the same scope
    # perception.providers.ground_truth_heat_detector_provider.
    # GroundTruthHeatDetectorProvider already documents for its own
    # Ground Truth adapter -- Rate-of-Rise is out of scope for both.

    # Degrees Celsius, matching the units HazardNodeState.temperature
    # already uses throughout this codebase. 57.0 mirrors
    # GroundTruthHeatDetectorProvider.DEFAULT_ALARM_THRESHOLD's own
    # documented ~57C/135F fixed-temperature placeholder -- restated
    # independently here (not imported) per this codebase's own "same
    # vocabulary, independently restated" convention for a value this
    # small, rather than creating a cross-package import for one float.
    activation_threshold: float = 57.0

    # =====================================================

    def __post_init__(self):

        self.object_type = "HeatDetector"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_state(self, temperature: Optional[float], time: Optional[float] = None) -> DetectorState:

        if self.health_status != HealthStatus.OK:
            return DetectorState.FAULT

        if not self.active:
            return DetectorState.NORMAL

        if temperature is None:
            return DetectorState.NORMAL

        if temperature >= self.activation_threshold:

            if time is not None:
                self.last_activation_time = time

            return DetectorState.ALARM

        return DetectorState.NORMAL

    # =====================================================

    def to_dict(self):

        data = self._sensor_dict()

        data.update({
            "activation_threshold": self.activation_threshold,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._sensor_kwargs(data)

        kwargs.update(
            activation_threshold=data.get(
                "activation_threshold",
                57.0,
            ),
        )

        return cls(**kwargs)

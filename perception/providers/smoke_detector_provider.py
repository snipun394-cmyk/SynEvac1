from typing import List

from sensors.provider import DetectorProvider

from perception.models.smoke_detector_observation import SmokeDetectorReading


class SmokeDetectorProvider(DetectorProvider):

    # Specializes sensors.provider.DetectorProvider for smoke-type
    # devices specifically (Detector.detector_type == "Smoke") --
    # returns typed SmokeDetectorReadings instead of opaque
    # SensorReadings. Fusing multiple detectors' readings (smoke and
    # heat both covering the same node) into one ObservedNodeState is
    # a future Sensor Fusion stage's job, not this interface's -- this
    # class adds exactly one method and implements none of it.

    def alarm_states_at(self, time: float) -> List[SmokeDetectorReading]:

        raise NotImplementedError

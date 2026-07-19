from typing import List

from sensors.provider import DetectorProvider

from perception.models.heat_detector_observation import HeatDetectorReading


class HeatDetectorProvider(DetectorProvider):

    # Specializes sensors.provider.DetectorProvider for heat-type
    # devices specifically (Detector.detector_type == "Heat") --
    # returns typed HeatDetectorReadings instead of opaque
    # SensorReadings. Fusing multiple detectors' readings into one
    # ObservedNodeState is a future Sensor Fusion stage's job, not
    # this interface's -- this class adds exactly one method and
    # implements none of it.

    def alarm_states_at(self, time: float) -> List[HeatDetectorReading]:

        raise NotImplementedError

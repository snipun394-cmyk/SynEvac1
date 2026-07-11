from typing import List

from sensors.reading import SensorReading


class SensorProvider:

    # The one interface every real-world OR simulated data source is
    # meant to implement to get raw sensor data into the system --
    # mirrors HazardSource/HazardProvider's role as a single seam a
    # whole future subsystem plugs into. Deliberately independent of
    # Navigation, Pathfinding, Simulation, Behavior, Hazard Evolution,
    # and Occupancy: this is only a contract for "what does a sensor
    # give us", never for what that data means to any downstream
    # subsystem. This file, and every class in it, must stay free of
    # that coupling -- see sensors/replay.py for the one place a
    # concrete provider is deliberately allowed to bridge into the
    # Hazard/Occupancy pipelines.

    def readings_at(self, time: float) -> List[SensorReading]:

        raise NotImplementedError


class CameraProvider(SensorProvider):

    # Specializes SensorProvider for camera-shaped devices (CCTV,
    # vision models, a future RTSP/ONVIF+YOLO pipeline) -- still
    # returns only raw SensorReading objects, never a person-count or
    # any interpreted value directly. Exists so a future concrete
    # camera integration has a type to subclass that says "this is a
    # camera feed" in its own right, without adding any method beyond
    # SensorProvider's own -- CameraProvider and DetectorProvider stay
    # distinguishable by type even though their contract is identical.

    pass


class DetectorProvider(SensorProvider):

    # Specializes SensorProvider for detector-shaped devices (smoke/
    # heat/flame/gas, multi-criteria detectors, beam detectors, Manual
    # Call Points, a future Fire Alarm Control Panel feed). Same
    # non-committal shape as CameraProvider.

    pass


class NullSensorProvider(SensorProvider):

    # V1's trivial default -- proves the interface without any real
    # data source behind it, same role NullHazardSource plays for
    # HazardSource: has no opinion about anything, ever.

    def readings_at(self, time: float) -> List[SensorReading]:

        return []


class NullCameraProvider(CameraProvider):

    def readings_at(self, time: float) -> List[SensorReading]:

        return []


class NullDetectorProvider(DetectorProvider):

    def readings_at(self, time: float) -> List[SensorReading]:

        return []

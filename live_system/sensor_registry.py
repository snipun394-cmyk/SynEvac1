from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading


class SensorKind(Enum):

    # Every sensor family this phase names as a *future* concrete
    # implementation target (see each Sensor subclass below) --
    # deliberately not open-ended: a fifth kind is a new enum member
    # plus a new Sensor subclass, never a free string a registry
    # caller could misspell.

    CCTV = auto()
    SMOKE_DETECTOR = auto()
    HEAT_DETECTOR = auto()
    FIRE_ALARM_CONTROL_PANEL = auto()


# =====================================================
# Abstract sensor interfaces -- mirrors PerceptionProvider/
# HazardProvider/DecisionEngine's established "one seam, no concrete
# implementation yet" pattern (raise NotImplementedError). No
# networking, no ONVIF/RTSP, no hardware SDK, and no YOLO/vision model
# lives here or anywhere in this phase -- a concrete sensor (a real
# CCTV feed, a real FACP integration) is future work that satisfies
# this exact contract, constructed and injected by whoever wires up a
# real deployment. read() returning None means "no reading this
# cycle" (device offline/unavailable) -- never a fabricated stale or
# zero reading.
# =====================================================


class Sensor:

    def __init__(self, sensor_id: str):

        self.sensor_id = sensor_id

    # =====================================================

    def read(self, time: float):

        raise NotImplementedError


class CCTVSensor(Sensor):

    def read(self, time: float) -> Optional[CameraFrameObservation]:

        raise NotImplementedError


class SmokeDetectorSensor(Sensor):

    def read(self, time: float) -> Optional[SmokeDetectorReading]:

        raise NotImplementedError


class HeatDetectorSensor(Sensor):

    def read(self, time: float) -> Optional[HeatDetectorReading]:

        raise NotImplementedError


@dataclass(frozen=True)
class FACPReading:

    # Perception's own PerceptionSystemStatus carries exactly one field
    # sourced from a Fire Alarm Control Panel: panel_communication_ok
    # (see perception/models/building_observation.py) -- restated here
    # as this sensor kind's own reading shape rather than importing
    # PerceptionSystemStatus itself, since a single bool-plus-timestamp
    # reading is not that whole assembled type. active_alarm_zone_ids
    # is the FACP's own zone-addressed alarm bank, independent of (and
    # potentially more authoritative than) individual detector
    # readings -- included because a real FACP is commonly the system
    # of record for "which zones are currently in alarm," not merely a
    # communications-health reporter.

    panel_id: str
    timestamp: float
    panel_communication_ok: bool = True
    active_alarm_zone_ids: List[str] = field(default_factory=list)


class FireAlarmControlPanelSensor(Sensor):

    def read(self, time: float) -> Optional[FACPReading]:

        raise NotImplementedError


# =====================================================


@dataclass(frozen=True)
class SensorReadings:

    # One cycle's worth of raw readings, grouped exactly the way
    # perception.fusion.sensor_fusion.SensorFusion.fuse() already wants
    # them (camera_observations/smoke_detector_readings/
    # heat_detector_readings) plus this package's own FACPReading list
    # -- the direct output of SensorRegistry.read_all(), and the direct
    # input live_system.integration's PerceptionGateway composes
    # SensorFusion with.

    timestamp: float

    camera_observations: List[CameraFrameObservation] = field(default_factory=list)
    smoke_detector_readings: List[SmokeDetectorReading] = field(default_factory=list)
    heat_detector_readings: List[HeatDetectorReading] = field(default_factory=list)
    facp_readings: List[FACPReading] = field(default_factory=list)


# =====================================================


_KIND_BY_CLASS = {
    CCTVSensor: SensorKind.CCTV,
    SmokeDetectorSensor: SensorKind.SMOKE_DETECTOR,
    HeatDetectorSensor: SensorKind.HEAT_DETECTOR,
    FireAlarmControlPanelSensor: SensorKind.FIRE_ALARM_CONTROL_PANEL,
}


class UnknownSensorKindError(Exception):

    # Raised by register() for any Sensor that is not an instance of
    # one of the four abstract interfaces above -- a registry caller
    # gets a clear, immediate error instead of a sensor silently never
    # appearing in any read_all() bucket.

    pass


class SensorRegistry:

    # Maintains registered sensors, grouped by SensorKind -- purely
    # bookkeeping and dispatch, never networking (Phase 2's own scope
    # boundary): registering a sensor is a local method call, and
    # read_all() is a local loop over already-injected Sensor
    # instances, not a poll over a real device network.

    def __init__(self):

        self._sensors: Dict[str, Sensor] = {}
        self._kind_of: Dict[str, SensorKind] = {}

    # =====================================================

    def register(self, sensor: Sensor) -> None:

        kind = self._kind_of_sensor(sensor)

        if sensor.sensor_id in self._sensors:

            raise ValueError(
                f"A sensor with sensor_id={sensor.sensor_id!r} is already registered -- "
                f"unregister it first if this is meant to replace it."
            )

        self._sensors[sensor.sensor_id] = sensor
        self._kind_of[sensor.sensor_id] = kind

    # =====================================================

    def unregister(self, sensor_id: str) -> None:

        self._sensors.pop(sensor_id, None)
        self._kind_of.pop(sensor_id, None)

    # =====================================================

    def is_registered(self, sensor_id: str) -> bool:

        return sensor_id in self._sensors

    # =====================================================

    def sensor(self, sensor_id: str) -> Sensor:

        return self._sensors[sensor_id]

    # =====================================================

    def sensors_of_kind(self, kind: SensorKind) -> List[Sensor]:

        return [
            sensor for sensor_id, sensor in self._sensors.items()
            if self._kind_of[sensor_id] == kind
        ]

    # =====================================================

    def all_sensors(self) -> List[Sensor]:

        return list(self._sensors.values())

    # =====================================================

    def __len__(self) -> int:

        return len(self._sensors)

    # =====================================================

    def read_all(self, time: float) -> SensorReadings:

        # A sensor whose read() returns None this cycle contributes
        # nothing -- never a stale carry-forward, never a fabricated
        # default reading. Matches CameraProvider/HazardProvider's own
        # "absence means no opinion" convention throughout this
        # codebase.

        camera_observations = [
            reading for sensor in self.sensors_of_kind(SensorKind.CCTV)
            if (reading := sensor.read(time)) is not None
        ]
        smoke_detector_readings = [
            reading for sensor in self.sensors_of_kind(SensorKind.SMOKE_DETECTOR)
            if (reading := sensor.read(time)) is not None
        ]
        heat_detector_readings = [
            reading for sensor in self.sensors_of_kind(SensorKind.HEAT_DETECTOR)
            if (reading := sensor.read(time)) is not None
        ]
        facp_readings = [
            reading for sensor in self.sensors_of_kind(SensorKind.FIRE_ALARM_CONTROL_PANEL)
            if (reading := sensor.read(time)) is not None
        ]

        return SensorReadings(
            timestamp=time,
            camera_observations=camera_observations,
            smoke_detector_readings=smoke_detector_readings,
            heat_detector_readings=heat_detector_readings,
            facp_readings=facp_readings,
        )

    # =====================================================

    def _kind_of_sensor(self, sensor: Sensor) -> SensorKind:

        for sensor_cls, kind in _KIND_BY_CLASS.items():
            if isinstance(sensor, sensor_cls):
                return kind

        raise UnknownSensorKindError(
            f"{type(sensor).__name__} is not a recognized Sensor subclass -- expected an "
            f"instance of one of {[cls.__name__ for cls in _KIND_BY_CLASS]}."
        )

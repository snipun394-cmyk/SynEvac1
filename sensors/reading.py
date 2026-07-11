from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SensorReading:

    # One raw, protocol-agnostic reading from one physical or
    # simulated sensor at one point in time -- the single shape every
    # future adapter (a detector protocol, an RTSP/ONVIF+YOLO camera
    # pipeline, a Fire Alarm Control Panel feed, a Manual Call Point,
    # a firefighter's manual report, a BLE/UWB or WiFi localization
    # feed) is meant to normalize its raw data into before anything
    # else in the system ever sees it.
    #
    # `payload` is deliberately opaque here: SensorProvider and its
    # subclasses never interpret it, only carry it. Turning a payload
    # into a HazardContribution or an OccupancyObservation is a job
    # for a future translation layer built on top of this contract,
    # not for this class -- keeps this module free of any opinion
    # about what a reading *means*.

    sensor_id: str
    timestamp: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

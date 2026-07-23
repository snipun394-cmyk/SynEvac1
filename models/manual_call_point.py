from dataclasses import dataclass
from typing import Optional

from models.sensor_asset import DetectorState, HealthStatus, SensorAsset


@dataclass
class ManualCallPoint(SensorAsset):

    # Manual Call Point (MCP) / pull station -- the physical fire-alarm
    # device a building occupant activates by hand. Reuses the exact
    # same SensorAsset foundation SmokeDetector/HeatDetector already
    # establish (models/sensor_asset.py's own docstring names "Manual
    # Call Point" directly as a future candidate that fits this shape
    # unchanged) -- same id/name/floor_id/zone_ids/position/mount_
    # height/active/mode/connection/health_status/installation_date/
    # last_activation_time, nothing duplicated.
    #
    # UNLIKE SmokeDetector/HeatDetector, an MCP has no continuous
    # external hazard reading to compare against a threshold -- there is
    # no "smoke level" or "temperature" a Ground Truth provider computes
    # for it. Activation is a direct, binary HUMAN ACTION on the device
    # itself, so `activated` is stored directly as the device's own
    # intrinsic state (mirroring a real pull station, which stays
    # visibly activated/latched until a technician physically resets
    # it) rather than passed into compute_state() as an external
    # reading the way smoke_level/temperature are. This is why
    # compute_state() below takes no reading parameter, unlike its two
    # siblings.

    activated: bool = False

    # =====================================================

    def __post_init__(self):

        self.object_type = "ManualCallPoint"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def activate(self):

        self.activated = True

    # =====================================================

    def restore(self):

        # The physical device's own reset (a technician re-arming the
        # pull station at the box itself) -- deliberately separate from
        # facp.engine.SimulatedFACP.reset() (the PANEL's own operator
        # reset), the same "device-level vs. panel-level" split
        # SimulatedFACP.manual_alarm()'s own docstring already draws for
        # its pre-existing no-backing-asset manual-alarm path. Resetting
        # the panel while this device is still physically activated must
        # never silently clear it -- see facp/engine.py's own state
        # machine notes.

        self.activated = False

    # =====================================================

    def compute_state(self, time: Optional[float] = None) -> DetectorState:

        if self.health_status != HealthStatus.OK:
            return DetectorState.FAULT

        if not self.active:
            return DetectorState.NORMAL

        if self.activated:

            if time is not None:
                self.last_activation_time = time

            return DetectorState.ALARM

        return DetectorState.NORMAL

    # =====================================================

    def to_dict(self):

        data = self._sensor_dict()

        data.update({
            "activated": self.activated,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._sensor_kwargs(data)

        kwargs.update(
            activated=data.get(
                "activated",
                False,
            ),
        )

        return cls(**kwargs)

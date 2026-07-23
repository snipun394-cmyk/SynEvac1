from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.sensor_asset import HealthStatus


class EmergencyLightAvailability:

    # The one thing SynEvac can honestly say about an Emergency Light
    # right now: is it currently able to provide egress lighting. NOT a
    # hazard/alarm concept (no ALARM state -- a light has no "detection"
    # of anything) and deliberately NOT wired into route safety/
    # Pathfinding (Phase 8's own explicit scope boundary -- "unavailable"
    # is an honest maintenance/status fact, never itself evidence a
    # route is unsafe; that would require a real safety-policy model and
    # real evidence this codebase does not have yet).

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    FAULT = "FAULT"

    ALL = (AVAILABLE, UNAVAILABLE, FAULT)


@dataclass
class EmergencyLight(EngineeringAsset):

    # A building safety OUTPUT asset -- not a sensor (it detects
    # nothing, so it is NOT SensorAsset-based, unlike SmokeDetector/
    # HeatDetector/ManualCallPoint). Reuses EngineeringAsset exactly as
    # Camera/DynamicEvacuationSign do (id/name/floor_id/zone_ids/
    # position/mount_height/active/mode/connection) -- a physical device
    # placed in the building with the same asset-management needs as
    # every other engineering asset.
    #
    # health_status reuses models.sensor_asset.HealthStatus's own
    # existing OK/Fault/Offline vocabulary directly (Phase 6's own
    # "reuse existing status vocabulary" instruction) -- added here
    # explicitly rather than through SensorAsset, since this asset is
    # not a sensor and must not inherit SensorAsset's own installation_
    # date/last_activation_time fields (an alarm-activation timestamp
    # has no meaning for a light).
    #
    # No lux/photometric calculation, no battery-runtime physics -- only
    # what this codebase can honestly represent today (Phase 6's own
    # explicit scope boundary).

    health_status: str = HealthStatus.OK

    # Exposed for the Property Panel combo box, same convention
    # Detector.DETECTOR_TYPES/Speaker.SPEAKER_TYPES already use for a
    # small closed set. Purely descriptive -- never affects
    # compute_availability() below.
    LIGHT_TYPES = (
        "Wall Mounted",
        "Ceiling Mounted",
        "Recessed",
    )

    light_type: str = "Wall Mounted"

    # =====================================================

    def __post_init__(self):

        self.object_type = "EmergencyLight"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_availability(self) -> str:

        # Deliberately simple and total -- no external reading, no
        # hazard/perception input of any kind, mirroring this asset's
        # own "output/status asset, not a sensor" role. FAULT (the
        # device itself reports a problem) takes priority over a merely
        # inactive/offline light, the same "device fault outranks
        # anything else" priority SensorAsset-based compute_state()
        # methods already establish.

        if self.health_status == HealthStatus.FAULT:
            return EmergencyLightAvailability.FAULT

        if not self.active or self.health_status == HealthStatus.OFFLINE:
            return EmergencyLightAvailability.UNAVAILABLE

        return EmergencyLightAvailability.AVAILABLE

    # =====================================================

    def to_dict(self):

        data = self._asset_dict()

        data.update({
            "health_status": self.health_status,
            "light_type": self.light_type,
        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        kwargs = cls._asset_kwargs(data)

        kwargs.update(
            health_status=data.get(
                "health_status",
                HealthStatus.OK,
            ),

            light_type=data.get(
                "light_type",
                "Wall Mounted",
            ),
        )

        return cls(**kwargs)

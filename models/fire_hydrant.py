from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.fire_safety_asset import PassiveFireSafetyAvailability, compute_passive_availability
from models.sensor_asset import HealthStatus


@dataclass
class FireHydrant(EngineeringAsset):

    # An internal fire hydrant / landing valve outlet point -- the
    # connection point firefighters attach a hose to, not a hydraulic
    # network. No pump, tank, pipe routing, pressure, or flow-rate
    # modeling of any kind -- see docs/architecture/
    # fire_suppression_safety_assets.md's "no fabricated hydraulic
    # performance" boundary. Reuses EngineeringAsset + health_status
    # directly, same shape as FireExtinguisher/HoseReel (see
    # models.fire_safety_asset's own module docstring for why these
    # three share compute_passive_availability() rather than each
    # restating it).

    health_status: str = HealthStatus.OK

    # A small, standard closed set (Property Panel combo box), purely
    # descriptive.
    HYDRANT_TYPES = (
        "Wet Riser Landing Valve",
        "Dry Riser Landing Valve",
        "External Hydrant",
    )

    hydrant_type: str = "Wet Riser Landing Valve"

    # =====================================================

    def __post_init__(self):

        self.object_type = "FireHydrant"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_availability(self) -> str:

        return compute_passive_availability(self.active, self.health_status)

    # =====================================================

    def to_dict(self):

        data = self._asset_dict()

        data.update({
            "health_status": self.health_status,
            "hydrant_type": self.hydrant_type,
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

            hydrant_type=data.get(
                "hydrant_type",
                "Wet Riser Landing Valve",
            ),
        )

        return cls(**kwargs)

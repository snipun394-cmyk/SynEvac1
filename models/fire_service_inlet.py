from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.fire_safety_asset import PassiveFireSafetyAvailability, compute_passive_availability
from models.sensor_asset import HealthStatus


@dataclass
class FireServiceInlet(EngineeringAsset):

    # A Fire Service Inlet / Breeching Inlet -- the external connection
    # point a fire engine pumps into to boost/supply the building's
    # fire water system. A passive resource, same shape as
    # FireExtinguisher/FireHydrant/HoseReel (models/fire_safety_asset.py):
    # no continuous reading, only availability. No external fire-engine
    # simulation and no assumed pressure/flow of any kind -- this class
    # only represents that the connection point exists, where, and
    # whether it is currently usable.

    health_status: str = HealthStatus.OK

    INLET_TYPES = (
        "Wet Riser Inlet",
        "Dry Riser Inlet",
        "Sprinkler System Inlet",
    )

    inlet_type: str = "Wet Riser Inlet"

    # =====================================================

    def __post_init__(self):

        self.object_type = "FireServiceInlet"

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
            "inlet_type": self.inlet_type,
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

            inlet_type=data.get(
                "inlet_type",
                "Wet Riser Inlet",
            ),
        )

        return cls(**kwargs)

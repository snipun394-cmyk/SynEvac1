from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.fire_safety_asset import PassiveFireSafetyAvailability, compute_passive_availability
from models.sensor_asset import HealthStatus


@dataclass
class HoseReel(EngineeringAsset):

    # A fixed hose reel installation point -- identity, location, and
    # usability only. No hose-reach geometry, no coverage-radius
    # modeling: nothing in this codebase claims a zone is "protected"
    # merely because a HoseReel exists there (see docs/architecture/
    # fire_suppression_safety_assets.md). Reuses EngineeringAsset +
    # health_status directly, same shape as FireExtinguisher/
    # FireHydrant.
    #
    # Deliberately no hose_reel_type field -- investigation found no
    # standards data this codebase can meaningfully use for a hose
    # reel classification the way EmergencyLight's mounting style or
    # FireExtinguisher's agent type are meaningfully descriptive; kept
    # minimal rather than inventing a closed set with no consumer.

    health_status: str = HealthStatus.OK

    # =====================================================

    def __post_init__(self):

        self.object_type = "HoseReel"

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
        )

        return cls(**kwargs)

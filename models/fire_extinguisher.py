from dataclasses import dataclass

from models.engineering_asset import EngineeringAsset
from models.fire_safety_asset import PassiveFireSafetyAvailability, compute_passive_availability
from models.sensor_asset import HealthStatus


@dataclass
class FireExtinguisher(EngineeringAsset):

    # A passive, manually-operated fire-safety RESOURCE -- not a sensor
    # (it detects nothing) and not an automatically-activating device
    # (unlike Sprinkler, see models.sprinkler.Sprinkler's own
    # docstring for that distinction). Reuses EngineeringAsset exactly
    # as EmergencyLight/Camera/DynamicEvacuationSign do; adds
    # health_status directly (not through SensorAsset -- same reasoning
    # models.emergency_light.EmergencyLight already documents: this
    # asset has no installation_date/last_activation_time concept an
    # alarm-reporting sensor needs).
    #
    # Never automatically extinguishes anything -- see
    # docs/architecture/fire_suppression_safety_assets.md's own
    # "digital-twin asset state != physical fire-suppression effect"
    # boundary. This model only ever answers "does this extinguisher
    # exist, where is it, and is it currently usable".

    health_status: str = HealthStatus.OK

    # A small, standard closed set (Property Panel combo box), purely
    # descriptive -- never affects compute_availability() below, the
    # same "descriptive only" role models.emergency_light.EmergencyLight.
    # LIGHT_TYPES already establishes for its own light_type field.
    EXTINGUISHER_TYPES = (
        "Water",
        "Foam",
        "CO2",
        "Dry Powder",
        "Wet Chemical",
    )

    extinguisher_type: str = "Water"

    # =====================================================

    def __post_init__(self):

        self.object_type = "FireExtinguisher"

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
            "extinguisher_type": self.extinguisher_type,
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

            extinguisher_type=data.get(
                "extinguisher_type",
                "Water",
            ),
        )

        return cls(**kwargs)

from dataclasses import dataclass
from typing import Optional

from models.engineering_asset import EngineeringAsset
from models.sensor_asset import HealthStatus


class TankOperationalState:

    # Deliberately conservative: LOW_LEVEL/EMPTY are only ever reported
    # when a real current_level_liters measurement says so. A tank with
    # no measurement configured (current_level_liters is None -- "not
    # measured/unknown", never a fabricated 0.0) reports AVAILABLE
    # unless its own health/active state says otherwise -- the honest
    # default is "nothing contradicts normal operation", never a
    # guessed LOW_LEVEL/EMPTY the model has no evidence for.

    AVAILABLE = "AVAILABLE"
    LOW_LEVEL = "LOW_LEVEL"
    EMPTY = "EMPTY"
    FAULT = "FAULT"
    UNAVAILABLE = "UNAVAILABLE"

    ALL = (AVAILABLE, LOW_LEVEL, EMPTY, FAULT, UNAVAILABLE)


@dataclass
class FireWaterTank(EngineeringAsset):

    # A fire water storage tank -- an EngineeringAsset (not a sensor,
    # not a pump): it has no continuous alarm-driven reading and no
    # run state, only a health/level fact about a static reservoir.
    # Capacity/level units are explicit LITERS throughout (Phase 3's
    # own "make units explicit" instruction) -- never left ambiguous.
    #
    # No refill simulation, no consumption-rate modeling, no hydraulic
    # duration calculation: current_level_liters is a plain caller-
    # reported fact (mirrors every other intrinsic-state field in this
    # codebase -- ManualCallPoint.activated, PumpAsset.running), never
    # derived from anything this framework simulates.

    health_status: str = HealthStatus.OK

    capacity_liters: float = 0.0

    # None = not measured/unknown -- the honest default. A negative
    # value or a value exceeding capacity_liters is never silently
    # clamped by this class; validation_framework/Property Panel input
    # handling is where a caller would reject one, not here (this is a
    # plain data container, the same "no domain validation on the
    # constructor itself" convention every other model dataclass in
    # this codebase already follows).
    current_level_liters: Optional[float] = None

    # A conservative, documented placeholder (Phase 3's own
    # "if current water quantity is represented... validation strict"
    # instruction, applied the same way HeatDetector.activation_threshold's
    # own 57.0C placeholder is documented) -- below this fraction of
    # capacity_liters, the tank is reported LOW_LEVEL rather than
    # AVAILABLE.
    LOW_LEVEL_FRACTION = 0.2

    # =====================================================

    def __post_init__(self):

        self.object_type = "FireWaterTank"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def compute_state(self) -> str:

        if self.health_status != HealthStatus.OK:
            return TankOperationalState.FAULT

        if not self.active:
            return TankOperationalState.UNAVAILABLE

        if self.current_level_liters is None:
            return TankOperationalState.AVAILABLE

        if self.current_level_liters <= 0:
            return TankOperationalState.EMPTY

        if self.capacity_liters > 0 and self.current_level_liters < self.LOW_LEVEL_FRACTION * self.capacity_liters:
            return TankOperationalState.LOW_LEVEL

        return TankOperationalState.AVAILABLE

    # =====================================================

    def to_dict(self):

        data = self._asset_dict()

        data.update({
            "health_status": self.health_status,
            "capacity_liters": self.capacity_liters,
            "current_level_liters": self.current_level_liters,
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

            capacity_liters=data.get(
                "capacity_liters",
                0.0,
            ),

            current_level_liters=data.get(
                "current_level_liters",
                None,
            ),
        )

        return cls(**kwargs)

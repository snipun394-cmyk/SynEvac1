from dataclasses import dataclass

from models.pump_asset import PumpAsset


@dataclass
class JockeyPump(PumpAsset):

    # A smaller-duty pressure-maintenance pump on the same Fire Water
    # System as a FirePump -- genuinely a pump (shares PumpAsset, see
    # that class's own docstring for why this is a real subclass
    # relationship, not a coincidental shape overlap). Its existence in
    # this model must never be read as establishing system pressure by
    # itself -- RUNNING means only "the digital twin reports this pump
    # as running" (PumpAsset.compute_state()'s own docstring), exactly
    # as honest and exactly as limited as FirePump's own RUNNING state.

    def __post_init__(self):

        self.object_type = "JockeyPump"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def to_dict(self):

        return self._pump_dict()

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        return cls(**cls._pump_kwargs(data))

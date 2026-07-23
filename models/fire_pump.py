from dataclasses import dataclass

from models.pump_asset import PumpAsset


@dataclass
class FirePump(PumpAsset):

    # The primary fire pump supplying pressure to a Fire Water System
    # (see models/fire_water_system.py). See PumpAsset's own docstring
    # for the full run-state/control-mode semantics shared with
    # JockeyPump -- nothing is added here beyond object_type/move_to/
    # to_dict/from_dict, the same minimal-subclass shape SmokeDetector/
    # HeatDetector already establish over SensorAsset.

    def __post_init__(self):

        self.object_type = "FirePump"

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

from dataclasses import dataclass, field

from models.base_object import BaseObject
from utils.geometry import Point


@dataclass
class AssemblyPoint(BaseObject):
    position: Point = field(default_factory=lambda: Point(0.0, 0.0))
    capacity: int = 0

    def __post_init__(self):
        self.object_type = "AssemblyPoint"

    def to_dict(self):
        data = super().to_dict()
        data.update({
            "position": {
                "x": self.position.x,
                "y": self.position.y
            },
            "capacity": self.capacity,
        })
        return data
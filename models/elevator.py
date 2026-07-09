from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Elevator(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    width: float = 2.0
    depth: float = 2.0

    capacity: int = 15

    speed: float = 1.5

    current_floor: int = 0

    is_operational: bool = True

    evacuation_enabled: bool = False

    # =====================================================

    def __post_init__(self):

        self.object_type = "Elevator"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "width": self.width,

            "depth": self.depth,

            "capacity": self.capacity,

            "speed": self.speed,

            "current_floor": self.current_floor,

            "is_operational": self.is_operational,

            "evacuation_enabled": self.evacuation_enabled,

        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        return cls(

            id=data["id"],

            name=data.get(
                "name",
                "",
            ),

            properties=data.get(
                "properties",
                {},
            ),

            created_at=data.get(
                "created_at",
                "",
            ),

            modified_at=data.get(
                "modified_at",
                "",
            ),

            position=tuple(
                data.get(
                    "position",
                    (0.0, 0.0),
                )
            ),

            floor_id=data.get(
                "floor_id",
                "",
            ),

            width=data.get(
                "width",
                2.0,
            ),

            depth=data.get(
                "depth",
                2.0,
            ),

            capacity=data.get(
                "capacity",
                15,
            ),

            speed=data.get(
                "speed",
                1.5,
            ),

            current_floor=data.get(
                "current_floor",
                0,
            ),

            is_operational=data.get(
                "is_operational",
                True,
            ),

            evacuation_enabled=data.get(
                "evacuation_enabled",
                False,
            ),
        )
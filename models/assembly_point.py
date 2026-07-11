from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class AssemblyPoint(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    length: float = 10.0
    width: float = 10.0

    # Optional -- 0 means unspecified/no capacity limit, same
    # convention Zone uses for max_occupancy.
    capacity: int = 0

    description: str = ""

    active: bool = True

    # =====================================================

    def __post_init__(self):

        self.object_type = "AssemblyPoint"

    # =====================================================

    @property
    def area(self):

        return self.length * self.width

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "length": self.length,

            "width": self.width,

            "capacity": self.capacity,

            "description": self.description,

            "active": self.active,

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

            length=data.get(
                "length",
                10.0,
            ),

            width=data.get(
                "width",
                10.0,
            ),

            capacity=data.get(
                "capacity",
                0,
            ),

            description=data.get(
                "description",
                "",
            ),

            active=data.get(
                "active",
                True,
            ),
        )

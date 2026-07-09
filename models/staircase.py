from dataclasses import dataclass, field

from models.base_object import BaseObject
from utils.geometry import Point, Polygon


@dataclass
class Staircase(BaseObject):

    polygon: Polygon = field(default_factory=Polygon)

    floor_id: str = ""

    connected_floor_id: str = ""

    width: float = 1.50

    direction: str = "Both"

    capacity: int = 100

    # =====================================================

    def __post_init__(self):

        self.object_type = "Staircase"

    # =====================================================

    @property
    def center(self):

        if not self.polygon.points:
            return (0.0, 0.0)

        x = sum(
            p.x for p in self.polygon.points
        ) / len(self.polygon.points)

        y = sum(
            p.y for p in self.polygon.points
        ) / len(self.polygon.points)

        return (x, y)

    # =====================================================

    @property
    def area(self):

        return self.polygon.area

    # =====================================================

    def move(self, dx, dy):

        self.polygon.move(dx, dy)

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "polygon": self.polygon.to_list(),

            "floor_id": self.floor_id,

            "connected_floor_id": self.connected_floor_id,

            "width": self.width,

            "direction": self.direction,

            "capacity": self.capacity,

        })

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        polygon = Polygon()

        for x, y in data.get(
            "polygon",
            [],
        ):

            polygon.add(
                Point(x, y)
            )

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

            polygon=polygon,

            floor_id=data.get(
                "floor_id",
                "",
            ),

            connected_floor_id=data.get(
                "connected_floor_id",
                "",
            ),

            width=data.get(
                "width",
                1.5,
            ),

            direction=data.get(
                "direction",
                "Both",
            ),

            capacity=data.get(
                "capacity",
                100,
            ),
        )
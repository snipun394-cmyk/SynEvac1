from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Staircase(BaseObject):

    start_point: tuple = (0.0, 0.0)
    end_point: tuple = (0.0, 0.0)

    floor_id: str = ""

    width: float = 1.50

    # Placeholder only -- full multi-floor vertical connectivity
    # (Navigation Graph inter-floor edges) is deferred. This just
    # records which floor the stair leads to.
    connected_floor_id: str = ""

    def __post_init__(self):

        self.object_type = "Staircase"

    # =====================================================

    @property
    def center(self):

        return (
            (
                self.start_point[0]
                + self.end_point[0]
            )
            / 2,
            (
                self.start_point[1]
                + self.end_point[1]
            )
            / 2,
        )

    # =====================================================

    @property
    def length(self):

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        return (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

    # =====================================================

    def move(self, dx, dy):

        self.start_point = (
            self.start_point[0] + dx,
            self.start_point[1] + dy,
        )

        self.end_point = (
            self.end_point[0] + dx,
            self.end_point[1] + dy,
        )

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update(
            {
                "start_point": self.start_point,
                "end_point": self.end_point,
                "floor_id": self.floor_id,
                "width": self.width,
                "connected_floor_id": self.connected_floor_id,
            }
        )

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

            start_point=tuple(
                data.get(
                    "start_point",
                    (0.0, 0.0),
                )
            ),

            end_point=tuple(
                data.get(
                    "end_point",
                    (0.0, 0.0),
                )
            ),

            floor_id=data.get(
                "floor_id",
                "",
            ),

            width=data.get(
                "width",
                1.50,
            ),

            connected_floor_id=data.get(
                "connected_floor_id",
                "",
            ),
        )

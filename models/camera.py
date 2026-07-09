from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Camera(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    rotation: float = 0.0

    field_of_view: float = 90.0

    range: float = 25.0

    active: bool = True

    # =====================================================

    def __post_init__(self):

        self.object_type = "Camera"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def rotate(self, angle):

        self.rotation = angle

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "rotation": self.rotation,

            "field_of_view": self.field_of_view,

            "range": self.range,

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

            rotation=data.get(
                "rotation",
                0.0,
            ),

            field_of_view=data.get(
                "field_of_view",
                90.0,
            ),

            range=data.get(
                "range",
                25.0,
            ),

            active=data.get(
                "active",
                True,
            ),
        )
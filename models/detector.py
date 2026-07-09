from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Detector(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    detector_type: str = "Smoke"

    activation_temperature: float = 68.0

    coverage_radius: float = 7.5

    active: bool = True

    # =====================================================

    def __post_init__(self):

        self.object_type = "Detector"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "detector_type": self.detector_type,

            "activation_temperature": self.activation_temperature,

            "coverage_radius": self.coverage_radius,

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

            detector_type=data.get(
                "detector_type",
                "Smoke",
            ),

            activation_temperature=data.get(
                "activation_temperature",
                68.0,
            ),

            coverage_radius=data.get(
                "coverage_radius",
                7.5,
            ),

            active=data.get(
                "active",
                True,
            ),
        )
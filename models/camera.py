import math

from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Camera(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    rotation: float = 0.0

    horizontal_fov: float = 90.0

    max_range: float = 25.0

    mount_height: float = 3.0

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
    # Derived Coverage Geometry
    #
    # Never stored -- always recomputed from Position, Rotation,
    # Horizontal FOV and Max Range. A sector (fan) polygon: the
    # camera position, followed by points along the arc from
    # -fov/2 to +fov/2 around the facing direction. 0 degrees
    # points along +x, increasing clockwise -- matching Qt's own
    # rotation convention so the model and the graphics item
    # agree without either depending on the other.
    # =====================================================

    def coverage_polygon(self, segments=24):

        cx, cy = self.position

        half_fov = self.horizontal_fov / 2

        points = [(cx, cy)]

        for i in range(segments + 1):

            angle_degrees = (
                self.rotation
                - half_fov
                + (self.horizontal_fov * i / segments)
            )

            angle_radians = math.radians(angle_degrees)

            points.append(
                (
                    cx + self.max_range * math.cos(angle_radians),
                    cy + self.max_range * math.sin(angle_radians),
                )
            )

        return points

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "rotation": self.rotation,

            "horizontal_fov": self.horizontal_fov,

            "max_range": self.max_range,

            "mount_height": self.mount_height,

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

            horizontal_fov=data.get(
                "horizontal_fov",
                90.0,
            ),

            max_range=data.get(
                "max_range",
                25.0,
            ),

            mount_height=data.get(
                "mount_height",
                3.0,
            ),

            active=data.get(
                "active",
                True,
            ),
        )

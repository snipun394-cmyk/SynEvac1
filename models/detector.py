import math

from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Detector(BaseObject):

    position: tuple = (0.0, 0.0)

    floor_id: str = ""

    coverage_radius: float = 3.0

    mount_height: float = 3.0

    detector_type: str = "Smoke"

    active: bool = True

    # Valid Detector Type values, exposed for the Property Panel
    # combo box rather than hard-coding the list twice.
    DETECTOR_TYPES = (
        "Smoke",
        "Heat",
        "Flame",
        "Gas",
    )

    # =====================================================

    def __post_init__(self):

        self.object_type = "Detector"

    # =====================================================

    def move_to(self, x, y):

        self.position = (x, y)

    # =====================================================
    # Derived Coverage Geometry
    #
    # Never stored -- always recomputed from Position and
    # Coverage Radius. A circle centered on the detector's
    # position (no rotation concept for Detector, unlike Camera).
    # Returned as points around the perimeter so it mirrors
    # Camera.coverage_polygon()'s shape.
    # =====================================================

    def coverage_circle(self, segments=32):

        cx, cy = self.position

        points = []

        for i in range(segments):

            angle_radians = math.radians(360 * i / segments)

            points.append(
                (
                    cx + self.coverage_radius * math.cos(angle_radians),
                    cy + self.coverage_radius * math.sin(angle_radians),
                )
            )

        return points

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "position": self.position,

            "floor_id": self.floor_id,

            "coverage_radius": self.coverage_radius,

            "mount_height": self.mount_height,

            "detector_type": self.detector_type,

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

            coverage_radius=data.get(
                "coverage_radius",
                3.0,
            ),

            mount_height=data.get(
                "mount_height",
                3.0,
            ),

            detector_type=data.get(
                "detector_type",
                "Smoke",
            ),

            active=data.get(
                "active",
                True,
            ),
        )

from dataclasses import dataclass, field

from models.base_object import BaseObject


@dataclass
class Zone(BaseObject):

    # =====================================================
    # Geometry (meters)
    # =====================================================

    x: float = 0.0
    y: float = 0.0

    width: float = 0.0
    height: float = 0.0

    polygon: list = field(default_factory=list)

    # =====================================================
    # Metadata
    # =====================================================

    floor_id: str = ""
    zone_type: str = "Generic"

    max_occupancy: int = 0

    # =====================================================

    def __post_init__(self):

        self.object_type = "Zone"

    # =====================================================
    # Geometry
    # =====================================================

    @property
    def area(self):

        return self.width * self.height

    @property
    def perimeter(self):

        return 2 * (self.width + self.height)

    @property
    def center(self):

        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def top_left(self):

        return (
            self.x,
            self.y,
        )

    @property
    def top_right(self):

        return (
            self.x + self.width,
            self.y,
        )

    @property
    def bottom_right(self):

        return (
            self.x + self.width,
            self.y + self.height,
        )

    @property
    def bottom_left(self):

        return (
            self.x,
            self.y + self.height,
        )

    # =====================================================
    # Operations
    # =====================================================

    def move_to(self, x, y):

        self.x = x
        self.y = y

    def move_by(self, dx, dy):

        self.x += dx
        self.y += dy

    def resize(self, width, height):

        self.width = width
        self.height = height

    def contains(self, x, y):

        return (
            self.x <= x <= self.x + self.width
            and
            self.y <= y <= self.y + self.height
        )

    # =====================================================
    # Serialization
    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "x": self.x,
            "y": self.y,

            "width": self.width,
            "height": self.height,

            "polygon": self.polygon,

            "floor_id": self.floor_id,
            "zone_type": self.zone_type,
            "max_occupancy": self.max_occupancy,

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

            x=data.get(
                "x",
                0.0,
            ),

            y=data.get(
                "y",
                0.0,
            ),

            width=data.get(
                "width",
                0.0,
            ),

            height=data.get(
                "height",
                0.0,
            ),

            polygon=data.get(
                "polygon",
                [],
            ),

            floor_id=data.get(
                "floor_id",
                "",
            ),

            zone_type=data.get(
                "zone_type",
                "Generic",
            ),

            max_occupancy=data.get(
                "max_occupancy",
                0,
            ),
        )
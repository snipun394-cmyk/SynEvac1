from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Obstacle(BaseObject):

    # Top-left, in meters -- same convention as Zone. Not exposed
    # as an editable Property Panel field (the spec lists only
    # Length/Width/Type/Traversability/Traversal Cost/Active);
    # placement/repositioning happens via the Tool and Move.
    x: float = 0.0
    y: float = 0.0

    length: float = 2.0
    width: float = 2.0

    floor_id: str = ""

    obstacle_type: str = "Barrier"

    traversability: str = "Blocked"

    # Used by future simulation/navigation movement-cost
    # calculations. Never interpreted here.
    traversal_cost: float = 1.0

    active: bool = True

    OBSTACLE_TYPES = (
        "Barrier",
        "Furniture",
        "Equipment",
        "Construction",
        "Restricted Area",
        "Other",
    )

    TRAVERSABILITY_OPTIONS = (
        "Passable",
        "Reduced Width",
        "Blocked",
    )

    # =====================================================

    def __post_init__(self):

        self.object_type = "Obstacle"

    # =====================================================

    def move_to(self, x, y):

        self.x = x
        self.y = y

    # =====================================================

    def resize(self, length, width):

        self.length = length
        self.width = width

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update({

            "x": self.x,

            "y": self.y,

            "length": self.length,

            "width": self.width,

            "floor_id": self.floor_id,

            "obstacle_type": self.obstacle_type,

            "traversability": self.traversability,

            "traversal_cost": self.traversal_cost,

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

            x=data.get(
                "x",
                0.0,
            ),

            y=data.get(
                "y",
                0.0,
            ),

            length=data.get(
                "length",
                2.0,
            ),

            width=data.get(
                "width",
                2.0,
            ),

            floor_id=data.get(
                "floor_id",
                "",
            ),

            obstacle_type=data.get(
                "obstacle_type",
                "Barrier",
            ),

            traversability=data.get(
                "traversability",
                "Blocked",
            ),

            traversal_cost=data.get(
                "traversal_cost",
                1.0,
            ),

            active=data.get(
                "active",
                True,
            ),
        )

from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Door(BaseObject):

    start_point: tuple = (0.0, 0.0)
    end_point: tuple = (0.0, 0.0)

    floor_id: str = ""

    width: float = 0.90

    door_type: str = "Standard"

    normally_open: bool = False
    locked: bool = False

    active: bool = True

    # Connectivity -- the two engineering spaces (Zones) this
    # Door joins. Empty string means "not connected yet". Zone
    # touching geometrically never implies connectivity on its
    # own; only an explicit Door creates a traversable link. The
    # Navigation Graph will later derive graph edges from these
    # two references -- never resolved or validated here.
    zone_a_id: str = ""
    zone_b_id: str = ""

    DOOR_TYPES = (
        "Standard",
        "Fire Door",
        "Sliding",
        "Double Leaf",
    )

    # =====================================================

    def __post_init__(self):

        self.object_type = "Door"

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
                "door_type": self.door_type,
                "normally_open": self.normally_open,
                "locked": self.locked,
                "active": self.active,
                "zone_a_id": self.zone_a_id,
                "zone_b_id": self.zone_b_id,
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
                0.90,
            ),

            door_type=data.get(
                "door_type",
                "Standard",
            ),

            normally_open=data.get(
                "normally_open",
                False,
            ),

            locked=data.get(
                "locked",
                False,
            ),

            active=data.get(
                "active",
                True,
            ),

            zone_a_id=data.get(
                "zone_a_id",
                "",
            ),

            zone_b_id=data.get(
                "zone_b_id",
                "",
            ),
        )

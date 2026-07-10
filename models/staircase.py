import math

from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Staircase(BaseObject):

    # Reviewed for a future single-staircase (Staircase + Landing)
    # redesign: today this object lives only in the "from" floor's
    # Floor.stairs and is never rendered on to_floor_id's floor.
    # Adding a `to_position` field (a landing coordinate on the
    # destination floor) needs no migration -- to_dict()/from_dict()
    # already follow the additive-field, default-via-.get() pattern
    # used by every object in this codebase, so older .syn files
    # would simply fall back to the field's default. Ownership can
    # stay on Floor.stairs; GraphicsScene would just need to also
    # render a landing marker on whichever floor matches
    # to_floor_id by scanning the Building it already has a
    # reference to. No serialization changes are required to leave
    # this open.
    start_point: tuple = (0.0, 0.0)
    end_point: tuple = (0.0, 0.0)

    from_floor_id: str = ""
    to_floor_id: str = ""

    width: float = 1.50

    # Project-wide default, until Project exposes a real setting.
    DEFAULT_ANGLE_DEGREES = 35.0

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
    # Derived traversal properties
    #
    # Never stored -- Building remains the single source of
    # truth for elevation (itself derived, never stored -- see
    # Building.floor_elevation()). Both take `building` so they
    # can resolve from_floor_id/to_floor_id themselves rather
    # than the Stair holding a Floor reference. If to_floor_id
    # is unset ("None" chosen deliberately, not auto-picked --
    # see PropertyPanel._populate_to_floor_combo()), this
    # correctly returns 0.0 rather than an arbitrary distance.
    # =====================================================

    def vertical_height(self, building):

        from_floor = building.get_floor(self.from_floor_id)
        to_floor = building.get_floor(self.to_floor_id)

        if from_floor is None or to_floor is None:
            return 0.0

        return abs(
            building.floor_elevation(to_floor)
            - building.floor_elevation(from_floor)
        )

    # =====================================================

    def travel_distance(self, building):

        height = self.vertical_height(building)

        angle_radians = math.radians(
            self.DEFAULT_ANGLE_DEGREES
        )

        return height / math.sin(angle_radians)

    # =====================================================

    def to_dict(self):

        data = super().to_dict()

        data.update(
            {
                "start_point": self.start_point,
                "end_point": self.end_point,
                "from_floor_id": self.from_floor_id,
                "to_floor_id": self.to_floor_id,
                "width": self.width,
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

            from_floor_id=data.get(
                "from_floor_id",
                "",
            ),

            to_floor_id=data.get(
                "to_floor_id",
                "",
            ),

            width=data.get(
                "width",
                1.50,
            ),
        )

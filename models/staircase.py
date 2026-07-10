import math

from dataclasses import dataclass

from models.base_object import BaseObject


@dataclass
class Staircase(BaseObject):

    # A Staircase is one physical connector spanning exactly two
    # floors -- one object, one id, rendered on both floors (see
    # GraphicsScene.rebuild_scene()). from_position/to_position
    # are each in their OWN floor's coordinate space; there is no
    # meaningful single "length" or "center" between them the way
    # there is for a same-floor line like Exit/Door, so those
    # properties (and a unified move()) were deliberately dropped
    # when this was redesigned from a single-floor line.
    #
    # A future "Stairwell" grouping object (multiple flights
    # belonging to one physical stairwell spanning >2 floors) can
    # be layered on top of this without changing what a Staircase
    # is or how Navigation Graph reads it: Stairwell would just
    # hold a list of Staircase ids, each Staircase still being the
    # one flight = one graph edge unit it is today.
    from_position: tuple = (0.0, 0.0)
    to_position: tuple = (0.0, 0.0)

    from_floor_id: str = ""
    to_floor_id: str = ""

    # Connectivity -- the Zone at each end this Staircase actually
    # opens into. Empty string means "not connected yet". Same
    # convention as Door.zone_a_id/zone_b_id and Exit.zone_id: never
    # inferred from geometry (from_position/to_position are just
    # coordinates for drawing, not a zone lookup), only ever set
    # explicitly. The Navigation Graph derives a Zone <-> Zone edge
    # (on two different floors) from these two references -- never
    # resolved or validated here.
    from_zone_id: str = ""
    to_zone_id: str = ""

    width: float = 1.50

    # Project-wide default, until Project exposes a real setting.
    DEFAULT_ANGLE_DEGREES = 35.0

    def __post_init__(self):

        self.object_type = "Staircase"

    # =====================================================
    # Derived traversal properties
    #
    # Never stored -- Building remains the single source of
    # truth for elevation (itself derived, never stored -- see
    # Building.floor_elevation()). Both take `building` so they
    # can resolve from_floor_id/to_floor_id themselves rather
    # than the Stair holding a Floor reference. If to_floor_id
    # is unset, this correctly returns 0.0 rather than an
    # arbitrary distance.
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
                "from_position": self.from_position,
                "to_position": self.to_position,
                "from_floor_id": self.from_floor_id,
                "to_floor_id": self.to_floor_id,
                "from_zone_id": self.from_zone_id,
                "to_zone_id": self.to_zone_id,
                "width": self.width,
            }
        )

        return data

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        # Backward compatibility: pre-redesign .syn files stored
        # the entrance point as "start_point". Those files never
        # had a real cross-floor landing coordinate at all -- their
        # "end_point" was a second point on the SAME floor as
        # start_point, meaningless as a position on to_floor_id --
        # so when "to_position" is missing, default it to the
        # entrance point rather than trusting the old "end_point",
        # which would silently misplace the landing on a floor it
        # was never actually measured against. The user can drag
        # it to the correct spot once, same as any other move.
        from_position = tuple(
            data.get(
                "from_position",
                data.get(
                    "start_point",
                    (0.0, 0.0),
                ),
            )
        )

        to_position = tuple(
            data.get(
                "to_position",
                from_position,
            )
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

            from_position=from_position,

            to_position=to_position,

            from_floor_id=data.get(
                "from_floor_id",
                "",
            ),

            to_floor_id=data.get(
                "to_floor_id",
                "",
            ),

            from_zone_id=data.get(
                "from_zone_id",
                "",
            ),

            to_zone_id=data.get(
                "to_zone_id",
                "",
            ),

            width=data.get(
                "width",
                1.50,
            ),
        )

import math

from dataclasses import dataclass
from typing import Optional, Tuple

from models.base_object import BaseObject


@dataclass
class StairObservableRegion:

    # Observable Stair Perception milestone -- the smallest EXPLICIT,
    # camera-observable footprint a Staircase can carry, ONE PER FLOOR
    # SIDE (see Staircase.from_observable_region/to_observable_region
    # below). This is deliberately NOT a claim about the physical stair
    # tread/riser footprint, and NOT a Zone: it is only the 2D, top-down
    # region within THIS floor's local coordinate plane inside which a
    # calibrated CCTV detection is honestly considered "on this
    # Staircase" for perception purposes -- an axis-aligned rectangle
    # centered on the stair's own from_position/to_position anchor for
    # that side, with an explicit width/depth an operator authors
    # (Designer) or a calibration workflow measures. Never inferred from
    # Staircase.width or any other existing field -- audit's own "no
    # geometry may be fabricated at runtime" requirement (see
    # docs/architecture/stair_camera_perception_audit.md Phase 2).

    center_x: float = 0.0
    center_y: float = 0.0

    width: float = 0.0   # meters, along floor-plan local X
    depth: float = 0.0   # meters, along floor-plan local Y

    def contains(self, x: float, y: float) -> bool:

        half_width = self.width / 2.0
        half_depth = self.depth / 2.0

        return (
            (self.center_x - half_width) <= x <= (self.center_x + half_width)
            and
            (self.center_y - half_depth) <= y <= (self.center_y + half_depth)
        )

    # =====================================================

    def to_dict(self):

        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "depth": self.depth,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        return cls(
            center_x=data.get("center_x", 0.0),
            center_y=data.get("center_y", 0.0),
            width=data.get("width", 0.0),
            depth=data.get("depth", 0.0),
        )


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

    # Observable Stair Perception milestone -- OPTIONAL, one per floor
    # side, each in that side's own floor-local coordinate space (same
    # "no shared coordinate system between from/to" constraint
    # from_position/to_position already document above). None is the
    # honest default and stays None until explicitly authored -- a
    # missing region means STAIR LOCALIZATION UNAVAILABLE for that side,
    # never an invented footprint (see contains_world_point() below and
    # docs/architecture/live_stair_perception.md).
    from_observable_region: Optional[StairObservableRegion] = None
    to_observable_region: Optional[StairObservableRegion] = None

    # Project-wide default, until Project exposes a real setting.
    DEFAULT_ANGLE_DEGREES = 35.0

    def __post_init__(self):

        self.object_type = "Staircase"

    # =====================================================
    # Observable region lookup (Observable Stair Perception milestone)
    #
    # A Staircase is one object with a DIFFERENT observable region per
    # floor side (mirrors from_position/to_position's own per-side
    # convention) -- observable_region_for_floor() picks the correct one
    # for a given floor_id, or None if that floor_id is neither end of
    # this Staircase, or that side simply has no region authored yet.
    # This is what makes a "wrong floor" query naturally return no
    # match, without any caller needing to check floor identity itself.
    # =====================================================

    def observable_region_for_floor(self, floor_id: str) -> Optional[StairObservableRegion]:

        if floor_id == self.from_floor_id:
            return self.from_observable_region

        if floor_id == self.to_floor_id:
            return self.to_observable_region

        return None

    # =====================================================

    def contains_world_point(self, floor_id: str, world_position: Tuple[float, float]) -> bool:

        region = self.observable_region_for_floor(floor_id)

        if region is None:
            return False

        x, y = world_position

        return region.contains(x, y)

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
                "from_observable_region": (
                    self.from_observable_region.to_dict()
                    if self.from_observable_region is not None
                    else None
                ),
                "to_observable_region": (
                    self.to_observable_region.to_dict()
                    if self.to_observable_region is not None
                    else None
                ),
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

            # Observable Stair Perception milestone -- absent in every
            # .syn file predating this milestone (and legitimately absent
            # in any project where an operator has not authored one yet).
            # None, never a fabricated default region -- see
            # StairObservableRegion's own docstring.
            from_observable_region=(
                StairObservableRegion.from_dict(data["from_observable_region"])
                if data.get("from_observable_region") is not None
                else None
            ),

            to_observable_region=(
                StairObservableRegion.from_dict(data["to_observable_region"])
                if data.get("to_observable_region") is not None
                else None
            ),
        )

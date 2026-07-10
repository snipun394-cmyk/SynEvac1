from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from models.floor import Floor


@dataclass
class Building:

    name: str

    id: str = field(default_factory=lambda: str(uuid4()))

    floors: list[Floor] = field(default_factory=list)

    # =====================================================

    def add_floor(self, floor: Floor):

        self.floors.append(floor)

    # =====================================================

    def remove_floor(self, floor: Floor):

        if floor in self.floors:
            self.floors.remove(floor)

    # =====================================================

    def get_floor(self, floor_id: str):

        for floor in self.floors:

            if floor.id == floor_id:
                return floor

        return None

    # =====================================================

    @property
    def floor_count(self):

        return len(self.floors)

    # =====================================================
    # Ordering
    #
    # display_order, not list position, is the source of
    # truth for floor sequence. Consumers should read floor
    # order through ordered_floors() rather than iterating
    # self.floors directly.
    # =====================================================

    def ordered_floors(self):

        return sorted(
            self.floors,
            key=lambda floor: floor.display_order,
        )

    # =====================================================

    def _next_display_order(self):

        if not self.floors:
            return 0

        return max(
            floor.display_order
            for floor in self.floors
        ) + 1

    # =====================================================
    # Elevation
    #
    # Never stored on Floor -- always derived from the
    # cumulative height of every floor below this one in
    # display_order. The lowest floor therefore always resolves
    # to 0.0 with no special-casing required. Reordering floors
    # or editing any floor's height automatically changes every
    # elevation above it, since nothing here is cached.
    # =====================================================

    def floor_elevation(self, floor):

        if floor not in self.floors:
            return 0.0

        elevation = 0.0

        for other in self.ordered_floors():

            if other is floor:
                return elevation

            elevation += other.height

        return elevation

    # =====================================================
    # Floor Management
    # =====================================================

    def create_floor(self, name, **kwargs):

        floor = Floor(
            name=name,
            display_order=self._next_display_order(),
            **kwargs,
        )

        self.add_floor(floor)

        return floor

    # =====================================================

    def rename_floor(self, floor, name):

        if floor not in self.floors:
            return

        floor.rename(name)

    # =====================================================

    def delete_floor(self, floor):

        if floor not in self.floors:
            return None

        ordered_before = self.ordered_floors()
        index = ordered_before.index(floor)

        self.remove_floor(floor)

        ordered_after = self.ordered_floors()

        if not ordered_after:
            return None

        # Prefer the floor that was directly below the
        # deleted one; fall back to whatever now occupies
        # its position. This is a suggestion only -- Building
        # does not track "the" active floor itself, that
        # remains session/UI state (see architecture review).
        fallback_index = max(
            0,
            min(index - 1, len(ordered_after) - 1),
        )

        return ordered_after[fallback_index]

    # =====================================================

    def duplicate_floor(self, floor, name=None):

        if floor not in self.floors:
            return None

        # Implementation detail only: reuses to_dict()/
        # from_dict() for this milestone's deep copy. Not a
        # pattern to be relied on by other systems -- a
        # dedicated clone mechanism is expected later.
        new_floor = Floor.from_dict(floor.to_dict())

        new_floor.id = str(uuid4())
        new_floor.name = name or f"{floor.name} (Copy)"
        new_floor.display_order = self._next_display_order()

        now = datetime.now().isoformat()

        # Internal same-floor references (Door.zone_a_id/zone_b_id)
        # must follow their target onto the duplicate -- otherwise
        # a duplicated Door would still point at the *original*
        # floor's zones, a dangling reference this method should
        # never produce. Build the id remap before the re-ID loop
        # touches anything, so both the old and new zone ids are
        # known before any zone gets its new id assigned. Any
        # future object type that references another object on the
        # same floor should follow this same two-step shape: map
        # first, then patch after the main re-ID loop below.
        zone_id_map = {
            zone.id: str(uuid4())
            for zone in new_floor.zones
        }

        for collection in (
            new_floor.zones,
            new_floor.exits,
            new_floor.stairs,
            new_floor.elevators,
            new_floor.cameras,
            new_floor.detectors,
            new_floor.assembly_points,
            new_floor.obstacles,
            new_floor.doors,
        ):

            for engineering_object in collection:

                if collection is new_floor.zones:
                    engineering_object.id = zone_id_map[engineering_object.id]
                else:
                    engineering_object.id = str(uuid4())

                engineering_object.floor_id = new_floor.id
                engineering_object.created_at = now
                engineering_object.modified_at = now

        for door in new_floor.doors:

            if door.zone_a_id in zone_id_map:
                door.zone_a_id = zone_id_map[door.zone_a_id]

            if door.zone_b_id in zone_id_map:
                door.zone_b_id = zone_id_map[door.zone_b_id]

        self.add_floor(new_floor)

        return new_floor

    # =====================================================

    def move_floor_up(self, floor):

        if floor not in self.floors:
            return

        ordered = self.ordered_floors()
        index = ordered.index(floor)

        if index == 0:
            return

        other = ordered[index - 1]

        floor.display_order, other.display_order = (
            other.display_order,
            floor.display_order,
        )

    # =====================================================

    def move_floor_down(self, floor):

        if floor not in self.floors:
            return

        ordered = self.ordered_floors()
        index = ordered.index(floor)

        if index == len(ordered) - 1:
            return

        other = ordered[index + 1]

        floor.display_order, other.display_order = (
            other.display_order,
            floor.display_order,
        )

    # =====================================================

    def to_dict(self):

        return {
            "id": self.id,
            "name": self.name,
            "floors": [
                floor.to_dict()
                for floor in self.floors
            ],
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data):

        building = cls(
            id=data["id"],
            name=data["name"],
        )

        for floor_data in data.get("floors", []):

            building.add_floor(
                Floor.from_dict(floor_data)
            )

        return building

# Registry of navigable-space types a Door can connect.
#
# Door.zone_a_id/zone_b_id (see models/door.py) already just store a
# plain object id -- they always have, and that never needed to
# change here. What used to be missing was a way to resolve that id
# against more than one Floor collection (only floor.zones) and to
# describe the object generically in the UI. This module is the one
# place that knows the full list of connectable types and how to find
# them; adding a future type (Outdoor Area, Refuge Area, Stair Lobby)
# means adding one entry here, not touching Door, the Property Panel,
# or the Navigation Graph builder again.

ZONE = "Zone"
ASSEMBLY_POINT = "AssemblyPoint"

CONNECTABLE_SPACE_TYPES = (
    ZONE,
    ASSEMBLY_POINT,
)

# object_type string -> human-readable label. Keys match the exact
# object_type each model's __post_init__ already sets (Zone ->
# "Zone", AssemblyPoint -> "AssemblyPoint"), so this is the only
# place display wording ("Assembly Point", with a space) diverges
# from the internal type string.
SPACE_TYPE_LABELS = {
    ZONE: "Zone",
    ASSEMBLY_POINT: "Assembly Point",
}


def spaces_of_type(floor, space_type):

    if space_type == ZONE:
        return floor.zones

    if space_type == ASSEMBLY_POINT:
        return floor.assembly_points

    return []


def all_connectable_spaces(floor):

    # Yields (space_type, object) for every connectable space on a
    # floor, Zones first then Assembly Points -- a stable order for
    # combo boxes and node generation alike.
    for space_type in CONNECTABLE_SPACE_TYPES:

        for obj in spaces_of_type(floor, space_type):

            yield space_type, obj


def label_for(space_type, name):

    return (
        f"{SPACE_TYPE_LABELS.get(space_type, space_type)}: "
        f"{name}"
    )

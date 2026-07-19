import math

from visibility.geometry import (
    EPSILON,
    closest_point_on_segment,
    perpendicular_distance_to_line,
)


# How close a Door/Exit's own center point must project to a Zone's
# perimeter edge to be treated as the opening carved into that edge.
# A generous-but-not-unlimited tolerance: Doors/Exits are drawn by the
# user roughly on the wall between the two spaces they connect (see
# navigation/graph_builder.py's own door/exit walking-distance
# comments), not always pixel-perfect on the Zone rectangle's exact
# boundary line.
DOOR_WALL_TOLERANCE_M = 0.5


def zone_perimeter_edges(zone):

    # Zone is always an axis-aligned rectangle for engineering
    # purposes (Zone.contains()/center/corners all already work this
    # way -- see models/zone.py; the separate `polygon` field is a
    # Designer-only vertex-editing concern, not read by any
    # engineering model). Four edges, corner to corner, in order.

    corners = [
        zone.top_left,
        zone.top_right,
        zone.bottom_right,
        zone.bottom_left,
    ]

    return [
        (corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]


def _openings_for_zone(zone, doors, exits):

    # A Door/Exit is an opening in THIS zone's wall only when it is
    # explicitly connected to it -- same "connectivity is explicit,
    # never inferred from geometry" convention Door.zone_a_id/
    # zone_b_id and Exit.zone_id already establish for the Navigation
    # Graph (see navigation/graph_builder.py). Geometry only decides
    # *where on the perimeter* the opening falls, never *whether*
    # one exists.

    openings = []

    for door in doors:

        if door.zone_a_id == zone.id or door.zone_b_id == zone.id:
            openings.append((door.center, door.width))

    for exit_obj in exits:

        if exit_obj.zone_id == zone.id:
            openings.append((exit_obj.center, exit_obj.width))

    return openings


def _edge_length(edge):

    (x1, y1), (x2, y2) = edge

    return math.hypot(x2 - x1, y2 - y1)


def _carve_edge(edge, openings):

    # Returns the opaque sub-segments of `edge` remaining after
    # removing a gap for every opening whose center projects close
    # enough to this edge (see DOOR_WALL_TOLERANCE_M). An opening
    # belonging to this zone but actually drawn against a different
    # edge of it (e.g. a door on the north wall while this is the
    # south wall) simply doesn't project close enough here and is
    # correctly ignored for this edge.

    edge_start, edge_end = edge

    length = _edge_length(edge)

    if length < EPSILON:
        return []

    gaps = []

    for center, width in openings:

        _, _, t = closest_point_on_segment(center, edge_start, edge_end)
        distance = perpendicular_distance_to_line(center, edge_start, edge_end)

        if distance > DOOR_WALL_TOLERANCE_M:
            continue

        half_t = (width / 2) / length

        gaps.append((max(0.0, t - half_t), min(1.0, t + half_t)))

    if not gaps:
        return [edge]

    gaps.sort()

    merged = [list(gaps[0])]

    for gap_start, gap_end in gaps[1:]:

        if gap_start <= merged[-1][1] + EPSILON:
            merged[-1][1] = max(merged[-1][1], gap_end)
        else:
            merged.append([gap_start, gap_end])

    x1, y1 = edge_start
    x2, y2 = edge_end
    dx, dy = x2 - x1, y2 - y1

    def point_at(t):
        return (x1 + dx * t, y1 + dy * t)

    segments = []
    cursor = 0.0

    for gap_start, gap_end in merged:

        if gap_start > cursor + EPSILON:
            segments.append((point_at(cursor), point_at(gap_start)))

        cursor = max(cursor, gap_end)

    if cursor < 1.0 - EPSILON:
        segments.append((point_at(cursor), point_at(1.0)))

    return segments


def zone_wall_segments(zone, doors, exits):

    openings = _openings_for_zone(zone, doors, exits)

    segments = []

    for edge in zone_perimeter_edges(zone):
        segments.extend(_carve_edge(edge, openings))

    return segments


def obstacle_segments(obstacle):

    # Only a "Blocked" Obstacle is opaque to sight -- "Passable"/
    # "Reduced Width" already mean a person (and by the same physical
    # reasoning, a sightline) can get through it. No gap-carving here:
    # unlike a Zone wall, an Obstacle has no Door/Exit concept cut
    # into it.

    if obstacle.traversability != "Blocked":
        return []

    x, y = obstacle.x, obstacle.y

    corners = [
        (x, y),
        (x + obstacle.length, y),
        (x + obstacle.length, y + obstacle.width),
        (x, y + obstacle.width),
    ]

    return [
        (corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]


def floor_opaque_segments(floor):

    # Every wall/barrier segment on one floor a camera's line of
    # sight can be blocked by -- Zone perimeters (minus their Door/
    # Exit openings) plus Blocked Obstacles. Reused as-is by
    # VisibilityEngine; never mutated, never cached across calls (a
    # Floor's zones/doors/exits/obstacles can change between two
    # calls, and this is cheap enough to rebuild every time -- see
    # visibility/engine.py's own performance notes).

    segments = []

    for zone in floor.zones:
        segments.extend(zone_wall_segments(zone, floor.doors, floor.exits))

    for obstacle in floor.obstacles:
        segments.extend(obstacle_segments(obstacle))

    return segments
